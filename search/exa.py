"""Async Exa Search implementation: screened evidence, persistent cache and audit.

SDK/API: https://exa.ai/docs/reference/search
Price fallback: https://exa.ai/docs/admin/pricing (checked 2026-11-14).
"""

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from core.config import RunConfig
from core.schemas import Evidence
from search.policy import PolicyDenied, SearchPolicy

# Auto search, <=10 results, highlights, no summaries. Override as pricing changes.
EXA_ESTIMATED_COST_USD = 0.007


class SearchUnavailable(RuntimeError):
    """Search failed; this is distinct from a successful search without usable evidence."""

    def __init__(self, message: str, cost_reservation_usd: float = 0):
        super().__init__(message)
        self.cost_reservation_usd = cost_reservation_usd


class SearchSDK(Protocol):
    async def search(self, query: str, **kwargs): ...


class SearchResult(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    cost_usd: float = 0
    cost_basis: str = "cache"
    cache_hit: bool = False
    request_id: str | None = None
    blocked: list[str] = Field(default_factory=list)  # URLs the policy removed


class ExaSearch:
    """One run-scoped adapter for either the swarm or the single-agent baseline.

    find() implements ERD's Search interface. lookup() additionally returns cost and
    cache metadata for the experiment runner. No remote requests occur when
    ``config.search`` is false. ``estimated_cost_usd`` is the price recorded when Exa does
    not report one. Use an async context manager or call aclose(). Injected SDK clients
    remain caller-owned.
    """

    def __init__(
        self,
        policy: SearchPolicy,
        run_id: str,
        root: Path = Path("runs"),
        *,
        config: RunConfig | None = None,
        client: SearchSDK | None = None,
        timeout_s: float = 20,
        author_role: str = "experimenter",
        estimated_cost_usd: float = EXA_ESTIMATED_COST_USD,
    ):
        if not run_id or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("A run ID and positive finite timeout are required")
        if not math.isfinite(estimated_cost_usd) or estimated_cost_usd < 0:
            raise ValueError("The estimated cost must be finite and non-negative")
        self.policy = policy
        self.run_id = run_id
        self.author_role = author_role
        self.config = config or RunConfig()
        self.estimated_cost_usd = float(estimated_cost_usd)
        self._timeout = timeout_s
        self._client = client
        self._owns_client = client is None
        self._lock = asyncio.Lock()
        self._closed = False
        scope = hashlib.sha256(run_id.encode()).hexdigest()[:32]
        directory = Path(root) / scope
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "search.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                type TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL UNIQUE, spec TEXT NOT NULL,
                observation TEXT
            );
        """)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    def audit(self, event_type: str, payload: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO audit(ts,type,payload) VALUES(?,?,?)",
                (time.time(), event_type, json.dumps({"run_id": self.run_id, **payload})),
            )

    def _sdk(self):
        if self._client is None:
            key = os.environ.get("EXA_API_KEY")
            if not key:
                raise SearchUnavailable("Set EXA_API_KEY in the server environment")
            from exa_py import AsyncExa

            self._client = AsyncExa(api_key=key, timeout=self._timeout)
        return self._client

    async def find(self, query: str, k: int = 5) -> list[Evidence]:
        return (await self.lookup(query, k)).evidence

    async def lookup(self, query: str, k: int = 5) -> SearchResult:
        if self._closed:
            raise RuntimeError("Search is closed")
        if not 1 <= k <= 10:
            raise ValueError("Use between 1 and 10 documentation results")
        if not self.config.search:
            self.audit("search.disabled", {"query": query})
            return SearchResult(cost_basis="disabled")
        started = time.monotonic()
        try:
            query = self.policy.check_query(query)
        except PolicyDenied as exc:
            self.audit("search.blocked", {"query": query, "reason": str(exc)})
            raise
        request = {
            "query": query,
            "num_results": k,
            "type": "auto",
            "contents": {"highlights": True},
            **self.policy.sdk_filters(),
        }
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "request": request,
                    "policy": self.policy.fingerprint,
                    "author_role": self.author_role,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        async with self._lock:
            cached = self.db.execute(
                "SELECT result FROM cache WHERE key=?", (cache_key,)
            ).fetchone()
            if cached:
                original = SearchResult.model_validate_json(cached["result"])
                result = original.model_copy(
                    update={
                        "cost_usd": 0.0,
                        "cost_basis": "cache",
                        "cache_hit": True,
                    },
                    deep=True,
                )
            else:
                attempted = False
                try:
                    async with asyncio.timeout(self._timeout):
                        sdk = self._sdk()
                        attempted = True
                        response = await sdk.search(**request)
                    result = self._screen(response, query, cache_key)
                except asyncio.CancelledError:
                    self.audit(
                        "search.error", {"query": query, "error": "cancelled", "cost_usd": None}
                    )
                    raise
                except Exception as exc:
                    self.audit(
                        "search.error",
                        {
                            "query": query,
                            "error": type(exc).__name__,
                            "cost_usd": None,
                            "cost_reservation_usd": self.estimated_cost_usd
                            if attempted
                            else 0,
                        },
                    )
                    if isinstance(exc, SearchUnavailable):
                        raise
                    raise SearchUnavailable(
                        "Documentation search failed; inspect the local audit",
                        self.estimated_cost_usd if attempted else 0,
                    ) from exc
                with self.db:
                    self.db.execute(
                        "INSERT INTO cache(key,result) VALUES(?,?)",
                        (cache_key, result.model_dump_json()),
                    )
            self.audit(
                "search.done",
                {
                    "query": query,
                    "urls": [ev.url for ev in result.evidence],
                    "evidence_ids": [ev.id for ev in result.evidence],
                    "cache_hit": result.cache_hit,
                    "cost_usd": result.cost_usd,
                    "cost_basis": result.cost_basis,
                    "request_id": result.request_id,
                    "policy_hash": self.policy.fingerprint,
                    "seconds": time.monotonic() - started,
                },
            )
            return result

    def _screen(self, response, query: str, cache_key: str) -> SearchResult:
        cost = getattr(getattr(response, "cost_dollars", None), "total", None)
        known_cost = isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0
        result = SearchResult(
            cost_usd=float(cost) if known_cost else self.estimated_cost_usd,
            cost_basis="provider_estimate" if known_cost else "configured_estimate",
            request_id=getattr(response, "request_id", None),
        )
        seen = set()
        for hit in response.results:
            url = hit.url
            try:
                self.policy.check_url(url)
                highlights = hit.highlights or []
                if not isinstance(highlights, list) or any(
                    not isinstance(h, str) for h in highlights
                ):
                    raise ValueError("Malformed highlights")
                claim = "\n".join(highlights)
                if not claim.strip():
                    raise PolicyDenied("No source highlights")
                if len(claim) > 16000:
                    raise PolicyDenied("Highlights exceed the evidence budget")
                self.policy.check_text((hit.title or "") + "\n" + claim)
            except PolicyDenied as exc:
                self.audit(
                    "search.result_blocked", {"query": query, "url": url, "reason": str(exc)}
                )
                if isinstance(url, str) and url not in result.blocked:
                    result.blocked.append(url)
                continue
            self.audit("search.result_allowed", {"query": query, "url": url})
            if url in seen:
                continue
            seen.add(url)
            digest = hashlib.sha256((cache_key + url + claim).encode()).hexdigest()
            result.evidence.append(
                Evidence(
                    id="search-" + digest,
                    kind="search",
                    claim=claim,
                    author_role=self.author_role,
                    source_ref=getattr(hit, "id", None) or url,
                    url=url,
                    conditions={
                        "query": query,
                        "policy_hash": self.policy.fingerprint,
                        "content_hash": hashlib.sha256(claim.encode()).hexdigest(),
                    },
                )
            )
        return result

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_client and self._client is not None:
                await self._client.aclose()
            self.db.close()
