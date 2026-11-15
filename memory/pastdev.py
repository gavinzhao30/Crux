"""past.dev Memory implementation with local-first durability and run-scoped readers.

API contract: https://past.dev/docs/memory-api/api-reference
Access scopes: https://past.dev/docs/memory-api/access-control
"""

import asyncio
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from core.schemas import Evidence
from memory.base import (
    ContentRejected,
    IngestionFailed,
    MemoryRecord,
    RecallUnavailable,
    RunContext,
)
from memory.instrumentation import MemoryTelemetry
from memory.store import MemoryStore

API_BASE = "https://api.past.dev/api/v1"


class _Receipt(BaseModel):
    ingestionId: str = Field(min_length=1)
    status: Literal["queued", "processing", "completed", "failed"]


class _Source(BaseModel):
    sourceId: str


class _RecallResult(BaseModel):
    sources: list[_Source] = Field(default_factory=list)
    artifact: dict = Field(default_factory=dict)


class _RecallResponse(BaseModel):
    results: list[_RecallResult]


class _Status(BaseModel):
    status: Literal["queued", "processing", "completed", "failed"]
    failure: dict | None = None


class PastMemory:
    """Use as an async context manager, or explicitly call aclose().

    write()/recall() implement ERD section 5's Memory interface. write_record()
    adds provenance, corrections and mandatory reviews without changing Evidence.
    Start one adapter per run/config/repeat. A successful write means durable locally,
    not indexed remotely. flush() waits for remote readiness and raises on failure.
    """

    def __init__(
        self,
        context: RunContext,
        root: Path = Path("runs"),
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 30,
        poll_interval_s: float = 0.2,
        max_tokens: int = 1500,
        background: bool = True,
        telemetry: MemoryTelemetry | None = None,
    ):
        if timeout_s <= 0 or poll_interval_s < 0 or max_tokens < 1:
            raise ValueError("Invalid memory timeout, poll interval or evidence budget")
        key = os.environ.get("PAST_API_KEY")
        if not key:
            raise ValueError("Set PAST_API_KEY in the server environment")
        self.context = context
        self.telemetry = telemetry or MemoryTelemetry("pastdev")
        self.store = MemoryStore(root, context)
        self._headers = {"Authorization": f"Bearer {key}"}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=10)
        self._timeout = timeout_s
        self._poll_interval = poll_interval_s
        self._max_tokens = max_tokens
        self._background = background
        self._scope_ready = False
        self._scope_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._submit_lock = asyncio.Lock()
        self._worker: asyncio.Task | None = None
        self._closed = False

    async def __aenter__(self):
        self._start_worker()
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    def _check_run(self, run_id: str) -> None:
        if self._closed:
            raise RuntimeError("Memory is closed")
        if run_id != self.context.run_id:
            raise ValueError("Memory adapter is bound to another run")

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        started = time.perf_counter()
        try:
            response = await self._client.request(
                method, API_BASE + path, json=body, headers=self._headers
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object from past.dev")  # noqa: TRY004
        except (httpx.HTTPError, ValueError) as exc:
            self.store.audit(
                "memory.request",
                {
                    "method": method,
                    "path": path,
                    "request": body,
                    "error": type(exc).__name__,
                    "seconds": time.perf_counter() - started,
                },
            )
            raise
        self.store.audit(
            "memory.request",
            {
                "method": method,
                "path": path,
                "request": body,
                "response": result,
                "seconds": time.perf_counter() - started,
            },
        )
        return result

    async def _ensure_scope(self) -> None:
        async with self._scope_lock:
            if not self._scope_ready:
                await self._request(
                    "PUT",
                    f"/audiences/{self.context.audience}",
                    {
                        "kind": "fixed",
                        "name": self.context.audience,
                        "identities": [self.context.identity(r) for r in self.context.roles],
                    },
                )
                self._scope_ready = True

    async def write(self, ev: Evidence, run_id: str, author_role: str) -> None:
        started = time.perf_counter()
        self._check_run(run_id)
        self.context.identity(author_role)
        if ev.author_role != author_role:
            raise ValueError("Evidence author must match the writing role")
        previous = self.store.by_evidence(ev.id)
        if previous:
            if previous.evidence != ev:
                raise ValueError("Evidence IDs are immutable; use a new ID with supersedes")
            self._start_worker()
            await self.telemetry.written(
                self.store, ev, time.perf_counter() - started, idempotent=True
            )
            return
        await self.write_record(
            MemoryRecord(
                event_id=ev.id,
                run_id=run_id,
                task_id=self.context.task_id,
                evidence=ev,
                event_type=ev.kind,
                evidence_ids=[ev.id],
                mandatory=ev.kind == "objection",
                input_hash=ev.conditions.get("input_hash"),
                code_hash=ev.conditions.get("code_hash"),
                environment_id=ev.conditions.get("environment_id"),
            )
        )

    async def write_record(self, record: MemoryRecord) -> None:
        started = time.perf_counter()
        self._check_run(record.run_id)
        if record.task_id != self.context.task_id:
            raise ValueError("Memory record belongs to another task")
        if not record.evidence.id.strip():
            raise ValueError("Evidence ID must not be empty")
        if (
            record.evidence.kind == "objection"
            or record.current_status == "invalid_analysis"
            or record.supersedes
        ):
            record = record.model_copy(update={"mandatory": True})
        author = self.context.identity(record.evidence.author_role)
        if record.evidence.id in record.supersedes:
            raise ValueError("A record cannot supersede itself")
        for evidence_id in record.supersedes:
            if self.store.by_evidence(evidence_id) is None:
                raise ValueError("A superseded evidence ID must exist in this run")
        content = record.model_dump_json(indent=2)
        if len(content) >= 16000:
            raise ValueError("Memory records must be under 16000 characters; use artifact pointers")
        source_id = (
            self.context.audience + ":ev:" + hashlib.sha256(record.evidence.id.encode()).hexdigest()
        )
        body = {
            "id": source_id,
            "content": content,
            "audience": self.context.audience,
            "identity": author,
            "timestamp": record.occurred_at.isoformat(),
            "label": f"{record.evidence.kind} by {record.evidence.author_role}",
            "metadata": {
                "run_id": record.run_id,
                "task_id": record.task_id,
                "event_id": record.event_id,
                "evidence_id": record.evidence.id,
            },
        }
        body["idempotencyKey"] = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()
        ).hexdigest()
        self.store.append(record, source_id, body)
        await self.telemetry.written(self.store, record.evidence, time.perf_counter() - started)
        self._start_worker()

    def _start_worker(self) -> None:
        if self._background and not self._closed and (not self._worker or self._worker.done()):
            self._worker = asyncio.create_task(self._upload_loop())

    async def _upload_loop(self) -> None:
        delay = 0.5
        while self.store.pending():
            try:
                await self.flush()
                delay = 0.5
            except ContentRejected:
                self.store.audit("memory.upload_blocked", {"reason": "content-rejected"})
                return
            except (httpx.HTTPError, ValueError, IngestionFailed, TimeoutError) as exc:
                self.store.audit("memory.upload_deferred", {"error": type(exc).__name__})
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def submit(self) -> None:
        """Send every queued body that has no ingestion receipt yet, without waiting for
        indexing. The provider indexes sent records on its side; receipts stay in the
        outbox so a later flush can confirm them. Bounded by the adapter timeout."""
        self._check_run(self.context.run_id)
        async with asyncio.timeout(self._timeout):
            await self._ensure_scope()
            await self._submit_unsent()

    async def _submit_unsent(self) -> None:
        async with self._submit_lock:
            for row in self.store.pending():
                if row["status"] == "rejected" or row["ingestion_id"] is not None:
                    continue
                try:
                    receipt = _Receipt.model_validate(
                        await self._request("POST", "/ingest", json.loads(row["body"]))
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    self.store.progress(
                        row["source_id"], attempts=row["attempts"] + 1, error=type(exc).__name__
                    )
                    raise
                self.store.progress(
                    row["source_id"], ingestion_id=receipt.ingestionId, status=receipt.status
                )

    async def flush(self) -> None:
        """Drain the durable queue with a bounded wait; safe to retry after a restart."""
        self._check_run(self.context.run_id)
        async with self._flush_lock, asyncio.timeout(self._timeout):
            if not self.store.pending():
                return
            await self._ensure_scope()
            # Send every new body before polling any, so none waits behind another's poll.
            await self._submit_unsent()
            rejected = False
            for row in self.store.pending():
                source_id = row["source_id"]
                if row["status"] == "rejected":
                    rejected = True
                    continue
                self.store.progress(source_id, attempts=row["attempts"] + 1)
                try:
                    status = row["status"]
                    ingestion_id = row["ingestion_id"]
                    if ingestion_id is None:
                        receipt = _Receipt.model_validate(
                            await self._request("POST", "/ingest", json.loads(row["body"]))
                        )
                        ingestion_id = receipt.ingestionId
                        status = receipt.status
                        self.store.progress(source_id, ingestion_id=ingestion_id, status=status)
                    if status == "failed":
                        await self._request("POST", f"/ingest/{quote(ingestion_id, safe='')}/retry")
                    while status != "completed":
                        result = _Status.model_validate(
                            await self._request("GET", f"/ingest/{quote(ingestion_id, safe='')}")
                        )
                        status = result.status
                        self.store.progress(source_id, status=status)
                        if status == "failed":
                            code = (result.failure or {}).get("code", "unknown")
                            if code == "content-rejected":
                                self.store.progress(source_id, status="rejected", error=code)
                                rejected = True
                                break
                            raise IngestionFailed(f"Ingestion failed: {code}")
                        if status != "completed":
                            # Records written while this one indexes are sent meanwhile.
                            await self._submit_unsent()
                            await asyncio.sleep(self._poll_interval)
                    if status == "completed":
                        self.store.progress(source_id, status="completed", error=None)
                except (httpx.HTTPError, ValueError, IngestionFailed, TimeoutError) as exc:
                    self.store.progress(source_id, error=type(exc).__name__)
                    raise
            if rejected:
                raise ContentRejected("Rejected content needs a new corrected event")

    async def recall(self, query: str, run_id: str, reader_role: str, k: int = 8) -> list[Evidence]:
        self._check_run(run_id)
        identity = self.context.identity(reader_role)
        if not query.strip() or not 1 <= k <= 200:
            raise ValueError("Recall needs a query and k between 1 and 200")
        self._start_worker()
        started = time.perf_counter()
        remote_ids: list[str] = []
        failed = False
        try:
            async with asyncio.timeout(self._timeout):
                await self._ensure_scope()
                response = _RecallResponse.model_validate(
                    await self._request(
                        "POST",
                        "/recall",
                        {
                            "query": query,
                            "identity": identity,
                            "limit": k,
                            "maxTokens": self._max_tokens,
                        },
                    )
                )
                # Only source IDs resolving to this run's canonical records are trusted.
                for result in response.results:
                    if not result.artifact.get("supersededAt"):
                        remote_ids.extend(s.sourceId for s in result.sources)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, TimeoutError):
            failed = True
        rows = self.store.eligible()
        mandatory, pending, recalled = [], [], []
        by_source = {row["source_id"]: row for row in rows}
        for row in rows:
            record = MemoryRecord.model_validate_json(row["record"])
            if record.mandatory:
                mandatory.append(record.evidence)
            elif row["status"] != "completed":
                pending.append(record.evidence)
        for source_id in remote_ids:
            if source_id in by_source:
                recalled.append(
                    MemoryRecord.model_validate_json(by_source[source_id]["record"]).evidence
                )
        # Mandatory reviews are never silently truncated. Pending data is an explicit
        # freshness buffer, not a second semantic retriever. Remote ranking is kept.
        selected = {ev.id: ev for ev in reversed(mandatory)}
        for ev in pending + recalled:
            if len(selected) >= max(k, len(mandatory)):
                break
            selected.setdefault(ev.id, ev)
        await self.telemetry.recalled(
            self.store,
            reader_role,
            query,
            list(selected.values()),
            time.perf_counter() - started,
            source_ids=remote_ids,
            fallback="pending_and_mandatory" if failed else None,
            remote_status="failed" if failed else "ok",
        )
        if failed and not selected:
            raise RecallUnavailable("Remote recall failed; no pending or mandatory local evidence")
        return [ev.model_copy(deep=True) for ev in selected.values()]

    async def aclose(self) -> None:
        """Stop uploads without discarding pending work; resume via flush on restart."""
        if self._closed:
            return
        self._closed = True
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        if self._owns_client:
            await self._client.aclose()
        self.store.close()
