"""Identical memory-prompt formatting and measurement for the two required arms."""

import hashlib
import inspect
import json
import math
from collections.abc import Callable

from core.schemas import Evidence
from memory.store import MemoryStore


def format_memory(evidence: list[Evidence]) -> str:
    """This exact fragment is counted and can be inserted into the agent prompt."""
    if not evidence:
        return ""
    return json.dumps(
        [ev.model_dump(mode="json") for ev in evidence],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


class MemoryTelemetry:
    """Optional exact tokenizer and event callback; default counts are labeled estimates.

    The callback receives EVENTS.md-compatible payloads plus measurement fields.
    It may be sync or async. Sink failures are logged without breaking scientific work.
    """

    def __init__(
        self,
        backend: str,
        *,
        token_counter: Callable[[str], int] | None = None,
        token_method: str = "estimate_utf8_bytes_div4",
        on_event: Callable | None = None,
    ):
        if token_counter is not None and token_method == "estimate_utf8_bytes_div4":
            raise ValueError("Name the tokenizer used by an exact token counter")
        if token_counter is None and token_method != "estimate_utf8_bytes_div4":
            raise ValueError("An exact tokenizer needs a token counter")
        self.backend = backend
        self.token_counter = token_counter
        self.token_method = token_method
        self.on_event = on_event
        self.events: list[tuple[str, dict]] = []

    async def _emit(self, store: MemoryStore, event: str, payload: dict) -> None:
        payload = {**payload, "backend": self.backend}
        store.audit(event, payload)
        self.events.append((event, payload))
        if self.on_event:
            try:
                result = self.on_event(event, json.loads(json.dumps(payload)))
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - a sink must not break scientific work
                store.audit("memory.event_sink_error", {"error": type(exc).__name__})

    async def written(
        self, store: MemoryStore, evidence: Evidence, seconds: float, **extra
    ) -> None:
        await self._emit(
            store,
            "memory.written",
            {
                "run_id": store.context.run_id,
                "evidence_id": evidence.id,
                "author_role": evidence.author_role,
                "seconds": seconds,
                "latency_ms": seconds * 1000,
                "durability": "local",
                **extra,
            },
        )

    async def recalled(
        self,
        store: MemoryStore,
        reader: str,
        query: str,
        evidence: list[Evidence],
        seconds: float,
        **extra,
    ) -> dict:
        prompt = format_memory(evidence)
        count = (
            self.token_counter(prompt)
            if self.token_counter
            else math.ceil(len(prompt.encode("utf-8")) / 4)
        )
        if not isinstance(count, int) or count < 0:
            raise ValueError("Token counters must return a nonnegative integer")
        payload = {
            "run_id": store.context.run_id,
            "reader_role": reader,
            "query": query,
            "evidence_ids": [ev.id for ev in evidence],
            "authors": [ev.author_role for ev in evidence],
            "source_refs": [ev.source_ref for ev in evidence],
            "seconds": seconds,
            "latency_ms": seconds * 1000,
            "prompt_tokens": count,
            "token_method": self.token_method,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            **extra,
        }
        await self._emit(store, "memory.recalled", payload)
        return payload

    def summary(self) -> dict:
        recalls = [p for event, p in self.events if event == "memory.recalled"]
        writes = [p for event, p in self.events if event == "memory.written"]
        latencies = [p["latency_ms"] for p in recalls]
        return {
            "recall_calls": len(recalls),
            "write_calls": len(writes),
            "recall_latency_p50_ms": percentile(latencies, 0.5),
            "recall_latency_p95_ms": percentile(latencies, 0.95),
            "write_latency_p50_ms": percentile([p["latency_ms"] for p in writes], 0.5),
            "write_latency_p95_ms": percentile([p["latency_ms"] for p in writes], 0.95),
            "prompt_tokens": sum(p["prompt_tokens"] for p in recalls),
            "token_method": self.token_method,
            "cross_role_recall_calls": sum(
                any(a != p["reader_role"] for a in p["authors"]) for p in recalls
            ),
        }
