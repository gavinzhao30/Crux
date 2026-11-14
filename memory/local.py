"""Role-only history baseline: no API key, semantic retriever or cross-role context."""

import hashlib
import time
from pathlib import Path
from typing import Literal

from core.schemas import Evidence
from memory.base import MemoryRecord, RunContext
from memory.instrumentation import MemoryTelemetry
from memory.store import MemoryStore


class LocalMemory:
    def __init__(
        self,
        context: RunContext,
        root: Path = Path("runs"),
        *,
        mode: Literal["role_only"] = "role_only",
        telemetry: MemoryTelemetry | None = None,
    ):
        if mode != "role_only":
            raise ValueError("DU-17 implements only the role_only baseline")
        self.context = context
        # Separate physical ledgers prevent an arm accidentally inheriting another's outbox.
        self.store = MemoryStore(Path(root) / "local_role_only", context)
        self.telemetry = telemetry or MemoryTelemetry("local_role_only")
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    def _check_run(self, run_id: str) -> None:
        if self._closed:
            raise RuntimeError("Memory is closed")
        if run_id != self.context.run_id:
            raise ValueError("Memory adapter is bound to another run")

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
        self.context.identity(record.evidence.author_role)
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
        if record.evidence.id in record.supersedes:
            raise ValueError("A record cannot supersede itself")
        for evidence_id in record.supersedes:
            if self.store.by_evidence(evidence_id) is None:
                raise ValueError("A superseded evidence ID must exist in this run")
        if len(record.model_dump_json(indent=2)) >= 16000:
            raise ValueError("Memory records must be under 16000 characters; use artifact pointers")
        source_id = (
            self.context.audience + ":ev:" + hashlib.sha256(record.evidence.id.encode()).hexdigest()
        )
        self.store.append(record, source_id)
        await self.telemetry.written(self.store, record.evidence, time.perf_counter() - started)

    async def recall(self, query: str, run_id: str, reader_role: str, k: int = 8) -> list[Evidence]:
        self._check_run(run_id)
        self.context.identity(reader_role)
        if not query.strip() or not 1 <= k <= 200:
            raise ValueError("Recall needs a query and k between 1 and 200")
        started = time.perf_counter()
        records = [
            MemoryRecord.model_validate_json(row["record"])
            for row in reversed(self.store.eligible())
        ]
        # Filter authors BEFORE selecting or counting mandatory records. Another role's
        # objection must not enter this baseline through the local freshness buffer.
        own = [r for r in records if r.evidence.author_role == reader_role]
        mandatory = [r.evidence for r in own if r.mandatory]
        ordinary = [r.evidence for r in own if not r.mandatory]
        selected = mandatory + ordinary[: max(0, k - len(mandatory))]
        await self.telemetry.recalled(
            self.store,
            reader_role,
            query,
            selected,
            time.perf_counter() - started,
            remote_status="not_applicable",
            fallback=None,
        )
        return [ev.model_copy(deep=True) for ev in selected]

    async def flush(self) -> None:
        self._check_run(self.context.run_id)

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            self.store.close()
