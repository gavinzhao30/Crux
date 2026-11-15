"""Run-scoped memory for the orchestrator: one past.dev audience (or local ledger) per run.

``core.orchestrator`` passes the run id on every ``write``/``recall``, while the backends
bind to one run when they are built. ``RunMemory`` builds its backend on the first call,
from that run id, so ``build_task9_run`` can choose the backend before the run id exists.
The journal lives with the run, under ``<runs_dir>/<run_id>/memory/``.

Memory is advisory. A failed write or recall is logged and kept in ``errors`` (and in the
journal's audit table once it exists), never raised into the run, and a failed recall
returns nothing. Recalled records never update beliefs: the orchestrator only puts them
in prompts and dedupes them by evidence id.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from core.schemas import Evidence
from memory.base import DEFAULT_ROLES, RunContext
from memory.instrumentation import MemoryTelemetry

log = logging.getLogger(__name__)

BACKENDS = ("pastdev", "local_role_only")
# The ERD's scientific roles plus the other authors the orchestrator and labs use.
RUN_ROLES = (*DEFAULT_ROLES, "lab", "search", "proposer", "deliverer")
# A memory record must stay under 16,000 characters (past.dev's limit, enforced by the
# backends). A claim is serialised once inside the record, with JSON escaping, so long
# claims (a Task 9 profile is a JSON document) are cut here with a pointer to the run.
MAX_CLAIM_CHARS = 6000
MAX_QUERY_CHARS = 2000


class RunMemory:
    """``core.interfaces.Memory`` for one run, on past.dev or the role-only local ledger.

    ``backend`` names the arm for ``memory.*`` events. ``options`` go to ``PastMemory``
    (for example ``timeout_s`` or a test ``client``). Close with ``aclose()``: it sends
    every record past.dev has not received yet (for up to ``close_send_s``) and stops
    without waiting for indexing, which past.dev finishes on its side. Receipts and anything
    unsent stay in the run's journal.
    """

    def __init__(
        self,
        backend: str,
        runs_dir: Path | str = "runs",
        *,
        task_id: str,
        config_id: str = "default",
        repeat: int = 0,
        roles: tuple[str, ...] = RUN_ROLES,
        close_send_s: float = 10.0,
        **options: Any,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"memory backend must be one of {BACKENDS}, not {backend!r}")
        if backend == "local_role_only" and options:
            raise ValueError("provider options are not valid for the local memory arm")
        self.backend = backend
        self.runs_dir = Path(runs_dir)
        self.task_id = task_id
        self.config_id = config_id
        self.repeat = repeat
        self.roles = tuple(roles)
        self.close_send_s = close_send_s
        self.options = dict(options)
        self.run_id: str | None = None
        self.memory: Any = None
        self.telemetry = MemoryTelemetry(backend)
        self.errors: list[dict] = []
        self._lock = asyncio.Lock()
        self._closed = False

    # --- core.interfaces.Memory ---------------------------------------------------

    async def write(self, ev: Evidence, run_id: str, author_role: str) -> None:
        memory = await self._bind(run_id)
        if memory is None:
            return
        try:
            await memory.write(_fit(ev), run_id, author_role)
        except Exception as exc:  # noqa: BLE001 - memory must never stop the run
            self._error("write", exc, evidence_id=ev.id, author_role=author_role)

    async def recall(
        self, query: str, run_id: str, reader_role: str, k: int = 8
    ) -> list[Evidence]:
        memory = await self._bind(run_id)
        if memory is None:
            return []
        text = " ".join(str(query).split())[:MAX_QUERY_CHARS] or "evidence for this run"
        try:
            return list(await memory.recall(text, run_id, reader_role, k=max(1, min(k, 200))))
        except Exception as exc:  # noqa: BLE001 - a failed recall means no extra context
            self._error("recall", exc, reader_role=reader_role)
            return []

    # --- lifecycle ----------------------------------------------------------------

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        memory = self.memory
        if memory is None:
            return
        submit = getattr(memory, "submit", None)  # the local arm has nothing to send
        if submit is not None:
            try:
                await asyncio.wait_for(submit(), self.close_send_s)
            except Exception as exc:  # noqa: BLE001 - unsent records stay in the journal
                self._error("submit", exc)
        try:
            await memory.aclose()
        except Exception as exc:  # noqa: BLE001
            self._error("close", exc)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def summary(self) -> dict:
        """Latency percentiles, call counts, prompt-size estimates and cross-role recalls."""
        return {
            **self.telemetry.summary(),
            "backend": self.backend,
            "run_id": self.run_id,
            "errors": len(self.errors),
        }

    # --- helpers ------------------------------------------------------------------

    async def _bind(self, run_id: str) -> Any:
        if self._closed:
            return None
        if self.memory is None and self.run_id is None:
            async with self._lock:
                if self.memory is None and self.run_id is None:
                    self.run_id = run_id
                    self.memory = self._open(run_id)
        if run_id != self.run_id:
            self._error("bind", ValueError(f"bound to run {self.run_id}, called for {run_id}"))
            return None
        return self.memory

    def _open(self, run_id: str) -> Any:
        from core.config import RunConfig
        from memory.factory import create_memory

        try:
            context = RunContext(
                run_id=run_id,
                task_id=self.task_id,
                config_id=self.config_id,
                repeat=self.repeat,
                roles=self.roles,
            )
            return create_memory(
                context,
                RunConfig(memory=self.backend),
                self.runs_dir / run_id / "memory",
                telemetry=self.telemetry,
                **self.options,
            )
        except Exception as exc:  # noqa: BLE001 - the run goes on without memory
            self._error("open", exc)
            return None

    def _error(self, operation: str, exc: BaseException, **fields: Any) -> None:
        entry = {"operation": operation, "error": f"{type(exc).__name__}: {exc}"[:500], **fields}
        self.errors.append(entry)
        log.warning("memory %s failed (%s): %s", operation, self.backend, entry["error"])
        store = getattr(self.memory, "store", None)
        if store is not None:
            try:
                store.audit("memory.degraded", entry)
            except Exception:  # noqa: BLE001, S110 - the journal may already be closed
                pass


def _fit(ev: Evidence) -> Evidence:
    """``ev`` with its claim cut to ``MAX_CLAIM_CHARS``; the full record stays in the run."""
    if len(ev.claim) <= MAX_CLAIM_CHARS:
        return ev
    note = f" [... cut from {len(ev.claim)} characters; full record {ev.id} in the run]"
    return ev.model_copy(
        update={
            "claim": ev.claim[: MAX_CLAIM_CHARS - len(note)] + note,
            "conditions": {**ev.conditions, "claim_chars": len(ev.claim)},
        }
    )


def memory_from_config(
    cfg: Any,
    *,
    task_id: str,
    env: Mapping[str, str] | None = None,
    **options: Any,
) -> RunMemory | None:
    """The run's memory for ``cfg.memory``, or ``None`` when that backend is unavailable.

    ``pastdev`` needs ``PAST_API_KEY``; without it the run has no shared memory (a warning
    is logged and no network call is made). ``local_role_only`` needs no key. ``vector``
    has no backend yet. ``cfg.runs_dir`` and ``cfg.name`` give the journal folder and the
    audience's config id.
    """
    mode = getattr(cfg, "memory", "pastdev")
    runs_dir = Path(getattr(cfg, "runs_dir", "runs"))
    config_id = str(getattr(cfg, "name", "default") or "default")
    if mode == "local_role_only":
        return RunMemory(mode, runs_dir, task_id=task_id, config_id=config_id)
    if mode == "pastdev":
        if not (os.environ if env is None else env).get("PAST_API_KEY"):
            log.warning("memory=pastdev but PAST_API_KEY is not set: running without memory")
            return None
        return RunMemory(
            mode,
            runs_dir,
            task_id=task_id,
            config_id=config_id,
            **{"timeout_s": 15.0, **options},
        )
    log.warning("memory=%r has no backend yet: running without memory", mode)
    return None
