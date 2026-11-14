"""Starting runs in the background.

The server only needs ``RunLauncher.start``. The default implementation lazily imports
the engine (``core.*``) so this module imports cleanly even while the engine is absent.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LauncherUnavailable(RuntimeError):
    """The engine cannot be imported or started (the server answers 503)."""


class InvalidRunRequest(ValueError):
    """The run request names something the engine rejects (the server answers 400)."""


class UnknownLab(InvalidRunRequest):
    """The requested lab name is not registered (the server answers 400)."""


class RunLauncher(Protocol):
    async def start(
        self, lab: str, config: str, seed: int, overrides: dict, run_id: str
    ) -> None: ...


def new_run_id(now: datetime | None = None) -> str:
    """``r-YYYYMMDD-HHMMSS-xxxx`` with four random hex characters."""
    now = now or datetime.now()
    return f"r-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


# Lab registry: lab name -> attribute of ``core.fakes_engine`` that builds the lab.
# Real labs get registered here later (e.g. by adding a factory under a new key).
FAKE_LABS: dict[str, str] = {
    "fake": "FakeLab",
    "fake-surprise": "FakeSurpriseLab",
    "fake-numeric": "FakeNumericLab",
}


class DefaultLauncher:
    """Runs ``core.orchestrator.run`` as an asyncio task inside the server process."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._tasks: set[asyncio.Task] = set()  # keep references so tasks are not collected

    async def start(self, lab: str, config: str, seed: int, overrides: dict, run_id: str) -> None:
        if lab not in FAKE_LABS:
            raise UnknownLab(f"unknown lab {lab!r}; known: {', '.join(sorted(FAKE_LABS))}")
        coro = self._engine_coroutine(lab, config, seed, overrides, run_id)
        (self.runs_dir / run_id).mkdir(parents=True, exist_ok=True)
        task = asyncio.create_task(coro, name=f"run-{run_id}")
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task) -> None:
        """Forget the task; a crash is already recorded as ``run.error`` in its event file."""
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("%s ended with an error: %r", task.get_name(), task.exception())

    # ------------------------------------------------------------------
    # Engine integration: the ONLY place that knows core.orchestrator's API.
    # Adjust this function if the orchestrator signature changes.
    # ------------------------------------------------------------------
    def _engine_coroutine(
        self, lab: str, config: str, seed: int, overrides: dict[str, Any], run_id: str
    ) -> Coroutine[Any, Any, Any]:
        try:
            from core import config as core_config
            from core import fakes_engine, orchestrator
        except ImportError as exc:  # engine not written yet, or a dependency is missing
            raise LauncherUnavailable(f"engine is not importable: {exc}") from exc

        try:
            cfg = core_config.load_config(
                config, overrides=overrides, seed=seed, runs_dir=str(self.runs_dir)
            )
        except (KeyError, ValueError, TypeError) as exc:  # unknown preset or bad override
            raise InvalidRunRequest(f"invalid config: {exc}") from exc
        try:
            lab_obj = getattr(fakes_engine, FAKE_LABS[lab])(seed=seed)
            deps = fakes_engine.fake_deps(lab_obj, seed=seed)
        except (AttributeError, TypeError) as exc:
            raise LauncherUnavailable(f"engine API mismatch: {exc}") from exc
        return orchestrator.run(lab_obj, cfg, deps, run_id=run_id)
