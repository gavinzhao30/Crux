"""Single-writer run logs and live subscriptions for one asyncio event loop."""

import asyncio
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from core.schemas import RunEvent


class EventBus:
    """One instance owns one new run. Subscribers receive future events in file order."""

    def __init__(
        self,
        run_id: str,
        root: Path = Path("runs"),
        *,
        clock: Callable[[], float] = time.time,
    ):
        if not re.fullmatch(r"[a-zA-Z0-9_-][a-zA-Z0-9._-]*", run_id):
            raise ValueError("run_id must be a single directory name")
        self.run_id = run_id
        self.path = root / run_id / "events.jsonl"
        self._clock = clock
        self._seq = 0
        self._closed = False
        self._subscribers: set[asyncio.Queue[RunEvent]] = set()

    async def emit(self, event_type: str, payload: dict) -> RunEvent:
        if self._closed:
            raise ValueError("run has already ended")
        if (self._seq == 0) != (event_type == "run.started"):
            raise ValueError("run.started must appear exactly once, first")
        event = RunEvent(
            run_id=self.run_id,
            seq=self._seq,
            ts=self._clock(),
            type=event_type,
            payload=payload,
        )
        line = event.model_dump_json()
        event = RunEvent.model_validate_json(line)
        # No await between assigning the sequence, persisting it, and notifying readers.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open(
            "x" if self._seq == 0 else "a", encoding="utf-8", newline="\n"
        ) as log:
            log.write(line + "\n")
        self._seq += 1
        self._closed = event_type in {"run.finished", "run.error"}
        for queue in self._subscribers:
            queue.put_nowait(event.model_copy(deep=True))
        return event

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[AsyncIterator[RunEvent]]:
        """Register on context entry and release the queue even if the reader cancels."""
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        already_closed = self._closed
        self._subscribers.add(queue)

        async def stream() -> AsyncIterator[RunEvent]:
            if already_closed:
                return
            while True:
                event = await queue.get()
                yield event
                if event.type in {"run.finished", "run.error"}:
                    return

        try:
            yield stream()
        finally:
            self._subscribers.discard(queue)
