"""Regenerate the offline UI replays with: uv run python -m scripts.make_demo_fixture."""

import asyncio
import itertools
import tempfile
from pathlib import Path

from core.events import EventBus
from scripts.demo_loop import record_demo


async def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        for run_id, fail in [("demo", False), ("demo-error", True)]:
            clock = itertools.count(1_794_700_000).__next__
            bus = EventBus(run_id, Path(temp), clock=clock)
            await record_demo(bus, fail=fail)
            target = destination / f"{run_id}.jsonl"
            target.write_bytes(bus.path.read_bytes())


if __name__ == "__main__":
    asyncio.run(generate(Path("web/fixtures")))
