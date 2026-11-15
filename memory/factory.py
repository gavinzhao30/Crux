"""One explicit config switch, retaining the same Memory protocol for both arms."""

from pathlib import Path

from core.config import RunConfig
from memory.base import RunContext
from memory.instrumentation import MemoryTelemetry
from memory.local import LocalMemory
from memory.pastdev import PastMemory


def create_memory(
    context: RunContext,
    config: RunConfig,
    root: Path = Path("runs"),
    *,
    telemetry: MemoryTelemetry | None = None,
    **pastdev_options,
) -> LocalMemory | PastMemory:
    if config.memory == "local_role_only":
        if pastdev_options:
            raise ValueError("Provider options are not valid for the local memory arm")
        return LocalMemory(context, root, telemetry=telemetry)
    return PastMemory(context, root, telemetry=telemetry, **pastdev_options)
