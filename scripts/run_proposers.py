"""Run DU-9 on agent-visible lab/profile JSON and write the proposal events."""

import argparse
import asyncio
import json
from pathlib import Path

from agents.llm import LLM
from agents.proposer import propose
from core.config import RunConfig
from core.events import EventBus
from core.fakes import FakeMemory
from core.schemas import Evidence
from core.tracing import TraceSession, current_run


async def run(args):
    data = json.loads(args.input.read_text(encoding="utf-8"))
    config = RunConfig.model_validate(data.get("config", {}))
    bus = EventBus(args.run_id)
    await bus.emit(
        "run.started",
        {
            "lab": data["brief"].get("lab", "bixbench3"),
            "goal": data["brief"]["goal"],
            "config": config.model_dump(mode="json"),
        },
    )
    tracing = TraceSession(args.run_id, lambda: "propose")
    token = current_run.set(tracing)
    try:
        result = await propose(
            data["brief"],
            [Evidence.model_validate(ev) for ev in data["profile"]],
            llm=LLM(config.llm),
            memory=FakeMemory(),
            bus=bus,
            config=config,
        )
        print(result.model_dump_json(indent=2))
    except Exception as error:
        await bus.emit(
            "run.error", {"message": type(error).__name__, "where": "proposers"}
        )
        raise
    finally:
        try:
            await tracing.flush()
            bus.path.with_name("traces.json").write_text(
                json.dumps({"traces": tracing.refs, "tracing": tracing.status()}, indent=2),
                encoding="utf-8",
            )
        finally:
            current_run.reset(token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    asyncio.run(run(parser.parse_args()))
