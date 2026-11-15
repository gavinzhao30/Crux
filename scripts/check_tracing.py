"""Run a synthetic categorical scenario with live traced design calls.

    uv run --env-file .env python scripts/check_tracing.py --run-id du45-trace-smoke

This checks infrastructure, not the scientific Task 9 hero run. The lab, other
roles and likelihoods are deterministic fixtures. Only the designer calls a model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.llm import CHEAP_MODEL, LLM, LLMConfig
from core.config import load_config
from core.fakes_engine import FakeAgents, FakeLab, fake_deps
from core.orchestrator import run
from core.replay import replay_check


class Choice(BaseModel):
    experiment_id: str
    reason: str


class LiveDesigner(FakeAgents):
    def __init__(self, lab, llm):
        super().__init__(lab, llm=None)
        self.live_llm = llm

    async def design(self, question, hypotheses, evidence, history):
        proposals = await super().design(question, hypotheses, evidence, history)
        [choice] = await self.live_llm.structured(
            "designer", Choice,
            [{"role": "user", "content": json.dumps({
                "task": "Choose the most useful available diagnostic. Use an exact candidate ID.",
                "question": question.model_dump(mode="json"),
                "hypotheses": [h.model_dump(mode="json") for h in hypotheses],
                "evidence": [e.model_dump(mode="json") for e in evidence],
                "candidates": [e.model_dump(mode="json") for e in proposals],
            })}],
            step=f"design:{question.id}",
        )
        if choice.experiment_id not in {p.id for p in proposals}:
            raise ValueError("Designer returned an ID outside the supplied candidates")
        return sorted(proposals, key=lambda p: p.id != choice.experiment_id)


async def record(run_id):
    required = ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                "LANGFUSE_BASE_URL", "LANGFUSE_PROJECT_ID")
    if any(not os.getenv(name) for name in required):
        raise ValueError("Load the required OpenAI and Langfuse settings from .env")
    config = load_config(selection="model_pick", runs_dir="runs", max_llm_calls=8)
    config.llm = LLMConfig(role_models={"default": CHEAP_MODEL}, cache=False, max_retries=1)
    llm = LLM(config.llm)
    lab = FakeLab()
    deps = fake_deps(lab)
    deps.agents = LiveDesigner(lab, llm)
    deps.predictor.llm = None
    deps.llm = llm
    await run(lab, config, deps, run_id=run_id)
    result = replay_check(Path("runs") / run_id)
    print(result.summary())
    manifest = json.loads((Path("runs") / run_id / "manifest.json").read_text())
    print("Live model calls:", manifest["usage"]["calls"])
    print("Telemetry status:", manifest["tracing"])
    for ref in manifest["traces"]:
        print("Model trace:", ref["url"])
    return 0 if result.ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        return asyncio.run(record(args.run_id))
    except Exception as exc:  # noqa: BLE001 - keep credential-bearing SDK errors out of logs
        print(f"Tracing check failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
