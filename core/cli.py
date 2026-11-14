"""Run the offline fakes end to end.

    python -m core.cli --lab fake|fake-surprise|fake-numeric --config default --seed 0 --runs-dir runs

Prints the run id, the number of events, which questions settled (or ended robust or
unresolved) and the path of the events file. No network access and no API key are needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from core.config import PRESETS, load_config
from core.fakes_engine import (
    FakeClock,
    FakeLab,
    FakeNumericLab,
    FakeSurpriseLab,
    fake_deps,
)
from core.orchestrator import run

FAKE_LABS: dict[str, type[FakeLab | FakeNumericLab]] = {
    "fake": FakeLab,
    "fake-surprise": FakeSurpriseLab,
    "fake-numeric": FakeNumericLab,
}
LABS = tuple(FAKE_LABS)


def make_lab(name: str, seed: int = 0, latency_s: float = 0.0) -> FakeLab | FakeNumericLab:
    """Build a fake lab by its CLI name."""
    if name not in FAKE_LABS:
        raise ValueError(f"unknown lab {name!r}; choose from {', '.join(LABS)}")
    return FAKE_LABS[name](seed=seed, latency_s=latency_s)


def summarise(events_path: Path) -> dict[str, list[str]]:
    """Question id -> outcome lines, read back from the event file."""
    outcome: dict[str, list[str]] = {
        "settled": [],
        "robust": [],
        "unresolved": [],
    }
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        p = event["payload"]
        if event["type"] == "question.settled":
            outcome["settled"].append(
                f"{p['question_id']} -> {p['hypothesis_id']} ({p['posterior']:.3f})"
            )
        elif event["type"] == "question.robust":
            outcome["robust"].append(f"{p['question_id']} -> {p['answer_key']}")
        elif event["type"] == "question.unresolved":
            outcome["unresolved"].append(f"{p['question_id']}: {p['reason']}")
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.cli", description=__doc__.split("\n")[0])
    parser.add_argument("--lab", choices=LABS, default="fake")
    parser.add_argument("--config", choices=sorted(PRESETS), default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--fake-clock", action="store_true", help="use a deterministic clock (reproducible files)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, seed=args.seed, runs_dir=args.runs_dir)
    lab = make_lab(args.lab, seed=args.seed)
    deps = fake_deps(lab, seed=args.seed, clock=FakeClock() if args.fake_clock else time.time)
    report = asyncio.run(run(lab, cfg, deps, run_id=args.run_id))

    events_path = Path(args.runs_dir) / report.run_id / "events.jsonl"
    n_events = sum(1 for line in events_path.read_text(encoding="utf-8").splitlines() if line)
    outcome = summarise(events_path)
    print(f"run id:     {report.run_id}")
    print(f"events:     {n_events}")
    for key in ("settled", "robust", "unresolved"):
        print(f"{key + ':':<11} {'; '.join(outcome[key]) or '-'}")
    print(f"cost:       ${report.cost_usd:.4f} (fake)")
    print(f"events at:  {events_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
