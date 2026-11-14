"""Write the recorded demo runs the UI replays.

    python scripts/make_demo_runs.py

Runs the offline fakes and copies each run's events file to ``runs/demo/fake_categorical.jsonl``
and ``runs/demo/fake_numeric.jsonl``. The runs themselves happen in a temporary directory with a
relative ``runs`` folder, so artifact paths in the files are relative and carry no local paths.
Agents and labs pause briefly so the recorded timestamps replay at a watchable pace. The runs
use the ``default`` preset with lessons on, so the replays include ``lesson.written``. Each file
is checked with ``core.replay`` before the script exits.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so `python scripts/make_demo_runs.py` finds `core`

DEMOS = (("fake", "fake_categorical.jsonl"), ("fake-numeric", "fake_numeric.jsonl"))
LATENCY_S = 0.25


async def _record(lab_name: str, seed: int) -> str:
    from core.cli import make_lab
    from core.config import load_config
    from core.fakes_engine import fake_deps
    from core.orchestrator import run

    lab = make_lab(lab_name, seed=seed, latency_s=LATENCY_S)
    deps = fake_deps(lab, seed=seed, clock=time.time, latency_s=LATENCY_S)
    cfg = load_config("default", seed=seed, runs_dir="runs", lessons=True)
    report = await run(lab, cfg, deps)
    return report.run_id


def main() -> int:
    from core.replay import replay_check

    out_dir = ROOT / "runs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    here = Path.cwd()
    ok = True
    with tempfile.TemporaryDirectory(prefix="crux-demo-") as tmp:
        os.chdir(tmp)
        try:
            for lab_name, file_name in DEMOS:
                run_id = asyncio.run(_record(lab_name, seed=0))
                target = out_dir / file_name
                shutil.copyfile(Path(tmp) / "runs" / run_id / "events.jsonl", target)
                result = replay_check(target)
                ok = ok and result.ok
                print(result.summary())
        finally:
            os.chdir(here)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
