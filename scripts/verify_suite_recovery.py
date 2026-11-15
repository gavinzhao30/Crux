"""Live infrastructure fixtures: worker process crash and coordinator recovery, no model calls."""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from eval.run_suite import (
    DurableEventBus,
    app,
    atomic_json,
    checkpoint,
    exclude_runtime_data,
    image,
    volume,
)


@app.function(image=image.add_local_python_source("scripts", ignore=exclude_runtime_data), volumes={"/checkpoints": volume}, timeout=180, max_containers=1)
def crash_probe(job_id, crash):
    directory = Path("/checkpoints/jobs") / job_id
    directory.mkdir(parents=True, exist_ok=True)
    asyncio.run(DurableEventBus(job_id, Path("/checkpoints/jobs")).emit("run.started", {"fixture": True}))
    artifacts = directory / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "partial.txt").write_text("acknowledged work survives process loss")
    atomic_json(directory / "status.json", {"state": "running", "fixture": True})
    asyncio.run(checkpoint())
    if crash:
        time.sleep(150)
    atomic_json(directory / "result.json", {"state": "complete", "status": "finished", "fixture": True,
        "cost_usd": 0, "model_calls": 0, "experiments": 0,
        "heldout_predictions": {"20": [1.0]*10}})
    asyncio.run(checkpoint())
    return "persisted"


def main():
    output = Path("eval/results/harness-recovery/du23-20261115-v2").resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [{"job_id": f"du23-recovery-20261115-v2-{suffix}", "suite": "boxing", "case_id": "1000",
             "arm": "fixture", "seed": 1000, "repeat": 0} for suffix in ["crash", "complete"]]
    atomic_json(output / "jobs.json", {"suite": "boxing", "jobs": jobs, "fixture_only": True})
    with app.run():
        call = crash_probe.spawn(jobs[0]["job_id"], True)
        ready = False
        for _ in range(120):
            try:
                b"".join(volume.read_file(f"/jobs/{jobs[0]['job_id']}/status.json"))
                ready = True
                break
            except FileNotFoundError:
                time.sleep(1)
        assert ready, "Probe did not reach its persisted checkpoint"
        call.cancel(terminate_containers=True)
        try:
            call.get(timeout=15)
        except Exception as exc:  # noqa: BLE001 - container termination is intentional
            print("Expected worker termination:", type(exc).__name__, flush=True)
        crash_probe.remote(jobs[1]["job_id"], False)
    # A new coordinator process recovers exclusively from persisted state.
    subprocess.run([sys.executable, "-m", "eval.run_suite", "boxing", "--recover", "--output", str(output)], check=True)
    rows = [json.loads((output / "jobs" / f"{j['job_id']}.json").read_text()) for j in jobs]
    assert rows[0]["score"] == 0 and rows[0]["status"] == "interrupted"
    assert rows[1]["grade_status"] == "graded" and rows[1]["score"] > 0
    for job in jobs:
        evidence = output / "evidence" / job["job_id"]
        assert (evidence / "events.jsonl").is_file()
        assert (evidence / "artifacts/partial.txt").read_text() == "acknowledged work survives process loss"
    atomic_json(output / "verification.json", {"worker_process_crash": "pass", "fresh_coordinator_recovery": "pass",
        "committed_events_and_artifacts": "pass", "expected_jobs": 2, "recorded_jobs": len(rows),
        "model_calls": 0, "fixture_only": True})
    print("Live crash/recovery verification passed")


if __name__ == "__main__":
    main()
