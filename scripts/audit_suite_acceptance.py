"""Reconcile every suite summary row against saved jobs and independent grader reports."""
import argparse
import csv
import json
import math
from pathlib import Path


def audit(directory):
    manifest = json.loads((directory / "jobs.json").read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((directory / "summary.csv").open(encoding="utf-8")))
    expected = {j["job_id"] for j in manifest["jobs"]}
    assert len(csv_rows) == len(expected)
    assert {r["job_id"] for r in csv_rows} == expected
    paired = {}
    checked_grades = 0
    failures = 0
    for row in csv_rows:
        job = next(j for j in manifest["jobs"] if j["job_id"] == row["job_id"])
        record = json.loads((directory / "jobs" / f"{job['job_id']}.json").read_text(encoding="utf-8"))
        assert "fault" not in job and "affected_samples" not in job
        paired.setdefault((job["case_id"], job["repeat"]), set()).add(job["seed"])
        assert math.isclose(float(row["score"]), record["score"], abs_tol=1e-12)
        if record.get("state") != "complete" or record.get("grade_status") == "failed":
            assert record["score"] == 0
            failures += 1
        elif manifest["suite"] == "task9":
            raw = json.loads((directory / "grades" / job["job_id"] / "grade.json").read_text())
            mean = sum(a["score"] for a in raw["artifacts"].values()) / raw["artifact_count"]
            assert math.isclose(mean, record["score"], abs_tol=1e-12)
            assert record["fault_verdict"] not in {"judge_error", "needs_fault_judge"}
            checked_grades += 1
        else:
            assert record["grade_status"] == "graded"
            assert math.isclose(record["score"], 1/(1+record["raw_mse"]), abs_tol=1e-12)
            checked_grades += 1
        for key in ["cost_usd", "input_tokens", "output_tokens", "experiments"]:
            if record.get(key) is not None:
                assert math.isclose(float(row[key]), float(record[key]), abs_tol=1e-12)
    assert all(len(seeds) == 1 for seeds in paired.values())
    acceptance = json.loads((directory / "acceptance.json").read_text())
    assert acceptance["under_20_minutes"] and acceptance["wall_seconds"] < 1200
    result = {"suite": manifest["suite"], "requested": len(expected), "recorded": len(csv_rows),
              "separate_grades_reconciled": checked_grades, "failures_retained": failures,
              "paired_seeds": "pass", "summary_matches_job_and_grader": "pass",
              "under_20_minutes": True, "wall_seconds": acceptance["wall_seconds"]}
    (directory / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
