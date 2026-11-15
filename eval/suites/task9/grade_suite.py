"""Grade submitted Task 9 case runs and join evaluator-only fault reviews."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from eval.grade_bixbench3 import grade_task9
from labs.bixbench3.task9_contract import PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--grader-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--arms", nargs="+", help="Requested arm names, including arms with no submitted runs"
    )
    args = parser.parse_args()
    cases = json.loads(args.truth.read_text(encoding="utf-8"))["cases"]
    runs = json.loads(args.runs.read_text(encoding="utf-8"))
    if set(runs) - cases.keys():
        raise ValueError("Run index contains unknown cases")
    submitted_arms = {arm for case in runs.values() for arm in case}
    arms = sorted(set(args.arms) if args.arms is not None else submitted_arms)
    if submitted_arms - set(arms):
        raise ValueError("Run index contains an arm absent from --arms")
    if not arms or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", arm) for arm in arms):
        raise ValueError("Expected at least one simple arm name")
    rows = []
    for case_id, truth in sorted(cases.items()):
        if args.split != "all" and truth["split"] != args.split:
            continue
        for arm in arms:
            run = runs.get(case_id, {}).get(arm)
            row = {
                "case_id": case_id,
                "arm": arm,
                "split": truth["split"],
                "fault": truth["fault"],
                "mean": 0.0,
                "fraction_at_least_0_80": 0.0,
                "status": "missing",
                "review": None,
            }
            if run is not None:
                row["review"] = run.get("review")
                artifact_dir = Path(run["artifacts"]) if run.get("artifacts") else None
                if (
                    run.get("status") in {"failed", "error"}
                    or artifact_dir is None
                    or not artifact_dir.is_dir()
                ):
                    row["status"] = "failed"
                else:
                    result = grade_task9(
                        artifacts_dir=artifact_dir,
                        references=args.references,
                        output_dir=args.output_dir / case_id / arm,
                        grader_repo=args.grader_repo,
                        python=args.python,
                    )
                    row.update(
                        status="graded",
                        mean=result["mean"],
                        fraction_at_least_0_80=result["fraction_at_least_0_80"],
                    )
            rows.append(row)
    coverage = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        planted = [row["mean"] for row in arm_rows if row["fault"] != "clean"]
        coverage[arm] = {
            "requested": len(arm_rows),
            "graded": sum(row["status"] == "graded" for row in arm_rows),
            "failed": sum(row["status"] == "failed" for row in arm_rows),
            "missing": sum(row["status"] == "missing" for row in arm_rows),
            "mean": mean(row["mean"] for row in arm_rows) if arm_rows else None,
            "clean": [row for row in arm_rows if row["fault"] == "clean"],
            "planted_mean": mean(planted) if planted else None,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (args.output_dir / "suite-grade.json").write_text(
        json.dumps({"protocol": PROTOCOL, "coverage": coverage, "cases": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
