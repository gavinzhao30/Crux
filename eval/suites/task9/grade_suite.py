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
    args = parser.parse_args()
    cases = json.loads(args.truth.read_text(encoding="utf-8"))["cases"]
    runs = json.loads(args.runs.read_text(encoding="utf-8"))
    if set(runs) - cases.keys():
        raise ValueError("Run index contains unknown cases")
    arms = sorted({arm for case in runs.values() for arm in case})
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
                "mean": None,
                "fraction_at_least_0_80": None,
                "review": None,
            }
            if run is not None:
                result = grade_task9(
                    artifacts_dir=Path(run["artifacts"]),
                    references=args.references,
                    output_dir=args.output_dir / case_id / arm,
                    grader_repo=args.grader_repo,
                    python=args.python,
                )
                row.update(
                    mean=result["mean"],
                    fraction_at_least_0_80=result["fraction_at_least_0_80"],
                    review=run.get("review"),
                )
            rows.append(row)
    coverage = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        scored = [row for row in arm_rows if row["mean"] is not None]
        coverage[arm] = {
            "requested": len(arm_rows),
            "graded": len(scored),
            "clean": [row for row in scored if row["fault"] == "clean"],
            "planted_mean": mean([row["mean"] for row in scored if row["fault"] != "clean"])
            if any(row["fault"] != "clean" for row in scored)
            else None,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (args.output_dir / "suite-grade.json").write_text(
        json.dumps({"protocol": PROTOCOL, "coverage": coverage, "cases": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
