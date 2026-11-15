"""Run every DE-from-counts diagnostic on one case folder and show each outcome.

    uv run python -m diagnostics.de_from_counts.run_all path/to/case [--json out.json]

A case folder holds ``tagseq_featurecounts_matrix.tsv`` and ``samples.tsv`` (see the
"Real data" guide in Linear). For every fork and diagnostic this prints the outcome bin, the
bin a clean dataset should land in (the most likely bin under the fork's null option), and
the flagged samples. On data with no known problems, every row should match; a mismatch is
either a false alarm (a threshold to fix) or a real problem in the data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from diagnostics.de_from_counts.library import library_questions, script_code
from labs.bixbench3.bins import choose_bin


def clean_bin(design: dict, null_key: str) -> str | None:
    """The bin a dataset without the fault should land in, from the library's rows."""
    row = (design.get("exact") or design.get("stated") or {}).get(null_key)
    return max(row, key=row.get) if row else None


def run_one(
    code: str, case: Path, python: str, timeout_s: int
) -> tuple[dict | None, str]:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "diagnostic.py"
        script.write_text(code, encoding="utf-8")
        try:
            done = subprocess.run(
                [python, str(script)],
                env={
                    **os.environ,
                    "CRUX_DATA_DIR": str(case),
                    "PYTHONIOENCODING": "utf-8",
                },
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout_s} s"
    lines = [x for x in done.stdout.splitlines() if x.startswith("RESULT:")]
    if done.returncode != 0 or len(lines) != 1:
        tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
        return None, f"exit {done.returncode}: {tail[:200]}"
    return json.loads(lines[0].removeprefix("RESULT:")), ""


def run_all(
    case: Path, *, python: str | None = None, timeout_s: int = 120
) -> list[dict]:
    rows = []
    for question in library_questions():
        null_key = question.proposal.options[0].answer_key
        for exp in question.diagnostics:
            name = (exp.design or {}).get("diagnostic", exp.id)
            result, error = run_one(
                script_code(name), case, python or sys.executable, timeout_s
            )
            outcome = None
            if result is not None:
                try:
                    outcome = choose_bin(((b.id, b.rule) for b in exp.bins), result)
                except (ValueError, KeyError, TypeError) as exc:
                    error = f"no bin: {exc}"
            expected = clean_bin(exp.design or {}, null_key)
            rows.append(
                {
                    "fork": question.key,
                    "diagnostic": name,
                    "outcome": outcome,
                    "clean_outcome": expected,
                    "matches_clean": outcome is not None and outcome == expected,
                    "flagged_samples": (result or {}).get("flagged_samples", []),
                    "error": error,
                    "result": result,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "case", type=Path, help="folder with the counts and samples.tsv"
    )
    parser.add_argument(
        "--json", type=Path, help="also write every row, with full results"
    )
    args = parser.parse_args(argv)
    rows = run_all(args.case.resolve())
    width = max(len(r["diagnostic"]) for r in rows)
    for r in rows:
        mark = "ok " if r["matches_clean"] else ("ERR" if r["error"] else "!! ")
        got = r["outcome"] or r["error"]
        flagged = ", ".join(map(str, r["flagged_samples"][:6]))
        more = (
            f" +{len(r['flagged_samples']) - 6}"
            if len(r["flagged_samples"]) > 6
            else ""
        )
        print(
            f"{mark} {r['fork']:<16} {r['diagnostic']:<{width}}  {got:<20} (clean: {r['clean_outcome']})"
            + (f"  flagged: {flagged}{more}" if flagged else "")
        )
    n_ok = sum(r["matches_clean"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} diagnostics land where clean data should")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
