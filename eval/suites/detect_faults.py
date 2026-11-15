"""Score the diagnostics on a planted suite: caught, missed and false alarms per fault.

    uv run python -m eval.suites.detect_faults ../data/suites/covid --workers 6

For every case it runs all 12 diagnostics (``diagnostics.de_from_counts.run_all``) and reads
the evaluator truth key, which only this scorer sees. A fault is **caught** when a
diagnostic of its fork lands in the fault option's likeliest bin. A **false alarm** is any
other fork landing in its own fault bin. That includes the clean case, so problems already
in the public data show up there and should be read against it.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from diagnostics.de_from_counts.library import library_questions
from diagnostics.de_from_counts.run_all import run_all


def fault_bins() -> dict[str, set[str]]:
    """fork -> the bins its fault option most likely produces, one per diagnostic."""
    out: dict[str, set[str]] = {}
    for question in library_questions():
        fault_key = question.proposal.options[1].answer_key
        for exp in question.diagnostics:
            design = exp.design or {}
            row = (design.get("exact") or design.get("stated") or {}).get(fault_key)
            if row:
                out.setdefault(question.key, set()).add(
                    f"{exp.design['diagnostic']}:{max(row, key=row.get)}"
                )
    return out


def score(suite: Path, *, workers: int = 4) -> dict:
    truth = json.loads((suite / "evaluator" / "truth.json").read_text(encoding="utf-8"))
    bins = fault_bins()
    cases = sorted(truth["cases"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(
            zip(
                cases,
                pool.map(lambda c: run_all(suite / "cases" / c), cases),
                strict=True,
            )
        )
    rows = []
    for case in cases:
        info = truth["cases"][case]
        fired = {
            r["fork"]
            for r in results[case]
            if f"{r['diagnostic']}:{r['outcome']}" in bins.get(r["fork"], set())
        }
        fork = info["fork"]
        rows.append(
            {
                "case": case,
                "fault": info["fault"],
                "caught": fork in fired if fork else None,
                "false_alarms": sorted(fired - {fork}),
                "fired": sorted(fired),
            }
        )
    return {"source": truth["source"], "skipped": truth["skipped"], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = score(args.suite, workers=args.workers)
    print(f"source: {report['source']}; skipped faults: {report['skipped'] or 'none'}")
    for r in sorted(report["rows"], key=lambda r: r["fault"]):
        verdict = (
            "-" if r["caught"] is None else ("caught" if r["caught"] else "MISSED")
        )
        alarms = ", ".join(r["false_alarms"]) or "none"
        print(f"{r['fault']:<18} {verdict:<7} false alarms: {alarms}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
