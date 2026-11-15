"""Calibration of the outcome predictors on real data with planted faults (DU-54).

Each method gives P(bin | true option) for every record, exactly as ``eval/eval_predictor.py``
scores them (same rows, same epsilon floor, same metrics):

* ``library``: the library's exact or stated rows.
* ``sampled``: gpt-4.1-mini x8 (``SampledPredictor``). Its rows depend only on the diagnostic
  and the option, never on the data, so DU-29's cached rows are reused as they are.
* ``base``: Qwen/Qwen3-8B before fine-tuning.
* ``old``: the DU-29 fine-tune, trained on synthetic cases only.
* ``new``: the DU-54 fine-tune, trained on synthetic cases plus real airway and pasilla cases.
* ``new_clean``: the same, but without the real records of the ``DOUBTFUL`` pairs. It was
  trained after ``new`` had been scored on COVID (its sample-label rows had got worse, and the
  airway label forks are in ``DOUBTFUL``), so it is a second look at the test set.

Evaluation sets:

* ``real_test``: COVID (Expression Atlas E-ENAD-46). No model was trained on it.
* ``real_test_trusted``: the same without the doubtful (dataset, fork) pairs in ``DOUBTFUL``,
  where the public data already has a problem, so the "no fault" answer key is doubtful. The
  pairs were fixed before any model was scored; ``unplanted_fault_rate`` in the report shows
  how often each fork lands in its fault bin when no fault was planted, as a check.
* ``real_other``: airway (GSE52778) and pasilla (GSE18508). Unseen by library, sampled, base
  and old; the new model trained on them, so it is not scored there.
* ``synthetic_test`` and ``synthetic_shift``: DU-29's held-out synthetic sets, to check that
  adding real data did not cost anything on synthetic cases.

    set -a; . ../Crux/.env; set +a
    uv run --group tinker python -m eval.eval_predictor_real --methods base,old,new
    uv run python -m eval.eval_predictor_real --report-only --notes notes.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from agents.tinker_predictor import BASE_MODEL, logprobs_to_row
from eval.eval_predictor import (
    FORK_ORDER,
    PREFILL_PRICE,
    estimate_tinker_tokens,
    load_cache,
    method_rows,
    record_key,
    run_tinker,
    summarise,
)
from eval.predictor_data.dataset import DATA_DIR, add_to_ledger, load_split, read_ledger
from eval.predictor_data.make_real_examples import DOUBTFUL, MIX_DIR, REAL_DIR

MIX_CLEAN_DIR = Path("runs/predictor_mix_clean")

RESULTS = Path("eval/results")
METHODS = ("library", "sampled", "base", "old", "new", "new_clean")
LABELS = {
    "library": "Library rows (exact or stated)",
    "sampled": "gpt-4.1-mini sampled x8",
    "base": "Qwen3-8B base",
    "old": "Fine-tune on synthetic only (DU-29)",
    "new": "Fine-tune on synthetic + real (DU-54)",
    "new_clean": "Fine-tune on synthetic + real, doubtful forks dropped (DU-54)",
}
SETS = ("real_test", "real_test_trusted", "real_other", "synthetic_test", "synthetic_shift")
# where each method was trained: it is not scored on these sets
TRAINED_ON = {"new": {"real_other"}, "new_clean": {"real_other"}}


def trusted(record: dict) -> bool:
    return (record.get("dataset"), record["fork"]) not in DOUBTFUL


def load_sets(real: Path, synthetic: Path) -> dict[str, list[dict]]:
    test = load_split("real_test", real)
    return {
        "real_test": test,
        "real_test_trusted": [r for r in test if trusted(r)],
        "real_other": load_split("real_train", real) + load_split("real_val", real),
        "synthetic_test": load_split("test", synthetic),
        "synthetic_shift": load_split("shift", synthetic),
    }


def tinker_rows(*caches: dict[str, dict]) -> dict[str, dict[str, dict[str, float]]]:
    out = {}
    for cache in caches:
        for key, entry in cache.items():
            out[key] = {k: logprobs_to_row(lp, 0.0) for k, lp in entry["logprobs"].items()}
    return out


def rows_for(method: str, sets: dict, real: Path, synthetic: Path) -> dict | None:
    """record key -> {answer_key: raw row}, from every cache that holds the method."""
    every = [r for records in sets.values() for r in records]
    if method in ("library", "sampled"):
        return method_rows(method, every, synthetic)
    if method == "base":
        rows = tinker_rows(load_cache(synthetic, "base"), load_cache(real, "base"))
    elif method == "old":
        rows = tinker_rows(load_cache(synthetic, "finetuned"), load_cache(real, "old"))
    else:
        rows = tinker_rows(load_cache(real, method))
    return rows or None


def unplanted_fault_rate(records: list[dict]) -> dict[str, dict[str, dict]]:
    """dataset -> fork -> how often a diagnostic lands in its fault bin with no fault planted."""
    from eval.suites.detect_faults import fault_bins

    bins = fault_bins()
    counts: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in records:
        if r["observed_bin"] is None or r["fork"] == _planted_fork(r):
            continue
        hit = f"{r['diagnostic']}:{r['observed_bin']}" in bins.get(r["fork"], set())
        counts[r["dataset"]][r["fork"]][0] += hit
        counts[r["dataset"]][r["fork"]][1] += 1
    return {
        ds: {
            fork: {"rate": hits / n, "n": n}
            for fork, (hits, n) in sorted(forks.items())
        }
        for ds, forks in sorted(counts.items())
    }


def _planted_fork(record: dict) -> str | None:
    from diagnostics.de_from_counts.synth import FAULT_FORK

    return FAULT_FORK.get(record["fault"]) if record.get("fault") else None


def evaluate(real: Path, synthetic: Path, eps: float) -> dict:
    sets = load_sets(real, synthetic)
    report: dict = {
        "eps": eps,
        "sets": {
            name: {"records": len(rs), "cases": len({r["case_id"] for r in rs})}
            for name, rs in sets.items()
        },
        "methods": {},
        "doubtful": {f"{d}/{f}": why for (d, f), why in DOUBTFUL.items()},
    }
    for method in METHODS:
        rows = rows_for(method, sets, real, synthetic)
        if rows is None:
            continue
        report["methods"][method] = {
            name: summarise(records, rows, eps)
            for name, records in sets.items()
            if name not in TRAINED_ON.get(method, set())
        }
    real_records = sets["real_test"] + sets["real_other"]
    report["unplanted_fault_rate"] = unplanted_fault_rate(real_records)
    return report


# --- report -----------------------------------------------------------------------------


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(report: dict, name: str, group: str) -> list[str]:
    lines = [
        "| Method | n | Log loss | Brier | Accuracy | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        scores = report["methods"].get(method, {}).get(name)
        if scores is None:
            if method in report["methods"]:
                lines.append(f"| {LABELS[method]} | trained on this set | | | | |")
            continue
        m = scores.get(group) or {}
        if not m.get("n"):
            continue
        lines.append(
            f"| {LABELS[method]} | {m['n']} | {_fmt(m['log_loss'])} | {_fmt(m['brier'])} "
            f"| {_fmt(m['accuracy'])} | {_fmt(m['ece'])} |"
        )
    return lines


def markdown(report: dict, notes: dict) -> str:
    out = ["# Outcome predictors on real data with planted faults (DU-54)", ""]
    for para in notes.get("summary", []):
        out += [para, ""]
    out += ["## Data", ""]
    for para in notes.get("data", []):
        out += [para, ""]
    out += ["| Set | Cases | Records |", "|---|---|---|"]
    for name in SETS:
        s = report["sets"][name]
        out.append(f"| `{name}` | {s['cases']} | {s['records']} |")
    out.append("")
    titles = {
        "real_test": "COVID, held out from every model",
        "real_test_trusted": "COVID without the doubtful forks",
        "real_other": "Airway and pasilla (in the new model's training data)",
        "synthetic_test": "Synthetic test (DU-29)",
        "synthetic_shift": "Synthetic shift (DU-29)",
    }
    for name in SETS:
        for group, label in (
            ("stated", "forks where the library has only stated rows"),
            ("overall", "all forks"),
        ):
            out += [f"## {titles[name]}: {label}", ""]
            out += _table(report, name, group)
            out.append("")
    out += ["## Per fork on COVID (log loss)", ""]
    methods = [m for m in METHODS if "real_test" in report["methods"].get(m, {})]
    out += [
        "| Fork | " + " | ".join(LABELS[m] for m in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for fork in FORK_ORDER:
        cells = []
        for m in methods:
            v = report["methods"][m]["real_test"].get(fork, {}).get("log_loss")
            cells.append(_fmt(v))
        out.append(f"| {fork} | " + " | ".join(cells) + " |")
    out += ["", "## How often a fork lands in its fault bin with no fault planted", ""]
    out += [
        "Doubtful pairs, fixed before scoring: "
        + "; ".join(f"`{k}` ({v})" for k, v in report["doubtful"].items())
        + ".",
        "",
        "| Dataset | " + " | ".join(FORK_ORDER) + " |",
        "|---|" + "---|" * len(FORK_ORDER),
    ]
    for ds, forks in report["unplanted_fault_rate"].items():
        cells = [
            _fmt(forks[f]["rate"], 2) if f in forks else "-" for f in FORK_ORDER
        ]
        out.append(f"| {ds} | " + " | ".join(cells) + " |")
    out.append("")
    for para in notes.get("caveats", []):
        out += [para, ""]
    spend = report.get("spend") or {}
    if spend:
        out += [
            "## Spend",
            "",
            "Estimated from tokens at list prices: "
            + ", ".join(f"{k} ${v:.2f}" for k, v in spend.items())
            + ".",
            "",
        ]
    return "\n".join(out).rstrip() + "\n"


# --- driver -----------------------------------------------------------------------------


def _model_path(root: Path) -> str:
    return json.loads((root / "model.json").read_text(encoding="utf-8"))["model_path"]


def score(method: str, sets: dict, args) -> None:
    """Score the records a Tinker method still lacks, within the spend cap."""
    if method == "base":
        target, records = BASE_MODEL, sets["real_test"] + sets["real_other"]
    elif method == "old":
        target, records = _model_path(args.synthetic), sets["real_test"] + sets["real_other"]
    elif method in ("new", "new_clean"):
        target = _model_path(args.mix if method == "new" else args.mix_clean)
        records = sets["real_test"] + sets["synthetic_test"] + sets["synthetic_shift"]
    else:
        return
    done = load_cache(args.real, method)
    todo = [r for r in records if record_key(r) not in done]
    est = estimate_tinker_tokens(todo) * PREFILL_PRICE / 1e6
    spent = float(read_ledger(args.real).get("total_usd", 0.0))
    print(f"{method}: {len(todo)} records to score, about ${est:.2f} "
          f"(ledger ${spent:.2f}, cap ${args.max_usd:.2f})", flush=True)
    if est + spent > args.max_usd:
        print(f"Refusing {method}: the estimate exceeds --max-usd", flush=True)
        return
    usd = asyncio.run(
        run_tinker(method, target, records, args.real, concurrency=args.concurrency)
    )
    if usd:
        add_to_ledger(args.real, f"eval-{method}", usd, {"provider": "tinker"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", type=Path, default=REAL_DIR)
    parser.add_argument("--synthetic", type=Path, default=DATA_DIR)
    parser.add_argument("--mix", type=Path, default=MIX_DIR)
    parser.add_argument("--mix-clean", type=Path, default=MIX_CLEAN_DIR)
    parser.add_argument("--methods", default="base,old,new")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--max-usd", type=float, default=8.0, help="Tinker scoring cap")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--notes", type=Path, default=None, help="JSON with report prose")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.report_only:
        sets = load_sets(args.real, args.synthetic)
        for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
            score(method, sets, args)
    report = evaluate(args.real, args.synthetic, args.eps)
    notes = json.loads(args.notes.read_text(encoding="utf-8")) if args.notes else {}
    spend = {
        "Tinker scoring (this report)": read_ledger(args.real).get("total_usd", 0.0),
        "Tinker training (new model)": read_ledger(args.mix).get("total_usd", 0.0),
        "Tinker training (new_clean model)": read_ledger(args.mix_clean).get(
            "total_usd", 0.0
        ),
    }
    report["spend"] = {k: float(v) for k, v in spend.items() if v}
    for name, root in (("old", args.synthetic), ("new", args.mix), ("new_clean", args.mix_clean)):
        path = root / "model.json"
        if path.is_file():
            report.setdefault("models", {})[name] = json.loads(path.read_text("utf-8"))
    for name in ("manifest.json",):
        path = args.real / name
        if path.is_file():
            report["real_data"] = json.loads(path.read_text("utf-8"))
    for name, root in (("new", args.mix), ("new_clean", args.mix_clean)):
        path = root / "mix.json"
        if path.is_file():
            report.setdefault("mix", {})[name] = json.loads(path.read_text("utf-8"))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "predictor_real_data.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    text = markdown(report, notes)
    (args.results / "predictor_real_data.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
