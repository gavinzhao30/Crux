"""Calibration of outcome predictors on held-out synthetic Task 9 cases (DU-29).

For every record in the ``test`` and ``shift`` splits (one case x one library diagnostic,
with the fork's true option and the observed bin), each method gives P(bin | true option):

* ``library``: the library's ``design["exact"]`` or ``design["stated"]`` row.
* ``sampled``: ``agents.predictor.SampledPredictor`` (gpt-4.1-mini, 8 samples). It sees the
  diagnostic and the orchestrator's context (goal and question) but not the data, so its rows
  depend only on (diagnostic, option): ``--sampled-repeats`` independent rows are drawn per
  pair (cache off) and records use them round-robin. Spend is capped by ``--openai-max-usd``.
* ``base``: Qwen/Qwen3-8B before fine-tuning, scored like ``TinkerPredictor`` (sum of the
  completion's token log-probabilities per bin, softmax over bins).
* ``finetuned``: the LoRA model in ``runs/predictor_data/model.json``, scored the same way.

Every method scores every diagnostic ("model only": exact rows are removed before asking the
model predictors), and all rows get the predictor's floor ``(1 - eps) * p + eps / n_bins``
(``--eps``, default 0.1, the engine default). Metrics, overall and per fork: log loss (nats),
multi-class Brier, argmax accuracy and top-label ECE (10 bins); plus option sensitivity, the
mean total-variation distance between a diagnostic's rows under the fork's two options (a
predictor that ignores the option scores 0 and makes every diagnostic look uninformative).

Model outputs are cached under ``runs/predictor_data/eval_<method>.jsonl``, so methods can run
at different times; the report always uses every cached method.

    set -a; . /path/to/.env; set +a
    uv run --group tinker python -m eval.eval_predictor --methods library,sampled,base,finetuned
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import zlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from agents.tinker_predictor import BASE_MODEL, TinkerPredictor, logprobs_to_row
from eval.predictor_data.dataset import (
    DATA_DIR,
    add_to_ledger,
    library_index,
    load_split,
    read_ledger,
)

RESULTS = Path("eval/results")
METHODS = ("library", "sampled", "base", "finetuned")
LABELS = {
    "library": "(a) library exact/stated rows",
    "sampled": "(b) SampledPredictor, gpt-4.1-mini x8",
    "base": "(c) Qwen3-8B base, logprob scoring",
    "finetuned": "(d) Qwen3-8B LoRA fine-tune",
}
STATED_FORKS = ("sample_labels", "group_direction")
PREFILL_PRICE = 0.195  # $ per million prompt tokens, Qwen3-8B on Tinker
OPENAI_PRICE = {"gpt-4.1-mini": (0.40, 1.60)}  # $ per million input, output tokens
FORK_ORDER = (
    "sample_labels",
    "duplicates",
    "sample_quality",
    "cell_line",
    "gene_ids",
    "group_direction",
)


# --- rows -----------------------------------------------------------------------------


def floor(row: dict[str, float], eps: float) -> dict[str, float]:
    total = sum(row.values())
    n = len(row)
    return {b: (1.0 - eps) * v / total + eps / n for b, v in row.items()}


def library_row(diagnostic: str, answer_key: str) -> dict[str, float]:
    design = library_index()[diagnostic].experiment.design or {}
    table = design.get("exact") or design.get("stated") or {}
    return dict(table[answer_key])


def model_only(diagnostic: str):
    """The library experiment without its exact/stated rows, so a predictor scores it."""
    exp = library_index()[diagnostic].experiment.model_copy(deep=True)
    exp.design = {
        k: v for k, v in (exp.design or {}).items() if k not in ("exact", "stated")
    }
    return exp


def record_key(record: dict) -> str:
    return f"{record['case_id']}|{record['diagnostic']}"


# --- metrics --------------------------------------------------------------------------


def metrics(pairs: Iterable[tuple[dict[str, float], str]]) -> dict:
    """Log loss (nats), Brier, argmax accuracy and top-label ECE over (row, observed bin)."""
    pairs = list(pairs)
    if not pairs:
        return {"n": 0}
    loss = brier = correct = 0.0
    confidences, hits = [], []
    impossible = 0
    for row, observed in pairs:
        p = row.get(observed, 0.0)
        impossible += p <= 0.0
        loss += -math.log(max(p, 1e-12))
        brier += sum((v - (1.0 if b == observed else 0.0)) ** 2 for b, v in row.items())
        top = max(row, key=row.get)
        hit = top == observed
        correct += hit
        confidences.append(row[top])
        hits.append(hit)
    n = len(pairs)
    return {
        "n": n,
        "log_loss": loss / n,
        "brier": brier / n,
        "accuracy": correct / n,
        "ece": ece(confidences, hits),
        "impossible": impossible,
    }


def ece(confidences: list[float], hits: list[bool], n_bins: int = 10) -> float:
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, h in zip(confidences, hits, strict=True):
        buckets[min(n_bins - 1, int(c * n_bins))].append((c, h))
    total = len(confidences)
    return sum(
        len(items)
        / total
        * abs(
            sum(c for c, _ in items) / len(items)
            - sum(h for _, h in items) / len(items)
        )
        for items in buckets.values()
    )


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a[k] - b.get(k, 0.0)) for k in a)


# --- model predictions ------------------------------------------------------------------


def cache_path(root: Path, method: str) -> Path:
    return root / f"eval_{method}.jsonl"


def load_cache(root: Path, method: str) -> dict[str, dict]:
    path = cache_path(root, method)
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            out[entry["key"]] = entry
    return out


def estimate_tinker_tokens(records: list[dict], prompt_tokens: float = 240.0) -> float:
    index = library_index()
    return sum(
        2 * len(index[r["diagnostic"]].experiment.bins) * prompt_tokens for r in records
    )


async def run_tinker(
    method: str,
    target: str,
    records: list[dict],
    root: Path,
    *,
    concurrency: int,
    chunk: int = 200,
) -> float:
    """Score both fork options for every record not yet cached; returns $ spent (estimate)."""
    done = load_cache(root, method)
    todo = [r for r in records if record_key(r) not in done]
    if not todo:
        return 0.0
    predictor = TinkerPredictor(target, eps=0.0, max_concurrency=concurrency)
    index = library_index()
    spent_tokens = 0

    async def one(record: dict) -> dict:
        entry = index[record["diagnostic"]]
        exp = model_only(record["diagnostic"])
        rows = {}
        for key in (entry.null_key, entry.fault_key):
            rows[key] = await predictor.score(
                exp, entry.options[key], [record["profile"]]
            )
        return {"key": record_key(record), "logprobs": rows}

    with cache_path(root, method).open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), chunk):
            batch = todo[start : start + chunk]
            before = predictor.prompt_tokens
            results = await asyncio.gather(*(one(r) for r in batch))
            spent_tokens += predictor.prompt_tokens - before
            for result in results:
                handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(
                f"{method}: {start + len(batch)}/{len(todo)} records, "
                f"{spent_tokens / 1e6:.2f}M prompt tokens",
                flush=True,
            )
    return spent_tokens * PREFILL_PRICE / 1e6


async def run_sampled(
    records: list[dict], root: Path, *, repeats: int, max_usd: float
) -> float:
    """``repeats`` independent SampledPredictor rows per (diagnostic, option), cache off."""
    from agents.llm import LLM, LLMConfig
    from agents.predictor import SampledPredictor
    from labs.bixbench3.task9_contract import load_prompt

    done = load_cache(root, "sampled")
    goal = load_prompt()[0]
    config = LLMConfig(
        cache=False, prices_per_mtok=dict(OPENAI_PRICE), max_concurrency=16
    )
    llm = LLM(config)
    predictor = SampledPredictor(llm, n=8, eps=0.0)
    index = library_index()
    diagnostics = sorted({r["diagnostic"] for r in records})
    with cache_path(root, "sampled").open("a", encoding="utf-8") as handle:
        for repeat in range(repeats):
            jobs = []
            for diagnostic in diagnostics:
                key = f"{diagnostic}|{repeat}"
                if key in done:
                    continue
                entry = index[diagnostic]
                context = f"Goal: {goal}\nQuestion: {entry.question_text}"
                hyps = [entry.options[entry.null_key], entry.options[entry.fault_key]]
                jobs.append(
                    (key, predictor.predict(model_only(diagnostic), hyps, context))
                )
            if not jobs:
                continue
            if llm.usage_totals()["cost_usd"] >= max_usd:
                print("sampled: OpenAI spend cap reached; stopping repeats", flush=True)
                break
            tables = await asyncio.gather(*(job for _, job in jobs))
            for (key, _), table in zip(jobs, tables, strict=True):
                diagnostic = key.split("|")[0]
                entry = index[diagnostic]
                rows = {
                    h_key: table[entry.options[h_key].id]
                    for h_key in (entry.null_key, entry.fault_key)
                }
                handle.write(json.dumps({"key": key, "rows": rows}) + "\n")
            handle.flush()
            totals = llm.usage_totals()
            print(
                f"sampled: repeat {repeat + 1}/{repeats}, ${totals['cost_usd']:.3f} so far",
                flush=True,
            )
    return float(llm.usage_totals()["cost_usd"])


# --- assembling rows ----------------------------------------------------------------------


def method_rows(
    method: str, records: list[dict], root: Path
) -> dict[str, dict[str, dict[str, float]]] | None:
    """record key -> {answer_key: raw row} for every option scored, or None if not run."""
    index = library_index()
    if method == "library":
        return {
            record_key(r): {
                k: library_row(r["diagnostic"], k)
                for k in (
                    index[r["diagnostic"]].null_key,
                    index[r["diagnostic"]].fault_key,
                )
            }
            for r in records
        }
    cache = load_cache(root, method)
    if not cache:
        return None
    if method == "sampled":
        repeats: dict[str, list[dict]] = defaultdict(list)
        for key, entry in sorted(cache.items()):
            repeats[key.split("|")[0]].append(entry["rows"])
        out = {}
        for r in records:
            draws = repeats.get(r["diagnostic"])
            if draws:
                pick = zlib.crc32(r["case_id"].encode()) % len(draws)
                out[record_key(r)] = draws[pick]
        return out
    return {
        key: {k: logprobs_to_row(lp, 0.0) for k, lp in entry["logprobs"].items()}
        for key, entry in cache.items()
    }


def summarise(
    records: list[dict], rows: dict[str, dict[str, dict[str, float]]], eps: float
) -> dict:
    """Overall, per-fork and stated-fork metrics plus option sensitivity."""
    index = library_index()
    pairs: dict[str, list] = defaultdict(list)
    sensitivity: dict[str, list[float]] = defaultdict(list)
    for r in records:
        table = rows.get(record_key(r))
        if table is None or r["answer_key"] not in table:
            continue
        fork = r["fork"]
        row = floor(table[r["answer_key"]], eps)
        for group in ("overall", fork, "stated" if fork in STATED_FORKS else "exact"):
            pairs[group].append((row, r["observed_bin"]))
        entry = index[r["diagnostic"]]
        if entry.null_key in table and entry.fault_key in table:
            tv = total_variation(
                floor(table[entry.null_key], eps), floor(table[entry.fault_key], eps)
            )
            for group in (
                "overall",
                fork,
                "stated" if fork in STATED_FORKS else "exact",
            ):
                sensitivity[group].append(tv)
    out = {}
    for group, items in pairs.items():
        out[group] = metrics(items)
        tvs = sensitivity.get(group) or []
        out[group]["option_tv"] = sum(tvs) / len(tvs) if tvs else None
    return out


def evaluate(root: Path, eps: float, methods: Iterable[str]) -> dict:
    splits = {"test": load_split("test", root), "shift": load_split("shift", root)}
    for kind in ("weak_effect", "three_replicates"):
        splits[f"shift:{kind}"] = [
            r for r in splits["shift"] if r["shift_kind"] == kind
        ]
    report: dict = {"eps": eps, "methods": {}, "splits": {}}
    for split, records in splits.items():
        report["splits"][split] = {
            "records": len(records),
            "cases": len({r["case_id"] for r in records}),
        }
    for method in methods:
        rows = method_rows(method, splits["test"] + splits["shift"], root)
        if rows is None:
            continue
        report["methods"][method] = {
            split: summarise(records, rows, eps) for split, records in splits.items()
        }
        raw = {
            split: summarise(records, rows, 0.0) for split, records in splits.items()
        }
        report["methods"][method]["_raw_overall"] = {
            split: raw[split].get("overall", {}) for split in splits
        }
    return report


# --- report -----------------------------------------------------------------------------


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(report: dict, notes: dict) -> str:
    methods = [m for m in METHODS if m in report["methods"]]
    lines = [
        "# Outcome predictor calibration (DU-29)",
        "",
        (
            "P(observed bin | true option) on held-out synthetic Task 9 cases. Each case plants "
            "zero or one fault; all 12 library diagnostics run on it. Every method scores every "
            f"diagnostic, with the predictor floor eps = {report['eps']} applied to all rows. "
            "Log loss in nats; lower is better for log loss, Brier and ECE. `option TV` is the "
            "mean total-variation distance between a diagnostic's rows under the fork's two "
            "options (0 = the predictor ignores the option)."
        ),
        "",
        "Splits: "
        + ", ".join(
            f"`{s}` {v['cases']} cases / {v['records']} records"
            for s, v in report["splits"].items()
        )
        + ". `test` shares the training knob distribution; `shift` uses knob ranges never "
        "seen in training (half weak knockout effect, half three replicates per group).",
        "",
    ]
    header = "| method | split | n | log loss | Brier | accuracy | ECE | option TV |"
    rule = "|---|---|---|---|---|---|---|---|"
    lines += ["## All diagnostics", "", header, rule]
    for split in ("test", "shift", "shift:weak_effect", "shift:three_replicates"):
        for m in methods:
            s = report["methods"][m][split].get("overall", {})
            lines.append(
                f"| {LABELS[m]} | {split} | {s.get('n', 0)} | {_fmt(s.get('log_loss'))} | "
                f"{_fmt(s.get('brier'))} | {_fmt(s.get('accuracy'))} | {_fmt(s.get('ece'))} | "
                f"{_fmt(s.get('option_tv'))} |"
            )
    lines += [
        "",
        "## Stated-row forks only (sample_labels, group_direction)",
        "",
        (
            "These four diagnostics have no exact rows. Since PR #22 both predictors use the "
            "library's stated rows for them by default; with `use_stated_rows=False` "
            "TinkerPredictor fills them instead. On the other eight, exact library rows win."
        ),
        "",
        header,
        rule,
    ]
    for split in ("test", "shift"):
        for m in methods:
            s = report["methods"][m][split].get("stated", {})
            lines.append(
                f"| {LABELS[m]} | {split} | {s.get('n', 0)} | {_fmt(s.get('log_loss'))} | "
                f"{_fmt(s.get('brier'))} | {_fmt(s.get('accuracy'))} | {_fmt(s.get('ece'))} | "
                f"{_fmt(s.get('option_tv'))} |"
            )
    lines += [
        "",
        "## Per fork, log loss (test / shift)",
        "",
        "| fork | " + " | ".join(LABELS[m] for m in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for fork in FORK_ORDER:
        cells = []
        for m in methods:
            t = report["methods"][m]["test"].get(fork, {}).get("log_loss")
            s = report["methods"][m]["shift"].get(fork, {}).get("log_loss")
            cells.append(f"{_fmt(t)} / {_fmt(s)}")
        lines.append(f"| {fork} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Per fork, accuracy (test / shift)",
        "",
        "| fork | " + " | ".join(LABELS[m] for m in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for fork in FORK_ORDER:
        cells = []
        for m in methods:
            t = report["methods"][m]["test"].get(fork, {}).get("accuracy")
            s = report["methods"][m]["shift"].get(fork, {}).get("accuracy")
            cells.append(f"{_fmt(t)} / {_fmt(s)}")
        lines.append(f"| {fork} | " + " | ".join(cells) + " |")
    if "library" in report["methods"]:
        raw = report["methods"]["library"]["_raw_overall"]
        lines += [
            "",
            "Library rows taken literally (no floor): "
            + ", ".join(
                f"{raw[s].get('impossible', 0)} of {raw[s].get('n', 0)} {s} records"
                for s in ("test", "shift")
            )
            + " landed in a bin the row gives probability 0.",
        ]
    model_methods = [m for m in ("sampled", "base", "finetuned") if m in methods]
    if model_methods:
        lines += [
            "",
            "## Without the floor (eps = 0), all diagnostics",
            "",
            "| method | split | log loss | Brier | accuracy | ECE |",
            "|---|---|---|---|---|---|",
        ]
        for split in ("test", "shift"):
            for m in model_methods:
                s = report["methods"][m]["_raw_overall"][split]
                lines.append(
                    f"| {LABELS[m]} | {split} | {_fmt(s.get('log_loss'))} | "
                    f"{_fmt(s.get('brier'))} | {_fmt(s.get('accuracy'))} | "
                    f"{_fmt(s.get('ece'))} |"
                )
    if notes.get("interpretation"):
        lines += ["", "## Interpretation", "", notes["interpretation"]]
    if notes.get("setup"):
        lines += ["", "## Setup", "", notes["setup"]]
    return "\n".join(lines) + "\n"


# --- driver -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibration of the outcome predictors"
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--sampled-repeats", type=int, default=4)
    parser.add_argument("--openai-max-usd", type=float, default=3.0)
    parser.add_argument(
        "--max-usd", type=float, default=15.0, help="Tinker cap incl. ledger"
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--notes", type=Path, default=None, help="JSON with report prose"
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.data)
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    records = load_split("test", root) + load_split("shift", root)

    if not args.report_only:
        for method in wanted:
            if method == "sampled":
                usd = asyncio.run(
                    run_sampled(
                        records,
                        root,
                        repeats=args.sampled_repeats,
                        max_usd=args.openai_max_usd,
                    )
                )
                if usd:
                    add_to_ledger(
                        root, "eval-sampled-openai", usd, {"provider": "openai"}
                    )
            elif method in ("base", "finetuned"):
                if method == "base":
                    target = BASE_MODEL
                else:
                    model = json.loads(
                        (root / "model.json").read_text(encoding="utf-8")
                    )
                    target = model["model_path"]
                done = load_cache(root, method)
                todo = [r for r in records if record_key(r) not in done]
                est = estimate_tinker_tokens(todo) * PREFILL_PRICE / 1e6
                ledger = read_ledger(root)
                tinker_spent = sum(
                    e["usd"] for e in ledger["entries"] if e.get("provider") != "openai"
                )
                print(
                    f"{method}: {len(todo)} records to score, estimated ${est:.2f} "
                    f"(Tinker ledger ${tinker_spent:.2f}, cap ${args.max_usd:.2f})",
                    flush=True,
                )
                if est + tinker_spent > args.max_usd:
                    print(f"Refusing {method}: estimate exceeds --max-usd", flush=True)
                    continue
                usd = asyncio.run(
                    run_tinker(
                        method, target, records, root, concurrency=args.concurrency
                    )
                )
                if usd:
                    add_to_ledger(root, f"eval-{method}", usd, {"provider": "tinker"})

    report = evaluate(root, args.eps, METHODS)
    notes = json.loads(args.notes.read_text(encoding="utf-8")) if args.notes else {}
    report["notes"] = notes
    report["spend"] = read_ledger(root)
    model_json = root / "model.json"
    if model_json.is_file():
        model = json.loads(model_json.read_text(encoding="utf-8"))
        report["model"] = {k: v for k, v in model.items() if k != "model_path"}
        report["model"]["model_path"] = model.get("model_path")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "predictor_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.results / "predictor_calibration.md").write_text(
        markdown(report, notes), encoding="utf-8"
    )
    print(markdown(report, notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
