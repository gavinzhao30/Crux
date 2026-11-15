"""Compare Task 9 stated, sampled and Jev outcome probabilities on real-count cases.

Run with the Task 9 fault suite, including its private evaluator truth:

    python -m eval.calibrate_task9 --suite /path/to/suite --work-dir runs/du31

Omit `--suite` to reuse collected observations and the per-provider caches. Set
`OPENAI_API_KEY` and `TYPESAFE_API_KEY` in the shell to score both model arms.
Use `--methods stated` for an offline report or `--collect-only` to record outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx

from agents.jev_predictor import JevPredictor, discover_model, probability_row
from agents.llm import CHEAP_MODEL, LLM, LLMConfig
from agents.predictor import SampledPredictor
from agents.tinker_predictor import profile_summary
from core.eig import expected_info_gain, score
from core.orchestrator import RELEVANCE_TV
from diagnostics.de_from_counts.run_all import run_all
from diagnostics.de_from_counts.synth import FAULT_FORK
from eval.eval_predictor import OPENAI_PRICE, floor, metrics, model_only, record_key
from eval.predictor_data.dataset import library_index
from eval.predictor_data.make_examples import profile_case

METHODS = ("stated", "sampled", "jev")
EPS = 0.1


def collect_case(case: Path, spec: dict) -> list[dict]:
    profile = profile_summary(profile_case(case))
    fault_fork = FAULT_FORK.get(spec["fault"])
    records = []
    for result in run_all(case):
        entry = library_index()[result["diagnostic"]]
        records.append(
            {
                "case_id": case.name,
                "split": spec["split"],
                "diagnostic": result["diagnostic"],
                "answer_key": entry.fault_key
                if fault_fork == entry.fork
                else entry.null_key,
                "observed_bin": result["outcome"],
                "error": result["error"],
                "profile": profile,
            }
        )
    return records


def collect(suite: Path, records_path: Path) -> None:
    cases = json.loads((suite / "evaluator/truth.json").read_text(encoding="utf-8"))[
        "cases"
    ]
    names = sorted(cases)
    with ProcessPoolExecutor(max_workers=4) as pool:
        batches = pool.map(
            collect_case,
            [suite / "cases" / name for name in names],
            [cases[n] for n in names],
        )
        temp = records_path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for records in batches:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
                print(f"Recorded {records[0]['case_id']}", flush=True)
        temp.replace(records_path)


def eligible_records(records: list[dict]) -> list[dict]:
    return [
        r
        for r in records
        if r["observed_bin"] is not None
        and (library_index()[r["diagnostic"]].experiment.design or {}).get("stated")
    ]


def stated_rows(records: list[dict]) -> dict:
    return {
        record_key(r): library_index()[r["diagnostic"]].experiment.design["stated"]
        for r in records
    }


async def predict_rows(predictor, records: list[dict], *, concurrency: int) -> dict:
    limit = asyncio.Semaphore(concurrency)
    inputs = {
        record_key(record): (record["diagnostic"], record["profile"])
        for record in records
    }

    async def one(diagnostic: str, profile: str):
        entry = library_index()[diagnostic]
        options = list(entry.options.values())
        context = (
            f"Data profile: {profile}\nQuestion: {entry.question_text}\nOptions: "
            + "; ".join(option.statement for option in options)
        )
        experiment = model_only(diagnostic)
        async with limit:
            # Outcomes and the planted answer key stay in the evaluator.
            rows = await predictor.predict(experiment, options, context)
        bins = [b.id for b in experiment.bins]
        return {
            key: probability_row(rows[h.id], bins) for key, h in entry.options.items()
        }

    unique = list(dict.fromkeys(inputs.values()))
    predictions = await asyncio.gather(*(one(*item) for item in unique))
    shared = dict(zip(unique, predictions, strict=True))
    return {key: shared[item] for key, item in inputs.items()}


def reliability(pairs: list[tuple[dict[str, float], str]]) -> list[dict]:
    buckets = defaultdict(list)
    for row, observed in pairs:
        top = max(row, key=row.get)
        confidence = row[top]
        buckets[min(9, int(10 * confidence))].append((confidence, top == observed))
    return [
        {
            "low": index / 10,
            "high": (index + 1) / 10,
            "n": len(values),
            "mean_confidence": sum(p for p, _ in values) / len(values),
            "observed_accuracy": sum(hit for _, hit in values) / len(values),
        }
        for index, values in sorted(buckets.items())
    ]


def first_choice(rows: list[dict], tables: dict) -> str | None:
    ranked = {}
    for r in rows:
        name = r["diagnostic"]
        entry = library_index()[name]
        table = tables[record_key(r)]
        bins = [b.id for b in entry.experiment.bins]
        option_rows = [
            floor(table[key], EPS) for key in (entry.null_key, entry.fault_key)
        ]
        a, b = ([option[bin_id] for bin_id in bins] for option in option_rows)
        tv = sum(abs(p - q) for p, q in zip(a, b, strict=True)) / 2
        if tv < RELEVANCE_TV:
            continue
        eig = expected_info_gain(
            [0.425, 0.425, 0.15], [a, b, [1 / len(bins)] * len(bins)]
        )
        if eig >= 0.05:
            ranked[name] = score(eig, entry.experiment.est_cost_s)
    return max(ranked, key=ranked.get) if ranked else None


def evaluate(records: list[dict], tables: dict[str, dict], statuses: dict) -> dict:
    stated_diagnostics = sorted({r["diagnostic"] for r in eligible_records(records)})
    out = {
        "eps": EPS,
        "stated_diagnostics": stated_diagnostics,
        "status": statuses,
        "splits": {},
    }
    for split in ("dev", "test"):
        all_rows = [r for r in records if r["split"] == split]
        eligible = eligible_records(all_rows)
        matched = [
            r for r in eligible if all(record_key(r) in t for t in tables.values())
        ]
        groups = defaultdict(list)
        for r in matched:
            groups[(r["case_id"], library_index()[r["diagnostic"]].fork)].append(r)
        ranked = [g for g in groups.values() if len(g) > 1]
        baseline = [first_choice(g, tables["stated"]) for g in ranked]
        summary = {
            "executed": len(all_rows),
            "invalid": sum(r["observed_bin"] is None for r in all_rows),
            "eligible": len(eligible),
            "matched": len(matched),
            "methods": {},
        }
        for method, table in tables.items():
            pairs = [
                (floor(table[record_key(r)][r["answer_key"]], EPS), r["observed_bin"])
                for r in matched
            ]
            choices = [first_choice(g, table) for g in ranked]
            summary["methods"][method] = {
                **metrics(pairs),
                "reliability": reliability(pairs),
                "ranking_groups": len(ranked),
                "changed_vs_stated": sum(
                    a != b for a, b in zip(baseline, choices, strict=True)
                ),
                "no_choice": choices.count(None),
            }
        out["splits"][split] = summary
    test = out["splits"]["test"]
    if (
        set(tables) == set(METHODS)
        and test["matched"] == test["eligible"]
        and test["matched"]
    ):
        losses = {m: test["methods"][m]["log_loss"] for m in METHODS}
        out["heldout_log_loss_winner"] = min(losses, key=losses.get)
        out["jev_beats_both"] = losses["jev"] < min(losses["stated"], losses["sampled"])
    else:
        out["heldout_log_loss_winner"] = None
        out["jev_beats_both"] = None
    return out


def markdown(report: dict) -> str:
    lines = [
        "# Task 9 outcome calibration (DU-31)",
        "",
        (
            "Observed bins come from executing the library diagnostics locally on the real-count "
            "cases. The matched comparison uses diagnostics with library-stated rows "
            "and conditions on the evaluator's known option. All arms use the same 0.1 probability "
            "floor. Lower log loss and Brier are better."
        ),
        "",
        "Stated probability tables are available for "
        + ", ".join(f"`{name}`" for name in report["stated_diagnostics"])
        + ".",
        "",
        (
            "Offline rankings replay the first choice within each eligible case and fork, with "
            "the same priors, candidate costs and minimum EIG. They count changes relative to "
            "Stated, including decisions to skip every candidate."
        ),
        "",
    ]
    for split, group in report["splits"].items():
        lines.extend(
            [
                f"## {split.title()}",
                "",
                (
                    f"{group['executed']} diagnostic executions, {group['invalid']} invalid, "
                    f"{group['matched']} matched with stated rows."
                ),
                "",
                "| Predictor | n | Log loss | Brier | ECE | First choices changed |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            item = group["methods"].get(method, {})
            if item.get("n"):
                lines.append(
                    f"| {method.title()} | {item['n']} | {item['log_loss']:.4f} | "
                    f"{item['brier']:.4f} | {item['ece']:.4f} | "
                    f"{item['changed_vs_stated']}/{item['ranking_groups']} |"
                )
            else:
                lines.append(f"| {method.title()} | 0 | — | — | — | — |")
        lines.extend(
            [
                "",
                "| Predictor | Confidence bucket | n | Mean confidence | Accuracy |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for method, item in group["methods"].items():
            for bucket in item["reliability"]:
                lines.append(
                    f"| {method.title()} | {bucket['low']:.1f}–{bucket['high']:.1f} | "
                    f"{bucket['n']} | {bucket['mean_confidence']:.3f} | "
                    f"{bucket['observed_accuracy']:.3f} |"
                )
        lines.append("")
    lines.extend(
        f"- {method.title()} — {status}" for method, status in report["status"].items()
    )
    winner = report["heldout_log_loss_winner"]
    if winner:
        lines.extend(["", f"Lowest held-out log loss belongs to **{winner.title()}**."])
        if not report["jev_beats_both"]:
            lines.extend(
                [
                    "",
                    (
                        "Keep the sampled predictor as the fallback where library probabilities "
                        "are unavailable. For a direct fallback comparison, evaluate Jev and "
                        "eight sampled draws on diagnostic–hypothesis pairs with no library row. "
                        "Give both predictors the same inputs, score held-out log loss against "
                        "observed bins, and replay first diagnostic choices."
                    ),
                ]
            )
    else:
        lines.extend(
            ["", "The three-way decision requires full matched held-out coverage."]
        )
    return "\n".join(lines) + "\n"


async def compare(records: list[dict], work: Path, methods: set[str]) -> dict:
    eligible = eligible_records(records)
    tables = {"stated": stated_rows(eligible)}
    statuses = {"stated": "Library-authored approximate probability tables"}
    if "sampled" in methods:
        config = LLMConfig(
            cache_dir=work / "sampled-cache",
            max_concurrency=16,
            prices_per_mtok=OPENAI_PRICE,
            role_models={"default": CHEAP_MODEL},
        )
        llm = LLM(config)
        tables["sampled"] = await predict_rows(
            SampledPredictor(llm, n=8, eps=0.0), eligible, concurrency=8
        )
        totals = llm.usage_totals()
        statuses["sampled"] = (
            f"{CHEAP_MODEL}, eight draws per option, "
            f"{totals['calls'] - totals['cached_calls']} fresh calls, "
            f"${totals['cost_usd']:.3f} estimated"
        )
    if "jev" in methods:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            statuses["jev"] = "Pending TypeSafe API key"
        else:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {key}"}, timeout=30
            ) as client:
                try:
                    model = await discover_model(client)
                    tables["jev"] = await predict_rows(
                        JevPredictor(client, model, work / "jev-cache"),
                        eligible,
                        concurrency=4,
                    )
                except httpx.HTTPStatusError as exc:
                    statuses["jev"] = (
                        f"Pending after TypeSafe HTTP {exc.response.status_code}"
                    )
                else:
                    statuses["jev"] = (
                        f"{model}, one batched Choice request per experiment"
                    )
    return evaluate(records, tables, statuses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=Path("eval/results"))
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    methods = set(args.methods.split(","))
    if methods - set(METHODS):
        parser.error("Choose methods from stated,sampled,jev")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    path = args.work_dir / "records.jsonl"
    if args.suite is not None:
        collect(args.suite.resolve(), path)
    if not path.is_file():
        parser.error("Pass --suite to collect real diagnostic outcomes first")
    if args.collect_only:
        return
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    report = asyncio.run(compare(records, args.work_dir, methods))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "task9_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.results / "task9_calibration.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(f"Calibration reports written to {args.results}")


if __name__ == "__main__":
    main()
