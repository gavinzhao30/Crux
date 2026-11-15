"""Predictor records from real RNA-seq datasets with planted faults (DU-54).

The DU-29 predictor was trained and tested only on synthetic cases. This builds the same
records from the real case folders of ``eval/real_data/prepare_cases.py``:

1. **Plant.** ``eval/suites/plant_faults.py`` plants the six faults on every real case, over
   several seeds (each seed blinds and shuffles the samples and picks different affected
   samples). Faults a design cannot support are skipped and listed in the manifest.
2. **Run.** Every planted case is profiled with the lab's ``_PROFILE`` and goes through all 12
   library diagnostics in-process, exactly as ``make_examples.make_case`` does for synthetic
   cases. One JSONL record per case and diagnostic, with the same schema, so the trainer and
   the evaluators take them unchanged.
3. **Label.** ``answer_key`` comes from the planter's private truth file: the fault option for
   the planted fault's fork, else the fork's null option.

Splits are by dataset, never by case. COVID (E-ENAD-46) is held out entirely as ``real_test``.
Airway (both views of GSE52778, which are the same data, so always together) and pasilla are
``real_train``, except the cases of the last seed, which are ``real_val``.

    uv run python -m eval.real_data.prepare_cases --out ../data
    uv run python -m eval.predictor_data.make_real_examples generate --cases ../data/cases
    uv run python -m eval.predictor_data.make_real_examples mix --upweight 2

``mix`` writes a training root (``train``/``val``) with the synthetic training data plus the
real training data (repeated ``--upweight`` times), for ``eval/train_tinker_predictor.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from agents.tinker_predictor import profile_summary
from diagnostics.de_from_counts.synth import FAULT_FORK
from eval.predictor_data.dataset import DATA_DIR, load_split
from eval.predictor_data.make_examples import _experiments, profile_case, run_code
from eval.suites.plant_faults import plant
from labs.bixbench3.bins import choose_bin

REAL_DIR = Path("runs/predictor_real")
MIX_DIR = Path("runs/predictor_mix")
# real case folder -> dataset; the two airway views are the same samples
DATASETS = {
    "covid-lung-colon": "covid",
    "airway-treatment-groups": "airway",
    "airway-donor-x-treatment": "airway",
    "pasilla-knockdown": "pasilla",
}
TEST_DATASETS = ("covid",)
# (dataset, fork) pairs whose "no fault" answer key is doubtful because the public data itself
# has a problem there. Fixed before any model was scored; reported separately, and dropped from
# training by ``mix --drop-doubtful``.
DOUBTFUL = {
    ("covid", "sample_quality"): "library SRR12816735 is low complexity",
    ("airway", "sample_labels"): "albuterol labels look swapped in donors N052611, N080611",
    ("airway", "group_direction"): "albuterol labels look swapped in donors N052611, N080611",
}
SEEDS = (101, 102, 103, 104, 105)
REAL_SPLITS = ("real_train", "real_val", "real_test")


def split_for(dataset: str, seed: int, seeds: tuple[int, ...]) -> str:
    if dataset in TEST_DATASETS:
        return "real_test"
    return "real_val" if seed == seeds[-1] else "real_train"


# --- planting ---------------------------------------------------------------------------


def plant_suites(
    cases_dir: Path,
    out: Path,
    seeds: tuple[int, ...],
    instances: int,
    datasets: dict[str, str] = DATASETS,
) -> list[dict]:
    """Plant every real case for every seed (kept if already planted); one spec per case."""
    specs = []
    for source, dataset in datasets.items():
        case = cases_dir / source
        if not case.is_dir():
            raise FileNotFoundError(f"{case} is missing: run eval.real_data.prepare_cases")
        for seed in seeds:
            suite = out / "suites" / f"{source}-s{seed}"
            truth_path = suite / "evaluator" / "truth.json"
            if not truth_path.is_file():
                shutil.rmtree(suite, ignore_errors=True)
                suite.parent.mkdir(parents=True, exist_ok=True)
                plant(case, suite, seed=seed, control="Ctrl", instances=instances)
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            for name, entry in sorted(truth["cases"].items()):
                fault = None if entry["fault"] == "clean" else entry["fault"]
                specs.append(
                    {
                        "case_id": f"{source}-s{seed}-{name}",
                        "dataset": dataset,
                        "source": source,
                        "seed": seed,
                        "split": split_for(dataset, seed, seeds),
                        "fault": fault,
                        "implicated": list(entry.get("affected_samples") or []),
                        "folder": str(suite / "cases" / name),
                        "skipped": truth.get("skipped") or {},
                    }
                )
    return specs


# --- running ----------------------------------------------------------------------------


def make_real_case(spec: dict, scratch: str | None = None) -> list[dict]:
    """Profile one planted real case and run every diagnostic on a scratch copy."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    folder = Path(tempfile.mkdtemp(prefix="crux-real-", dir=scratch))
    try:
        shutil.copytree(spec["folder"], folder, dirs_exist_ok=True)
        profile = profile_case(folder)
        summary = profile_summary([profile])
        fault_fork = FAULT_FORK.get(spec["fault"]) if spec["fault"] else None
        records = []
        for question, exp in _experiments():
            keys = [o.answer_key for o in question.proposal.options]
            answer = keys[1] if question.key == fault_fork else keys[0]
            result, error, observed = None, None, None
            try:
                result = run_code(exp.code, folder)
                observed = choose_bin(((b.id, b.rule) for b in exp.bins), result)
            except Exception as exc:  # noqa: BLE001 - an invalid run: kept, never trained on
                error = f"{type(exc).__name__}: {exc}"[:500]
            records.append(
                {
                    "case_id": spec["case_id"],
                    "split": spec["split"],
                    "shift_kind": spec["dataset"],
                    "dataset": spec["dataset"],
                    "source": spec["source"],
                    "seed": spec["seed"],
                    "knobs": None,
                    "layout": "real",
                    "fault": spec["fault"],
                    "fault_line": None,
                    "fault_severity": None,
                    "implicated": spec["implicated"],
                    "fork": question.key,
                    "diagnostic": exp.design["diagnostic"],
                    "experiment_id": exp.id,
                    "answer_key": answer,
                    "observed_bin": observed,
                    "result": result,
                    "error": error,
                    "profile": profile,
                    "profile_summary": summary,
                }
            )
        return records
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def generate(
    cases_dir: Path,
    out: Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    instances: int = 4,
    workers: int | None = None,
    datasets: dict[str, str] = DATASETS,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    specs = plant_suites(cases_dir, out, seeds, instances, datasets)
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    handles = {
        s: (out / f"{s}.jsonl").open("w", encoding="utf-8", newline="\n")
        for s in REAL_SPLITS
    }
    cases: dict[str, Counter] = defaultdict(Counter)
    invalid: Counter = Counter()
    try:
        if workers == 1:
            batches = map(make_real_case, specs)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            batches = pool.map(make_real_case, specs, chunksize=2)
        try:
            for spec, records in zip(specs, batches, strict=True):
                for record in records:
                    handles[record["split"]].write(json.dumps(record, sort_keys=True) + "\n")
                    invalid[spec["dataset"]] += record["observed_bin"] is None
                cases[f"{spec['split']}|{spec['dataset']}"][spec["fault"] or "clean"] += 1
                done = sum(sum(c.values()) for c in cases.values())
                if done % 25 == 0:
                    print(f"{done}/{len(specs)} cases", flush=True)
        finally:
            if workers != 1:
                pool.shutdown()
    finally:
        for handle in handles.values():
            handle.close()
    skipped = {s["source"]: s["skipped"] for s in specs if s["skipped"]}
    manifest = {
        "seeds": list(seeds),
        "instances": instances,
        "datasets": datasets,
        "test_datasets": list(TEST_DATASETS),
        "cases": {k: dict(sorted(v.items())) for k, v in sorted(cases.items())},
        "invalid_records": dict(invalid),
        "skipped_faults": skipped,
        "seconds": round(time.monotonic() - started, 1),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


# --- the training mix -------------------------------------------------------------------


def is_doubtful(record: dict) -> bool:
    return (record.get("dataset"), record["fork"]) in DOUBTFUL


def build_mix(
    synthetic: Path,
    real: Path,
    out: Path,
    *,
    upweight: int = 2,
    seed: int = 54,
    drop_doubtful: bool = False,
) -> dict:
    """``train`` = synthetic train + real_train x ``upweight``; ``val`` = both vals, shuffled.

    The trainer shuffles the validation records and keeps a fixed-size subset, so mixing them
    here puts real cases into early stopping in proportion. ``drop_doubtful`` leaves out the
    real records of the ``DOUBTFUL`` (dataset, fork) pairs, whose labels are not trustworthy.
    """
    if upweight < 1:
        raise ValueError("upweight must be >= 1")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    real_train, real_val = load_split("real_train", real), load_split("real_val", real)
    dropped = 0
    if drop_doubtful:
        before = len(real_train) + len(real_val)
        real_train = [r for r in real_train if not is_doubtful(r)]
        real_val = [r for r in real_val if not is_doubtful(r)]
        dropped = before - len(real_train) - len(real_val)
    syn_train = load_split("train", synthetic)
    train = syn_train + real_train * upweight
    rng.shuffle(train)
    val = load_split("val", synthetic) + real_val
    rng.shuffle(val)
    for name, records in (("train", train), ("val", val)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    info = {
        "synthetic_train": len(syn_train),
        "real_train": len(real_train),
        "upweight": upweight,
        "train": len(train),
        "val": len(val),
        "real_val": sum(r.get("layout") == "real" for r in val),
        "dropped_doubtful": dropped,
    }
    (out / "mix.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="plant the real cases and run the diagnostics")
    gen.add_argument("--cases", type=Path, default=Path("../data/cases"))
    gen.add_argument("--out", type=Path, default=REAL_DIR)
    gen.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    gen.add_argument("--instances", type=int, default=4)
    gen.add_argument("--workers", type=int, default=None)
    mix = sub.add_parser("mix", help="synthetic + real training root for the trainer")
    mix.add_argument("--synthetic", type=Path, default=DATA_DIR)
    mix.add_argument("--real", type=Path, default=REAL_DIR)
    mix.add_argument("--out", type=Path, default=MIX_DIR)
    mix.add_argument("--upweight", type=int, default=2)
    mix.add_argument("--drop-doubtful", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        manifest = generate(
            args.cases, args.out, seeds=seeds, instances=args.instances, workers=args.workers
        )
        print(json.dumps(manifest["cases"], indent=1))
        print(f"done in {manifest['seconds']} s", flush=True)
    else:
        info = build_mix(
            args.synthetic,
            args.real,
            args.out,
            upweight=args.upweight,
            drop_doubtful=args.drop_doubtful,
        )
        print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
