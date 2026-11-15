"""Generate predictor training and evaluation records from synthetic Task 9 cases (DU-29).

Each case draws realism knobs (``diagnostics.de_from_counts.synth.Knobs``), plants zero or one
fault (about a third clean, the rest spread evenly over the six faults and both cell lines),
profiles the data with the lab's own ``_PROFILE`` code and runs all 12 library diagnostics
in-process (``exec`` with ``CRUX_DATA_DIR`` set and stdout captured), cases spread over a
process pool. One JSONL record per case and diagnostic:

``case_id, split, shift_kind, seed, knobs, layout, fault, fault_line, implicated, fork,
diagnostic, experiment_id, answer_key, observed_bin, result, error, profile,
profile_summary``

``answer_key`` is the true option for the diagnostic's fork: the fault option when that fork's
fault was planted, else the fork's null option. ``profile`` is the parsed ``_PROFILE`` result
(what ``SwarmTask9Lab.profile_facts`` holds) and ``profile_summary`` its compact line.

Splits are by case. ``train``/``val``/``test`` share one knob distribution; ``shift`` cases
come from ranges never used in training: half ``weak_effect`` (knockout effect sd 0.2-0.45
instead of 0.5-1.2) and half ``three_replicates`` (training uses two).

    uv run python -m eval.predictor_data.make_examples --train 900 --val 120 --test 200 --shift 200
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from agents.tinker_predictor import profile_summary
from diagnostics.de_from_counts import library_questions
from diagnostics.de_from_counts.synth import (
    FAULT_FORK,
    FAULTS,
    GROUPS,
    LINE_FREE_FAULTS,
    Knobs,
    featurecounts_case,
    plant,
    simulate,
    suite_case,
    write_case,
)
from execution.local import rewrite_paths
from labs.bixbench3.bins import choose_bin
from labs.bixbench3.task9 import _PROFILE

DEFAULT_OUT = Path("runs/predictor_data")
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2, "shift": 3}
SHIFT_KINDS = ("weak_effect", "three_replicates")
# Three clean slots per six faults: about a third of the cases are clean.
_FAULT_PATTERN = (None, None, None, *FAULTS)


# --- running code in-process ----------------------------------------------------------


def run_code(code: str, data_dir: Path) -> dict:
    """Run one diagnostic (or the profile) here, as the sandbox would; parse ``RESULT:``."""
    os.environ["CRUX_DATA_DIR"] = str(data_dir)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<diagnostic>", "exec"), {"__name__": "__main__"})  # noqa: S102
    lines = [
        line for line in buffer.getvalue().splitlines() if line.startswith("RESULT:")
    ]
    if len(lines) != 1:
        raise ValueError(f"expected one RESULT line, got {len(lines)}")
    return json.loads(lines[0].removeprefix("RESULT:"))


def profile_case(data_dir: Path) -> dict:
    """The lab's ``_PROFILE`` with ``/data`` pointed at ``data_dir`` (as ``LocalExecutor`` does)."""
    roots = {"data": data_dir, "outputs": data_dir, "work": data_dir}
    return run_code(rewrite_paths(_PROFILE, roots), data_dir)


# --- cases ----------------------------------------------------------------------------


def _log_uniform(rng, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_knobs(rng, shift_kind: str | None = None) -> tuple[Knobs, dict]:
    """In-distribution knobs, or a shifted variant; every draw happens either way."""
    shared = bool(rng.random() < 0.75)
    rho = float(rng.uniform(0.4, 0.9))
    strong_target = bool(rng.random() < 0.8)
    target = float(rng.uniform(2.5, 5.0) if strong_target else rng.uniform(0.0, 1.5))
    knobs = Knobs(
        n_genes=round(_log_uniform(rng, 2000, 12000)),
        depth=_log_uniform(rng, 4e5, 2e7),
        depth_spread=(float(rng.uniform(0.4, 0.8)), float(rng.uniform(1.2, 2.0))),
        replicates=2,
        effect=float(rng.uniform(0.5, 1.2)),
        signature_frac=float(rng.uniform(0.04, 0.12)),
        line_effect=float(rng.uniform(0.6, 2.0)),
        clone_effect=float(rng.uniform(0.4, 1.0)),
        dispersion=_log_uniform(rng, 0.01, 0.2),
        target_log2_fold=target,
        **Knobs.sharing(shared, rho),
    )
    weak = float(rng.uniform(0.2, 0.45))
    if shift_kind == "weak_effect":
        knobs = replace(knobs, effect=weak)
    elif shift_kind == "three_replicates":
        knobs = replace(knobs, replicates=3)
    elif shift_kind is not None:
        raise ValueError(shift_kind)
    extras = {
        "shared_effects": shared,
        "layout": "featurecounts" if rng.random() < 0.3 else "suite",
        "low_depth_factor": _log_uniform(rng, 1 / 100, 1 / 15),
        "gene_version_frac": _log_uniform(rng, 0.005, 0.2),
    }
    return knobs, extras


def case_specs(split: str, n: int, base_seed: int) -> list[dict]:
    """Deterministic case specs for one split: seeds, balanced faults and shift kinds."""
    rng = np.random.default_rng([base_seed, SPLIT_INDEX[split], 0])
    faults = [_FAULT_PATTERN[i % len(_FAULT_PATTERN)] for i in range(n)]
    rng.shuffle(faults)
    specs = []
    for i in range(n):
        seed = int(
            np.random.SeedSequence(
                [base_seed, SPLIT_INDEX[split], i + 1]
            ).generate_state(1)[0]
        )
        specs.append(
            {
                "case_id": f"{split}-{i:05d}",
                "split": split,
                "shift_kind": SHIFT_KINDS[i % 2] if split == "shift" else None,
                "seed": seed,
                "fault": faults[i],
            }
        )
    return specs


def _experiments():
    return [(q, e) for q in library_questions() for e in q.diagnostics]


def make_case(spec: dict, scratch: str | None = None) -> list[dict]:
    """Simulate one case, plant its fault, profile it and run every diagnostic."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    rng = np.random.default_rng(spec["seed"])
    knobs, extras = sample_knobs(rng, spec["shift_kind"])
    data_seed = int(rng.integers(0, 2**31 - 1))
    counts, genes, design = simulate(data_seed, knobs=knobs)
    if extras["layout"] == "featurecounts":
        sheet, frame = featurecounts_case(counts, genes, design)
    else:
        sheet, frame = suite_case(counts, genes, design, data_seed)
    fault = spec["fault"]
    fault_line, implicated = None, []
    if fault is not None:
        lines = list(GROUPS)
        fault_line = "T98G" if fault in LINE_FREE_FAULTS else str(rng.choice(lines))
        sheet, frame, implicated = plant(
            fault,
            sheet,
            frame,
            fault_line,
            rng,
            low_depth_factor=extras["low_depth_factor"],
            gene_version_frac=extras["gene_version_frac"],
        )
        if fault in LINE_FREE_FAULTS:
            fault_line = None
    folder = Path(tempfile.mkdtemp(prefix="crux-case-", dir=scratch))
    try:
        write_case(folder, sheet, frame, comment=extras["layout"] == "featurecounts")
        profile = profile_case(folder)
        summary = profile_summary([profile])
        records = []
        fault_fork = FAULT_FORK.get(fault) if fault else None
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
                    "shift_kind": spec["shift_kind"],
                    "seed": spec["seed"],
                    "knobs": {
                        **knobs.to_dict(),
                        "shared_effects": extras["shared_effects"],
                    },
                    "layout": extras["layout"],
                    "fault": fault,
                    "fault_line": fault_line,
                    "fault_severity": {
                        "low_depth_factor": extras["low_depth_factor"],
                        "gene_version_frac": extras["gene_version_frac"],
                    }
                    if fault in ("low_depth", "gene_version")
                    else None,
                    "implicated": implicated,
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


# --- driver ---------------------------------------------------------------------------


def generate(
    sizes: dict[str, int],
    out: Path,
    *,
    base_seed: int = 2029,
    workers: int | None = None,
    scratch: str | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 2))
    started = time.monotonic()
    manifest: dict = {"base_seed": base_seed, "splits": {}}
    for split, n in sizes.items():
        if n <= 0:
            continue
        specs = case_specs(split, n, base_seed)
        path = out / f"{split}.jsonl"
        bins: dict[str, Counter] = defaultdict(Counter)
        errors = 0
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            if workers == 1:
                batches = map(make_case, specs, [scratch] * len(specs))
            else:
                pool = ProcessPoolExecutor(max_workers=workers)
                batches = pool.map(
                    make_case, specs, [scratch] * len(specs), chunksize=4
                )
            try:
                for records in batches:
                    for record in records:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        key = f"{record['diagnostic']}|{record['answer_key']}"
                        bins[key][record["observed_bin"] or "<error>"] += 1
                        errors += record["observed_bin"] is None
            finally:
                if workers != 1:
                    pool.shutdown()
        faults = Counter(s["fault"] or "clean" for s in specs)
        manifest["splits"][split] = {
            "cases": n,
            "records": n * len(_experiments()),
            "invalid_records": errors,
            "faults": dict(sorted(faults.items())),
            "bins": {k: dict(v) for k, v in sorted(bins.items())},
        }
        print(f"{split}: {n} cases, {errors} invalid records -> {path}", flush=True)
    manifest["seconds"] = round(time.monotonic() - started, 1)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=900)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--shift", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    sizes = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "shift": args.shift,
    }
    manifest = generate(sizes, args.out, base_seed=args.seed, workers=args.workers)
    print(f"done in {manifest['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
