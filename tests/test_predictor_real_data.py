"""DU-54: predictor records from real datasets with planted faults, the training mix and the
real-data evaluator. Small synthetic case folders stand in for the real datasets; no network.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from diagnostics.de_from_counts.synth import (
    FAULT_FORK,
    simulate,
    suite_case,
    write_case,
)
from eval.eval_predictor_real import DOUBTFUL, evaluate, main, trusted
from eval.predictor_data.dataset import library_index
from eval.predictor_data.make_real_examples import build_mix, generate, split_for

TINY = {"tiny-held-out": "covid", "tiny-train": "pasilla"}


def tiny_cases(root: Path) -> Path:
    for i, name in enumerate(TINY):
        counts, genes, design = simulate(30 + i, n_genes=150)
        write_case(root / name, *suite_case(counts, genes, design, 30 + i))
    return root


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_splits_are_by_dataset_and_the_last_seed_is_validation():
    seeds = (1, 2, 3)
    assert {split_for("covid", s, seeds) for s in seeds} == {"real_test"}
    assert [split_for("airway", s, seeds) for s in seeds] == [
        "real_train",
        "real_train",
        "real_val",
    ]


def test_generate_labels_each_fork_from_the_private_truth(tmp_path: Path):
    cases = tiny_cases(tmp_path / "cases")
    out = tmp_path / "real"
    manifest = generate(cases, out, seeds=(1, 2), instances=1, workers=1, datasets=TINY)
    test = [json.loads(x) for x in (out / "real_test.jsonl").read_text().splitlines()]
    train = [json.loads(x) for x in (out / "real_train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (out / "real_val.jsonl").read_text().splitlines()]

    # the held-out dataset never reaches training, and the last seed is validation
    assert {r["dataset"] for r in test} == {"covid"}
    assert {r["dataset"] for r in train + val} == {"pasilla"}
    assert {r["seed"] for r in train} == {1} and {r["seed"] for r in val} == {2}
    assert manifest["invalid_records"] == {"covid": 0, "pasilla": 0}

    for record in test + train + val:
        entry = library_index()[record["diagnostic"]]
        planted = FAULT_FORK.get(record["fault"]) if record["fault"] else None
        expected = entry.fault_key if planted == record["fork"] else entry.null_key
        assert record["answer_key"] == expected
        assert record["observed_bin"] in {b.id for b in entry.experiment.bins}
        assert record["layout"] == "real" and "shape" in record["profile"]
    per_case = {}
    for record in test:
        per_case.setdefault(record["case_id"], []).append(record)
    assert all(len(rs) == 12 for rs in per_case.values())
    faults = {rs[0]["fault"] for rs in per_case.values()}
    assert None in faults and "duplicate_library" in faults
    for rs in per_case.values():
        if rs[0]["fault"] == "duplicate_library":
            seen = {r["diagnostic"]: r["observed_bin"] for r in rs}
            assert seen["duplicates_identical"] == "duplicate"

    # planting is kept: a second run reuses the suites and gives the same records
    again = generate(cases, out, seeds=(1, 2), instances=1, workers=1, datasets=TINY)
    assert again["cases"] == manifest["cases"]
    assert [json.loads(x) for x in (out / "real_test.jsonl").read_text().splitlines()] == test


def test_mix_repeats_real_training_records_and_mixes_validation(tmp_path: Path):
    syn, real = tmp_path / "syn", tmp_path / "real"
    rec = {"observed_bin": "x"}
    write_jsonl(syn / "train.jsonl", [dict(rec, case_id=f"s{i}") for i in range(3)])
    write_jsonl(syn / "val.jsonl", [dict(rec, case_id="sv")])
    write_jsonl(real / "real_train.jsonl", [dict(rec, case_id=f"r{i}", layout="real") for i in range(2)])
    write_jsonl(real / "real_val.jsonl", [dict(rec, case_id="rv", layout="real")])
    info = build_mix(syn, real, tmp_path / "mix", upweight=3)
    assert info == {
        "synthetic_train": 3,
        "real_train": 2,
        "upweight": 3,
        "train": 9,
        "val": 2,
        "real_val": 1,
        "dropped_doubtful": 0,
    }
    train = [json.loads(x) for x in (tmp_path / "mix" / "train.jsonl").read_text().splitlines()]
    assert sorted(r["case_id"] for r in train).count("r0") == 3


def test_mix_can_drop_the_doubtful_forks(tmp_path: Path):
    syn, real = tmp_path / "syn", tmp_path / "real"
    rec = {"observed_bin": "x", "layout": "real", "dataset": "airway"}
    write_jsonl(syn / "train.jsonl", [{"observed_bin": "x", "case_id": "s"}])
    write_jsonl(syn / "val.jsonl", [])
    write_jsonl(
        real / "real_train.jsonl",
        [dict(rec, case_id="a", fork="sample_labels"), dict(rec, case_id="b", fork="duplicates")],
    )
    write_jsonl(real / "real_val.jsonl", [dict(rec, case_id="c", fork="group_direction")])
    info = build_mix(syn, real, tmp_path / "mix", upweight=1, drop_doubtful=True)
    assert info["dropped_doubtful"] == 2 and info["real_train"] == 1 and info["val"] == 0
    train = [json.loads(x) for x in (tmp_path / "mix" / "train.jsonl").read_text().splitlines()]
    assert {r["case_id"] for r in train} == {"s", "b"}


def _records(prefix: str, dataset: str | None, n_cases: int) -> list[dict]:
    """One record per (case, library diagnostic), observing the null option's first bin."""
    out = []
    for i in range(n_cases):
        for name, entry in library_index().items():
            out.append(
                {
                    "case_id": f"{prefix}-{i}",
                    "dataset": dataset,
                    "shift_kind": dataset,
                    "fault": None,
                    "fork": entry.fork,
                    "diagnostic": name,
                    "answer_key": entry.null_key,
                    "observed_bin": entry.experiment.bins[0].id,
                    "profile": {},
                }
            )
    return out


def _cache(path: Path, records: list[dict]) -> None:
    lines = []
    for r in records:
        entry = library_index()[r["diagnostic"]]
        bins = [b.id for b in entry.experiment.bins]
        lp = {b: (-0.1 if b == r["observed_bin"] else -3.0) for b in bins}
        lines.append({"key": f"{r['case_id']}|{r['diagnostic']}",
                      "logprobs": {entry.null_key: lp, entry.fault_key: lp}})
    write_jsonl(path, lines)


def test_real_evaluator_scores_each_set_and_skips_where_a_model_trained(tmp_path: Path):
    real, syn, mix = tmp_path / "real", tmp_path / "syn", tmp_path / "mix"
    test, other = _records("covid", "covid", 2), _records("air", "airway", 1)
    write_jsonl(real / "real_test.jsonl", test)
    write_jsonl(real / "real_train.jsonl", other)
    write_jsonl(real / "real_val.jsonl", [])
    s_test, s_shift = _records("test", None, 1), _records("shift", None, 1)
    write_jsonl(syn / "test.jsonl", s_test)
    write_jsonl(syn / "shift.jsonl", s_shift)
    _cache(syn / "eval_base.jsonl", s_test + s_shift)
    _cache(real / "eval_base.jsonl", test + other)
    _cache(real / "eval_new.jsonl", test + s_test + s_shift)

    report = evaluate(real, syn, 0.1)
    assert set(report["methods"]) == {"library", "base", "new"}  # no sampled or old caches
    assert "real_other" not in report["methods"]["new"]
    assert "real_other" in report["methods"]["base"]
    doubtful = [r for r in test if not trusted(r)]
    assert doubtful and all(("covid", r["fork"]) in DOUBTFUL for r in doubtful)
    sets = report["sets"]
    assert sets["real_test_trusted"]["records"] == len(test) - len(doubtful)
    base = report["methods"]["base"]["real_test"]["overall"]
    assert base["n"] == len(test) and math.isfinite(base["log_loss"])
    assert base["accuracy"] == 1.0  # the cached rows favour the observed bin
    rates = report["unplanted_fault_rate"]
    assert set(rates) == {"covid", "airway"}

    results = tmp_path / "results"
    assert main(["--real", str(real), "--synthetic", str(syn), "--mix", str(mix),
                 "--report-only", "--results", str(results)]) == 0
    text = (results / "predictor_real_data.md").read_text(encoding="utf-8")
    assert "COVID, held out from every model" in text and "trained on this set" in text
    assert json.loads((results / "predictor_real_data.json").read_text())["sets"]
