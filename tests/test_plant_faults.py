"""The generic fault planter (DU-49) on a design that is not task 9."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eval.suites.plant_faults import COUNTS, FAULT_FORK, SAMPLES, plant


def make_case(
    root, *, lines=("liver", "lung", "colon"), conds=("Ctrl", "Drug"), reps=3
):
    rng = np.random.default_rng(1)
    rows = [(line, cond) for line in lines for cond in conds for _ in range(reps)]
    ids = [f"orig{i}" for i in range(len(rows))]
    sheet = pd.DataFrame(rows, columns=["cell_line", "condition"])
    sheet.insert(0, "sample_id", ids)
    sheet["count_column"] = [f"col_{i}" for i in ids]
    counts = pd.DataFrame(
        rng.integers(100, 5000, size=(120, len(rows))), columns=sheet["count_column"]
    )
    counts.insert(0, "Geneid", [f"ENSG{i:011d}" for i in range(120)])
    root.mkdir(parents=True)
    sheet.to_csv(root / SAMPLES, sep="\t", index=False)
    counts.to_csv(root / COUNTS, sep="\t", index=False)
    return root


def read(folder):
    return (
        pd.read_csv(folder / COUNTS, sep="\t"),
        pd.read_csv(folder / SAMPLES, sep="\t"),
    )


def test_each_case_changes_exactly_the_declared_samples(tmp_path):
    case = make_case(tmp_path / "case")
    truth = plant(case, tmp_path / "suite", seed=3)
    assert truth["skipped"] == {}
    assert sorted(i["fault"] for i in truth["cases"].values()) == sorted(
        ["clean"] + [f for f in FAULT_FORK for _ in range(2)]
    )
    clean = next(c for c, i in truth["cases"].items() if i["fault"] == "clean")
    base_counts, base_sheet = read(tmp_path / "suite" / "cases" / clean)
    assert not base_sheet["sample_id"].str.contains("orig").any()  # ids are blinded
    for name, info in truth["cases"].items():
        counts, sheet = read(tmp_path / "suite" / "cases" / name)
        assert set(sheet.columns) == {"sample_id", "cell_line", "condition"}
        changed_cols = set(counts.columns[(counts != base_counts).any(axis=0)]) - {
            "Geneid"
        }
        changed_rows = set(sheet.loc[(sheet != base_sheet).any(axis=1), "sample_id"])
        expected = set(info["affected_samples"])
        if info["fault"] == "duplicate_library":
            # Both copies are implicated, but only the overwritten one changes.
            expected = {info["affected_samples"][1]}
        assert changed_cols | changed_rows == expected, info
        if info["fault"] == "sample_swap":
            a, b = info["affected_samples"]
            assert (counts[a] == base_counts[b]).all() and (
                counts[b] == base_counts[a]
            ).all()
        if info["fault"] == "gene_version":
            assert counts["Geneid"].str.contains(r"\.1$").sum() == 6  # 5% of 120, added


def test_deterministic_and_truth_kept_apart(tmp_path):
    case = make_case(tmp_path / "case")
    one, two = plant(case, tmp_path / "a", seed=5), plant(case, tmp_path / "b", seed=5)
    assert one == two
    for folder in (tmp_path / "a" / "cases").iterdir():
        assert {p.name for p in folder.iterdir()} == {COUNTS, SAMPLES}
        twin = tmp_path / "b" / "cases" / folder.name
        assert (folder / COUNTS).read_bytes() == (twin / COUNTS).read_bytes()
    stored = json.loads((tmp_path / "a" / "evaluator" / "truth.json").read_text())
    assert stored["cases"] == one["cases"]


def test_unsupported_faults_are_skipped_with_a_reason(tmp_path):
    case = make_case(tmp_path / "case", lines=("S2",), reps=2)
    truth = plant(case, tmp_path / "suite", seed=1)
    assert set(truth["skipped"]) == {"cell_line_swap"}
    assert "two or more cell lines" in truth["skipped"]["cell_line_swap"]
