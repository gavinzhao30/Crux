"""Plant the six DE faults on any count dataset and keep the truth key private (DU-49).

    uv run python -m eval.suites.plant_faults --case ../data/cases/covid-lung-colon \
        --output ../data/suites/covid --seed 49

``eval/suites/task9/make_cases.py`` plants the same faults but only on task 9's design.
This works on any case folder (``tagseq_featurecounts_matrix.tsv`` + ``samples.tsv`` with
``sample_id``, ``cell_line``, ``condition`` and optional ``count_column``):

* ``sample_swap``: exchange the count columns of a control and a treated library in one line.
* ``label_inversion``: exchange the condition labels of a line's control and treated groups.
* ``duplicate_library``: copy one treated library over its replicate.
* ``low_depth``: thin one treated library to 1/50 of its reads.
* ``cell_line_swap``: exchange the cell_line labels of controls from two lines (2+ lines only).
* ``gene_version``: on 5% of genes, strip version suffixes if the ids have them, else add one
  (skipped for numeric Entrez ids, which never carry versions).

Faults the design can't support are skipped and listed in the truth file with the reason.
Output, deterministic for a seed: ``cases/<case-NN>/`` with only counts and samples (sample
ids blinded and shuffled), and ``evaluator/truth.json`` (fault, affected blinded ids, fork),
which must stay away from agents.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

COUNTS, SAMPLES = "tagseq_featurecounts_matrix.tsv", "samples.tsv"
FAULT_FORK = {
    "sample_swap": "sample_labels",
    "label_inversion": "group_direction",
    "duplicate_library": "duplicates",
    "low_depth": "sample_quality",
    "cell_line_swap": "cell_line",
    "gene_version": "gene_ids",
}
_VERSIONED = re.compile(r"\.\d+$")


def load_case(case: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts with ``gene_id`` (+ optional ``gene_symbol``) and one column per sample,
    renamed to the sample ids, and the sample sheet in the same order."""
    sheet = pd.read_csv(case / SAMPLES, sep="\t", dtype=str, keep_default_na=False)
    raw = pd.read_csv(case / COUNTS, sep="\t", comment="#")
    gene = "Geneid" if "Geneid" in raw else raw.columns[0]
    columns = sheet["count_column"] if "count_column" in sheet else sheet["sample_id"]
    counts = raw[list(columns)].astype("int64")
    counts.columns = list(sheet["sample_id"])
    counts.insert(0, "gene_id", raw[gene].astype(str))
    if "gene_symbol" in raw:
        counts.insert(1, "gene_symbol", raw["gene_symbol"])
    return counts, sheet[["sample_id", "cell_line", "condition"]].reset_index(drop=True)


def _groups(sheet: pd.DataFrame, control: str) -> dict[str, dict[str, list[str]]]:
    """line -> condition -> sample ids, for lines that have the control condition."""
    out: dict[str, dict[str, list[str]]] = {}
    for (line, cond), part in sheet.groupby(["cell_line", "condition"]):
        out.setdefault(line, {})[cond] = part["sample_id"].tolist()
    return {line: conds for line, conds in out.items() if control in conds}


def plant_one(fault: str, counts: pd.DataFrame, sheet: pd.DataFrame, rng, control: str):
    """Apply one fault to copies. Returns (counts, sheet, affected, None) or a skip reason."""
    counts, sheet = counts.copy(), sheet.copy()
    groups = _groups(sheet, control)
    treated = {
        line: [c for c, ids in conds.items() if c != control and ids]
        for line, conds in groups.items()
    }
    lines = [line for line in groups if treated[line]]
    if fault in {"sample_swap", "label_inversion"}:
        if not lines:
            return f"no line has both a {control!r} group and a treated group"
        line = str(rng.choice(lines))
        cond = str(rng.choice(treated[line]))
        ctrl_ids, cond_ids = groups[line][control], groups[line][cond]
        if fault == "sample_swap":
            a, b = str(rng.choice(ctrl_ids)), str(rng.choice(cond_ids))
            counts[[a, b]] = counts[[b, a]].to_numpy(copy=True)
            return counts, sheet, [a, b], None
        sheet.loc[sheet["sample_id"].isin(ctrl_ids), "condition"] = cond
        sheet.loc[sheet["sample_id"].isin(cond_ids), "condition"] = control
        return counts, sheet, ctrl_ids + cond_ids, None
    if fault in {"duplicate_library", "low_depth"}:
        options = [
            (ln, c) for ln in lines for c in treated[ln] if len(groups[ln][c]) >= 2
        ]
        if not options:
            return "no treated group has two or more replicates"
        line, cond = options[int(rng.integers(len(options)))]
        ids = groups[line][cond]
        if fault == "duplicate_library":
            source, target = (str(x) for x in rng.permutation(ids)[:2])
            counts[target] = counts[source].to_numpy(copy=True)
            return counts, sheet, [source, target], None
        sample = str(rng.choice(ids))
        counts[sample] = rng.binomial(counts[sample].to_numpy(), 1 / 50)
        return counts, sheet, [sample], None
    if fault == "cell_line_swap":
        if len(groups) < 2:
            return "needs controls from two or more cell lines"
        first, second = (str(x) for x in rng.permutation(sorted(groups))[:2])
        a = str(rng.choice(groups[first][control]))
        b = str(rng.choice(groups[second][control]))
        sheet.loc[sheet["sample_id"] == a, "cell_line"] = second
        sheet.loc[sheet["sample_id"] == b, "cell_line"] = first
        return counts, sheet, [a, b], None
    if fault == "gene_version":
        ids = counts["gene_id"]
        if ids.str.fullmatch(r"\d+").mean() > 0.5:
            return "numeric (Entrez) gene ids never carry versions"
        rows = rng.permutation(len(counts))[: max(1, min(1000, len(counts) // 20))]
        if ids.str.contains(_VERSIONED).mean() > 0.5:
            counts.loc[rows, "gene_id"] = ids.iloc[rows].str.replace(
                _VERSIONED, "", regex=True
            )
        else:
            counts.loc[rows, "gene_id"] = ids.iloc[rows] + ".1"
        return counts, sheet, [], None
    raise ValueError(f"unknown fault {fault!r}")


def plant(
    case: Path,
    output: Path,
    *,
    seed: int = 49,
    control: str = "Ctrl",
    instances: int = 2,
) -> dict:
    """Write a planted suite for ``case``; return the truth key (also in evaluator/)."""
    if output.exists():
        raise FileExistsError(f"choose a new suite folder: {output}")
    rng = np.random.default_rng(seed)
    counts, sheet = load_case(case)
    # Blind and shuffle sample ids once, so ids carry no condition or line.
    order = rng.permutation(len(sheet))
    blind = {
        old: f"S{i:02d}"
        for i, old in enumerate(sheet["sample_id"].iloc[order], start=1)
    }
    sheet = sheet.iloc[order].reset_index(drop=True)
    sheet["sample_id"] = sheet["sample_id"].map(blind)
    counts = counts.rename(columns=blind)
    head = [c for c in ("gene_id", "gene_symbol") if c in counts]
    counts = counts[head + sheet["sample_id"].tolist()]

    variants: list[tuple[str, int]] = [("clean", 0)]
    variants += [(fault, k) for fault in FAULT_FORK for k in range(instances)]
    case_ids = rng.permutation(np.arange(1, len(variants) + 1))
    truth: dict = {"seed": seed, "source": case.name, "cases": {}, "skipped": {}}
    staging = output.with_name(output.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    for number, (fault, instance) in zip(case_ids, variants, strict=True):
        if fault == "clean":
            planted = (counts, sheet, [], None)
        else:
            planted = plant_one(fault, counts, sheet, rng, control)
        if isinstance(planted, str):
            truth["skipped"][fault] = planted
            continue
        case_counts, case_sheet, affected, _ = planted
        folder = staging / "cases" / f"case-{number:02d}"
        folder.mkdir(parents=True)
        out = case_counts.rename(columns={"gene_id": "Geneid"})
        out.to_csv(folder / COUNTS, sep="\t", index=False, lineterminator="\n")
        case_sheet.to_csv(folder / SAMPLES, sep="\t", index=False, lineterminator="\n")
        truth["cases"][folder.name] = {
            "fault": fault,
            "instance": instance,
            "fork": FAULT_FORK.get(fault),
            "affected_samples": affected,
        }
    evaluator = staging / "evaluator"
    evaluator.mkdir(parents=True)
    (evaluator / "truth.json").write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output)
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--control", default="Ctrl")
    parser.add_argument("--instances", type=int, default=2)
    args = parser.parse_args()
    truth = plant(
        args.case,
        args.output,
        seed=args.seed,
        control=args.control,
        instances=args.instances,
    )
    print(f"{len(truth['cases'])} cases written; skipped: {truth['skipped'] or 'none'}")


if __name__ == "__main__":
    main()
