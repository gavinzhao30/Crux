"""Generate the frozen Task 9 downstream cases and a separate evaluator truth key."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from execution.staging import COUNTS, SAMPLES

GROUPS = {
    "T98G": ("Ctrl", "C10", "F7", "G2", "D4"),
    "HEK293": ("Ctrl", "D10", "E6"),
}
DEV_FAULTS = {"sample_swap", "duplicate_library", "gene_version"}


def _load(counts_path: Path, samples_path: Path, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(samples_path, sep="\t", dtype=str, keep_default_na=False)
    columns = samples.get("count_column", samples["sample_id"])
    if columns.duplicated().any() or samples["sample_id"].duplicated().any():
        raise ValueError("Sample identifiers and count columns must be unique")
    groups = samples.groupby(["cell_line", "condition"]).size().to_dict()
    if groups != {(cell, condition): 2 for cell, values in GROUPS.items() for condition in values}:
        raise ValueError("Expected the 16 Task 9 samples, with two replicates per condition")
    original = pd.read_csv(counts_path, sep="\t", comment="#")
    gene_column = "Geneid" if "Geneid" in original else original.columns[0]
    selected = original[columns.tolist()]
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Expected finite non-negative integer counts")
    if original[gene_column].isna().any() or original[gene_column].duplicated().any():
        raise ValueError("Expected unique gene IDs")
    # Blind the condition-bearing count column names in all arms and cases.
    order = rng.permutation(len(samples))
    samples = samples.iloc[order][["cell_line", "condition"]].reset_index(drop=True)
    ids = [f"S{i:02d}" for i in range(1, len(samples) + 1)]
    samples.insert(0, "sample_id", ids)
    counts = selected.iloc[:, order].astype("int64")
    counts.columns = ids
    counts.insert(0, "gene_id", original[gene_column].astype(str))
    if "gene_symbol" in original:
        counts.insert(1, "gene_symbol", original["gene_symbol"])
    return counts, samples


def _variants(counts, samples, rng):
    grouped = defaultdict(list)
    for index, row in samples.iterrows():
        grouped[row.cell_line, row.condition].append(index)
    cross_line = [
        (left, right)
        for left in samples.index
        if samples.at[left, "cell_line"] == "T98G"
        for right in samples.index
        if samples.at[right, "cell_line"] == "HEK293"
        if samples.at[left, "condition"] == samples.at[right, "condition"] == "Ctrl"
    ]
    versioned = counts.index[counts["gene_id"].str.contains(r"\.\d+$")].to_numpy()
    gene_rows = versioned if len(versioned) >= 2 else counts.index.to_numpy()
    if len(gene_rows) < 2:
        raise ValueError("Need at least two genes to create gene ID cases")
    gene_rows = rng.permutation(gene_rows)
    for i, cell in enumerate(GROUPS):
        ctrl = grouped[cell, "Ctrl"]
        condition = rng.choice(GROUPS[cell][1:])
        treated = grouped[cell, condition]
        swap = [int(rng.choice(ctrl)), int(rng.choice(treated))]
        yield "sample_swap", i, swap, []
        yield "label_inversion", i, [*ctrl, *treated], []
        source, target = rng.permutation(treated).tolist()
        yield "duplicate_library", i, [source, target], []
        yield "low_depth", i, [source], []
        yield "cell_line_swap", i, list(cross_line[i]), []
        rows = gene_rows[i::2][: max(1, min(1000, len(gene_rows) // 20))].tolist()
        yield "gene_version", i, [], rows


def _apply(fault, indices, genes, counts, samples, rng):
    ids = samples.loc[indices, "sample_id"].tolist()
    if fault == "sample_swap":
        counts.loc[:, ids] = counts[ids[::-1]].to_numpy()
    elif fault == "label_inversion":
        samples.loc[indices, "condition"] = samples.loc[
            indices[2:] + indices[:2], "condition"
        ].to_numpy()
    elif fault == "duplicate_library":
        counts[ids[1]] = counts[ids[0]].to_numpy()
    elif fault == "low_depth":
        counts[ids[0]] = rng.binomial(counts[ids[0]].to_numpy(), 1 / 50)
    elif fault == "cell_line_swap":
        samples.loc[indices, "cell_line"] = samples.loc[indices[::-1], "cell_line"].to_numpy()
    elif fault == "gene_version":

        def change_version(gene):
            root, dot, suffix = gene.rpartition(".")
            return root if dot and suffix.isdigit() else f"{gene}.1"

        counts.loc[genes, "gene_id"] = counts.loc[genes, "gene_id"].map(change_version)
    return ids


def make_cases(counts_path: Path, samples_path: Path, output: Path, seed: int = 43) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a new suite directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts, samples = _load(counts_path, samples_path, rng)
    variants = [("clean", 0, [], []), *_variants(counts, samples, rng)]
    opaque_ids = rng.permutation(np.arange(1, len(variants) + 1))
    truth = {}
    with TemporaryDirectory(dir=output.parent) as temp:
        result = Path(temp) / "suite"
        for case_num, (fault, instance, indices, genes) in zip(opaque_ids, variants, strict=True):
            case_id = f"case-{case_num:02d}"
            case = result / "cases" / case_id
            case.mkdir(parents=True)
            case_counts, case_samples = counts.copy(), samples.copy()
            ids = _apply(fault, indices, genes, case_counts, case_samples, rng)
            case_counts.to_csv(case / COUNTS, sep="\t", index=False, lineterminator="\n")
            case_samples.to_csv(case / SAMPLES, sep="\t", index=False, lineterminator="\n")
            affected = ids[1:] if fault == "duplicate_library" else ids
            truth[case_id] = {
                "fault": fault,
                "instance": instance,
                "affected_samples": affected,
                "duplicate_source": ids[0] if fault == "duplicate_library" else None,
                "affected_gene_rows": genes,
                "split": "dev"
                if fault == "clean" or (instance == 0 and fault in DEV_FAULTS)
                else "test",
            }
        private = result / "evaluator"
        private.mkdir(mode=0o700)
        (private / "truth.json").write_text(
            json.dumps({"seed": seed, "cases": truth}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    make_cases(args.counts, args.samples, args.output, args.seed)


if __name__ == "__main__":
    main()
