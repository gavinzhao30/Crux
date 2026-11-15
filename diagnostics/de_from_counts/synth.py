"""Synthetic Task 9 count data and planted faults for the DE-from-counts diagnostics.

Shared by ``tests/test_diagnostics_de.py`` and the predictor training data
(``eval/predictor_data/make_examples.py``).

The data follow the Task 9 design (T98G Ctrl, C10, F7, G2, D4 and HEK293 Ctrl, D10, E6) with
negative-binomial counts, a large cell line effect, a knockout signature partly shared by
both lines (including a target transcript lost in every knockout clone), and clone-specific
effects. :class:`Knobs` holds the realism settings; its defaults reproduce the original test
generator draw for draw, so ``simulate(seed)`` gives the same counts it always did.

Faults are applied to copies of the sample sheet and count frame by :func:`plant`, so a swap
really exchanges two columns. Each fault belongs to one library fork (``FAULT_FORK``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {"T98G": ("Ctrl", "C10", "F7", "G2", "D4"), "HEK293": ("Ctrl", "D10", "E6")}
COUNTS = "tagseq_featurecounts_matrix.tsv"
SAMPLES = "samples.tsv"
FAULT_FORK = {
    "sample_swap": "sample_labels",
    "label_inversion": "group_direction",
    "duplicate_library": "duplicates",
    "low_depth": "sample_quality",
    "cell_line_swap": "cell_line",
    "gene_version": "gene_ids",
}
FAULTS = tuple(FAULT_FORK)
# Faults whose placement does not depend on a chosen cell line.
LINE_FREE_FAULTS = frozenset({"cell_line_swap", "gene_version"})


@dataclass(frozen=True)
class Knobs:
    """Realism settings for :func:`simulate`. The defaults are the original generator.

    * ``n_genes``, ``depth``: matrix rows and the mean library size before the per-library
      factor drawn uniformly from ``depth_spread``.
    * ``replicates``: libraries per (cell line, condition) group (2 or 3).
    * ``effect``: standard deviation (natural log) of the knockout signature effects.
    * ``shared`` / ``private``: HEK293's signature is ``shared * T98G's + private * its own``
      draw; ``shared=0, private=1`` means condition effects are not shared across lines.
    * ``dispersion``: floor of the negative-binomial dispersion (biological noise); the
      Poisson-like term ``0.5 / sqrt(mean + 1)`` is added on top.
    * ``target_log2_fold``: loss of the knockout target in every clone (0: no target).
    """

    n_genes: int = 3000
    depth: float = 1.5e6
    depth_spread: tuple[float, float] = (0.6, 1.5)
    replicates: int = 2
    effect: float = 0.8
    signature_frac: float = 0.08
    shared: float = 0.7
    private: float = 0.71
    line_effect: float = 1.5
    clone_effect: float = 0.8
    clone_frac: float = 0.06
    dispersion: float = 0.02
    target_log2_fold: float = 4.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["depth_spread"] = list(self.depth_spread)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Knobs:
        values = dict(data)
        if "depth_spread" in values:
            values["depth_spread"] = tuple(values["depth_spread"])
        return cls(**values)

    @staticmethod
    def sharing(shared: bool, rho: float = 0.7) -> dict:
        """``shared``/``private`` weights for a cross-line signature correlation ``rho``."""
        if not shared:
            return {"shared": 0.0, "private": 1.0}
        return {"shared": rho, "private": math.sqrt(max(0.0, 1.0 - rho * rho))}


def design_of(replicates: int = 2) -> list[tuple[str, str]]:
    return [
        (line, cond)
        for line, conds in GROUPS.items()
        for cond in conds
        for _ in range(replicates)
    ]


def simulate(
    seed: int,
    n_genes: int | None = None,
    depth: float | None = None,
    knobs: Knobs | None = None,
):
    """Counts (genes x samples), gene IDs and the design, from a negative-binomial model.

    ``n_genes`` and ``depth`` override the matching ``knobs`` fields.
    """
    k = knobs or Knobs()
    if n_genes is not None:
        k = replace(k, n_genes=n_genes)
    if depth is not None:
        k = replace(k, depth=depth)
    n = k.n_genes
    rng = np.random.default_rng(seed)
    design = design_of(k.replicates)
    base = rng.normal(0.0, 2.0, n)  # natural-log abundance
    line_effect = {
        "T98G": np.zeros(n),
        "HEK293": rng.normal(0.0, k.line_effect, n) * (rng.random(n) < 0.5),
    }
    signature = rng.random(n) < k.signature_frac
    knockout_t = rng.normal(0.0, k.effect, n) * signature
    knockout_h = (
        k.shared * knockout_t + k.private * rng.normal(0.0, k.effect, n)
    ) * signature
    knockout = {"T98G": knockout_t, "HEK293": knockout_h}
    base[0] = np.log(
        200e-6 * np.exp(base).sum()
    )  # knockout target, ~200 CPM in controls
    line_effect["HEK293"][0] = 0.0
    for line in GROUPS:
        knockout[line][0] = -k.target_log2_fold * np.log(
            2
        )  # lost in every knockout clone
    clone = {
        (line, cond): rng.normal(0.0, k.clone_effect, n)
        * (rng.random(n) < k.clone_frac)
        for line, conds in GROUPS.items()
        for cond in conds[1:]
    }
    columns = []
    for line, cond in design:
        log_mu = base + line_effect[line]
        if cond != "Ctrl":
            log_mu = log_mu + knockout[line] + clone[line, cond]
        share = np.exp(log_mu - log_mu.max())
        mean = share / share.sum() * k.depth * rng.uniform(*k.depth_spread)
        dispersion = k.dispersion + 0.5 / np.sqrt(mean + 1)
        columns.append(rng.poisson(rng.gamma(1 / dispersion, dispersion * mean)))
    counts = np.column_stack(columns).astype(np.int64)
    versions = rng.integers(1, 20, n)
    genes = [f"ENSG{100000 + 7 * i:011d}.{versions[i]}" for i in range(n)]
    genes[-1] += "_PAR_Y"  # GENCODE-style tail on one versioned ID
    return counts, genes, design


def suite_case(counts, genes, design, seed):
    """The make_cases layout: shuffled, blinded sample IDs and a ``gene_id`` column."""
    order = np.random.default_rng(seed).permutation(len(design))
    ids = [f"S{i:02d}" for i in range(1, len(design) + 1)]
    sheet = pd.DataFrame(
        {
            "sample_id": ids,
            "cell_line": [design[k][0] for k in order],
            "condition": [design[k][1] for k in order],
        }
    )
    frame = pd.DataFrame(counts[:, order], columns=ids)
    frame.insert(0, "gene_id", genes)
    frame.insert(1, "gene_symbol", [f"SYM{i}" for i in range(len(genes))])
    return sheet, frame


def featurecounts_case(counts, genes, design):
    """The published layout: featureCounts header, annotation columns, ``count_column``."""
    seen: dict[tuple[str, str], int] = {}
    ids, paths = [], []
    for line, cond in design:
        seen[line, cond] = seen.get((line, cond), 0) + 1
        ids.append(f"{line}_{cond}_{seen[line, cond]}")
        paths.append(f"/align/{ids[-1]}.Aligned.sortedByCoord.out.bam")
    sheet = pd.DataFrame(
        {
            "sample_id": ids,
            "cell_line": [d[0] for d in design],
            "condition": [d[1] for d in design],
            "count_column": paths,
        }
    )
    n = len(genes)
    frame = pd.DataFrame(
        {
            "Geneid": genes,
            "Chr": ["chr1"] * n,
            "Start": np.arange(n) * 100 + 1,
            "End": np.arange(n) * 100 + 90,
            "Strand": ["+"] * n,
            "Length": [90] * n,
        }
    )
    frame = pd.concat([frame, pd.DataFrame(counts, columns=paths)], axis=1)
    return sheet, frame


def write_case(
    folder: Path, sheet: pd.DataFrame, frame: pd.DataFrame, comment: bool = False
):
    folder.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(folder / SAMPLES, sep="\t", index=False, lineterminator="\n")
    with (folder / COUNTS).open("w", encoding="utf-8", newline="\n") as handle:
        if comment:
            handle.write(
                "# Program:featureCounts v2.0.6; Command:featureCounts -a genes.gtf\n"
            )
        frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")


def plant(
    fault: str,
    sheet: pd.DataFrame,
    frame: pd.DataFrame,
    line: str,
    rng,
    *,
    low_depth_factor: float = 1 / 50,
    gene_version_frac: float | None = None,
):
    """Apply one fault to copies; return them with the implicated sample IDs.

    Works on both layouts (count columns named by ``sample_id`` or by ``count_column``).
    ``low_depth_factor`` is the binomial thinning of the degraded library;
    ``gene_version_frac`` the share of gene IDs whose version is stripped (default: 5%,
    at most 1000).
    """
    sheet, frame = sheet.copy(), frame.copy()
    column = (
        dict(zip(sheet["sample_id"], sheet["count_column"], strict=True))
        if "count_column" in sheet.columns
        else {s: s for s in sheet["sample_id"]}
    )

    def members(cell, cond):
        mask = (sheet["cell_line"] == cell) & (sheet["condition"] == cond)
        return sheet.loc[mask, "sample_id"].tolist()

    treated = str(rng.choice(GROUPS[line][1:]))
    if fault == "sample_swap":
        a, b = (
            str(rng.choice(members(line, "Ctrl"))),
            str(rng.choice(members(line, treated))),
        )
        ca, cb = column[a], column[b]
        frame[[ca, cb]] = frame[[cb, ca]].to_numpy(copy=True)
        return sheet, frame, [a, b]
    if fault == "label_inversion":
        ctrl, other = members(line, "Ctrl"), members(line, treated)
        sheet.loc[sheet["sample_id"].isin(ctrl), "condition"] = treated
        sheet.loc[sheet["sample_id"].isin(other), "condition"] = "Ctrl"
        return sheet, frame, ctrl + other
    if fault == "duplicate_library":
        picked = rng.permutation(members(line, treated)).tolist()
        source, target = str(picked[0]), str(picked[1])
        frame[column[target]] = frame[column[source]].to_numpy(copy=True)
        return sheet, frame, [source, target]
    if fault == "low_depth":
        sample = str(rng.choice(members(line, treated)))
        frame[column[sample]] = rng.binomial(
            frame[column[sample]].to_numpy(), low_depth_factor
        )
        return sheet, frame, [sample]
    if fault == "cell_line_swap":
        a = str(rng.choice(members("T98G", "Ctrl")))
        b = str(rng.choice(members("HEK293", "Ctrl")))
        sheet.loc[sheet["sample_id"] == a, "cell_line"] = "HEK293"
        sheet.loc[sheet["sample_id"] == b, "cell_line"] = "T98G"
        return sheet, frame, [a, b]
    if fault == "gene_version":
        gene_col = "Geneid" if "Geneid" in frame.columns else "gene_id"
        if gene_version_frac is None:
            size = max(1, min(1000, len(frame) // 20))
        else:
            size = max(1, min(len(frame) - 1, round(gene_version_frac * len(frame))))
        rows = rng.permutation(len(frame))[:size]
        frame.loc[rows, gene_col] = frame.loc[rows, gene_col].str.replace(
            r"\.\d+$", "", regex=True
        )
        return sheet, frame, []
    raise ValueError(fault)
