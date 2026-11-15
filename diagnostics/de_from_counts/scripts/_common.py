"""Shared input loading and helpers for the DE-from-counts diagnostics.

``library.py`` inlines this file in place of each script's ``from _common import ...`` line,
so every diagnostic runs as one self-contained file in the sandbox. Run a script directly
(``python scripts/duplicates.py``) and the import resolves to this file instead.

Inputs follow the Task 9 layout: ``samples.tsv`` (``sample_id``, ``cell_line``,
``condition`` and an optional ``count_column``) and ``tagseq_featurecounts_matrix.tsv``
(gene IDs in ``Geneid`` or the first column, ``#`` comment lines allowed). Only pandas and
numpy are used. Each script prints exactly one ``RESULT: {json}`` line with flat values.
"""

import json
import math
import os
import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(os.environ.get("CRUX_DATA_DIR", "/data"))
SAMPLES_FILE = "samples.tsv"
COUNTS_FILE = "tagseq_featurecounts_matrix.tsv"
CONTROL = "Ctrl"


class Inputs:
    """The sample sheet and count matrix, aligned column by column."""

    def __init__(self, sample_ids, lines, conditions, genes, counts):
        self.sample_ids = sample_ids  # list[str], sample sheet order
        self.lines = lines  # np.ndarray[str], labelled cell line per sample
        self.conditions = conditions  # np.ndarray[str], labelled condition per sample
        self.genes = genes  # pd.Series[str], one ID per matrix row
        self.counts = counts  # np.ndarray[float], genes x samples


def load_samples():
    samples = pd.read_csv(DATA_DIR / SAMPLES_FILE, sep="\t", dtype=str, keep_default_na=False)
    samples.columns = [str(c).strip() for c in samples.columns]
    for column in samples.columns:
        samples[column] = samples[column].astype(str).str.strip()
    missing = {"sample_id", "cell_line", "condition"} - set(samples.columns)
    if missing:
        raise ValueError(f"Sample sheet lacks columns: {sorted(missing)}")
    use_count_column = "count_column" in samples.columns and bool(
        (samples["count_column"] != "").all()
    )
    field = "count_column" if use_count_column else "sample_id"
    if samples["sample_id"].duplicated().any() or samples[field].duplicated().any():
        raise ValueError("Sample IDs and count columns must be unique")
    return samples, field


def read_matrix(usecols=None):
    return pd.read_csv(
        DATA_DIR / COUNTS_FILE, sep="\t", comment="#", low_memory=False, usecols=usecols
    )


def gene_column(frame):
    return "Geneid" if "Geneid" in frame.columns else frame.columns[0]


def load_inputs():
    samples, field = load_samples()
    columns = samples[field].tolist()
    frame = read_matrix()
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise ValueError(f"Count matrix lacks sample columns: {absent[:5]}")
    counts = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Counts must be finite and non-negative")
    return Inputs(
        sample_ids=samples["sample_id"].tolist(),
        lines=samples["cell_line"].to_numpy(dtype=str),
        conditions=samples["condition"].to_numpy(dtype=str),
        genes=frame[gene_column(frame)].astype(str),
        counts=counts,
    )


def library_sizes(counts):
    return counts.sum(axis=0)


def shallow_libraries(counts, min_ratio=0.1):
    """Boolean mask of libraries below ``min_ratio`` of the median library size."""
    sizes = library_sizes(counts)
    return sizes < min_ratio * max(float(np.median(sizes)), 1.0)


def to_cpm(counts):
    sizes = library_sizes(counts)
    return counts / np.where(sizes > 0, sizes, 1.0) * 1e6


def log_cpm(cpm, prior=1.0):
    return np.log2(cpm + prior)


def correlation(matrix):
    """Pearson correlation between the columns of a genes x samples matrix."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.sqrt((centred**2).sum(axis=0))
    norms = np.where(norms > 0, norms, 1.0)
    unit = centred / norms
    return np.clip(unit.T @ unit, -1.0, 1.0)


def column_ranks(matrix):
    """Ranks within each column (ties broken by order), for Spearman correlation."""
    order = np.argsort(matrix, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(matrix.shape[0], dtype=float)
    for j in range(matrix.shape[1]):
        ranks[order[:, j], j] = rows
    return ranks


def top_variable(matrix, n):
    """Row indices of the ``n`` most variable rows."""
    variance = matrix.var(axis=1)
    return np.argsort(-variance, kind="mergesort")[: min(n, matrix.shape[0])]


def top_magnitude(matrix, n):
    """Row indices of the ``n`` rows with the largest mean square.

    Unlike variance across columns, this does not favour rows where the columns disagree, so
    selecting genes this way does not bias the sign of correlations between the columns.
    """
    power = (matrix**2).mean(axis=1)
    return np.argsort(-power, kind="mergesort")[: min(n, matrix.shape[0])]


def assign_cell_lines(inputs, n_genes=2000):
    """Leave-one-out nearest-centroid cell line for every sample.

    Profiles are log2 CPM on well-expressed genes (mean CPM 10 or more), centred per gene. Each sample is compared with
    the centroid of every labelled cell line computed without that sample, so one or two
    mislabelled samples barely move the centroids. Returns the assigned line per sample,
    the correlation with the sample's labelled line and with its best other line.
    """
    cpm = to_cpm(inputs.counts)
    keep = cpm.mean(axis=1) >= 10.0
    values = log_cpm(cpm[keep])
    values = values[top_variable(values, n_genes)]
    values = values - values.mean(axis=1, keepdims=True)
    names = sorted(set(inputs.lines))
    assigned, own, other = [], [], []
    for j in range(values.shape[1]):
        scores = {}
        for name in names:
            members = [k for k in range(values.shape[1]) if inputs.lines[k] == name and k != j]
            if members:
                centroid = values[:, members].mean(axis=1)
                scores[name] = float(correlation(np.column_stack([values[:, j], centroid]))[0, 1])
        best = max(scores, key=scores.get)
        labelled = scores.get(inputs.lines[j], -1.0)
        rest = [v for k, v in scores.items() if k != inputs.lines[j]]
        assigned.append(best)
        own.append(labelled)
        other.append(max(rest) if rest else -1.0)
    return np.array(assigned, dtype=str), np.array(own), np.array(other)


def groups_of(keys):
    """Map each distinct key to the indices that carry it, in first-seen order."""
    out = {}
    for index, key in enumerate(keys):
        out.setdefault(key, []).append(index)
    return out


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 6) if math.isfinite(value) else None
    return value if value is None or isinstance(value, str) else str(value)


def emit(result):
    print("RESULT: " + json.dumps(_plain(result), allow_nan=False, sort_keys=True))
