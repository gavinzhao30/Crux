"""Sample labels, check 2: does every library's nearest neighbour share its condition?

Within each cell line (assigned from the data), expressed genes are centred per gene and
libraries are compared by Spearman correlation. A library is misplaced when its nearest
neighbour carries a different condition label. ``swap_resolves`` is true when exchanging
the labels of one pair of libraries (same line, different conditions) leaves no library
misplaced, the signature of two swapped samples.

Libraries below a tenth of the median depth are left out (``skipped_samples``), and a
library with no remaining replicate in its line is not judged (``unjudged_samples``).
"""

import numpy as np
from _common import (
    assign_cell_lines,
    column_ranks,
    correlation,
    emit,
    load_inputs,
    log_cpm,
    shallow_libraries,
    to_cpm,
)

MIN_MEAN_CPM = 10.0


def misplaced(neighbour, judged, conditions):
    return [k for k in judged if conditions[neighbour[k]] != conditions[k]]


def judged_libraries(neighbour, lines, conditions):
    """Libraries with a nearest neighbour and at least one replicate in their line."""
    return [
        k
        for k in range(len(conditions))
        if neighbour[k] >= 0
        and any(
            j != k and neighbour[j] >= 0 and lines[j] == lines[k] and conditions[j] == conditions[k]
            for j in range(len(conditions))
        )
    ]


def main():
    inputs = load_inputs()
    lines, _, _ = assign_cell_lines(inputs)
    cpm = to_cpm(inputs.counts)
    use = ~shallow_libraries(inputs.counts)
    neighbour = np.full(len(inputs.sample_ids), -1)
    for line in sorted(set(lines)):
        idx = np.flatnonzero((lines == line) & use)
        if len(idx) < 2:
            continue
        sub = cpm[:, idx]
        values = log_cpm(sub[sub.mean(axis=1) >= MIN_MEAN_CPM])
        values = values - values.mean(axis=1, keepdims=True)
        corr = correlation(column_ranks(values))
        np.fill_diagonal(corr, -np.inf)
        neighbour[idx] = idx[np.argmax(corr, axis=1)]
    conditions = list(inputs.conditions)
    judged = judged_libraries(neighbour, lines, conditions)
    wrong = misplaced(neighbour, judged, conditions)
    resolving = []
    if wrong:
        n = len(conditions)
        for a in range(n):
            for b in range(a + 1, n):
                if (
                    neighbour[a] < 0
                    or neighbour[b] < 0
                    or lines[a] != lines[b]
                    or conditions[a] == conditions[b]
                ):
                    continue
                trial = list(conditions)
                trial[a], trial[b] = trial[b], trial[a]
                if not misplaced(neighbour, judged_libraries(neighbour, lines, trial), trial):
                    resolving.append(f"{inputs.sample_ids[a]}|{inputs.sample_ids[b]}")
    ids = inputs.sample_ids
    return {
        "n_misplaced": len(wrong),
        "misplaced_samples": [ids[k] for k in wrong],
        "swap_resolves": bool(resolving),
        "resolving_swaps": resolving,
        "nearest_neighbours": [f"{ids[k]}>{ids[nn]}" for k, nn in enumerate(neighbour) if nn >= 0],
        "skipped_samples": [ids[k] for k in np.flatnonzero(~use)],
        "unjudged_samples": [ids[k] for k in range(len(ids)) if use[k] and k not in judged],
        "flagged_samples": [ids[k] for k in wrong] if resolving else [],
    }


if __name__ == "__main__":
    emit(main())
