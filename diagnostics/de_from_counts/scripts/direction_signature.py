"""Group direction, check 1: does each treated group share the other line's treatment signature?

Within each cell line (assigned from the data), every treated group's log fold change over
the line's control group is computed on genes expressed in both lines. Each treated group of
one line is correlated with each treated group of the other. When control and treated
labels are right and the treatments share biology across lines (the same knockout), every
cross-line correlation is positive. If one line's control and one treated group are
exchanged, that group's correlations with the other line all turn negative.

This is best effort: when the treatments share little biology across lines the
correlations sit near zero and the check reports unclear rather than guessing.
"""

import numpy as np
from _common import (
    CONTROL,
    assign_cell_lines,
    correlation,
    emit,
    load_inputs,
    log_cpm,
    to_cpm,
    top_magnitude,
)

MIN_MEAN_CPM = 10.0
N_GENES = 1000
INVERTED_MAX_CORR = -0.05
INVERTED_SCORE = -0.1


def main():
    inputs = load_inputs()
    lines, _, _ = assign_cell_lines(inputs)
    cpm = to_cpm(inputs.counts)
    logx = log_cpm(cpm)
    names = sorted(set(lines))
    result = {
        "mean_cross_corr": 0.0,
        "min_cross_corr": 0.0,
        "worst_group": "",
        "worst_group_score": 0.0,
        "worst_group_max_corr": 0.0,
        "n_pairs": 0,
        "group_scores": [],
        "flagged_samples": [],
    }
    if len(names) != 2:
        return result
    keep = np.ones(cpm.shape[0], dtype=bool)
    fold, members = {}, {}
    for line in names:
        idx = np.flatnonzero(lines == line)
        keep &= cpm[:, idx].mean(axis=1) >= MIN_MEAN_CPM
        ctrl = [k for k in idx if inputs.conditions[k] == CONTROL]
        if not ctrl:
            return result
        for condition in sorted({inputs.conditions[k] for k in idx} - {CONTROL}):
            group = [k for k in idx if inputs.conditions[k] == condition]
            fold[line, condition] = logx[:, group].mean(axis=1) - logx[:, ctrl].mean(axis=1)
            members[line, condition] = group + ctrl
    keys = list(fold)
    left = [k for k in keys if k[0] == names[0]]
    right = [k for k in keys if k[0] == names[1]]
    if not left or not right or keep.sum() < 10:
        return result
    matrix = np.column_stack([fold[k][keep] for k in keys])
    corr = correlation(matrix[top_magnitude(matrix, N_GENES)])
    pos = {k: i for i, k in enumerate(keys)}
    cross = np.array([[corr[pos[a], pos[b]] for b in right] for a in left])
    entries = {a: cross[i, :] for i, a in enumerate(left)}
    entries.update({b: cross[:, j] for j, b in enumerate(right)})
    scores = {k: float(v.mean()) for k, v in entries.items()}
    worst = min(scores, key=scores.get)
    worst_max = float(entries[worst].max())
    inverted = worst_max < INVERTED_MAX_CORR and scores[worst] <= INVERTED_SCORE
    result.update(
        mean_cross_corr=float(cross.mean()),
        min_cross_corr=float(cross.min()),
        worst_group=f"{worst[0]}:{worst[1]}",
        worst_group_score=scores[worst],
        worst_group_max_corr=worst_max,
        n_pairs=int(cross.size),
        group_scores=[f"{k[0]}:{k[1]}={scores[k]:.3f}" for k in keys],
        flagged_samples=sorted(inputs.sample_ids[k] for k in members[worst]) if inverted else [],
    )
    return result


if __name__ == "__main__":
    emit(main())
