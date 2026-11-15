"""Sample labels, check 1: search for the one label exchange that best explains the data.

Within each cell line (assigned from the data), log2 CPM of expressed genes is centred per
gene and the top variable genes give a library-by-library correlation matrix. For every
pair of libraries with different condition labels, ``gain`` is how much exchanging their
labels raises the correlation of each with its replicates. ``gap`` is the line's typical
replicate correlation minus its typical between-condition correlation, measured on
whichever labelling (as given or exchanged) is more coherent. ``swap_ratio = gain / gap``
is about 2 for a real exchange between distinct groups and near or below 0 when the labels
are right. A line whose ``gap`` is under 0.1 has too little structure to judge.

Libraries below a tenth of the median depth are left out (``skipped_samples``): they are
too noisy to place, and the quality fork deals with them.

With two replicates per group, exchanging (a, b) and exchanging their partners (a', b')
explain the data equally well: one restores the labels, the other leaves both groups
inverted. Both are reported; ``preferred_pair`` breaks the tie when the other cell line's
treatment signature clearly favours one (correlation margin above 0.05), else it is empty.
"""

import numpy as np
from _common import (
    CONTROL,
    assign_cell_lines,
    correlation,
    emit,
    groups_of,
    load_inputs,
    log_cpm,
    shallow_libraries,
    to_cpm,
    top_magnitude,
    top_variable,
)

MIN_MEAN_CPM = 10.0
N_GENES = 1000
SWAP_RATIO = 1.0
PREFERENCE_MARGIN = 0.05


def line_correlation(cpm, idx):
    sub = cpm[:, idx]
    values = log_cpm(sub[sub.mean(axis=1) >= MIN_MEAN_CPM])
    values = values - values.mean(axis=1, keepdims=True)
    return correlation(values[top_variable(values, N_GENES)])


def gap(corr, conditions):
    within = []
    for members in groups_of(conditions).values():
        pairs = [corr[a, b] for i, a in enumerate(members) for b in members[i + 1 :]]
        if pairs:
            within.append(float(np.mean(pairs)))
    n = len(conditions)
    between = [
        corr[a, b] for a in range(n) for b in range(a + 1, n) if conditions[a] != conditions[b]
    ]
    if not within or not between:
        return 0.0
    return float(np.median(within) - np.median(between))


def best_exchange(corr, conditions):
    groups = groups_of(conditions)
    best = None
    for a in range(len(conditions)):
        for b in range(a + 1, len(conditions)):
            if conditions[a] == conditions[b]:
                continue
            rest_a = [k for k in groups[conditions[a]] if k != a]
            rest_b = [k for k in groups[conditions[b]] if k != b]
            if not rest_a or not rest_b:
                continue
            gain = float(
                corr[a, rest_b].mean()
                + corr[b, rest_a].mean()
                - corr[a, rest_a].mean()
                - corr[b, rest_b].mean()
            )
            if best is None or gain > best[0] + 1e-12:
                partner = (rest_a[0], rest_b[0]) if len(rest_a) == len(rest_b) == 1 else None
                best = (gain, a, b, partner)
    return best


def treatment_signature(logx, use, lines, conditions, line):
    """Mean log fold change of a line's treated groups over its control, or None."""
    idx = [k for k in np.flatnonzero(lines == line) if use[k]]
    ctrl = [k for k in idx if conditions[k] == CONTROL]
    treated = sorted({conditions[k] for k in idx} - {CONTROL})
    if not ctrl or not treated:
        return None
    base = logx[:, ctrl].mean(axis=1)
    return np.mean(
        [logx[:, [k for k in idx if conditions[k] == c]].mean(axis=1) - base for c in treated],
        axis=0,
    )


def direction_score(logx, cpm, use, lines, conditions, line, other):
    """Correlation of a line's treatment signature with another line's, or None."""
    mine = treatment_signature(logx, use, lines, conditions, line)
    theirs = treatment_signature(logx, use, lines, conditions, other)
    if mine is None or theirs is None:
        return None
    keep = (cpm[:, lines == line].mean(axis=1) >= MIN_MEAN_CPM) & (
        cpm[:, lines == other].mean(axis=1) >= MIN_MEAN_CPM
    )
    both = np.column_stack([mine[keep], theirs[keep]])
    return float(correlation(both[top_magnitude(both, N_GENES)])[0, 1])


def main():
    inputs = load_inputs()
    lines, _, _ = assign_cell_lines(inputs)
    cpm = to_cpm(inputs.counts)
    use = ~shallow_libraries(inputs.counts)
    names = sorted(set(lines))
    ids = inputs.sample_ids
    gaps, candidate = {}, None
    for line in names:
        idx = np.flatnonzero((lines == line) & use)
        conditions = [inputs.conditions[k] for k in idx]
        if len(idx) < 4 or len(set(conditions)) < 2:
            continue
        corr = line_correlation(cpm, idx)
        found = best_exchange(corr, conditions)
        if found is None:
            continue
        gain, a, b, partner = found
        coherent = list(conditions)
        if gain > 0:
            coherent[a], coherent[b] = coherent[b], coherent[a]
        gaps[line] = gap(corr, coherent)
        ratio = float(np.clip(gain / gaps[line], -10.0, 10.0)) if gaps[line] > 0 else 0.0
        if candidate is None or ratio > candidate["ratio"]:
            candidate = {
                "ratio": ratio,
                "gain": gain,
                "line": line,
                "pair": [int(idx[a]), int(idx[b])],
                "alternative": [int(idx[k]) for k in partner] if partner else [],
            }
    result = {
        "swap_ratio": 0.0,
        "swap_gain": 0.0,
        "candidate_gap": 0.0,
        "min_line_gap": min(gaps.values()) if gaps else 0.0,
        "candidate_line": "",
        "candidate_pair": [],
        "alternative_pair": [],
        "candidate_conditions": [],
        "preferred_pair": [],
        "skipped_samples": [ids[k] for k in np.flatnonzero(~use)],
        "flagged_samples": [],
    }
    if candidate is None:
        return result
    pair, alternative = candidate["pair"], candidate["alternative"]
    preferred = []
    others = [n for n in names if n != candidate["line"]]
    if alternative and others and CONTROL in {inputs.conditions[k] for k in pair}:
        logx = log_cpm(cpm)
        scores = []
        for option in (pair, alternative):
            labels = inputs.conditions.copy()
            labels[option[0]], labels[option[1]] = labels[option[1]], labels[option[0]]
            scores.append(
                direction_score(logx, cpm, use, lines, labels, candidate["line"], others[0])
            )
        if None not in scores and abs(scores[0] - scores[1]) > PREFERENCE_MARGIN:
            preferred = pair if scores[0] > scores[1] else alternative
    result.update(
        swap_ratio=candidate["ratio"],
        swap_gain=candidate["gain"],
        candidate_gap=gaps[candidate["line"]],
        candidate_line=candidate["line"],
        candidate_pair=[ids[k] for k in pair],
        alternative_pair=[ids[k] for k in alternative],
        candidate_conditions=[inputs.conditions[k] for k in pair],
        preferred_pair=[ids[k] for k in preferred],
        flagged_samples=sorted(ids[k] for k in pair + alternative)
        if candidate["ratio"] >= SWAP_RATIO
        else [],
    )
    return result


if __name__ == "__main__":
    emit(main())
