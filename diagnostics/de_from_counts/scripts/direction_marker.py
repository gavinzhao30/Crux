"""Group direction, check 2: which group keeps the gene that every other group lost?

A knockout removes (or sharply reduces) its target's transcript in every knocked-out group.
For each gene and cell line (assigned from the data), find the group with the highest mean
log2 CPM. A marker is a gene where, in every line, that top group is expressed (10 CPM or
more), at least 4-fold above every other group on average, and each of its libraries is at
least 2-fold above every library of the other groups. Markers whose top group is the
control in every line support the labels; markers whose top group is the control in all
lines but one point to that line's control and top group being exchanged.

This is best effort: with no such marker (no shared knockout target, or a transcript that
survives the knockout) the check reports unclear.
"""

import numpy as np
from _common import CONTROL, assign_cell_lines, emit, load_inputs, log_cpm, to_cpm

MIN_TOP_CPM = 10.0
MIN_GAP_LOG2 = 2.0
MIN_SAMPLE_GAP_LOG2 = 1.0


def main():
    inputs = load_inputs()
    lines, _, _ = assign_cell_lines(inputs)
    cpm = to_cpm(inputs.counts)
    logx = log_cpm(cpm)
    names = sorted(set(lines))
    candidate = np.ones(cpm.shape[0], dtype=bool)
    top_group = {}
    for line in names:
        idx = np.flatnonzero(lines == line)
        conditions = sorted({inputs.conditions[k] for k in idx})
        if len(conditions) < 2 or CONTROL not in conditions:
            candidate[:] = False
            break
        members = [[k for k in idx if inputs.conditions[k] == c] for c in conditions]
        means = np.column_stack([logx[:, m].mean(axis=1) for m in members])
        top = np.argmax(means, axis=1)
        ordered = np.sort(means, axis=1)
        # Every library of the top group must sit above every library of the other groups.
        rows = np.arange(len(top))
        lowest = np.column_stack([logx[:, m].min(axis=1) for m in members])
        highest = np.column_stack([logx[:, m].max(axis=1) for m in members])
        lowest_top = lowest[rows, top]
        highest[rows, top] = -np.inf
        highest_rest = highest.max(axis=1)
        candidate &= (
            (ordered[:, -1] - ordered[:, -2] >= MIN_GAP_LOG2)
            & (ordered[:, -1] >= np.log2(MIN_TOP_CPM + 1))
            & (lowest_top - highest_rest >= MIN_SAMPLE_GAP_LOG2)
        )
        top_group[line] = np.array(conditions, dtype=object)[top]
    concordant, discordant, votes = [], [], {}
    if len(names) >= 2:
        for g in np.flatnonzero(candidate):
            off = [line for line in names if top_group[line][g] != CONTROL]
            if not off:
                concordant.append(int(g))
            elif len(off) == 1:
                discordant.append(int(g))
                key = (off[0], top_group[off[0]][g])
                votes[key] = votes.get(key, 0) + 1
    flagged, suspect = [], ""
    if discordant and not concordant:
        line, group = max(votes, key=votes.get)
        suspect = f"{line}:{group}"
        flagged = sorted(
            inputs.sample_ids[k]
            for k in range(len(lines))
            if lines[k] == line and inputs.conditions[k] in (group, CONTROL)
        )
    genes = inputs.genes
    return {
        "n_ctrl_markers": len(concordant),
        "n_discordant_markers": len(discordant),
        "ctrl_markers": [genes.iloc[g] for g in concordant[:5]],
        "discordant_markers": [genes.iloc[g] for g in discordant[:5]],
        "suspect_group": suspect,
        "flagged_samples": flagged,
    }


if __name__ == "__main__":
    emit(main())
