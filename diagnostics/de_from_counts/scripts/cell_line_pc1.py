"""Cell line, check 2: unsupervised two-way split on the first principal component.

Log2 CPM of well-expressed genes, centred per gene and scaled to unit length per library,
first principal component, libraries split by the sign of their PC1 score. Each group is
named after the majority of its labels; libraries whose label disagrees with their group
are mislabelled. ``separation`` is the gap between group means over the pooled within-group
standard deviation; below 4 the cell lines are not cleanly separated on PC1 and the check
is unclear (``separation`` is 0 when a group is empty or its majority label is tied).

Libraries below a tenth of the median depth are left out (``skipped_samples``): their
low-count genes are biased downwards in log space, which can drag them across PC1. The
centroid check still places them.
"""

import numpy as np
from _common import emit, groups_of, load_inputs, log_cpm, shallow_libraries, to_cpm

MIN_MEAN_CPM = 10.0


def main():
    inputs = load_inputs()
    use = np.flatnonzero(~shallow_libraries(inputs.counts))
    lines = inputs.lines[use]
    cpm = to_cpm(inputs.counts[:, use])
    values = log_cpm(cpm[cpm.mean(axis=1) >= MIN_MEAN_CPM])
    values = values - values.mean(axis=1, keepdims=True)
    # Unit-length libraries: a noisy library cannot claim PC1 for itself.
    values = values / np.maximum(np.sqrt((values**2).sum(axis=0)), 1e-12)
    _, singular, vt = np.linalg.svd(values, full_matrices=False)
    scores = vt[0] * singular[0]
    # Genes are centred, so each group's scores keep the sign of its centroid; a noisy
    # library is pulled towards zero but not across it, unlike a two-means boundary.
    clusters = (scores > 0).astype(int)
    wrong, tie, names = [], False, {}
    for cluster, members in groups_of(clusters.tolist()).items():
        labels = [lines[k] for k in members]
        counts = {name: labels.count(name) for name in set(labels)}
        top = max(counts.values())
        winners = sorted(name for name, c in counts.items() if c == top)
        tie = tie or len(winners) > 1
        names[cluster] = winners[0]
        wrong += [k for k in members if lines[k] != winners[0]]
    separation = 0.0
    a, b = scores[clusters == 0], scores[clusters == 1]
    if not tie and len(set(names.values())) == 2 and len(a) and len(b):
        within = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        pooled = np.sqrt(within / max(len(scores) - 2, 1))
        separation = min(abs(a.mean() - b.mean()) / max(pooled, 1e-9), 1000.0)
    ids = [inputs.sample_ids[k] for k in use]
    wrong = sorted(wrong)
    return {
        "n_mismatched": len(wrong),
        "mismatched_samples": [ids[k] for k in wrong],
        "flagged_samples": [ids[k] for k in wrong],
        "suggested_cell_lines": [f"{ids[k]}:{names[int(clusters[k])]}" for k in wrong],
        "separation": float(separation),
        "pc1_variance_fraction": float(singular[0] ** 2 / (singular**2).sum()),
        "cluster_sizes": sorted(int((clusters == c).sum()) for c in (0, 1)),
        "skipped_samples": [
            s for s, keep in zip(inputs.sample_ids, ~shallow_libraries(inputs.counts)) if not keep
        ],
    }


if __name__ == "__main__":
    emit(main())
