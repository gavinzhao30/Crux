"""Cell line, check 1: leave-one-out nearest centroid against the cell_line column.

Each library is compared (log2 CPM, top variable genes, centred) with the centroid of every
labelled cell line computed without it. A library that sits closer to another line's
centroid than to its own is mislabelled. Distinct cell lines differ in thousands of genes,
so one or two wrong labels barely move the centroids.
"""

import numpy as np
from _common import assign_cell_lines, emit, load_inputs


def main():
    inputs = load_inputs()
    assigned, own, other = assign_cell_lines(inputs)
    wrong = [k for k in range(len(assigned)) if assigned[k] != inputs.lines[k]]
    margin = own - other
    return {
        "n_mismatched": len(wrong),
        "mismatched_samples": [inputs.sample_ids[k] for k in wrong],
        "flagged_samples": [inputs.sample_ids[k] for k in wrong],
        "suggested_cell_lines": [f"{inputs.sample_ids[k]}:{assigned[k]}" for k in wrong],
        "min_margin": float(margin.min()),
        "median_margin": float(np.median(margin)),
    }


if __name__ == "__main__":
    emit(main())
