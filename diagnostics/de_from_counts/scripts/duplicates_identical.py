"""Duplicates, check 1: count columns that are identical gene by gene.

Two independent libraries never agree on every gene. A pair is a duplicate when at least
99% of the genes with a count in either library carry the same count in both.
"""

from _common import emit, load_inputs

IDENTICAL_FRACTION = 0.99


def main():
    inputs = load_inputs()
    counts = inputs.counts
    n = counts.shape[1]
    pairs, best, best_pair = [], 0.0, []
    for i in range(n):
        for j in range(i + 1, n):
            either = (counts[:, i] > 0) | (counts[:, j] > 0)
            if not either.any():
                continue
            same = float((counts[either, i] == counts[either, j]).mean())
            if same > best:
                best, best_pair = same, [inputs.sample_ids[i], inputs.sample_ids[j]]
            if same >= IDENTICAL_FRACTION:
                pairs.append((inputs.sample_ids[i], inputs.sample_ids[j]))
    flagged = sorted({s for pair in pairs for s in pair})
    return {
        "n_duplicate_pairs": len(pairs),
        "duplicate_pairs": [f"{a}|{b}" for a, b in pairs],
        "max_identical_fraction": best,
        "closest_pair": best_pair,
        "flagged_samples": flagged,
        # Keep the first library of each pair; the copy cannot be told from the source.
        "suggested_exclusions": sorted({b for _, b in pairs}),
    }


if __name__ == "__main__":
    emit(main())
