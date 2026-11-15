"""Duplicates, check 2: library-size-normalised profiles that are numerically the same.

Pearson correlation of log2(CPM + 1) on expressed genes. Real replicates stay well below
0.9999 because of sampling noise alone; a copied (or rescaled) library reaches 1.
"""

from _common import correlation, emit, load_inputs, log_cpm, to_cpm

DUPLICATE_CORRELATION = 0.9999


def main():
    inputs = load_inputs()
    cpm = to_cpm(inputs.counts)
    keep = cpm.mean(axis=1) >= 1.0
    corr = correlation(log_cpm(cpm[keep]))
    n = corr.shape[0]
    upper = [(corr[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    best, bi, bj = max(upper)
    pairs = [
        (inputs.sample_ids[i], inputs.sample_ids[j])
        for r, i, j in upper
        if r >= DUPLICATE_CORRELATION
    ]
    second = sorted(r for r, _, _ in upper)[-len(pairs) - 1] if len(upper) > len(pairs) else 0.0
    return {
        "n_duplicate_pairs": len(pairs),
        "duplicate_pairs": [f"{a}|{b}" for a, b in pairs],
        "max_correlation": float(best),
        "closest_pair": [inputs.sample_ids[bi], inputs.sample_ids[bj]],
        "max_correlation_excluding_duplicates": float(second),
        "flagged_samples": sorted({s for pair in pairs for s in pair}),
        "suggested_exclusions": sorted({b for _, b in pairs}),
        "n_genes": int(keep.sum()),
    }


if __name__ == "__main__":
    emit(main())
