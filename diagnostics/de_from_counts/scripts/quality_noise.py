"""Sample quality, check 2: excess noise relative to the other libraries of the cell line.

For each library, the noise is one minus its median correlation (log2 CPM, expressed genes)
with the other libraries assigned to the same cell line. A library is degraded when its
noise is more than 2.5 times the cell line's median noise, whatever its read count.
Cell lines are assigned from the data, so a mislabelled cell line does not look noisy.
"""

import numpy as np
from _common import assign_cell_lines, correlation, emit, load_inputs, log_cpm, to_cpm

MAX_NOISE_RATIO = 2.5
MIN_MEAN_CPM = 10.0


def main():
    inputs = load_inputs()
    lines, _, _ = assign_cell_lines(inputs)
    cpm = to_cpm(inputs.counts)
    ratio = np.ones(len(inputs.sample_ids))
    for line in sorted(set(lines)):
        idx = np.flatnonzero(lines == line)
        if len(idx) < 3:
            continue
        keep = cpm[:, idx].mean(axis=1) >= MIN_MEAN_CPM
        corr = correlation(log_cpm(cpm[np.ix_(keep, idx)]))
        noise = np.array([1.0 - np.median(np.delete(corr[k], k)) for k in range(len(idx))])
        ratio[idx] = noise / max(float(np.median(noise)), 1e-9)
    flagged = [s for s, r in zip(inputs.sample_ids, ratio) if r > MAX_NOISE_RATIO]
    worst = int(np.argmax(ratio))
    return {
        "n_flagged": len(flagged),
        "flagged_samples": flagged,
        "max_noise_ratio": float(ratio.max()),
        "noisiest_sample": inputs.sample_ids[worst],
        "samples": inputs.sample_ids,
        "noise_ratio": ratio.tolist(),
    }


if __name__ == "__main__":
    emit(main())
