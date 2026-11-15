"""Sample quality, check 1: sequencing depth and detected genes per library.

A library is degraded when its total count is below a tenth of the median library, or it
detects fewer than half the median number of genes. Ordinary library-size spread in one
experiment is two- to four-fold; a failed or heavily down-sampled library is far outside it.
"""

import numpy as np
from _common import emit, library_sizes, load_inputs

MIN_DEPTH_RATIO = 0.1
MIN_DETECTED_RATIO = 0.5


def main():
    inputs = load_inputs()
    sizes = library_sizes(inputs.counts)
    detected = (inputs.counts > 0).sum(axis=0).astype(float)
    depth_ratio = sizes / max(float(np.median(sizes)), 1.0)
    detected_ratio = detected / max(float(np.median(detected)), 1.0)
    low = (depth_ratio < MIN_DEPTH_RATIO) | (detected_ratio < MIN_DETECTED_RATIO)
    flagged = [s for s, bad in zip(inputs.sample_ids, low) if bad]
    return {
        "n_flagged": len(flagged),
        "flagged_samples": flagged,
        "min_depth_ratio": float(depth_ratio.min()),
        "min_detected_ratio": float(detected_ratio.min()),
        "median_library_size": float(np.median(sizes)),
        "samples": inputs.sample_ids,
        "library_sizes": sizes.tolist(),
        "depth_ratio": depth_ratio.tolist(),
        "detected_genes": detected.tolist(),
    }


if __name__ == "__main__":
    emit(main())
