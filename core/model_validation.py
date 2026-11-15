"""Strict numeric-model contracts, shared by local and remote evaluators."""
import numpy as np

from core.model_sandbox import ModelError, SandboxedFamily


def validate_family(source, seed=0):
    family = SandboxedFamily.from_source("validation", source)
    for n, m in [(4, 2), (7, 5), (512, 49)]:
        params = np.asarray(family.sample_prior(np.random.default_rng(seed), n))
        if params.ndim != 2 or params.shape[0] != n or params.shape[1] < 1 or not np.isfinite(params).all():
            raise ModelError(f"sample_prior: expected ({n}, p>0), finite; actual {params.shape}")
        again = family.sample_prior(np.random.default_rng(seed), n)
        if not np.array_equal(params, again):
            raise ModelError("sample_prior: fixed seed is not reproducible")
        designs = np.random.default_rng(seed + 1).uniform(-2, 2, (m, 2))
        original_params, original_designs = params.copy(), designs.copy()
        prediction = np.asarray(family.predict(designs, params))
        if prediction.shape != (n, m) or not np.isfinite(prediction).all():
            raise ModelError(f"predict: parameters {params.shape}, designs {designs.shape}; expected {(n, m)}, finite; actual {prediction.shape}")
        if not np.array_equal(prediction, family.predict(designs, params)):
            raise ModelError("predict: fixed inputs must give deterministic outputs")
        if not np.array_equal(params, original_params) or not np.array_equal(designs, original_designs):
            raise ModelError("predict: modified caller inputs")
    return family
