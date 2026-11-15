"""Paired seed summaries. Failed episodes remain in every denominator."""

import numpy as np

FAILURE_ERROR = 1e12
ERROR_FLOOR = 1e-12
TARGET_ERROR = 1.0


def interval(values, *, log=False, seed=4100):
    values = np.asarray(values, dtype=float)
    if log:
        values = np.log10(np.maximum(values, ERROR_FLOOR))
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))]
    estimates = np.mean(draws, axis=1) if log else np.median(draws, axis=1)
    return {"estimate": float(np.mean(values) if log else np.median(values)),
            "low": float(np.quantile(estimates, .025)), "high": float(np.quantile(estimates, .975))}


def summarize(rows):
    result = []
    for arm, budget in sorted({(r["arm"], r["budget"]) for r in rows}):
        selected = [r for r in rows if r["arm"] == arm and r["budget"] == budget]
        errors = [r["mse"] if r["status"] == "ok" else FAILURE_ERROR for r in selected]
        result.append({"arm": arm, "budget": budget, "n": len(selected),
                       "failures": sum(r["status"] != "ok" for r in selected),
                       "median": interval(errors), "mean_log10": interval(errors, log=True)})
    return result


def paired_contrasts(rows):
    results = []
    for budget in sorted({r["budget"] for r in rows}):
        by_arm = {arm: {r["seed"]: r for r in rows if r["arm"] == arm and r["budget"] == budget}
                  for arm in ["crux_eig", "crux_random"]}
        seeds = sorted(set(by_arm["crux_eig"]) & set(by_arm["crux_random"]))
        if not seeds:
            continue
        def error(row):
            return row["mse"] if row["status"] == "ok" else FAILURE_ERROR
        differences = np.array([np.log10(max(error(by_arm["crux_eig"][s]), ERROR_FLOOR)) -
                               np.log10(max(error(by_arm["crux_random"][s]), ERROR_FLOOR)) for s in seeds])
        rng = np.random.default_rng(4141)
        boot = differences[rng.integers(0, len(seeds), (2000, len(seeds)))].mean(axis=1)
        results.append({"budget": budget, "n": len(seeds), "mean_log10_error_difference": float(differences.mean()),
                        "low": float(np.quantile(boot, .025)), "high": float(np.quantile(boot, .975))})
    return results
