"""Categorical experiment scoring and belief updates."""

from collections.abc import Sequence
from math import inf, isfinite
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .schemas import Experiment, Hypothesis, Question


def likelihood_from_samples(
    counts: ArrayLike, n_bins: int, eps: float = 0.1
) -> NDArray[np.float64]:
    """Estimate an outcome distribution from counts in bin order."""
    samples = np.asarray(counts, dtype=float)
    if n_bins < 1 or samples.shape != (n_bins,):
        raise ValueError("Provide one count per bin, with at least one bin")
    if (
        not np.all(np.isfinite(samples))
        or np.any(samples < 0)
        or np.any(samples != np.floor(samples))
    ):
        raise ValueError("Sample counts must be finite non-negative integers")
    if not 0 <= eps <= 1:
        raise ValueError("eps must be between zero and one")
    smoothed = (samples + 0.5) / (samples.sum() + 0.5 * n_bins)
    return (1 - eps) * smoothed + eps / n_bins


def _normalize_prior(prior: ArrayLike) -> NDArray[np.float64]:
    weights = np.asarray(prior, dtype=float)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Prior must be a vector of finite non-negative weights")
    total = weights.sum()
    if not isfinite(total) or total <= 0:
        raise ValueError("Prior must have finite positive mass")
    return weights / total


def _distributions(
    prior: ArrayLike, likelihood: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weights = _normalize_prior(prior)
    rows = np.asarray(likelihood, dtype=float)
    if rows.ndim != 2 or rows.shape[0] != weights.size or rows.shape[1] == 0:
        raise ValueError("Likelihood must have one row per hypothesis and at least one bin")
    if not np.all(np.isfinite(rows)) or np.any(rows < 0) or not np.allclose(rows.sum(axis=1), 1.0):
        raise ValueError("Each likelihood row must be a probability distribution")
    return weights, rows / rows.sum(axis=1, keepdims=True)


def _entropy(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    logs = np.zeros_like(probabilities)
    np.log2(probabilities, out=logs, where=probabilities > 0)
    return -np.sum(probabilities * logs, axis=-1)


def expected_info_gain(
    prior: ArrayLike, likelihood: ArrayLike, answer_keys: Sequence[str | None] | None = None
) -> float:
    """Return bits of information about hypotheses, or about keyed answers."""
    weights, rows = _distributions(prior, likelihood)
    if answer_keys is not None:
        if len(answer_keys) != weights.size:
            raise ValueError("Provide one answer key per hypothesis")
        groups: dict[str | int, list[int]] = {}
        for index, key in enumerate(answer_keys):
            groups.setdefault(index if key is None else key, []).append(index)
        joint = np.array([(weights[indices] @ rows[indices]) for indices in groups.values()])
        weights = joint.sum(axis=1)
        positive = weights > 0
        weights = weights[positive]
        rows = joint[positive] / weights[:, None]
    if weights.size == 1:
        return 0.0
    # Roundoff can leave a tiny negative residual for an uninformative experiment.
    return max(0.0, float(_entropy(weights @ rows) - weights @ _entropy(rows)))


def score(eig: float, cost_s: float, impact: float = 1.0, lam: float = 0.2) -> float:
    """Discount information gain by estimated experiment runtime."""
    if any(not isfinite(value) or value < 0 for value in (eig, cost_s, impact, lam)):
        raise ValueError(
            "Information gain, cost, impact and lambda must be finite and non-negative"
        )
    return impact * eig / (1 + lam * cost_s / 60)


def _weighted_outcome(
    prior: ArrayLike, likelihood: ArrayLike, observed_bin: int
) -> NDArray[np.float64]:
    weights, rows = _distributions(prior, likelihood)
    if not 0 <= observed_bin < rows.shape[1]:
        raise ValueError("Observed bin is outside the likelihood table")
    return weights * rows[:, observed_bin]


def update(prior: ArrayLike, likelihood: ArrayLike, observed_bin: int) -> NDArray[np.float64]:
    """Compute the posterior over individual hypotheses."""
    weighted = _weighted_outcome(prior, likelihood, observed_bin)
    evidence = weighted.sum()
    if evidence == 0:
        raise ValueError("Cannot update on an outcome with zero mixture probability")
    return weighted / evidence


def is_surprise(
    prior: ArrayLike,
    likelihood: ArrayLike,
    observed_bin: int,
    *,
    null_index: int | None = None,
) -> bool:
    """Flag a rare outcome or a posterior above 0.4 on the null hypothesis."""
    weighted = _weighted_outcome(prior, likelihood, observed_bin)
    if null_index is not None and not 0 <= null_index < weighted.size:
        raise ValueError("Null index is outside the prior")
    evidence = weighted.sum()
    return bool(
        evidence < 0.1 or (null_index is not None and weighted[null_index] > 0.4 * evidence)
    )


def rank(
    question: "Question", hypotheses: Sequence["Hypothesis"], experiments: Sequence["Experiment"]
) -> list["Experiment"]:
    """Set each candidate's EIG and score, and sort by score then experiment ID.

    The proposer supplies the initial posterior, including the null prior.
    The null's outcome distribution is uniform for every experiment.
    """
    active = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.question_id == question.id and hypothesis.status == "active"
    ]
    prior = [hypothesis.posterior for hypothesis in active]
    keys = [hypothesis.answer_key for hypothesis in active]
    scored = []
    for experiment in experiments:
        bin_ids = [outcome.id for outcome in experiment.bins]
        if not bin_ids or len(set(bin_ids)) != len(bin_ids):
            raise ValueError(f"Experiment {experiment.id} needs unique outcome bins")
        try:
            rows = [
                [1 / len(bin_ids)] * len(bin_ids)
                if hypothesis.is_null
                else [experiment.likelihood[hypothesis.id][bin_id] for bin_id in bin_ids]
                for hypothesis in active
            ]
        except KeyError as exc:
            raise ValueError(f"Experiment {experiment.id} is missing likelihood for {exc}") from exc
        eig = expected_info_gain(prior, rows, keys)
        scored.append((experiment, eig, score(eig, experiment.est_cost_s, question.impact)))

    for experiment, eig, value in scored:
        experiment.eig_bits = eig
        experiment.score = value
    scored.sort(key=lambda item: (-item[2], item[0].id))
    return [experiment for experiment, _, _ in scored]


def question_status(
    posterior: ArrayLike,
    best_eig: float | None,
    experiments_run: int,
    answer_keys: Sequence[str | None] | None = None,
) -> Literal["open", "robust", "settled", "unresolved"]:
    """Apply stop rules to the current options and the selected candidate's EIG."""
    weights = _normalize_prior(posterior)
    if answer_keys is not None:
        if len(answer_keys) != weights.size:
            raise ValueError("Provide one answer key per hypothesis")
        if answer_keys[0] is not None and all(key == answer_keys[0] for key in answer_keys):
            return "robust"
    if weights.max() >= 0.9:
        return "settled"
    if experiments_run >= 4 or best_eig is None or best_eig < 0.05:
        return "unresolved"
    return "open"


def e_value(p: float, kappa: float = 0.5) -> float:
    """Convert a valid p-value to a power-calibrator e-value."""
    if not 0 <= p <= 1 or not 0 < kappa < 1:
        raise ValueError("p must be in [0, 1] and kappa must be in (0, 1)")
    return inf if p == 0 else kappa * p ** (kappa - 1)
