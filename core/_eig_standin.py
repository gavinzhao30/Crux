"""Categorical scoring (ERD section 6.1, Brief A). A stand-in until DU-6 lands ``core/eig.py``.

The engine never imports this module directly: ``core/scoring.py`` exposes these functions and
conventions and is backed by DU-6's array-based ``core.eig`` when it is present (converting the
dicts below to arrays), else by this module. The conventions:

* ``prior``: hypothesis id -> probability (normalised internally; zero entries are ignored).
* ``likelihood``: hypothesis id -> bin id -> P(bin | hypothesis), the shape of
  ``Experiment.likelihood``. Rows are normalised; a missing bin counts as 0 and a missing or
  all-zero row as uniform over the bins seen in the table.
* ``answer_keys``: hypothesis id -> answer key or ``None``. Options sharing a key are summed
  before the entropy terms; ``None`` keeps an option on its own.
* Null hypotheses ("none of these", ``P(o | h0) = 1/B``) are named by ``null_ids`` or, when it
  is omitted, recognised by the id ``"h0"`` or a ``".h0"`` suffix. ``rank`` uses
  ``Hypothesis.is_null``.

Formulas (bits, Shannon entropy H)::

    P(o | h)  = (1 - eps) * (n_o + 0.5) / (N + 0.5 * B) + eps / B     # from N sampled predictions
    pbar(o)   = sum_h pi_h * P(o | h)
    EIG(e)    = H[pbar] - sum_h pi_h * H[P(. | h)]
    score(e)  = impact * EIG / (1 + lam * est_cost_s / 60)
    update:     pi'_h is proportional to pi_h * P(o_observed | h)
    surprise:   pbar(o_observed) < 0.1  or  pi'(h0) > 0.4
    e-value:    e = kappa * p ** (kappa - 1)                          # 0.5 / sqrt(p) by default

As in ``core/eig_exec.py``, the null hypothesis is left out of the EIG (the remaining prior is
renormalised) unless fewer than two real options remain; ``null_in_eig`` overrides this. Scoring
h0 literally makes an experiment on which every real option agrees look informative, because
h0 spreads its mass over every bin; it would also make Brief A's acceptance tests 1 and 2
impossible to satisfy. h0 still takes part in updates and in the surprise rule.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence

from core.schemas import Experiment, Hypothesis, Question

__all__ = [
    "e_value",
    "entropy_bits",
    "expected_info_gain",
    "is_surprise",
    "likelihood_from_samples",
    "predictive",
    "rank",
    "robust_answer",
    "score",
    "update",
]

SURPRISE_PBAR = 0.1
SURPRISE_NULL_MASS = 0.4


def entropy_bits(p: Iterable[float]) -> float:
    """Shannon entropy in bits of a probability vector (0 log 0 = 0)."""
    return -sum(x * math.log2(x) for x in p if x > 0.0)


def likelihood_from_samples(
    counts: Mapping[str, int] | Sequence[int], n_bins: int, eps: float = 0.1
) -> dict[str, float] | list[float]:
    """Smoothed P(bin | h) from N sampled predictions.

    ``counts`` is bin id -> count (every bin present, zeros allowed) or a sequence of length
    ``n_bins``. Returns the same shape: a dict over the same bin ids, or a list.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    keys: list[str] | None
    if isinstance(counts, Mapping):
        keys = list(counts)
        values = [counts[k] for k in keys]
    else:
        keys = None
        values = list(counts)
    if len(values) != n_bins:
        raise ValueError(f"expected counts for {n_bins} bins, got {len(values)}")
    if any(v < 0 for v in values):
        raise ValueError("counts must be non-negative")
    total = float(sum(values))
    probs = [(1.0 - eps) * (v + 0.5) / (total + 0.5 * n_bins) + eps / n_bins for v in values]
    if keys is None:
        return probs
    return dict(zip(keys, probs, strict=True))


def score(eig: float, cost_s: float, impact: float = 1.0, lam: float = 0.2) -> float:
    """Expected information per unit cost: ``impact * eig / (1 + lam * cost_s / 60)``."""
    return impact * eig / (1.0 + lam * max(cost_s, 0.0) / 60.0)


def e_value(p: float, kappa: float = 0.5) -> float:
    """Calibrator e = kappa * p ** (kappa - 1) for a p-value p (0.5 / sqrt(p) by default)."""
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must be in (0, 1)")
    return kappa * p ** (kappa - 1.0)


# --- helpers ------------------------------------------------------------------------------


def _is_null(hid: str, null_ids: Collection[str] | None) -> bool:
    if null_ids is not None:
        return hid in null_ids
    return hid == "h0" or hid.endswith(".h0")


def _normalise(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {k: float(v) for k, v in weights.items() if v is not None and float(v) > 0.0}
    total = sum(clean.values())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("prior must have positive, finite mass")
    return {k: v / total for k, v in clean.items()}


def _bins(likelihood: Mapping[str, Mapping[str, float]], ids: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for hid in ids:
        for b in likelihood.get(hid, {}):
            seen.setdefault(b, None)
    return list(seen)


def _rows(
    likelihood: Mapping[str, Mapping[str, float]], ids: Iterable[str], bins: list[str]
) -> dict[str, list[float]]:
    out = {}
    n = len(bins)
    for hid in ids:
        raw = likelihood.get(hid) or {}
        row = [max(float(raw.get(b, 0.0) or 0.0), 0.0) for b in bins]
        total = sum(row)
        out[hid] = [v / total for v in row] if total > 0 else [1.0 / n] * n
    return out


# --- scoring ------------------------------------------------------------------------------


def expected_info_gain(
    prior: Mapping[str, float],
    likelihood: Mapping[str, Mapping[str, float]],
    answer_keys: Mapping[str, str | None] | None = None,
    *,
    null_ids: Collection[str] | None = None,
    null_in_eig: bool | None = None,
) -> float:
    """Mutual information (bits) between the outcome and which answer is right."""
    pi = _normalise(prior)
    real = [h for h in pi if not _is_null(h, null_ids)]
    include_null = len(real) < 2 if null_in_eig is None else null_in_eig
    if not include_null:
        if not real:
            return 0.0
        pi = _normalise({h: pi[h] for h in real})
    bins = _bins(likelihood, pi)
    if len(bins) < 2:
        return 0.0
    rows = _rows(likelihood, pi, bins)

    groups: dict[str, list[str]] = {}
    for hid in pi:
        key = (answer_keys or {}).get(hid)
        groups.setdefault(f"key:{key}" if key is not None else f"id:{hid}", []).append(hid)

    n = len(bins)
    pbar = [sum(pi[h] * rows[h][j] for h in pi) for j in range(n)]
    conditional = 0.0
    for members in groups.values():
        w = sum(pi[h] for h in members)
        if w <= 0.0:
            continue
        mix = [sum(pi[h] * rows[h][j] for h in members) / w for j in range(n)]
        conditional += w * entropy_bits(mix)
    return max(entropy_bits(pbar) - conditional, 0.0)


def rank(
    question: Question,
    hypotheses: list[Hypothesis],
    experiments: list[Experiment],
    lam: float = 0.2,
) -> list[Experiment]:
    """Copies of ``experiments`` with ``eig_bits`` and ``score`` set, best score first.

    The prior is the posterior of the question's active options; ties keep input order.
    """
    live = [h for h in hypotheses if h.question_id == question.id and h.status != "rejected"]
    prior = {h.id: h.posterior for h in live if h.posterior > 0.0}
    if not prior:
        prior = {h.id: h.prior for h in live if h.prior > 0.0}
    keys = {h.id: h.answer_key for h in live}
    nulls = {h.id for h in live if h.is_null}
    scored = []
    for exp in experiments:
        eig = expected_info_gain(prior, exp.likelihood, keys, null_ids=nulls) if prior else 0.0
        s = score(eig, exp.est_cost_s, question.impact, lam)
        scored.append(exp.model_copy(update={"eig_bits": eig, "score": s}))
    return sorted(scored, key=lambda e: -(e.score or 0.0))


def robust_answer(
    hypotheses: Iterable[Hypothesis] | Mapping[str, str | None],
    *,
    null_ids: Collection[str] | None = None,
) -> str | None:
    """The shared answer key when two or more active real options all share one, else None."""
    if isinstance(hypotheses, Mapping):
        keys = [k for hid, k in hypotheses.items() if not _is_null(hid, null_ids)]
    else:
        keys = [h.answer_key for h in hypotheses if h.status != "rejected" and not h.is_null]
    if len(keys) < 2 or keys[0] is None or any(k != keys[0] for k in keys):
        return None
    return keys[0]


# --- updating -----------------------------------------------------------------------------


def predictive(
    prior: Mapping[str, float], likelihood: Mapping[str, Mapping[str, float]], observed_bin: str
) -> float:
    """pbar(observed_bin) under the prior mixture."""
    pi = _normalise(prior)
    bins = _bins(likelihood, pi)
    if observed_bin not in bins:
        bins.append(observed_bin)
    rows = _rows(likelihood, pi, bins)
    j = bins.index(observed_bin)
    return sum(pi[h] * rows[h][j] for h in pi)


def update(
    prior: Mapping[str, float], likelihood: Mapping[str, Mapping[str, float]], observed_bin: str
) -> dict[str, float]:
    """Bayes' rule. Returns the posterior over the prior's ids (zero-prior ids dropped).

    If no hypothesis gives the observed bin any probability the prior is returned unchanged
    (normalised); with a null hypothesis present that cannot happen.
    """
    pi = _normalise(prior)
    bins = _bins(likelihood, pi)
    if observed_bin not in bins:
        bins.append(observed_bin)
    rows = _rows(likelihood, pi, bins)
    j = bins.index(observed_bin)
    joint = {h: pi[h] * rows[h][j] for h in pi}
    total = sum(joint.values())
    if total <= 0.0:
        return pi
    return {h: v / total for h, v in joint.items()}


def is_surprise(
    prior: Mapping[str, float],
    likelihood: Mapping[str, Mapping[str, float]],
    observed_bin: str,
    *,
    null_ids: Collection[str] | None = None,
    pbar_threshold: float = SURPRISE_PBAR,
    null_threshold: float = SURPRISE_NULL_MASS,
) -> bool:
    """True when the observation was unlikely under the mixture or moved mass to h0."""
    if predictive(prior, likelihood, observed_bin) < pbar_threshold:
        return True
    post = update(prior, likelihood, observed_bin)
    null_mass = sum(p for h, p in post.items() if _is_null(h, null_ids))
    return null_mass > null_threshold
