"""Particle-based Bayesian experimental design for rival executable models (ERD section 6.2).

A hypothesis is a *family* of models written as code: ``predict(designs, params)`` plus a prior
sampler ``sample_prior(rng, n)``. The belief state holds a probability per family and, per
family, a cloud of weighted parameter particles. A broad "none of these" family ``h0``
(``y = c`` with a wide prior on ``c`` and four times the noise) is always added unless
``null_weight=0``, so data that no proposed family can explain shows up as belief moving to h0.

Observations are scalars: ``y = predict_k(d, theta) + Normal(0, sigma_k**2)`` where
``sigma_k = sigma * noise_scale_k``.

Design scoring is in bits, by Monte Carlo with ``n_sim`` simulated outcomes per family:

* ``goal="family"``: mutual information between the outcome and which family is true.
* ``goal="total"``: mutual information between the outcome and (family, parameters).

Which families the score is computed over: by default only the *proposed* families, with their
probabilities renormalised. h0 is a misspecification detector. It takes part in belief updates,
the predictive mixture and the surprise rule, but designs are not scored by how well they
separate "something else" from the proposals. Scoring h0 too would make a design where every
proposal predicts the same value look informative (h0 predicts a wide spread there). With a single
proposed family there is nothing else to discriminate, so h0 is then included. ``null_in_eig``
overrides the default.

All probabilities are handled as logs with a stable log-sum-exp. Underflow never yields NaN:
a family whose probability collapses is held at a floor of 1e-300, and a particle whose model
returns a non-finite prediction gets zero likelihood.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np

__all__ = [
    "LOG_PROB_FLOOR",
    "NULL_NAME",
    "NULL_NOISE_SCALE",
    "PROB_FLOOR",
    "SURPRISE_NULL_MASS",
    "SURPRISE_Z",
    "Design",
    "Goal",
    "ModelFamily",
    "ParticleBelief",
    "UpdateResult",
    "as_designs",
    "logsumexp",
]

NULL_NAME = "h0"
NULL_NOISE_SCALE = 4.0
SURPRISE_Z = 4.0
SURPRISE_NULL_MASS = 0.4
PROB_FLOOR = 1e-300
LOG_PROB_FLOOR = math.log(PROB_FLOOR)

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_LN2 = math.log(2.0)
_MAX_Z2 = 1e300  # caps squared z-scores so absurd values stay finite
_ACTIVE_MIN = 1e-12  # families below this (renormalised) mass are skipped when scoring designs
_SIM_STREAM = 0x5EED  # spawn-key tag of the design-simulation random stream
_CHUNK_ELEMENTS = 2_000_000  # bound on the (sources, sims, designs, particles) tensor per chunk
_JITTER = 0.1

Design = float | tuple[float, ...]
Goal = Literal["family", "total"]


@dataclass(frozen=True)
class ModelFamily:
    """One rival hypothesis: a parametric model written as code.

    ``predict(designs, params)`` gets designs of shape (n_designs, d) and params of shape
    (n_particles, p) and returns predictions of shape (n_particles, n_designs).
    ``sample_prior(rng, n)`` returns n parameter vectors, shape (n, p).
    """

    name: str
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]
    sample_prior: Callable[[np.random.Generator, int], np.ndarray]
    prior_weight: float = 1.0  # relative prior mass, normalised together with the other families
    noise_scale: float = 1.0  # this family's noise sd is sigma * noise_scale
    is_null: bool = False  # True only for the built-in "none of these" family


@dataclass(frozen=True)
class UpdateResult:
    """What one observation did to the belief.

    ``z_score``, ``pred_mean`` and ``pred_sd`` describe the mixture predictive *before* the
    update; ``family_posterior`` and ``surprise`` are evaluated *after* it.
    ``neg_log_pred`` is ``-log p(y)`` under the pre-update mixture (nats).
    """

    design: Design
    log_likelihood: dict[str, float]
    y: float
    family_posterior: dict[str, float]
    z_score: float
    neg_log_pred: float
    surprise: bool
    pred_mean: float
    pred_sd: float


class _Assessment(NamedTuple):
    design: Design
    y: float
    log_lik: list[np.ndarray]
    log_evidence: np.ndarray
    new_log_pi: np.ndarray
    pred_mean: float
    pred_var: float
    log_pred: float
    z_score: float
    surprise: bool


def logsumexp(a: Any, axis: int = -1) -> np.ndarray:
    """Stable ``log(sum(exp(a)))`` along ``axis``. All ``-inf`` inputs give ``-inf``, not NaN."""
    a = np.asarray(a, dtype=float)
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        total = np.sum(np.exp(a - peak), axis=axis, keepdims=True)
        out = np.log(total) + peak
    return np.squeeze(out, axis=axis)


def as_designs(designs: Any) -> np.ndarray:
    """Normalise candidate designs to a float array of shape (n_designs, d).

    A scalar is one 1-D design, a flat sequence is n scalar designs, and a sequence of
    tuples (or a 2-D array) is n designs of dimension d.
    """
    arr = np.asarray(designs, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(
            f"designs must be scalars, a 1-D sequence or (n, d); got shape {arr.shape}"
        )
    if arr.shape[0] and (arr.shape[1] == 0 or not np.all(np.isfinite(arr))):
        raise ValueError("designs must be finite and have at least one coordinate")
    return arr


def _single_design(design: Any) -> tuple[np.ndarray, Design]:
    """One design as shape (1, d), plus a hashable copy (float or tuple) for reporting."""
    arr = np.asarray(design, dtype=float)
    if arr.ndim == 0:
        frozen: Design = float(arr)
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == 1):
        arr = arr.reshape(1, -1)
        frozen = tuple(float(v) for v in arr[0])
    else:
        raise ValueError("a single design must be a scalar or a 1-D sequence of coordinates")
    if arr.shape[1] == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("a design must be finite and have at least one coordinate")
    return arr, frozen


def _candidate_items(candidates: Any) -> tuple[list[Design], np.ndarray]:
    designs = as_designs(candidates)
    if np.ndim(candidates) <= 1:
        items: list[Design] = [float(v) for v in designs[:, 0]]
    else:
        items = [tuple(float(v) for v in row) for row in designs]
    return items, designs


def _log_normal(y: float, mean: np.ndarray, sd: float) -> np.ndarray:
    """Gaussian log density. NaN means (an invalid prediction) give ``-inf``."""
    with np.errstate(over="ignore", invalid="ignore"):
        z = (y - mean) * (1.0 / sd)
        out = -0.5 * np.minimum(z * z, _MAX_Z2) - (math.log(sd) + _HALF_LOG_2PI)
    return np.where(np.isnan(out), -np.inf, out)


def _log_mixture_density(
    y: np.ndarray, means_t: np.ndarray, log_w: np.ndarray, sd: float, mask_nan: bool
) -> np.ndarray:
    """``log sum_m w_m N(y; means_t[j, m], sd^2)`` for y of shape (..., J), means_t (J, M).

    The hot loop of design scoring: it works in place on one (..., J, M) buffer. Inputs may
    hold NaN (invalid predictions; pass ``mask_nan=True``) but not infinities.
    """
    limit = 1e150 * sd
    bound = np.max(np.abs(y), where=~np.isnan(y), initial=0.0) + np.max(
        np.abs(means_t), where=~np.isnan(means_t), initial=0.0
    )
    t = y[..., None] - means_t
    if bound > limit:  # squared differences could overflow: cap them
        np.clip(t, -limit, limit, out=t)
    np.multiply(t, t, out=t)
    t *= -0.5 / (sd * sd)
    t += log_w - (math.log(sd) + _HALF_LOG_2PI)
    if mask_nan:
        t[np.isnan(t)] = -np.inf
    peak = np.max(t, axis=-1, keepdims=True)
    peak[~np.isfinite(peak)] = 0.0
    t -= peak
    np.exp(t, out=t)
    with np.errstate(divide="ignore"):
        return np.log(np.sum(t, axis=-1)) + peak[..., 0]


def _normalize_log(log_p: np.ndarray) -> np.ndarray:
    """Normalise log probabilities, holding every entry at or above the floor."""
    log_p = log_p - logsumexp(log_p)
    log_p = np.maximum(log_p, LOG_PROB_FLOOR)
    return log_p - logsumexp(log_p)


def _inverse_cdf(w: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Indices for uniforms ``u`` in [0, 1) under weights ``w``; never picks a zero weight."""
    cdf = np.cumsum(w)
    idx = np.searchsorted(cdf, u * cdf[-1], side="right")
    return np.minimum(idx, w.size - 1)


def _pick_by_weight(log_w: np.ndarray, u: np.ndarray) -> np.ndarray:
    return _inverse_cdf(np.exp(log_w - logsumexp(log_w)), u)


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = w.size
    return _inverse_cdf(w, (np.arange(n) + rng.random()) / n)


def _z_score(y: float, mean: float, var: float) -> float:
    if not (math.isfinite(var) and var > 0.0 and math.isfinite(mean)):
        return 0.0
    return (y - mean) / math.sqrt(var)


class ParticleBelief:
    """Belief over rival model families, each with weighted parameter particles.

    Args:
        families: the proposed model families (names must be unique; ``"h0"`` is reserved).
        sigma: observation noise standard deviation (each family scales it by ``noise_scale``).
        n_particles: particles per family.
        seed: seeds the prior draws and every later resampling / jitter step.
        null_weight: prior mass of the built-in h0 family; 0 disables it.
        n_sim: simulated outcomes per family per design when scoring designs.
        null_mean, null_scale: prior of h0's constant, ``c ~ Normal(null_mean, null_scale)``;
            ``null_scale`` defaults to ``10 * sigma``.
        null_in_eig: score designs over h0 as well as the proposals. ``None`` (default) means
            only when fewer than two families were proposed.
    """

    def __init__(
        self,
        families: Iterable[ModelFamily],
        sigma: float,
        n_particles: int = 500,
        seed: int | None = 0,
        null_weight: float = 0.15,
        n_sim: int = 64,
        *,
        null_mean: float = 0.0,
        null_scale: float | None = None,
        null_in_eig: bool | None = None,
    ) -> None:
        user = list(families)
        if not user:
            raise ValueError("at least one model family is required")
        sigma = float(sigma)
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError("sigma must be a positive finite number")
        if int(n_particles) != n_particles or n_particles < 2:
            raise ValueError("n_particles must be an integer >= 2")
        if int(n_sim) != n_sim or n_sim < 1:
            raise ValueError("n_sim must be an integer >= 1")
        null_weight = float(null_weight)
        if not 0.0 <= null_weight < 1.0:
            raise ValueError("null_weight must be in [0, 1)")

        names: set[str] = set()
        for fam in user:
            if not isinstance(fam, ModelFamily):
                raise TypeError(f"expected ModelFamily, got {type(fam).__name__}")
            if fam.is_null:
                raise ValueError("the null family is added automatically; pass null_weight instead")
            if fam.name in names:
                raise ValueError(f"duplicate family name {fam.name!r}")
            if null_weight > 0.0 and fam.name == NULL_NAME:
                raise ValueError(f"family name {NULL_NAME!r} is reserved for the null family")
            if not (math.isfinite(fam.prior_weight) and fam.prior_weight > 0.0):
                raise ValueError(f"family {fam.name!r}: prior_weight must be positive and finite")
            if not (math.isfinite(fam.noise_scale) and fam.noise_scale > 0.0):
                raise ValueError(f"family {fam.name!r}: noise_scale must be positive and finite")
            names.add(fam.name)

        self._sigma = sigma
        self._n_particles = int(n_particles)
        self._n_sim = int(n_sim)

        weights = np.array([fam.prior_weight for fam in user], dtype=float)
        log_pi = list(np.log1p(-null_weight) + np.log(weights / weights.sum()))
        fams = list(user)
        self._null_index: int | None = None
        if null_weight > 0.0:
            scale = 10.0 * sigma if null_scale is None else float(null_scale)
            if not (math.isfinite(scale) and scale > 0.0) or not math.isfinite(null_mean):
                raise ValueError("null_mean must be finite and null_scale positive")
            fams.append(_null_family(float(null_mean), scale, null_weight))
            log_pi.append(math.log(null_weight))
            self._null_index = len(fams) - 1
        self._families: tuple[ModelFamily, ...] = tuple(fams)
        self._log_pi = np.array(log_pi, dtype=float)
        self._sigmas = np.array([sigma * fam.noise_scale for fam in fams], dtype=float)

        if null_in_eig is None:
            null_in_eig = len(user) < 2
        self._eig_indices = [k for k, fam in enumerate(fams) if null_in_eig or not fam.is_null]

        self._seed_seq = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(self._seed_seq)
        n = self._n_particles
        self._params = [self._draw_prior(fam, n) for fam in fams]
        self._log_w = [np.full(n, -math.log(n)) for _ in fams]
        self._history: list[UpdateResult] = []

    # ------------------------------------------------------------------ read-only state

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(fam.name for fam in self._families)

    @property
    def null_name(self) -> str | None:
        return None if self._null_index is None else NULL_NAME

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def n_particles(self) -> int:
        return self._n_particles

    @property
    def history(self) -> list[UpdateResult]:
        return list(self._history)

    @property
    def n_observations(self) -> int:
        return len(self._history)

    def family_posterior(self) -> dict[str, float]:
        """Current probability of each family (proposals first, then h0)."""
        p = np.exp(self._log_pi)
        p = p / p.sum()
        return {fam.name: float(v) for fam, v in zip(self._families, p, strict=True)}

    def effective_sample_sizes(self) -> dict[str, float]:
        out = {}
        for fam, log_w in zip(self._families, self._log_w, strict=True):
            w = np.exp(log_w - logsumexp(log_w))
            out[fam.name] = float(1.0 / np.sum(w * w))
        return out

    def particles(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Copies of one family's particles (n, p) and their normalised weights (n,)."""
        k = self.family_names.index(name)
        log_w = self._log_w[k]
        return self._params[k].copy(), np.exp(log_w - logsumexp(log_w))

    def predictive(self, design: Any) -> tuple[float, float]:
        """Mean and variance of the mixture predictive at one design (variance includes noise)."""
        d, _ = _single_design(design)
        preds = [self._predict(k, d)[:, 0] for k in range(len(self._families))]
        return self._moments(preds)

    def predict_mean(self, design: Any) -> float:
        """Mixture predictive mean at one design."""
        return self.predictive(design)[0]

    def predict_means(self, designs: Any) -> np.ndarray:
        """Batch predictive means; sandboxed models execute once per family."""
        designs = as_designs(designs)
        numerator = np.zeros(len(designs))
        denominator = np.zeros(len(designs))
        for k in range(len(self._families)):
            predictions = self._predict(k, designs)
            valid = np.isfinite(predictions)
            weights = np.exp(self._log_w[k])[:, None] * valid
            totals = weights.sum(axis=0)
            means = (weights * np.where(valid, predictions, 0)).sum(axis=0) / np.maximum(totals, 1e-300)
            mass = np.exp(self._log_pi[k]) * (totals > 0)
            numerator += mass * means
            denominator += mass
        return numerator / np.maximum(denominator, 1e-300)

    # ------------------------------------------------------------------ design scoring

    def expected_information_gain(self, candidates: Any, goal: Goal = "family") -> np.ndarray:
        """EIG in bits for each candidate design, shape (n_designs,). Clipped at 0.

        Deterministic given the seed, the number of observations so far and the candidates.
        It uses its own random stream, so calling it never changes the update path.
        """
        if goal not in ("family", "total"):
            raise ValueError(f"goal must be 'family' or 'total', got {goal!r}")
        designs = as_designs(candidates)
        n_designs = designs.shape[0]
        eig = np.zeros(n_designs)
        if n_designs == 0:
            return eig

        idx = list(self._eig_indices)
        log_pi = self._log_pi[idx] - logsumexp(self._log_pi[idx])
        keep = np.exp(log_pi) > _ACTIVE_MIN
        idx = [k for k, ok in zip(idx, keep, strict=True) if ok]
        log_pi = log_pi[keep] - logsumexp(log_pi[keep])
        pi = np.exp(log_pi)
        n_src = len(idx)

        # Common random numbers: one draw for every family, shared by all candidate designs.
        rng = self._sim_rng()
        n_sim, n_fam = self._n_sim, len(self._families)
        strata = (np.arange(n_sim) + rng.random((n_fam, n_sim))) / n_sim
        eps = rng.standard_normal((n_fam, n_sim))[idx]
        picks = [_pick_by_weight(self._log_w[k], strata[k]) for k in idx]
        sig = self._sigmas[idx]
        log_w = [self._log_w[k] for k in idx]
        own_total = -(np.log(sig) + _HALF_LOG_2PI)[:, None] - 0.5 * eps * eps  # (src, sim)

        # One predict call per family for all candidates (a sandboxed family costs a process
        # per call); only the (src, sim, design, particle) tensor is processed in chunks.
        preds_t = [np.ascontiguousarray(self._predict(k, designs).T) for k in idx]  # (J, N)
        has_nan = any(bool(np.isnan(p).any()) for p in preds_t)
        chunk = max(1, _CHUNK_ELEMENTS // (n_src * n_sim * self._n_particles))
        diag = np.arange(n_src)
        for start in range(0, n_designs, chunk):
            stop = min(start + chunk, n_designs)
            block = [p[start:stop] for p in preds_t]  # each (n_block, n_particles)
            y = np.stack(
                [block[a][:, picks[a]].T + sig[a] * eps[a][:, None] for a in range(n_src)]
            )  # (src, sim, n_block)
            log_lik = np.empty(y.shape + (n_src,))
            for b in range(n_src):
                log_lik[..., b] = _log_mixture_density(
                    y, block[b], log_w[b], float(sig[b]), has_nan
                )
            log_py = logsumexp(log_lik + log_pi, axis=-1)  # (src, sim, n_block)
            if goal == "family":
                own = log_lik[diag, :, :, diag]
            else:
                own = own_total[:, :, None]
            with np.errstate(invalid="ignore"):
                term = own - log_py
            ok = np.isfinite(y) & np.isfinite(term)
            term = np.where(ok, term, 0.0)
            mean_term = term.sum(axis=1) / np.maximum(ok.sum(axis=1), 1)  # (src, n_block)
            eig[start:stop] = pi @ mean_term
        return np.maximum(eig / _LN2, 0.0)

    def rank_designs(
        self, candidates: Any, goal: Goal = "family", costs: Any = None
    ) -> list[tuple[Design, float, float]]:
        """Candidates as ``(design, eig_bits, score)`` sorted by score = eig / cost, descending.

        Designs come back as floats (scalar designs) or tuples of floats. Ties keep input order.
        """
        items, designs = _candidate_items(candidates)
        if not items:
            return []
        if costs is None:
            cost = np.ones(len(items))
        else:
            cost = np.asarray(costs, dtype=float).reshape(-1)
            if cost.shape != (len(items),):
                raise ValueError("costs must have one entry per candidate")
            if not np.all(np.isfinite(cost) & (cost > 0.0)):
                raise ValueError("costs must be positive and finite")
        eig = self.expected_information_gain(designs, goal)
        score = eig / cost
        order = sorted(range(len(items)), key=lambda j: -score[j])
        return [(items[j], float(eig[j]), float(score[j])) for j in order]

    # ------------------------------------------------------------------ observations

    def surprise(self, design: Any, y: float) -> bool:
        """Would observing ``y`` at ``design`` be a surprise? Pure query; state is unchanged."""
        return self._assess(design, y).surprise

    def update(self, design: Any, y: float) -> UpdateResult:
        """Condition the belief on observing ``y`` at ``design``."""
        a = self._assess(design, y)
        self._log_pi = a.new_log_pi
        for k in range(len(self._families)):
            if not math.isfinite(a.log_evidence[k]):
                continue  # no particle can explain y: leave this family's weights as they were
            log_w = self._log_w[k] + a.log_lik[k] - a.log_evidence[k]
            self._log_w[k] = log_w - logsumexp(log_w)
            self._maybe_resample(k)
        result = UpdateResult(
            design=a.design,
            log_likelihood={
                f.name: float(v) for f, v in zip(self._families, a.log_evidence, strict=True)
            },
            y=a.y,
            family_posterior=self.family_posterior(),
            z_score=a.z_score,
            neg_log_pred=-a.log_pred,
            surprise=a.surprise,
            pred_mean=a.pred_mean,
            pred_sd=math.sqrt(a.pred_var) if math.isfinite(a.pred_var) else math.inf,
        )
        self._history.append(result)
        return result

    # ------------------------------------------------------------------ internals

    def _assess(self, design: Any, y: float) -> _Assessment:
        d, frozen = _single_design(design)
        y = float(y)
        if not math.isfinite(y):
            raise ValueError("observation y must be finite")
        n_fam = len(self._families)
        preds = [self._predict(k, d)[:, 0] for k in range(n_fam)]
        log_lik = [_log_normal(y, preds[k], self._sigmas[k]) for k in range(n_fam)]
        log_ev = np.array([float(logsumexp(self._log_w[k] + log_lik[k])) for k in range(n_fam)])
        mean, var = self._moments(preds)
        log_joint = self._log_pi + log_ev
        log_pred = float(logsumexp(log_joint))
        new_log_pi = _normalize_log(log_joint) if math.isfinite(log_pred) else self._log_pi.copy()
        z = _z_score(y, mean, var)
        null_mass = 0.0
        if self._null_index is not None:
            p = np.exp(new_log_pi)
            null_mass = float(p[self._null_index] / p.sum())
        flag = abs(z) > SURPRISE_Z or null_mass > SURPRISE_NULL_MASS
        return _Assessment(frozen, y, log_lik, log_ev, new_log_pi, mean, var, log_pred, z, flag)

    def _moments(self, preds: list[np.ndarray]) -> tuple[float, float]:
        """Mixture mean and variance from per-family predictions at one design."""
        n_fam = len(self._families)
        pi = np.exp(self._log_pi)
        mus = np.zeros(n_fam)
        vs = np.zeros(n_fam)
        for k, f in enumerate(preds):
            finite = np.isfinite(f)
            w = np.exp(self._log_w[k]) * finite
            total = w.sum()
            if not total > 0.0:
                pi[k] = 0.0  # no valid prediction from this family
                continue
            fk = np.where(finite, f, 0.0)
            mus[k] = float(w @ fk) / total
            vs[k] = float(w @ (fk - mus[k]) ** 2) / total + self._sigmas[k] ** 2
        if not pi.sum() > 0.0:
            return 0.0, math.inf
        pi = pi / pi.sum()
        mean = float(pi @ mus)
        with np.errstate(over="ignore", invalid="ignore"):
            var = float(pi @ (vs + (mus - mean) ** 2))
        return mean, var

    def _predict(self, k: int, designs: np.ndarray) -> np.ndarray:
        fam = self._families[k]
        params = self._params[k]
        expected = (params.shape[0], designs.shape[0])
        out = np.asarray(fam.predict(designs, params), dtype=float)
        if out.shape != expected:
            try:
                if out.ndim != 2:
                    raise ValueError
                out = np.broadcast_to(out, expected)
            except ValueError:
                raise ValueError(
                    f"family {fam.name!r}: predict returned shape {out.shape}, expected {expected}"
                ) from None
        if not np.all(np.isfinite(out)):
            out = np.where(np.isfinite(out), out, np.nan)
        return out

    def _draw_prior(self, fam: ModelFamily, n: int) -> np.ndarray:
        params = np.asarray(fam.sample_prior(self._rng, n), dtype=float)
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        if params.ndim != 2 or params.shape[0] != n:
            raise ValueError(
                f"family {fam.name!r}: sample_prior returned shape {params.shape}, "
                f"expected ({n}, p)"
            )
        if not np.all(np.isfinite(params)):
            raise ValueError(f"family {fam.name!r}: sample_prior returned non-finite values")
        return np.array(params, dtype=float, copy=True)

    def _maybe_resample(self, k: int) -> None:
        n = self._n_particles
        w = np.exp(self._log_w[k])
        w = w / w.sum()
        if 1.0 / np.sum(w * w) >= n / 2.0:
            return
        params = self._params[k]
        with np.errstate(over="ignore", invalid="ignore"):
            mean = w @ params
            std = np.sqrt(np.maximum(w @ (params - mean) ** 2, 0.0))
        scale = np.where(np.isfinite(std) & (std > 0.0), _JITTER * std, 0.0)
        params = params[_systematic_resample(w, self._rng)]
        self._params[k] = params + self._rng.standard_normal(params.shape) * scale
        self._log_w[k] = np.full(n, -math.log(n))

    def _sim_rng(self) -> np.random.Generator:
        seq = np.random.SeedSequence(
            self._seed_seq.entropy, spawn_key=(_SIM_STREAM, self.n_observations)
        )
        return np.random.default_rng(seq)


def _null_family(mean: float, scale: float, weight: float) -> ModelFamily:
    """The broad "none of these" family: y = c with c ~ Normal(mean, scale)."""

    def predict(designs: np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.broadcast_to(params[:, :1], (params.shape[0], designs.shape[0]))

    def sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(mean, scale, size=(n, 1))

    return ModelFamily(
        name=NULL_NAME,
        predict=predict,
        sample_prior=sample_prior,
        prior_weight=weight,
        noise_scale=NULL_NOISE_SCALE,
        is_null=True,
    )
