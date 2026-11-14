"""Categorical scoring for the orchestrator: one dict-based interface, two backends.

The orchestrator and the fakes import only this module. Every function takes and returns
the dict shapes the engine stores:

* ``prior``: hypothesis id -> probability (normalised here; zero entries are dropped).
* ``likelihood``: hypothesis id -> bin id -> P(bin | hypothesis), the shape of
  ``Experiment.likelihood``. Rows are normalised; a missing bin counts as 0 and a missing
  or all-zero row as uniform over the bins seen in the table.
* ``answer_keys``: hypothesis id -> answer key or ``None`` (options sharing a key are one
  answer for the EIG).
* Null hypotheses ("none of these") are named by ``null_ids`` or, when it is omitted,
  recognised by the id ``"h0"`` or a ``".h0"`` suffix. ``rank`` uses ``Hypothesis.is_null``.

Backend. ``core.eig`` (DU-6, array based: ``prior`` and ``likelihood`` are arrays in a
fixed hypothesis and bin order, ``observed_bin`` is an index) is used when it imports and
passes a small probe; otherwise ``core._eig_standin``. ``BACKEND`` names the one in use.
For the array backend this module fixes the order (the prior's ids in insertion order;
bins in first-seen order over those rows, plus the observed bin if no row names it),
normalises the rows, calls ``core.eig`` and maps the arrays back to dicts.

Where ``core.eig`` differs from what the orchestrator relies on, this module keeps the
orchestrator's behaviour (the stand-in's conventions):

* EIG and the null option. ``core.eig.rank`` scores h0 with a uniform row. Here h0 is left
  out of the EIG (the rest renormalised) unless fewer than two real options remain, as in
  ``core/eig_exec.py``: scoring h0 makes a check on which every real option agrees look
  informative. ``rank`` therefore calls ``core.eig.expected_info_gain`` on the real rows
  instead of ``core.eig.rank``.
* ``rank`` returns scored copies sorted by score with ties in input order and takes
  ``lam``; ``core.eig.rank`` mutates its inputs, breaks ties by id and fixes ``lam=0.2``.
  It also keeps every option whose status is not ``rejected`` (accepted ones included),
  where ``core.eig.rank`` keeps ``status == "active"`` only.
* ``update`` returns the prior unchanged when no option gives the observed bin any
  probability; ``core.eig.update`` raises ``ValueError``.
* ``is_surprise`` takes any number of null ids (several are merged into one row, which
  leaves the predictive and the null mass unchanged) and custom thresholds (the stand-in
  is used for non-default ones); ``core.eig.is_surprise`` takes one ``null_index`` and
  fixes 0.1 and 0.4.
* ``score`` clamps a negative cost to 0 and accepts a negative EIG (roundoff from the
  particle scorer) by using the formula directly; ``core.eig.score`` raises on both.
* ``e_value`` keeps the stand-in's domain ``0 < p <= 1``; ``core.eig.e_value`` returns
  ``inf`` for ``p == 0``.
* ``likelihood_from_samples`` accepts a dict of counts and returns the same shape; counts
  that are not whole numbers use the stand-in formula (``core.eig`` rejects them).
* ``predictive`` and ``robust_answer`` have no ``core.eig`` counterpart and are computed
  here (``robust_answer`` is the stand-in's).
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from types import ModuleType

import numpy as np

from core import _eig_standin as standin
from core.schemas import Experiment, Hypothesis, Question

__all__ = [
    "BACKEND",
    "SURPRISE_NULL_MASS",
    "SURPRISE_PBAR",
    "Scorer",
    "e_value",
    "entropy_bits",
    "expected_info_gain",
    "is_compatible",
    "is_surprise",
    "likelihood_from_samples",
    "load_backend",
    "predictive",
    "rank",
    "robust_answer",
    "score",
    "update",
]

logger = logging.getLogger(__name__)

SURPRISE_PBAR = standin.SURPRISE_PBAR
SURPRISE_NULL_MASS = standin.SURPRISE_NULL_MASS
_REQUIRED = (
    "expected_info_gain",
    "update",
    "is_surprise",
    "score",
    "likelihood_from_samples",
    "e_value",
)


# --- dict conventions ----------------------------------------------------------------------


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
    likelihood: Mapping[str, Mapping[str, float]], ids: Sequence[str], bins: Sequence[str]
) -> list[list[float]]:
    out = []
    n = len(bins)
    for hid in ids:
        raw = likelihood.get(hid) or {}
        row = [max(float(raw.get(b, 0.0) or 0.0), 0.0) for b in bins]
        total = sum(row)
        out.append([v / total for v in row] if total > 0 else [1.0 / n] * n)
    return out


def _table(
    prior: Mapping[str, float],
    likelihood: Mapping[str, Mapping[str, float]],
    observed_bin: str,
) -> tuple[list[str], list[float], list[list[float]], int]:
    """Ids, weights, normalised rows and the observed bin's index, in a fixed order."""
    pi = _normalise(prior)
    ids = list(pi)
    bins = _bins(likelihood, ids)
    if observed_bin not in bins:
        bins.append(observed_bin)
    return ids, [pi[h] for h in ids], _rows(likelihood, ids, bins), bins.index(observed_bin)


# --- the backend ---------------------------------------------------------------------------


def is_compatible(module: ModuleType) -> bool:
    """True when ``module`` has DU-6's array API (checked on a tiny worked example)."""
    if not all(callable(getattr(module, name, None)) for name in _REQUIRED):
        return False
    rows = [[0.9, 0.1], [0.1, 0.9]]
    try:
        post = np.asarray(module.update([0.5, 0.5], rows, 0), dtype=float)
        eig = float(module.expected_info_gain([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], None))
        lik = np.asarray(module.likelihood_from_samples([8, 0], 2, 0.1), dtype=float)
        surprise = bool(module.is_surprise([0.5, 0.5], rows, 0, null_index=None))
        value = float(module.score(1.0, 60.0, 1.0, 0.2))
    except Exception:  # noqa: BLE001 - any failure means "not this API"
        return False
    return (
        post.shape == (2,)
        and abs(post[0] - 0.9) < 1e-9
        and abs(eig - 1.0) < 1e-9
        and lik.shape == (2,)
        and abs(lik[0] - 0.9) < 1e-9
        and not surprise
        and abs(value - 1.0 / 1.2) < 1e-9
    )


def load_backend(name: str = "core.eig") -> ModuleType | None:
    """The DU-6 module when it imports and is API-compatible, else ``None``."""
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        if getattr(exc, "name", None) != name:
            logger.warning("%s failed to import (%s); using the stand-in scorer", name, exc)
        return None
    if not is_compatible(module):
        logger.warning("%s does not match the array API; using the stand-in scorer", name)
        return None
    return module


class Scorer:
    """The dict-based scoring API over ``backend`` (``core.eig``) or, if ``None``, the
    stand-in. The module-level functions are the methods of one default instance."""

    def __init__(self, backend: ModuleType | None = None) -> None:
        self.backend = backend
        self.name = backend.__name__ if backend is not None else standin.__name__

    # --- pure helpers

    @staticmethod
    def entropy_bits(p: Iterable[float]) -> float:
        return standin.entropy_bits(p)

    @staticmethod
    def robust_answer(
        hypotheses: Iterable[Hypothesis] | Mapping[str, str | None],
        *,
        null_ids: Collection[str] | None = None,
    ) -> str | None:
        return standin.robust_answer(hypotheses, null_ids=null_ids)

    def predictive(
        self,
        prior: Mapping[str, float],
        likelihood: Mapping[str, Mapping[str, float]],
        observed_bin: str,
    ) -> float:
        """pbar(observed_bin) under the prior mixture."""
        if self.backend is None:
            return standin.predictive(prior, likelihood, observed_bin)
        _, weights, rows, j = _table(prior, likelihood, observed_bin)
        return float(sum(w * row[j] for w, row in zip(weights, rows, strict=True)))

    # --- scoring

    def likelihood_from_samples(
        self, counts: Mapping[str, int] | Sequence[int], n_bins: int, eps: float = 0.1
    ) -> dict[str, float] | list[float]:
        """Smoothed P(bin | h) from N sampled predictions, in the shape of ``counts``."""
        values = list(counts.values()) if isinstance(counts, Mapping) else list(counts)
        whole = all(float(v).is_integer() for v in values)
        if self.backend is None or not whole or len(values) != n_bins:
            return standin.likelihood_from_samples(counts, n_bins, eps)
        if any(v < 0 for v in values) or not 0.0 <= eps <= 1.0:
            return standin.likelihood_from_samples(counts, n_bins, eps)  # raises
        probs = [float(v) for v in self.backend.likelihood_from_samples(values, n_bins, eps)]
        if isinstance(counts, Mapping):
            return dict(zip(counts, probs, strict=True))
        return probs

    def score(self, eig: float, cost_s: float, impact: float = 1.0, lam: float = 0.2) -> float:
        """Expected information per unit cost: ``impact * eig / (1 + lam * cost_s / 60)``."""
        cost = max(cost_s, 0.0)
        valid = all(math.isfinite(v) and v >= 0.0 for v in (eig, cost, impact, lam))
        if self.backend is None or not valid:
            return standin.score(eig, cost_s, impact, lam)
        return float(self.backend.score(eig, cost, impact, lam))

    def e_value(self, p: float, kappa: float = 0.5) -> float:
        """Calibrator e = kappa * p ** (kappa - 1) for a p-value ``0 < p <= 1``."""
        if self.backend is None or not 0.0 < p <= 1.0 or not 0.0 < kappa < 1.0:
            return standin.e_value(p, kappa)
        return float(self.backend.e_value(p, kappa))

    def expected_info_gain(
        self,
        prior: Mapping[str, float],
        likelihood: Mapping[str, Mapping[str, float]],
        answer_keys: Mapping[str, str | None] | None = None,
        *,
        null_ids: Collection[str] | None = None,
        null_in_eig: bool | None = None,
    ) -> float:
        """Mutual information (bits) between the outcome and which answer is right."""
        if self.backend is None:
            return standin.expected_info_gain(
                prior, likelihood, answer_keys, null_ids=null_ids, null_in_eig=null_in_eig
            )
        pi = _normalise(prior)
        real = [h for h in pi if not _is_null(h, null_ids)]
        include_null = len(real) < 2 if null_in_eig is None else null_in_eig
        if not include_null:
            if not real:
                return 0.0
            pi = _normalise({h: pi[h] for h in real})
        ids = list(pi)
        bins = _bins(likelihood, ids)
        if len(bins) < 2:
            return 0.0
        keys = [(answer_keys or {}).get(h) for h in ids]
        weights = [pi[h] for h in ids]
        rows = _rows(likelihood, ids, bins)
        return float(self.backend.expected_info_gain(weights, rows, keys))

    def rank(
        self,
        question: Question,
        hypotheses: list[Hypothesis],
        experiments: list[Experiment],
        lam: float = 0.2,
    ) -> list[Experiment]:
        """Copies of ``experiments`` with ``eig_bits`` and ``score`` set, best score first.

        The prior is the posterior of the question's options that are not rejected; ties
        keep input order.
        """
        if self.backend is None:
            return standin.rank(question, hypotheses, experiments, lam)
        live = [h for h in hypotheses if h.question_id == question.id and h.status != "rejected"]
        prior = {h.id: h.posterior for h in live if h.posterior > 0.0}
        if not prior:
            prior = {h.id: h.prior for h in live if h.prior > 0.0}
        keys = {h.id: h.answer_key for h in live}
        nulls = {h.id for h in live if h.is_null}
        scored = []
        for exp in experiments:
            eig = (
                self.expected_info_gain(prior, exp.likelihood, keys, null_ids=nulls)
                if prior
                else 0.0
            )
            s = self.score(eig, exp.est_cost_s, question.impact, lam)
            scored.append(exp.model_copy(update={"eig_bits": eig, "score": s}))
        return sorted(scored, key=lambda e: -(e.score or 0.0))

    # --- updating

    def update(
        self,
        prior: Mapping[str, float],
        likelihood: Mapping[str, Mapping[str, float]],
        observed_bin: str,
    ) -> dict[str, float]:
        """Bayes' rule over the prior's ids (zero-prior ids dropped). If no option gives
        the observed bin any probability, the normalised prior is returned unchanged."""
        if self.backend is None:
            return standin.update(prior, likelihood, observed_bin)
        ids, weights, rows, j = _table(prior, likelihood, observed_bin)
        if sum(w * row[j] for w, row in zip(weights, rows, strict=True)) <= 0.0:
            return dict(zip(ids, weights, strict=True))
        post = self.backend.update(weights, rows, j)
        return {h: float(p) for h, p in zip(ids, post, strict=True)}

    def is_surprise(
        self,
        prior: Mapping[str, float],
        likelihood: Mapping[str, Mapping[str, float]],
        observed_bin: str,
        *,
        null_ids: Collection[str] | None = None,
        pbar_threshold: float = SURPRISE_PBAR,
        null_threshold: float = SURPRISE_NULL_MASS,
    ) -> bool:
        """True when the observation was unlikely under the mixture or moved mass to h0."""
        default = pbar_threshold == SURPRISE_PBAR and null_threshold == SURPRISE_NULL_MASS
        if self.backend is None or not default:
            return standin.is_surprise(
                prior,
                likelihood,
                observed_bin,
                null_ids=null_ids,
                pbar_threshold=pbar_threshold,
                null_threshold=null_threshold,
            )
        ids, weights, rows, j = _table(prior, likelihood, observed_bin)
        real = [i for i, h in enumerate(ids) if not _is_null(h, null_ids)]
        nulls = [i for i, h in enumerate(ids) if _is_null(h, null_ids)]
        w = [weights[i] for i in real]
        r = [rows[i] for i in real]
        null_index = None
        if nulls:
            mass = sum(weights[i] for i in nulls)
            width = len(rows[0])
            r.append([sum(weights[i] * rows[i][b] for i in nulls) / mass for b in range(width)])
            w.append(mass)
            null_index = len(w) - 1
        return bool(self.backend.is_surprise(w, r, j, null_index=null_index))


_DEFAULT = Scorer(load_backend())
BACKEND: str = _DEFAULT.name

entropy_bits = _DEFAULT.entropy_bits
robust_answer = _DEFAULT.robust_answer
predictive = _DEFAULT.predictive
likelihood_from_samples = _DEFAULT.likelihood_from_samples
score = _DEFAULT.score
e_value = _DEFAULT.e_value
expected_info_gain = _DEFAULT.expected_info_gain
rank = _DEFAULT.rank
update = _DEFAULT.update
is_surprise = _DEFAULT.is_surprise
