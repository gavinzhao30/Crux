"""Tests for core/eig_exec.py (particle EIG over executable models) and core/model_sandbox.py."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from core.eig_exec import ModelFamily, ParticleBelief, as_designs, logsumexp
from core.model_sandbox import (
    ModelError,
    ModelSourceError,
    ModelTimeout,
    SandboxedFamily,
    UnsafeModelCode,
    validate_model_source,
)

SIGMA = 0.1
GRID = np.linspace(0.0, 2.0, 17)


def a_prior(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(1.0, 0.2, size=(n, 1))


def linear(designs: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[:, :1] * designs[:, 0][None, :]


def quadratic(designs: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[:, :1] * designs[:, 0][None, :] ** 2


def cubic(designs: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[:, :1] * designs[:, 0][None, :] ** 3


LINEAR = ModelFamily("linear", linear, a_prior)
QUADRATIC = ModelFamily("quadratic", quadratic, a_prior)
CUBIC = ModelFamily("cubic", cubic, a_prior)

TRUTHS = {
    "linear": lambda x: 1.1 * x,
    "quadratic": lambda x: 1.1 * x**2,
}


def make_belief(**kwargs) -> ParticleBelief:
    return ParticleBelief([LINEAR, QUADRATIC], sigma=SIGMA, **kwargs)


def assert_finite_posterior(post: dict[str, float]) -> None:
    values = np.array(list(post.values()))
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert math.isclose(values.sum(), 1.0, rel_tol=0, abs_tol=1e-9)


# ---------------------------------------------------------------- 1. discriminating design


def test_top_design_discriminates_and_x0_is_uninformative():
    belief = make_belief()
    ranked = belief.rank_designs(GRID, goal="family")
    assert len(ranked) == len(GRID)
    assert ranked[0][0] >= 1.75
    eig_at = {design: eig for design, eig, _ in ranked}
    assert eig_at[0.0] < 0.05
    assert eig_at[2.0] > 0.5
    scores = [score for _, _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_null_in_eig_makes_x0_informative_about_h0():
    belief = make_belief(null_in_eig=True)
    eig = belief.expected_information_gain([0.0])
    assert eig[0] > 0.1  # h0 predicts a wide spread at x=0, the proposals predict 0


def test_costs_divide_the_score():
    belief = make_belief()
    costs = np.ones(len(GRID))
    costs[-1] = 100.0  # make x=2 expensive
    ranked = belief.rank_designs(GRID, goal="family", costs=costs)
    assert ranked[0][0] != 2.0
    for design, eig, score in ranked:
        cost = 100.0 if design == 2.0 else 1.0
        assert score == pytest.approx(eig / cost)
    with pytest.raises(ValueError):
        belief.rank_designs(GRID, costs=np.zeros(len(GRID)))


# ---------------------------------------------------------------- 2. convergence


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("truth", ["quadratic", "linear"])
def test_converges_to_true_family_within_six_observations(truth, seed):
    belief = make_belief(seed=seed)
    data_rng = np.random.default_rng(100 + seed)
    reached = None
    for step in range(1, 7):
        x = belief.rank_designs(GRID, goal="family")[0][0]
        y = TRUTHS[truth](x) + SIGMA * data_rng.standard_normal()
        result = belief.update(x, y)
        assert not result.surprise
        if result.family_posterior[truth] > 0.9:
            reached = step
            break
    assert reached is not None, belief.family_posterior()
    assert belief.n_observations == reached


# ---------------------------------------------------------------- 3. surprise


def test_surprise_fires_on_off_family_data():
    belief = make_belief()
    fired = False
    for _ in range(3):
        x = belief.rank_designs(GRID, goal="family")[0][0]
        y = math.sin(5.0 * x)
        before = belief.family_posterior()
        flag = belief.surprise(x, y)
        assert belief.family_posterior() == before  # pure query
        assert belief.n_observations == len(belief.history)
        result = belief.update(x, y)
        assert result.surprise == flag
        if result.surprise:
            fired = True
            break
    assert fired
    assert result.family_posterior["h0"] > 0.4 or abs(result.z_score) > 4


def test_surprise_on_far_outlier_uses_z_score_without_null():
    belief = make_belief(null_weight=0.0)
    assert "h0" not in belief.family_posterior()
    mean, var = belief.predictive(1.0)
    y = mean + 10.0 * math.sqrt(var)
    assert belief.surprise(1.0, y)
    assert not belief.surprise(1.0, mean)


# ---------------------------------------------------------------- 4. determinism


def test_same_seed_same_output_and_ranking_does_not_perturb_updates():
    data_rng = np.random.default_rng(5)
    xs = [2.0, 1.0, 1.5, 0.5, 2.0, 1.25]
    ys = [1.1 * x**2 + SIGMA * data_rng.standard_normal() for x in xs]

    a = make_belief(seed=11)
    b = make_belief(seed=11)
    assert a.rank_designs(GRID) == b.rank_designs(GRID)
    assert a.rank_designs(GRID) == a.rank_designs(GRID)  # repeated query, same state
    for x, y in zip(xs, ys, strict=True):
        a.rank_designs(GRID, goal="family")  # only `a` ranks between updates
        a.rank_designs(GRID, goal="total")
        ra, rb = a.update(x, y), b.update(x, y)
        assert ra == rb
    assert a.family_posterior() == b.family_posterior()
    for name in a.family_names:
        pa, wa = a.particles(name)
        pb, wb = b.particles(name)
        np.testing.assert_array_equal(pa, pb)
        np.testing.assert_array_equal(wa, wb)
    assert a.rank_designs(GRID, goal="family") == b.rank_designs(GRID, goal="family")
    assert a.rank_designs(GRID, goal="total") == b.rank_designs(GRID, goal="total")


def test_different_seed_gives_different_particles():
    a = make_belief(seed=1)
    b = make_belief(seed=2)
    assert not np.array_equal(a.particles("linear")[0], b.particles("linear")[0])


# ---------------------------------------------------------------- 5. h0 accounting


def test_initial_posterior_h0_mass_and_prior_weights():
    fams = [
        ModelFamily("linear", linear, a_prior, prior_weight=1.0),
        ModelFamily("quadratic", quadratic, a_prior, prior_weight=3.0),
    ]
    belief = ParticleBelief(fams, sigma=SIGMA)
    post = belief.family_posterior()
    assert list(post) == ["linear", "quadratic", "h0"]
    assert abs(post["h0"] - 0.15) < 1e-12
    assert abs(post["linear"] - 0.85 * 0.25) < 1e-12
    assert abs(post["quadratic"] - 0.85 * 0.75) < 1e-12

    no_null = ParticleBelief(fams, sigma=SIGMA, null_weight=0.0)
    post = no_null.family_posterior()
    assert list(post) == ["linear", "quadratic"]
    assert abs(post["linear"] - 0.25) < 1e-12


def test_constructor_validation():
    with pytest.raises(ValueError):
        ParticleBelief([], sigma=SIGMA)
    with pytest.raises(ValueError):
        ParticleBelief([LINEAR, LINEAR], sigma=SIGMA)
    with pytest.raises(ValueError):
        ParticleBelief([ModelFamily("h0", linear, a_prior)], sigma=SIGMA)
    with pytest.raises(ValueError):
        ParticleBelief([LINEAR], sigma=0.0)
    with pytest.raises(ValueError):
        ParticleBelief([LINEAR], sigma=SIGMA, null_weight=1.0)
    with pytest.raises(ValueError):
        ParticleBelief([ModelFamily("bad", linear, a_prior, prior_weight=0.0)], sigma=SIGMA)
    bad_shape = ModelFamily("bad", lambda d, p: np.zeros(3), a_prior)
    with pytest.raises(ValueError):
        ParticleBelief([bad_shape, LINEAR], sigma=SIGMA).rank_designs(GRID)


# ---------------------------------------------------------------- 6. numerical safety


def test_absurd_observation_is_nan_free_and_h0_dominates():
    belief = make_belief()
    result = belief.update(2.0, 1e6)
    assert_finite_posterior(result.family_posterior)
    assert result.family_posterior["h0"] > 0.99
    assert result.surprise
    for value in (result.z_score, result.neg_log_pred, result.pred_mean, result.pred_sd):
        assert math.isfinite(value)
    assert all(math.isfinite(v) for v in belief.effective_sample_sizes().values())
    for goal in ("family", "total"):
        eig = belief.expected_information_gain(GRID, goal=goal)
        assert np.all(np.isfinite(eig)) and np.all(eig >= 0.0)
    assert math.isfinite(belief.predict_mean(1.0))
    # and the belief keeps working afterwards
    result = belief.update(1.0, 1.0)
    assert_finite_posterior(result.family_posterior)


def test_absurd_observation_without_null_family():
    belief = make_belief(null_weight=0.0)
    for y in (1e6, -1e6, 1e200):
        result = belief.update(2.0, y)
        assert_finite_posterior(result.family_posterior)
        assert math.isfinite(result.neg_log_pred)
    for name in belief.family_names:
        params, weights = belief.particles(name)
        assert np.all(np.isfinite(params)) and np.all(np.isfinite(weights))
        assert weights.sum() == pytest.approx(1.0)


def test_non_finite_predictions_get_zero_weight():
    def log_model(designs, params):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(params[:, :1]) * designs[:, 0][None, :]

    def wide_prior(rng, n):
        return rng.normal(0.5, 1.0, size=(n, 1))  # many negative draws -> NaN predictions

    belief = ParticleBelief([ModelFamily("log", log_model, wide_prior), LINEAR], sigma=SIGMA)
    eig = belief.expected_information_gain(GRID)
    assert np.all(np.isfinite(eig))
    for x in (2.0, 1.0, 0.5):
        result = belief.update(x, 1.1 * x)
        assert_finite_posterior(result.family_posterior)
        assert math.isfinite(result.z_score)
    with pytest.raises(ValueError):
        belief.update(1.0, float("nan"))


def test_logsumexp_is_stable():
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf
    assert logsumexp(np.array([1000.0, 1000.0])) == pytest.approx(1000.0 + math.log(2.0))
    a = np.array([[-1e4, 0.0], [3.0, -np.inf]])
    np.testing.assert_allclose(logsumexp(a, axis=1), [math.log1p(math.exp(-1e4)), 3.0])


# ---------------------------------------------------------------- 7. total goal


def test_total_goal_eig_is_finite_and_non_negative():
    belief = make_belief()
    ranked = belief.rank_designs(GRID, goal="total")
    eig = np.array([e for _, e, _ in ranked])
    assert eig.shape == (len(GRID),)
    assert np.all(np.isfinite(eig)) and np.all(eig >= 0.0)
    eig_at = {design: e for design, e, _ in ranked}
    assert eig_at[0.0] < 0.05  # every particle predicts exactly 0 there
    assert eig_at[2.0] > 0.5
    family = belief.expected_information_gain([2.0], goal="family")[0]
    assert eig_at[2.0] >= family - 0.05  # family identity is part of the total


def test_designs_as_tuples():
    def plane(designs, params):
        return params[:, :1] * designs[:, :1].T + params[:, 1:2] * designs[:, 1:2].T

    def plane_prior(rng, n):
        return rng.normal(1.0, 0.2, size=(n, 2))

    def x_only(designs, params):
        return params[:, :1] * designs[:, :1].T

    belief = ParticleBelief(
        [ModelFamily("plane", plane, plane_prior), ModelFamily("x_only", x_only, a_prior)],
        sigma=SIGMA,
    )
    candidates = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    ranked = belief.rank_designs(candidates)
    assert {d for d, _, _ in ranked} == set(candidates)
    assert ranked[0][0] in {(0.0, 1.0), (1.0, 1.0)}  # the second coordinate separates them
    result = belief.update((0.0, 1.0), 1.0)
    assert result.design == (0.0, 1.0)
    assert as_designs(0.5).shape == (1, 1)
    assert as_designs([0.1, 0.2]).shape == (2, 1)
    assert as_designs(candidates).shape == (3, 2)


def test_eig_of_a_design_does_not_depend_on_the_other_candidates():
    belief = make_belief()
    fine = np.linspace(0.0, 2.0, 161)  # contains GRID at every 10th point; several chunks
    big = belief.expected_information_gain(fine, goal="family")
    small = belief.expected_information_gain(GRID, goal="family")
    np.testing.assert_allclose(big[::10], small, rtol=1e-12, atol=1e-12)


def test_ranking_speed():
    belief = ParticleBelief(
        [LINEAR, QUADRATIC, CUBIC], sigma=SIGMA, n_particles=500, n_sim=64, null_in_eig=True
    )
    designs = np.linspace(0.0, 2.0, 16)
    belief.rank_designs(designs)  # warm up
    start = time.perf_counter()
    belief.rank_designs(designs)
    assert time.perf_counter() - start < 1.0


# ---------------------------------------------------------------- 8. sandbox

LINEAR_SRC = """
import numpy as np


def predict(designs, params):
    print("printed output must not corrupt the result")
    return params[:, :1] * designs[:, 0][None, :]


def sample_prior(rng, n):
    return rng.normal(1.0, 0.2, size=(n, 1))
"""

LOOP_SRC = """
def predict(designs, params):
    while True:
        pass


def sample_prior(rng, n):
    return rng.normal(size=(n, 1))
"""

FAILING_SRC = """
def predict(designs, params):
    raise ValueError("boom from the model")


def sample_prior(rng, n):
    return rng.normal(size=(n, 1))
"""


def test_sandboxed_family_matches_in_process_model():
    fam = SandboxedFamily.from_source("linear_sbx", LINEAR_SRC)
    assert isinstance(fam, ModelFamily)
    designs = as_designs(GRID)
    params = a_prior(np.random.default_rng(0), 40)
    np.testing.assert_array_equal(fam.predict(designs, params), linear(designs, params))
    first = fam.sample_prior(np.random.default_rng(3), 10)
    second = fam.sample_prior(np.random.default_rng(3), 10)
    assert first.shape == (10, 1)
    np.testing.assert_array_equal(first, second)


def test_sandboxed_family_inside_a_belief():
    fam = SandboxedFamily.from_source("linear_sbx", LINEAR_SRC)
    belief = ParticleBelief([fam, QUADRATIC], sigma=SIGMA, n_particles=200)
    assert belief.rank_designs(GRID)[0][0] >= 1.75
    result = belief.update(2.0, 4.4)
    assert result.family_posterior["quadratic"] > 0.9


@pytest.mark.parametrize(
    "snippet",
    [
        "import os",
        "import subprocess",
        "from os import path",
        "import numpy.testing",
        "from numpy import *",
        "from numpy import load",
        "x = open('data.txt')",
        "x = __import__('os')",
        "x = eval('1 + 1')",
        "exec('x = 1')",
        "x = ().__class__",
        "x = getattr(object, '__subclasses__')",
        "g = getattr",
        "import numpy as np\nx = np.load('a.npy')",
        "import numpy as np\nnp.zeros(3).tofile('a.bin')",
        "x = (i for i in range(3)).gi_frame",
        "x = '{0.__class__}'",
    ],
)
def test_unsafe_sources_are_rejected(snippet):
    src = snippet + "\n" + LOOP_SRC
    with pytest.raises(UnsafeModelCode):
        validate_model_source(src)
    with pytest.raises(UnsafeModelCode):
        SandboxedFamily.from_source("bad", src)


def test_source_must_parse_and_define_both_functions():
    with pytest.raises(ModelSourceError):
        validate_model_source("def predict(:\n    pass")
    with pytest.raises(ModelSourceError):
        validate_model_source("def predict(designs, params):\n    return designs\n")
    validate_model_source(LINEAR_SRC)  # the safe one passes
    validate_model_source("import math\nx = getattr(math, 'pi')\n" + LOOP_SRC)


def test_sandbox_timeout():
    fam = SandboxedFamily.from_source("loop", LOOP_SRC, timeout_s=2.0)
    start = time.perf_counter()
    with pytest.raises(ModelTimeout):
        fam.predict(as_designs(GRID), np.ones((5, 1)))
    assert time.perf_counter() - start < 3.5


def test_sandbox_runtime_error_includes_stderr():
    fam = SandboxedFamily.from_source("failing", FAILING_SRC)
    with pytest.raises(ModelError, match="boom from the model"):
        fam.predict(as_designs(GRID), np.ones((5, 1)))
