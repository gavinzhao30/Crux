import asyncio

import numpy as np
import pytest

from core.eig_exec import ModelFamily, ParticleBelief
from core.model_sandbox import ModelError
from core.model_validation import validate_family
from eval.boxing_proposals import generate_proposals
from eval.boxing_study import Families, FamilyProposal

GOOD = '''import numpy as np
def sample_prior(rng, n):
    return rng.normal(size=(n, 6))
def predict(designs, params):
    return params[:, :1] + np.zeros((1, len(designs)))
'''


@pytest.mark.parametrize("code", [
    GOOD.replace("(n, 6)", "(n, 3, 2)"),
    GOOD.replace("params[:, :1] + np.zeros((1, len(designs)))", "np.zeros((len(params), 3, len(designs)))"),
    GOOD.replace("params[:, :1] + np.zeros((1, len(designs)))", "np.zeros((len(params), len(params), len(designs)))"),
    GOOD.replace("params[:, :1] + np.zeros((1, len(designs)))", "np.random.default_rng().normal(size=(len(params), len(designs)))"),
    GOOD.replace("return params[:, :1]", "params[0, 0] = 0\n    return params[:, :1]"),
    GOOD.replace("return params[:, :1]", "return np.nan * params[:, :1]"),
])
def test_invalid_models_are_rejected(code):
    with pytest.raises(ModelError):
        validate_family(code)


def test_repair_attempts_are_preserved_and_bounded(monkeypatch):
    bad = FamilyProposal(statement="constant", code=GOOD.replace("(n, 6)", "(n, 3, 2)"))
    good = FamilyProposal(statement="constant", code=GOOD)
    calls = []

    async def ask(llm, schema, task):
        calls.append(task)
        return Families(families=[bad, good]) if schema is Families else good

    monkeypatch.setattr("eval.boxing_study.ask", ask)
    result = asyncio.run(generate_proposals(None, 3))
    assert result["validation_error"] is None
    assert len(result["families"]) == 2
    assert [a["repair"] for a in result["attempts"]] == [0, 1, 0]
    assert result["attempts"][0]["error"]

    async def never_valid(llm, schema, task):
        return Families(families=[bad, bad]) if schema is Families else bad

    monkeypatch.setattr("eval.boxing_study.ask", never_valid)
    result = asyncio.run(generate_proposals(None, 3))
    assert result["families"] == [] and result["validation_error"]
    assert len(result["attempts"]) == 6


@pytest.mark.parametrize("density", [True, False])
def test_resample_moves_match_full_history_gaussian_posterior(density):
    family = ModelFamily("constant", lambda d, p: p[:, :1] + np.zeros((1, len(d))),
                         lambda rng, n: rng.normal(size=(n, 1)),
                         log_prior=(lambda p: -.5*(p*p).sum(axis=1)) if density else None)
    beliefs = [ParticleBelief([family], .5, n_particles=4096, seed=42, null_weight=0, move_steps=4) for _ in range(2)]
    for belief in beliefs:
        for _ in range(4):
            belief.update([0], 1)
        assert belief.predict_mean([0]) == pytest.approx(4 / 4.25, abs=.04)
        params, weights = belief.particles("constant")
        mean = weights @ params[:, 0]
        assert weights @ (params[:, 0] - mean)**2 == pytest.approx(.25/4.25, abs=.02)
    assert np.array_equal(beliefs[0].particles("constant")[0], beliefs[1].particles("constant")[0])


def test_eig_matches_independent_gaussian_mixture_quadrature():
    means = np.array([-1., 1.])
    family = ModelFamily("binary", lambda d, p: p + np.zeros((1, len(d))),
                         lambda rng, n: means[:, None])
    b = ParticleBelief([family], .5, n_particles=2, n_sim=8192, null_weight=0)
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    total = 0
    for mean in means:
        y = mean + np.sqrt(2)*.5*nodes
        own = -.5*((y-mean)/.5)**2 - np.log(.5*np.sqrt(2*np.pi))
        density = np.exp(-.5*((y[:, None]-means)/.5)**2).mean(axis=1)/(.5*np.sqrt(2*np.pi))
        total += .5 * weights @ (own - np.log(density)) / np.sqrt(np.pi)/np.log(2)
    assert b.expected_information_gain([[0]], goal="total")[0] == pytest.approx(total, abs=.025)


def test_rankings_noise_and_query_purity():
    family = ModelFamily("linear", lambda d, p: p*d[:, 0], lambda rng, n: rng.normal(size=(n, 1)))
    winners = []
    for n, sims in [(512, 64), (2048, 256)]:
        b = ParticleBelief([family], .5, n_particles=n, n_sim=sims, null_weight=0, move_steps=4)
        before = b.particles("linear")
        gains = b.expected_information_gain([[0], [1], [2]], goal="total")
        winners.append(int(np.argmax(gains)))
        assert gains[0] == pytest.approx(0, abs=1e-9)
        assert np.array_equal(gains, b.expected_information_gain([[0], [1], [2]], goal="total"))
        assert np.array_equal(before[0], b.particles("linear")[0])
        noisy = ParticleBelief([family], 2, n_particles=n, n_sim=sims, null_weight=0)
        assert noisy.expected_information_gain([[2]], goal="total")[0] < gains[2]
    assert winners == [2, 2]
