import asyncio
import csv

import numpy as np
import pytest

from agents.llm import LLM
from core.eig_exec import ModelFamily, ParticleBelief
from eval.boxing_metrics import FAILURE_ERROR, paired_contrasts, summarize
from eval.boxing_study import FamilyProposal, Point, ask, export_results, run_episode
from labs.boxing.location_finding import (
    LocationFinding,
    intensity,
    predict_sources,
    source_prior,
)


def test_fixed_formula_and_seeded_observations():
    assert intensity([0, 0], [[0, 0], [1, 0], [0, 1]]) == pytest.approx(10002.099800019998)
    a, b = LocationFinding(41), LocationFinding(41)
    assert np.array_equal(a.evaluation_points(), b.evaluation_points())
    assert a.observe([.25, -.5]) == b.observe([.25, -.5])
    with pytest.raises(ValueError):
        a.observe([float("nan"), 0])
    with pytest.raises(ValueError):
        a.score([0])


def test_single_family_needs_parameter_eig():
    belief = ParticleBelief([ModelFamily("truth", predict_sources, source_prior)], .5,
                            n_particles=32, n_sim=8, null_weight=0)
    assert belief.expected_information_gain([[0, 0]], goal="family")[0] < 1e-9
    assert belief.expected_information_gain([[0, 0]], goal="total")[0] > 0
    assert belief.predict_means([[0, 0], [1, 1]]) == pytest.approx(
        [belief.predict_mean([0, 0]), belief.predict_mean([1, 1])])


def test_total_eig_matches_analytic_linear_gaussian_reference():
    family = ModelFamily("linear", lambda designs, params: params * designs[:, 0],
                         lambda rng, n: rng.normal(size=(n, 1)))
    belief = ParticleBelief([family], .5, n_particles=4096, n_sim=1024,
                            seed=41, null_weight=0)
    gains = belief.expected_information_gain([[0], [1], [2]], goal="total")
    exact = .5 * np.log2(1 + np.array([0, 1, 4]) / .25)
    assert gains == pytest.approx(exact, abs=.08)


def test_failure_is_included_and_bootstrap_is_reproducible():
    rows = [{"arm": "x", "budget": 5, "status": "ok", "mse": 1},
            {"arm": "x", "budget": 5, "status": "failed", "mse": 0}]
    summary = summarize(rows)
    assert summary == summarize(rows)
    assert summary[0]["failures"] == 1
    assert summary[0]["median"]["estimate"] == (1 + FAILURE_ERROR) / 2


def test_free_arms_share_truth_and_evaluation_points(tmp_path):
    async def check():
        records = []
        for arm in ["random", "grid", "oracle"]:
            records.append(await run_episode(arm, 0, [2, 3], tmp_path / arm,
                                             LLM(), particles=32, simulations=4))
        return records
    records = asyncio.run(check())
    assert len({r["sources_sha256"] for r in records}) == 1
    assert len({r["test_points_sha256"] for r in records}) == 1
    assert all(all(x["status"] == "ok" for x in r["rows"]) for r in records)
    assert all(r["usage"]["calls"] == 0 for r in records)


def test_crux_runs_real_numeric_engine(tmp_path):
    code = """import numpy as np
def sample_prior(rng, n):
    return rng.uniform(0, 10, size=(n, 1))
def predict(designs, params):
    return params[:, :1] * np.ones((1, designs.shape[0]))
"""
    result = asyncio.run(run_episode("crux_eig", 0, [1], tmp_path / "crux", LLM(),
        [FamilyProposal(statement="constant", code=code)], particles=16, simulations=2))
    assert result["rows"][0]["status"] == "ok"
    assert len(result["history"]) == 1


def test_paired_contrast_preserves_seed_pairing_and_failed_checkpoints():
    rows = [{"arm": arm, "seed": seed, "budget": 5, "status": "ok", "mse": value}
            for seed in range(20) for arm, value in [("crux_random", 100), ("crux_eig", 10)]]
    result = paired_contrasts(rows)[0]
    assert result["n"] == 20
    assert result["mean_log10_error_difference"] == pytest.approx(-1)
    assert result["high"] == pytest.approx(-1)
    rows[1]["status"] = "failed"
    assert paired_contrasts(rows)[0]["mean_log10_error_difference"] > -1


def test_spend_cap_rejects_before_api_call():
    llm = LLM()
    llm.study_cap = 0
    with pytest.raises(RuntimeError, match="StudySpendLimit"):
        asyncio.run(ask(llm, Point, {"task": "choose"}))
    assert llm.usage_totals()["calls"] == 0


def test_export_keeps_shared_costs_and_censored_targets(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.boxing_study.plot", lambda *args: None)
    record = {"rows": [{"arm": "crux_eig", "seed": 1, "budget": 5,
                         "status": "failed", "mse": FAILURE_ERROR, "failure": "NoPrediction"}],
              "usage": {"cost_usd": 0}, "paired_proposal_usage": {"cost_usd": .02},
              "target_reached_at_checkpoint": None}
    export_results([record], {"cost_usd": .02}, tmp_path)
    [row] = list(csv.DictReader((tmp_path / "costs.csv").open()))
    assert float(row["shared_proposal_cost_usd"]) == .02
    assert row["target_censored"] == "True"
    assert row["target_reached_at_checkpoint"] == ""
    assert row["failed_checkpoints"] == "1"
