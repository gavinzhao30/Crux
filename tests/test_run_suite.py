"""DU-23 planning, isolation, and failure-accounting checks."""

import argparse
import asyncio
import json

import pytest

from agents.llm import LLM
from eval.boxing_study import run_episode
from eval.run_suite import _grade_boxing, _summary, plan
from labs.boxing.location_finding import LocationFinding


def args(tmp_path, suite="boxing", arms="single_agent,crux"):
    return argparse.Namespace(suite=suite, arms=arms, cases="dev", case_ids=None, repeats=2,
                              concurrency=2, timeout=1800, seed=7, suite_dir=tmp_path,
                              references=None, grader_repo=None, grader_python=None,
                              dry_run=True, model="gpt-4.1-mini")


def test_boxing_plan_pairs_seeds_and_rejects_equivalent_arms(tmp_path):
    planned = plan(args(tmp_path))
    assert len(planned["jobs"]) == 20
    pairs = {}
    for job in planned["jobs"]:
        pairs.setdefault((job["case_id"], job["repeat"]), set()).add(job["seed"])
    assert all(len(seeds) == 1 for seeds in pairs.values())
    assert set(planned["boxing_seed_splits"]["dev"]).isdisjoint(
        planned["boxing_seed_splits"]["test"])
    with pytest.raises(ValueError, match="cannot exercise critic"):
        plan(args(tmp_path, arms="crux_no_critic"))


def test_task9_plan_uses_private_truth_only_for_case_selection(tmp_path):
    for case_id in ("case-01", "case-02"):
        case = tmp_path / "cases" / case_id
        case.mkdir(parents=True)
        (case / "tagseq_featurecounts_matrix.tsv").write_text("data")
        (case / "samples.tsv").write_text("data")
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    (evaluator / "truth.json").write_text(json.dumps({"cases": {
        "case-01": {"split": "dev", "fault": "clean"},
        "case-02": {"split": "test", "fault": "sample_swap"},
    }}))
    planned = plan(args(tmp_path, suite="task9"))
    assert len(planned["jobs"]) == 4
    assert {j["case_id"] for j in planned["jobs"]} == {"case-01"}
    assert all("fault" not in job for job in planned["jobs"])
    assert planned["jobs"][0]["config"] is None


def test_boxing_grade_is_deferred_and_failures_score_zero():
    job = {"seed": 1000}
    env = LocationFinding(1000)
    predictions = [1.0] * len(env.evaluation_points())
    result = _grade_boxing(job, {"heldout_predictions": {"20": list(predictions)}})
    assert result["grade_status"] == "graded"
    assert 0 < result["score"] <= 1
    assert result["checkpoints"][0]["score"] == 0
    failed = _grade_boxing(job, {"status": "failed"})
    assert failed["score"] == 0
    assert failed["grade_status"] == "failed"


def test_boxing_episode_exports_predictions_without_scoring(tmp_path):
    record = asyncio.run(run_episode("random", 1000, [1, 2], tmp_path / "episode",
                                     LLM(), particles=16, simulations=2, defer_grading=True))
    assert [row["status"] for row in record["rows"]] == ["pending_grade", "pending_grade"]
    assert all(row["mse"] is None for row in record["rows"])
    assert set(record["heldout_predictions"]) == {1, 2}


def test_summary_keeps_failed_job_in_denominator(tmp_path):
    _summary([{"job_id": "good", "score": 0.7}, {"job_id": "crash", "score": 0.0}], tmp_path)
    rows = (tmp_path / "summary.csv").read_text().splitlines()
    assert len(rows) == 3
    assert "crash" in rows[-1]
