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


def test_durable_bus_acknowledges_only_after_checkpoint(tmp_path, monkeypatch):
    from eval import run_suite as suite

    seen = []

    async def commit():
        seen.append(json.loads((tmp_path / "run/events.jsonl").read_text()) ["seq"])

    monkeypatch.setattr(suite, "checkpoint", commit)
    bus = suite.DurableEventBus("run", tmp_path)
    event = asyncio.run(bus.emit("run.started", {}))
    assert seen == [0] and event.seq == 0


def test_crashed_boxing_job_cannot_receive_positive_score():
    result = _grade_boxing({"seed": 1000}, {"state": "failed", "heldout_predictions": {"20": [1.0]*10}})
    assert result["score"] == 0


def test_recover_fetches_evidence_even_without_terminal_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from eval import run_suite as suite

    job = {"job_id": "interrupted", "suite": "boxing", "case_id": "1000", "seed": 1000,
           "arm": "crux", "usage": {"cost_usd": .01, "input_tokens": 100, "output_tokens": 20}}
    (tmp_path / "jobs.json").write_text(json.dumps({"suite": "boxing", "jobs": [job]}))

    class Missing:
        def read_file(self, path):
            raise FileNotFoundError(path)

    fetched = []

    def download(job, output):
        path = output / "evidence" / job["job_id"] / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        (path / "partial.txt").write_text("survived")
        fetched.append(job["job_id"])

    monkeypatch.setattr(suite, "volume", Missing())
    monkeypatch.setattr(suite, "_download_job", download)
    suite.recover(SimpleNamespace(output=tmp_path, suite="boxing", suite_dir=None))
    result = json.loads((tmp_path / "jobs/interrupted.json").read_text())
    assert result["score"] == 0 and fetched == ["interrupted"]
    assert result["shared_proposal_input_tokens"] == 100

    class Completed:
        def read_file(self, path):
            yield json.dumps({"state": "complete", "status": "finished", "heldout_predictions": {"20": [1.0]*10}}).encode()

    monkeypatch.setattr(suite, "volume", Completed())
    suite.recover(SimpleNamespace(output=tmp_path, suite="boxing", suite_dir=None))
    updated = json.loads((tmp_path / "jobs/interrupted.json").read_text())
    assert updated["state"] == "complete" and updated["score"] > 0


def test_fault_review_cache_prevents_duplicate_model_calls(tmp_path):
    from eval.run_suite import fault_metrics

    cache = tmp_path / "fault.json"
    cache.write_text(json.dumps({"fault_verdict": "caught", "fault_detected": True}))
    assert fault_metrics({}, {}, tmp_path, cache=cache, allow_model_calls=False)["fault_detected"]


def test_task9_metrics_preserve_usage_and_command_count(tmp_path):
    from eval.run_suite import evidence_metrics

    (tmp_path / "baseline.json").write_text(json.dumps({"usd_unadjusted_for_cache": ".02", "calls": 2,
        "usage": [{"prompt_tokens": 10, "completion_tokens": 5}], "execution_log": [{"valid": True}]}))
    result = evidence_metrics(tmp_path)
    assert result["input_tokens"] == 10 and result["experiments"] == 1
    assert not result["usage_complete"]


def test_all_task9_presets_resolve_and_numeric_capabilities_are_explicit(tmp_path):
    from eval.run_suite import ARMS, BOXING_ARMS, resolved_config

    configs = {arm: resolved_config(arm, 12, str(tmp_path), 420) for arm in ARMS if arm != "single_agent"}
    assert configs["crux_random_selection"]["selection"] == "random"
    assert not configs["crux_no_critic"]["critic"]
    assert configs["crux_role_only_memory"]["memory"] == "local_role_only"
    assert configs["crux"]["selection"] == "eig" and configs["crux"]["critic"]
    assert set(ARMS) - set(BOXING_ARMS) == {"crux_no_critic", "crux_role_only_memory"}


def test_download_preserves_empty_artifact_directories(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from modal.types import FileEntryType

    from eval import run_suite as suite

    class EmptyOutputs:
        def iterdir(self, prefix, recursive):
            return [SimpleNamespace(path="jobs/empty/artifacts", type=FileEntryType.DIRECTORY)]

    monkeypatch.setattr(suite, "volume", EmptyOutputs())
    suite._download_job({"job_id": "empty"}, tmp_path)
    assert (tmp_path / "evidence/empty/artifacts").is_dir()
