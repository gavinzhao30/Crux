"""End-to-end tests of the run loop on the offline fakes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.cli import main as cli_main
from core.config import load_config
from core.fakes_engine import (
    FakeAgents,
    FakeClock,
    FakeLab,
    FakeNumericLab,
    FakePredictor,
    FakeSurpriseLab,
    fake_deps,
)
from core.orchestrator import Deps, run
from core.replay import main as replay_main
from core.replay import replay_check
from core.schemas import CriticFinding, Report


def run_fake(
    tmp_path: Path,
    lab=None,
    preset: str = "default",
    run_id: str = "r-test",
    deps: Deps | None = None,
    **overrides,
) -> tuple[Report, list[dict], Path]:
    lab = lab if lab is not None else FakeLab()
    deps = deps if deps is not None else fake_deps(lab, clock=FakeClock())
    cfg = load_config(preset, runs_dir=str(tmp_path), **overrides)
    report = asyncio.run(run(lab, cfg, deps, run_id=run_id))
    path = tmp_path / run_id / "events.jsonl"
    return report, events_of(path), path


def events_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def of_type(events: list[dict], etype: str) -> list[dict]:
    return [e["payload"] for e in events if e["type"] == etype]


def started_ids(events: list[dict]) -> list[str]:
    return [p["experiment_id"] for p in of_type(events, "experiment.started")]


# --- the main path ------------------------------------------------------------------------------


def test_categorical_run_settles_each_question_on_the_hidden_truth(tmp_path: Path) -> None:
    lab = FakeLab()
    report, events, _ = run_fake(tmp_path, lab)
    settled = {p["question_id"]: p["hypothesis_id"] for p in of_type(events, "question.settled")}
    assert settled == {"q1": "q1.reverse", "q2": "q2.batch"}
    assert all(p["posterior"] >= 0.9 for p in of_type(events, "question.settled"))
    assert events[-1]["payload"]["status"] == "ok"
    assert report.unresolved == []
    assert {r["hypothesis"] for r in report.rejected} == {
        "q1.unstranded",
        "q1.forward",
        "q2.no_batch",
    }
    assert all(r["evidence_id"] for r in report.rejected)
    # duplicates merged with summed priors, null option added to every question
    merged = of_type(events, "hypothesis.merged")
    assert merged[0]["kept"] == "q1.reverse" and merged[0]["merged"] == ["q1.reverse.p2"]
    assert merged[0]["priors"]["q1.reverse"] == pytest.approx(0.425)
    assert merged[0]["priors"]["q1.h0"] == pytest.approx(0.15)
    nulls = [
        p["hypothesis"]["id"]
        for p in of_type(events, "hypothesis.proposed")
        if p["hypothesis"]["is_null"]
    ]
    assert nulls == ["q1.h0", "q2.h0"]
    # the check on which every option agrees never runs; cheap search runs and blocks a leak
    assert "library_sizes" not in started_ids(events)
    assert "intronic_rate" not in started_ids(events)
    (search,) = of_type(events, "search.done")
    assert search["blocked"] and all("ground-truth" not in u for u in search["urls"])
    # the deliverable follows the settled answers and the critic's repair round
    assert [a.name for a in report.artifacts] == ["de_table.csv", "qc_summary.csv"]
    assert all(a.contract_ok for a in report.artifacts)
    qc = Path(report.artifacts[1].path).read_text(encoding="utf-8")
    assert "reverse" in qc and "~ cell_line + condition" in qc
    # one budget tick per model call plus one per experiment
    ticks = of_type(events, "budget.tick")
    assert ticks[-1]["llm_calls"] == len(lab_calls_of(events)) == 16
    assert len(ticks) == ticks[-1]["llm_calls"] + len(of_type(events, "experiment.finished"))
    assert report.cost_usd == pytest.approx(ticks[-1]["cost_usd"])


def lab_calls_of(events: list[dict]) -> list[int]:
    """The llm_calls counter at each tick where it went up (one per model call)."""
    out, last = [], 0
    for p in of_type(events, "budget.tick"):
        if p["llm_calls"] > last:
            out.append(p["llm_calls"])
            last = p["llm_calls"]
    return out


def test_question_order_follows_impact_times_entropy(tmp_path: Path) -> None:
    _, events, _ = run_fake(tmp_path)
    scored = [p["question_id"] for p in of_type(events, "experiment.scored")]
    assert scored[0] == "q1"  # higher impact and more options
    assert started_ids(events) == [
        "strand_fraction",
        "batch_search",
        "pca_cell_line",
        "residual_clusters",
    ]


def test_random_selection_changes_the_experiment_order(tmp_path: Path) -> None:
    _, eig_events, _ = run_fake(tmp_path, run_id="r-eig")
    _, rnd_events, _ = run_fake(tmp_path, preset="crux_random_selection", run_id="r-random")
    assert of_type(rnd_events, "run.started")[0]["config"]["selection"] == "random"
    assert started_ids(rnd_events) != started_ids(eig_events)
    for p in of_type(rnd_events, "experiment.scored"):
        assert p["chosen"] in {r["experiment_id"] for r in p["ranking"]}


def test_model_pick_runs_the_designers_first_surviving_proposal(tmp_path: Path) -> None:
    _, events, _ = run_fake(tmp_path, selection="model_pick")
    proposed_order = [p["experiment"]["id"] for p in of_type(events, "experiment.proposed")]
    for scored in of_type(events, "experiment.scored"):
        candidates = {r["experiment_id"] for r in scored["ranking"]}
        first = next(e for e in proposed_order if e in candidates)
        assert scored["chosen"] == first
    assert started_ids(events)[:2] == ["strand_fraction", "batch_search"]


def test_no_critic_emits_no_finding_and_delivers_once(tmp_path: Path) -> None:
    report, events, _ = run_fake(tmp_path, preset="crux_no_critic")
    assert not of_type(events, "critic.finding")
    assert len(of_type(events, "artifact.written")) == 2
    assert not report.artifacts[0].contract_ok  # nobody caught the missing padj column


def test_critic_repair_runs_exactly_two_delivery_rounds(tmp_path: Path) -> None:
    report, events, _ = run_fake(tmp_path)
    written = [p["artifact"] for p in of_type(events, "artifact.written")]
    # each round writes both artifacts, then re-emits the same versions with the verdict
    assert [a["name"] for a in written] == ["de_table.csv", "qc_summary.csv"] * 4
    r1_write, r1_verdict, r2_write, r2_verdict = (written[i : i + 2] for i in range(0, 8, 2))
    assert [a["sha256"] for a in r1_verdict] == [a["sha256"] for a in r1_write]
    assert [a["contract_ok"] for a in r1_verdict] == [False, True]  # de_table blocked
    assert r2_write[0]["sha256"] != r1_write[0]["sha256"]  # the repaired table is a new version
    assert [a["contract_ok"] for a in r2_verdict] == [True, True]  # both pass after repair
    (finding,) = of_type(events, "critic.finding")
    assert finding["finding"]["severity"] == "block"
    assert finding["finding"]["artifact"] == "de_table.csv"
    header = Path(report.artifacts[0].path).read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[-1] == "padj"
    recalls = [p for p in of_type(events, "memory.recalled") if p["reader_role"] == "critic"]
    assert len(recalls) == 2


def test_tiny_budget_ends_cleanly_as_budget_exhausted(tmp_path: Path) -> None:
    report, events, path = run_fake(tmp_path, max_llm_calls=3)
    assert events[-1]["type"] == "run.finished"
    assert events[-1]["payload"]["status"] == "budget_exhausted"
    assert "model calls" in report.methods_md
    assert set(report.unresolved) == {"q1", "q2"}
    assert of_type(events, "budget.tick")[-1]["llm_calls"] == 3
    assert replay_check(path).ok


def test_cost_cap_stops_after_an_expensive_experiment(tmp_path: Path) -> None:
    _, events, _ = run_fake(tmp_path, max_usd=0.011)
    assert events[-1]["payload"]["status"] == "budget_exhausted"
    assert of_type(events, "budget.tick")[-1]["cost_usd"] >= 0.011


def test_surprise_regenerates_options_and_settles_on_the_new_truth(tmp_path: Path) -> None:
    _, events, path = run_fake(tmp_path, FakeSurpriseLab())
    types = [e["type"] for e in events]
    first_surprise = types.index("surprise.raised")
    later = [
        e["payload"]["hypothesis"]["id"]
        for e in events[first_surprise:]
        if e["type"] == "hypothesis.proposed"
    ]
    assert later == ["q1.mixed_kits", "q1.gdna_contamination"]
    surprise = of_type(events, "surprise.raised")[0]
    assert surprise["question_id"] == "q1" and surprise["null_mass"] > 0.4
    rejected = {p["hypothesis_id"]: p for p in of_type(events, "hypothesis.rejected")}
    assert {"q1.unstranded", "q1.forward", "q1.reverse"} <= set(rejected)
    assert rejected["q1.reverse"]["evidence_id"]
    settled = {p["question_id"]: p["hypothesis_id"] for p in of_type(events, "question.settled")}
    assert settled == {"q1": "q1.mixed_kits", "q2": "q2.batch"}
    assert replay_check(path).ok  # the evidence replay over the new options is consistent


def test_numeric_run_settles_on_quadratic_and_emits_curves(tmp_path: Path) -> None:
    report, events, path = run_fake(tmp_path, FakeNumericLab())
    (settled,) = of_type(events, "question.settled")
    assert settled["hypothesis_id"] == "q1.quadratic" and settled["posterior"] >= 0.9
    curves = of_type(events, "model.curves")
    assert curves and len(curves[-1]["x"]) == 17
    assert set(curves[-1]["families"]) == {"q1.linear", "q1.quadratic", "q1.h0"}
    assert curves[-1]["families"]["q1.quadratic"]["weight"] >= 0.9
    assert len(curves[-1]["points"]) == len(of_type(events, "belief.updated"))
    scored = of_type(events, "experiment.scored")[0]
    assert len(scored["ranking"]) == 17 and scored["chosen"] == "x=2"
    assert [a.name for a in report.artifacts] == ["measurements.csv", "posterior.csv"]
    assert replay_check(path).ok


def test_numeric_random_selection_still_converges(tmp_path: Path) -> None:
    _, events, _ = run_fake(tmp_path, FakeNumericLab(), preset="crux_random_selection")
    outcome = of_type(events, "question.settled") + of_type(events, "question.unresolved")
    assert len(outcome) == 1
    assert of_type(events, "experiment.scored")[0]["chosen"] != "x=2"


def test_same_seed_gives_identical_event_files(tmp_path: Path) -> None:
    def normalised(run_id: str, lab) -> str:
        _, _, path = run_fake(tmp_path, lab, run_id=run_id)
        return path.read_text(encoding="utf-8").replace(run_id, "RUN")  # ts included

    assert normalised("r-same-a", FakeLab()) == normalised("r-same-b", FakeLab())
    assert normalised("r-num-a", FakeNumericLab()) == normalised("r-num-b", FakeNumericLab())


def test_replay_check_passes_and_catches_a_corrupted_posterior(tmp_path: Path) -> None:
    _, events, path = run_fake(tmp_path)
    result = replay_check(path)
    assert result.ok, result.summary()
    assert result.n_checked == len(of_type(events, "belief.updated")) == 4
    assert result.final_beliefs["q2"]["q2.batch"] == pytest.approx(0.942, abs=1e-3)
    assert replay_main([str(path)]) == 0

    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        event = json.loads(line)
        if event["type"] == "belief.updated":
            post = event["payload"]["posterior"]
            first = next(iter(post))
            post[first] += 1e-6
            lines[i] = json.dumps(event)
            break
    bad = tmp_path / "corrupt.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = replay_check(bad)
    assert not result.ok and "differs from the replay" in result.summary()
    assert replay_main([str(bad)]) == 1


def test_replay_check_catches_envelope_breaks(tmp_path: Path) -> None:
    _, _, path = run_fake(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    gap = tmp_path / "gap.jsonl"
    gap.write_text("\n".join(lines[:5] + lines[6:]) + "\n", encoding="utf-8")
    assert "seq" in replay_check(gap).summary()
    cut = tmp_path / "cut.jsonl"
    cut.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert not replay_check(cut).ok
    assert not replay_check(tmp_path / "missing.jsonl").ok


# --- rules on the side paths ------------------------------------------------------------------


def test_invalid_verdict_updates_nothing_and_the_spec_stays_spent(tmp_path: Path) -> None:
    lab = FakeLab()
    deps = fake_deps(lab, reject_experiments=("strand_fraction",))
    _, events, path = run_fake(tmp_path, lab, deps=deps)
    verdicts = {p["experiment_id"]: p["valid"] for p in of_type(events, "referee.verdict")}
    assert verdicts["strand_fraction"] is False
    assert all(p["experiment_id"] != "strand_fraction" for p in of_type(events, "belief.updated"))
    assert started_ids(events).count("strand_fraction") == 1
    assert started_ids(events)[1] == "strand_count_ratio"
    # Without the decisive check q1 stops at 0.89 and nothing informative is left to run.
    (q1,) = [p for p in of_type(events, "question.unresolved") if p["question_id"] == "q1"]
    assert q1["top"] == ["q1.reverse"] and "below 0.05 bits" in q1["reason"]
    assert replay_check(path).ok


def test_no_search_preset_never_searches(tmp_path: Path) -> None:
    _, events, _ = run_fake(tmp_path, preset="no_search")
    assert not of_type(events, "search.done")
    assert all(p["experiment"]["kind"] != "search" for p in of_type(events, "experiment.proposed"))
    settled = {p["question_id"]: p["hypothesis_id"] for p in of_type(events, "question.settled")}
    assert settled == {"q1": "q1.reverse", "q2": "q2.batch"}


def test_role_only_memory_hides_other_roles_records(tmp_path: Path) -> None:
    _, events, _ = run_fake(tmp_path, preset="crux_role_only_memory")
    recalls = of_type(events, "memory.recalled")
    assert recalls and all(set(p["authors"]) <= {p["reader_role"]} for p in recalls)
    _, shared, _ = run_fake(tmp_path, run_id="r-shared")
    designer = [p for p in of_type(shared, "memory.recalled") if p["reader_role"] == "designer"]
    assert any("lab" in p["authors"] for p in designer)


def test_exact_only_predictor_drops_sampled_experiments(tmp_path: Path) -> None:
    _, events, _ = run_fake(tmp_path, predictor="exact_only")
    sources = {p["experiment"]["likelihood_source"] for p in of_type(events, "experiment.proposed")}
    assert sources == {"exact"}
    assert not of_type(events, "search.done")


def test_without_null_option_no_h0_is_proposed(tmp_path: Path) -> None:
    _, events, path = run_fake(tmp_path, null_hypothesis=False)
    assert all(not p["hypothesis"]["is_null"] for p in of_type(events, "hypothesis.proposed"))
    assert len(of_type(events, "question.settled")) == 2
    assert replay_check(path).ok


class ReopeningAgents(FakeAgents):
    async def critic(self, artifacts, contract):
        await self._call("critic")
        self.critic_rounds += 1
        if self.critic_rounds == 1:
            return [
                CriticFinding(
                    id="f-design",
                    artifact="qc_summary.csv",
                    severity="block",
                    rule="design_formula",
                    message="the design formula is not justified by a check",
                    reopens_question="q2",
                )
            ]
        return []


def test_block_finding_can_reopen_a_question(tmp_path: Path) -> None:
    lab = FakeLab()
    deps = fake_deps(lab)
    deps.agents = ReopeningAgents(lab, llm=deps.llm)
    _, events, path = run_fake(tmp_path, lab, deps=deps)
    opened = [p["question"]["id"] for p in of_type(events, "question.opened")]
    assert opened == ["q1", "q2", "q2"]
    # two delivery rounds, each writing both artifacts and then re-emitting them with the verdict
    assert len(of_type(events, "artifact.written")) == 8
    assert of_type(events, "question.unresolved")[-1]["question_id"] == "q2"
    assert replay_check(path).ok


class SlowPredictor(FakePredictor):
    in_flight = 0
    peak = 0

    async def predict(self, exp, hypotheses, context):
        SlowPredictor.in_flight += 1
        SlowPredictor.peak = max(SlowPredictor.peak, SlowPredictor.in_flight)
        await asyncio.sleep(0.01)
        SlowPredictor.in_flight -= 1
        return await super().predict(exp, hypotheses, context)


def test_predictor_calls_run_concurrently(tmp_path: Path) -> None:
    lab = FakeLab()
    deps = fake_deps(lab)
    deps.predictor = SlowPredictor(lab, llm=deps.llm)
    _, events, _ = run_fake(tmp_path, lab, deps=deps)
    assert SlowPredictor.peak >= 3
    assert len(of_type(events, "question.settled")) == 2


def test_cancelling_a_run_finishes_it_as_stopped(tmp_path: Path) -> None:
    async def scenario() -> None:
        lab = FakeLab(latency_s=0.02)
        task = asyncio.create_task(
            run(lab, load_config(runs_dir=str(tmp_path)), fake_deps(lab), run_id="r-stop")
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    events = events_of(tmp_path / "r-stop" / "events.jsonl")
    assert events[-1]["type"] == "run.finished"
    assert events[-1]["payload"]["status"] == "stopped"


class LeakyAgents(FakeAgents):
    async def referee(self, experiment, observation):
        raise PermissionError("Incorrect API key provided: sk-proj-abc123XYZ****wxyz")


def test_run_error_is_last_and_redacts_key_shaped_text(tmp_path: Path) -> None:
    lab = FakeLab()
    deps = fake_deps(lab)
    deps.agents = LeakyAgents(lab, llm=deps.llm)
    with pytest.raises(PermissionError):
        run_fake(tmp_path, lab, deps=deps)
    events = events_of(tmp_path / "r-test" / "events.jsonl")
    assert events[-1]["type"] == "run.error"
    assert events[-1]["payload"]["where"] == "referee:strand_fraction"
    assert "sk-proj" not in events[-1]["payload"]["message"]
    assert "[redacted]" in events[-1]["payload"]["message"]
    assert deps.llm.on_usage is None


def test_usage_hook_is_restored_and_executor_closed(tmp_path: Path) -> None:
    lab = FakeLab()
    deps = fake_deps(lab)
    run_fake(tmp_path, lab, deps=deps)
    assert deps.llm.on_usage is None
    assert deps.executor.opened == deps.executor.closed == ["fake-sandbox-1"]


def test_cli_runs_each_fake_lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for lab in ("fake", "fake-surprise", "fake-numeric"):
        code = cli_main(
            ["--lab", lab, "--runs-dir", str(tmp_path), "--fake-clock", "--run-id", f"r-cli-{lab}"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert f"r-cli-{lab}" in out and "settled:" in out and "events.jsonl" in out
        assert replay_check(tmp_path / f"r-cli-{lab}" / "events.jsonl").ok


class SlowProposers(FakeAgents):
    """Charges a model call, then waits as a slow API would before returning."""

    last_type_before_return = ""

    async def propose(self, brief, evidence):
        result = await super().propose(brief, evidence)
        await asyncio.sleep(0.05)  # the loop is idle: queued ticks should reach the file now
        path = Path(self.lab.run_dir) / "events.jsonl"
        last = path.read_text(encoding="utf-8").splitlines()[-1]
        self.last_type_before_return = json.loads(last)["type"]
        return result


def test_ticks_reach_the_file_while_a_slow_model_call_is_in_flight(tmp_path: Path) -> None:
    lab = FakeLab()
    deps = fake_deps(lab, clock=FakeClock())
    deps.agents = agents = SlowProposers(lab, llm=deps.llm)
    _, events, path = run_fake(tmp_path, lab, deps=deps)
    assert agents.last_type_before_return == "budget.tick"
    ticks = of_type(events, "budget.tick")
    assert ticks[1]["llm_calls"] == 2  # both proposer samples, before question.opened
    types = [e["type"] for e in events]
    assert types.index("question.opened") > types.index("budget.tick")
    assert replay_check(path).ok


def test_unknown_event_types_are_refused(tmp_path: Path) -> None:
    from core.orchestrator import _Run

    async def scenario() -> None:
        lab = FakeLab()
        cfg = load_config(runs_dir=str(tmp_path))
        deps = fake_deps(lab)
        bus = deps.bus_factory("r-bad", tmp_path, clock=deps.clock)
        with pytest.raises(ValueError, match="unknown event type"):
            await _Run(lab, cfg, deps, "r-bad", bus)._emit("run.exploded")

    asyncio.run(scenario())
    assert not (tmp_path / "r-bad" / "events.jsonl").exists()
