"""Contract tests: schemas, the event envelope and payloads (docs/EVENTS.md), the event bus as
the engine uses it, budget and config, and Brief A's acceptance tests for the scorer stand-in."""

from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from core import _eig_standin as standin
from core import events as E
from core.budget import Budget, BudgetExhausted
from core.config import PRESETS, Config, EngineConfig, RunConfig, load_config
from core.events import EventBus, jsonable, new_run_id
from core.fakes_engine import FakeAgents, FakeClock, FakeLab, FakeNumericLab, fake_deps
from core.orchestrator import run
from core.schemas import (
    Artifact,
    BeliefUpdate,
    CriticFinding,
    Evidence,
    Experiment,
    Hypothesis,
    Observation,
    OutcomeBin,
    Question,
    Report,
    RunEvent,
    RunState,
    Trial,
    bin_for,
    eval_rule,
)

ROOT = Path(__file__).resolve().parents[1]

# --- the contract, encoded independently of the engine (docs/EVENTS.md) ---------------------

PAYLOAD_KEYS: dict[str, set[str]] = {
    "run.started": {"lab", "goal", "config", "started_at"},
    "profile.done": {"evidence"},
    "question.opened": {"question"},
    "hypothesis.proposed": {"hypothesis"},
    "hypothesis.merged": {"question_id", "kept", "merged", "priors"},
    "experiment.proposed": {"experiment"},
    "experiment.scored": {"question_id", "ranking", "chosen"},
    "experiment.started": {"experiment_id", "question_ids", "kind"},
    "experiment.log": {"experiment_id", "stream", "line"},
    "experiment.finished": {"observation"},
    "referee.verdict": {"experiment_id", "valid", "note"},
    "belief.updated": {"question_id", "experiment_id", "observed", "prior", "posterior"},
    "model.curves": {"question_id", "x", "families", "points"},
    "hypothesis.rejected": {"question_id", "hypothesis_id", "evidence_id", "reason"},
    "question.settled": {"question_id", "hypothesis_id", "posterior"},
    "question.robust": {"question_id", "answer_key"},
    "question.unresolved": {"question_id", "reason", "top"},
    "surprise.raised": {"question_id", "experiment_id", "reason", "z_score", "null_mass"},
    "search.done": {"experiment_id", "query", "urls", "blocked", "bin_id"},
    "memory.written": {"evidence_id", "author_role", "backend"},
    "memory.recalled": {
        "reader_role",
        "query",
        "evidence_ids",
        "authors",
        "backend",
        "latency_ms",
    },
    "artifact.written": {"artifact"},
    "critic.finding": {"finding"},
    "lesson.written": {"lesson"},
    "budget.tick": {"cost_usd", "llm_calls", "tokens_in", "tokens_out", "seconds"},
    "run.finished": {"report", "status"},
    "run.error": {"message", "where"},
}
NESTED = {
    "question.opened": ("question", Question),
    "hypothesis.proposed": ("hypothesis", Hypothesis),
    "experiment.proposed": ("experiment", Experiment),
    "experiment.finished": ("observation", Observation),
    "artifact.written": ("artifact", Artifact),
    "critic.finding": ("finding", CriticFinding),
    "lesson.written": ("lesson", Evidence),
    "run.finished": ("report", Report),
}
RANKING_KEYS = {"experiment_id", "kind", "goal", "eig_bits", "est_cost_s", "est_cost_usd", "score"}
NUMBER = (int, float)


def _is_prob_map(d: object) -> bool:
    return isinstance(d, dict) and all(
        isinstance(k, str) and isinstance(v, NUMBER) for k, v in d.items()
    )


def check_payload(etype: str, p: dict) -> None:
    """Raise AssertionError when ``p`` breaks the EVENTS.md payload for ``etype``."""
    optional = {"traces"}
    optional |= {"experiment.proposed": {"replay_prior"},
                 "experiment.started": {"spec_hash"},
                 "belief.updated": {"log_likelihood"}}.get(etype, set())
    assert set(p) - optional == PAYLOAD_KEYS[etype], f"{etype}: keys {sorted(p)}"
    if etype in NESTED:
        key, model = NESTED[etype]
        model.model_validate(p[key])
    if etype == "run.started":
        assert isinstance(p["config"], dict) and "selection" in p["config"]
        assert "role_models" in p["config"]["llm"]
        datetime.fromisoformat(p["started_at"])
    elif etype == "profile.done":
        for ev in p["evidence"]:
            Evidence.model_validate(ev)
    elif etype == "hypothesis.merged":
        assert isinstance(p["kept"], str) and isinstance(p["merged"], list)
        assert _is_prob_map(p["priors"]) and math.isclose(sum(p["priors"].values()), 1.0)
    elif etype == "experiment.scored":
        scores = [r["score"] for r in p["ranking"]]
        assert scores == sorted(scores, reverse=True), "ranking must be best first"
        assert all(set(r) == RANKING_KEYS for r in p["ranking"])
        assert p["chosen"] in {r["experiment_id"] for r in p["ranking"]}
    elif etype == "experiment.started":
        assert isinstance(p["question_ids"], list) and isinstance(p["kind"], str)
    elif etype == "experiment.log":
        assert p["stream"] in ("stdout", "stderr") and len(p["line"]) <= 500
    elif etype == "referee.verdict":
        assert isinstance(p["valid"], bool) and isinstance(p["note"], str)
    elif etype == "belief.updated":
        assert isinstance(p["observed"], (str, int, float))
        assert _is_prob_map(p["prior"]) and _is_prob_map(p["posterior"])
        assert math.isclose(sum(p["posterior"].values()), 1.0, abs_tol=1e-9)
    elif etype == "model.curves":
        n = len(p["x"])
        for fam in p["families"].values():
            assert set(fam) == {"mean", "sd", "weight"}
            assert len(fam["mean"]) == n and len(fam["sd"]) == n
        assert all(set(pt) == {"x", "y"} for pt in p["points"])
    elif etype == "hypothesis.rejected":
        assert p["evidence_id"] is None or isinstance(p["evidence_id"], str)
    elif etype == "question.settled":
        assert isinstance(p["posterior"], NUMBER)
    elif etype == "question.unresolved":
        assert isinstance(p["top"], list) and isinstance(p["reason"], str)
    elif etype == "surprise.raised":
        assert p["z_score"] is None or isinstance(p["z_score"], NUMBER)
        assert p["null_mass"] is None or isinstance(p["null_mass"], NUMBER)
    elif etype == "search.done":
        assert isinstance(p["urls"], list) and isinstance(p["blocked"], list)
        assert p["bin_id"] is None or isinstance(p["bin_id"], str)
    elif etype == "memory.recalled":
        assert len(p["authors"]) == len(p["evidence_ids"])
        assert isinstance(p["latency_ms"], NUMBER)
    elif etype == "budget.tick":
        assert isinstance(p["llm_calls"], int) and isinstance(p["tokens_in"], int)
        assert isinstance(p["cost_usd"], NUMBER) and isinstance(p["seconds"], NUMBER)
    elif etype == "run.finished":
        assert p["status"] in ("ok", "budget_exhausted", "stopped")


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class RobustAgents(FakeAgents):
    """Proposes q2's options with one shared answer key, so q2 is robust from the start."""

    async def propose(self, brief, evidence):
        questions, hyps = await super().propose(brief, evidence)
        return questions, [
            h.model_copy(update={"answer_key": "same"}) if h.question_id == "q2" else h
            for h in hyps
        ]


class BrokenAgents(FakeAgents):
    async def design(self, question, hypotheses, evidence, history):
        raise RuntimeError("designer crashed")


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict]]:
    """Five fake runs that together emit every event type in the contract."""
    runs_dir = tmp_path_factory.mktemp("runs")
    out = {}
    plain = {
        "categorical": FakeLab(),
        "surprise": FakeLab(surprise=True),
        "numeric": FakeNumericLab(),
    }
    for name, lab in plain.items():
        cfg = load_config("default", runs_dir=str(runs_dir), lessons=True)
        report = asyncio.run(run(lab, cfg, fake_deps(lab), run_id=f"r-contract-{name}"))
        out[name] = load_events(runs_dir / report.run_id / "events.jsonl")

    lab = FakeLab()
    deps = fake_deps(lab)
    deps.agents = RobustAgents(lab, llm=deps.llm)
    cfg = load_config(
        "default", runs_dir=str(runs_dir), settle=0.99, max_experiments_per_question=1
    )
    asyncio.run(run(lab, cfg, deps, run_id="r-contract-edge"))
    out["edge"] = load_events(runs_dir / "r-contract-edge" / "events.jsonl")

    lab = FakeLab()
    deps = fake_deps(lab)
    deps.agents = BrokenAgents(lab, llm=deps.llm)
    with pytest.raises(RuntimeError, match="designer crashed"):
        asyncio.run(run(lab, load_config(runs_dir=str(runs_dir)), deps, run_id="r-contract-error"))
    out["error"] = load_events(runs_dir / "r-contract-error" / "events.jsonl")
    return out


# --- schemas ----------------------------------------------------------------------------------


def _samples() -> list:
    bins = [
        OutcomeBin(id="hi", description="high", rule="x > 1"),
        OutcomeBin(id="lo", description="low", rule="x <= 1"),
    ]
    exp = Experiment(
        id="e1",
        question_ids=["q1"],
        kind="compute",
        goal="check x",
        language="python",
        code="print('RESULT: {\"x\": 2}')",
        bins=bins,
        likelihood={"q1.a": {"hi": 0.9, "lo": 0.1}},
        spec_hash="abc",
    )
    obs = Observation(experiment_id="e1", valid=True, value={"x": 2}, bin_id="hi", surprise=0.2)
    hyp = Hypothesis(
        id="q1.a",
        question_id="q1",
        statement="A",
        author="proposer-1",
        rationale="r",
        answer_key="a",
        prior=0.5,
        posterior=0.7,
        rejected_by=["ev-1"],
    )
    art = Artifact(name="t.csv", path="runs/x/artifacts/t.csv", sha256="0" * 64, findings=["ok"])
    return [
        Question(id="q1", text="Which?", kind="categorical", raised_by="proposer-1"),
        hyp,
        bins[0],
        exp,
        obs,
        BeliefUpdate(question_id="q1", experiment_id="e1", prior={"a": 0.5}, posterior={"a": 1.0}),
        Evidence(
            id="ev-1", kind="observation", claim="c", author_role="referee", conditions={"k": 1}
        ),
        art,
        CriticFinding(id="f1", artifact="t.csv", severity="warn", rule="r", message="m"),
        Report(
            run_id="r",
            artifacts=[art],
            methods_md="#",
            rejected=[{"hypothesis": "h"}],
            unresolved=["q2"],
            cost_usd=0.1,
            seconds=2.0,
        ),
        RunEvent(run_id="r", seq=0, ts=1.5, type="run.started", payload={"lab": "x"}),
        Trial(experiment=exp, observation=obs),
        RunState(
            run_id="r",
            run_dir="runs/r",
            lab="fake",
            hypotheses={"q1.a": hyp},
            experiments={"e1": exp},
        ),
    ]


@pytest.mark.parametrize("obj", _samples(), ids=lambda o: type(o).__name__)
def test_schema_round_trip(obj) -> None:
    again = type(obj).model_validate_json(obj.model_dump_json())
    assert again == obj
    assert type(obj).model_validate(json.loads(json.dumps(obj.model_dump(mode="json")))) == obj


def test_schema_defaults_follow_the_erd() -> None:
    q = Question(id="q", text="t", kind="model", raised_by="x")
    assert (q.impact, q.status) == (1.0, "open")
    e = Experiment(id="e", question_ids=[], kind="search", goal="g")
    assert (e.likelihood_source, e.est_cost_s, e.status, e.bins) == ("sampled", 30, "proposed", [])
    h = Hypothesis(
        id="h", question_id="q", statement="s", author="a", rationale="r", prior=0.1, posterior=0.1
    )
    assert (h.is_null, h.status, h.rejected_by, h.answer_key) == (False, "active", [], None)


def test_run_state_helpers() -> None:
    def mk(hid: str, p: float, **kw: object) -> Hypothesis:
        return Hypothesis(
            id=hid,
            question_id="q1",
            statement=hid,
            author="a",
            rationale="r",
            prior=p,
            posterior=p,
            **kw,
        )

    state = RunState(
        run_id="r",
        run_dir="d",
        lab="l",
        questions={"q1": Question(id="q1", text="t", kind="categorical", raised_by="a")},
        hypotheses={
            "a": mk("a", 0.3),
            "b": mk("b", 0.6),
            "h0": mk("h0", 0.1, is_null=True),
            "c": mk("c", 0.0, status="rejected"),
        },
    )
    assert [h.id for h in state.options("q1", active_only=True, include_null=False)] == ["a", "b"]
    assert state.best("q1").id == "b"
    assert state.answers()["q1"].id == "b"


@pytest.mark.parametrize(
    ("rule", "value", "expected"),
    [
        ("frac_sense > 0.85", {"frac_sense": 0.9}, True),
        ("0.35 <= frac_sense <= 0.65", {"frac_sense": 0.7}, False),
        ("'dutp' in text and n > 0", {"text": "a dutp kit", "n": 2}, True),
        ("not ('dutp' in text)", {"text": "nothing"}, True),
        ("abs(value - 1) < 0.1", 1.05, True),
        ("missing > 1", {"x": 1}, False),
        ("__import__('os')", {}, False),
        ("x.__class__", {"x": 1}, False),
        ("x ** 100000", {"x": 2}, False),
        ("1/0 > 0", {}, False),
        ("", {}, False),
    ],
)
def test_eval_rule_is_safe_and_total(rule: str, value: object, expected: bool) -> None:
    assert eval_rule(rule, value) is expected


def test_bin_for_takes_the_first_matching_bin() -> None:
    bins = [
        OutcomeBin(id="a", description="", rule="x > 1"),
        OutcomeBin(id="b", description="", rule="True"),
    ]
    assert bin_for(bins, {"x": 2}) == "a"
    assert bin_for(bins, {"x": 0}) == "b"
    assert bin_for(bins, None) is None


# --- event files from fake runs -------------------------------------------------------------


def test_every_event_matches_the_contract(fake_runs: dict[str, list[dict]]) -> None:
    seen: set[str] = set()
    for name, events in fake_runs.items():
        for e in events:
            assert set(e) == {"run_id", "seq", "ts", "type", "payload"}, name
            assert e["type"] in PAYLOAD_KEYS, f"{name}: unknown type {e['type']}"
            RunEvent.model_validate(e)
            check_payload(e["type"], e["payload"])
            seen.add(e["type"])
    assert seen == set(PAYLOAD_KEYS), f"not covered: {sorted(set(PAYLOAD_KEYS) - seen)}"


@pytest.mark.parametrize("name", ["fake_categorical.jsonl", "fake_numeric.jsonl"])
def test_recorded_demo_runs_match_the_contract(name: str) -> None:
    from core.replay import replay_check

    path = ROOT / "runs" / "demo" / name
    events = load_events(path)
    for e in events:
        assert set(e) == {"run_id", "seq", "ts", "type", "payload"}
        RunEvent.model_validate(e)
        check_payload(e["type"], e["payload"])
    assert events[-1]["type"] == "run.finished" and events[-1]["payload"]["status"] == "ok"
    assert replay_check(path).ok


def test_edge_runs_close_questions_as_robust_and_unresolved(
    fake_runs: dict[str, list[dict]],
) -> None:
    events = fake_runs["edge"]
    robust = [e["payload"] for e in events if e["type"] == "question.robust"]
    unresolved = [e["payload"] for e in events if e["type"] == "question.unresolved"]
    assert robust == [{"question_id": "q2", "answer_key": "same"}]
    assert [u["question_id"] for u in unresolved] == ["q1"]
    assert unresolved[0]["top"][0] == "q1.reverse"
    error = fake_runs["error"][-1]
    assert error["type"] == "run.error"
    assert error["payload"]["where"] == "design:q1"
    assert "designer crashed" in error["payload"]["message"]


def test_envelope_rules(fake_runs: dict[str, list[dict]]) -> None:
    for name, events in fake_runs.items():
        assert [e["seq"] for e in events] == list(range(len(events))), name
        assert len({e["run_id"] for e in events}) == 1
        assert events[0]["type"] == "run.started"
        assert events[-1]["type"] == ("run.error" if name == "error" else "run.finished")
        assert sum(e["type"] in ("run.finished", "run.error") for e in events) == 1
        ts = [e["ts"] for e in events]
        assert ts == sorted(ts)


def test_event_types_constants_match_the_contract() -> None:
    doc = (ROOT / "docs" / "EVENTS.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `([\w.]+)` \|", doc, re.MULTILINE))
    assert set(E.EVENT_TYPES) == set(PAYLOAD_KEYS) == documented
    assert len(E.EVENT_TYPES) == 27
    assert set(E.TERMINAL_EVENTS) == {"run.finished", "run.error"}


def test_invalid_runs_never_update_beliefs(fake_runs: dict[str, list[dict]]) -> None:
    for events in fake_runs.values():
        invalid = {
            e["payload"]["experiment_id"]
            for e in events
            if e["type"] == "referee.verdict" and not e["payload"]["valid"]
        }
        updated = {e["payload"]["experiment_id"] for e in events if e["type"] == "belief.updated"}
        assert not invalid & updated


def test_no_secret_in_any_payload(fake_runs: dict[str, list[dict]]) -> None:
    text = json.dumps(fake_runs)
    assert "sk-" not in text and "api_key" not in text.lower()


# --- event bus ----------------------------------------------------------------------------------


def test_event_bus_takes_jsonable_payloads_and_fans_out(tmp_path: Path) -> None:
    async def scenario() -> list:
        bus = EventBus("r-test", tmp_path, clock=FakeClock(start=10.0, step=0.5))
        question = Question(id="q", text="t", kind="model", raised_by="a")
        async with bus.subscribe() as stream:
            started = {"lab": "x", "goal": "g", "config": {}, "started_at": "2026-01-01"}
            await bus.emit(E.RUN_STARTED, jsonable(started))
            tick = {"cost_usd": float("nan"), "llm_calls": 1, "seconds": np.float64(0.5)}
            await bus.emit(E.BUDGET_TICK, jsonable({**tick, "tokens_in": 0, "tokens_out": 0}))
            await bus.emit(E.QUESTION_OPENED, jsonable({"question": question}))
            await bus.emit(E.RUN_ERROR, jsonable({"message": "m", "where": Path("a/b")}))
            return [e async for e in stream]

    received = asyncio.run(scenario())
    lines = load_events(tmp_path / "r-test" / "events.jsonl")
    assert [e["seq"] for e in lines] == [r.seq for r in received] == [0, 1, 2, 3]
    assert [e["ts"] for e in lines] == [10.0, 10.5, 11.0, 11.5]  # the injected clock
    assert lines[1]["payload"]["cost_usd"] is None  # NaN never reaches the file
    assert lines[1]["payload"]["seconds"] == 0.5
    assert lines[2]["payload"]["question"]["status"] == "open"
    assert lines[3]["payload"]["where"] == "a/b"


def test_new_run_id_format_and_jsonable() -> None:
    assert re.fullmatch(r"r-\d{8}-\d{6}-[0-9a-f]{4}", new_run_id())
    assert new_run_id(datetime(2026, 11, 14, 15, 30, 12)).startswith("r-20261114-153012-")
    assert jsonable({"p": Path("a/b"), "s": {2, 1}, "f": float("inf")}) == {
        "p": "a/b",
        "s": [1, 2],
        "f": None,
    }


# --- budget and config ------------------------------------------------------------------------


def test_budget_counts_calls_cost_and_time() -> None:
    clock = FakeClock(start=0.0, step=0.0)
    b = Budget(max_usd=1.0, max_llm_calls=3, max_seconds=100, clock=clock)
    b.on_usage({"input_tokens": 10, "output_tokens": 5, "cost_usd": None})
    b.on_usage({"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.25})
    b.on_experiment(30.0, 0.5)
    assert b.snapshot() == {
        "cost_usd": 0.75,
        "llm_calls": 2,
        "tokens_in": 11,
        "tokens_out": 6,
        "seconds": 0.0,
    }
    assert b.unpriced_calls == 1
    b.check()
    b.on_usage({"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    with pytest.raises(BudgetExhausted, match="model calls"):
        b.check()
    timed = Budget(10, 10, 5, clock=FakeClock(start=0.0, step=3.0))
    timed.check()
    with pytest.raises(BudgetExhausted, match="s cap"):
        timed.check()


def test_budget_accepts_llm_usage_objects() -> None:
    from agents.llm import Usage

    b = Budget(1.0, 10, 100)
    b.on_usage(
        Usage(
            role="r",
            model="m",
            input_tokens=3,
            output_tokens=4,
            cost_usd=0.01,
            cached=False,
            seconds=0.1,
        )
    )
    assert (b.llm_calls, b.tokens_in, b.tokens_out) == (1, 3, 4)


def test_config_presets_and_overrides() -> None:
    assert set(PRESETS) == {
        "default",
        "crux_random_selection",
        "crux_no_critic",
        "crux_role_only_memory",
        "no_search",
    }
    assert issubclass(EngineConfig, RunConfig) and Config is EngineConfig
    assert set(RunConfig.model_fields) < set(EngineConfig.model_fields)
    assert RunConfig().model_dump() == Config().model_dump(include=set(RunConfig.model_fields))
    cfg = Config()
    assert (cfg.settle, cfg.min_eig_bits, cfg.max_experiments_per_question) == (0.9, 0.05, 4)
    assert (cfg.max_critic_rounds, cfg.null_prior, cfg.eps, cfg.cost_lambda) == (2, 0.15, 0.1, 0.2)
    assert (cfg.n_predictor_samples, cfg.max_usd, cfg.max_llm_calls, cfg.max_seconds) == (
        8,
        2.0,
        40,
        1800,
    )
    assert load_config("crux_random_selection").selection == "random"
    assert load_config("crux_no_critic").critic is False
    assert load_config("crux_role_only_memory").memory == "local_role_only"
    assert (cfg.memory, cfg.lessons, cfg.proposers) == ("pastdev", False, 3)  # RunConfig's
    assert load_config("no_search", seed=7).search is False
    assert load_config("no_search", seed=7).seed == 7
    assert load_config("default", overrides={"settle": 0.95}, seed=2).settle == 0.95
    with pytest.raises(KeyError):
        load_config("nope")
    with pytest.raises(ValueError):
        Config(selection="vote")
    with pytest.raises(ValueError):
        Config(not_a_flag=True)
    with pytest.raises(ValueError):
        Config(memory="role_only")  # RunConfig's value set applies


# --- Brief A acceptance tests (core/_eig_standin.py) ---------------------------------------------


def _renormalised(prior: dict[str, float]) -> dict[str, float]:
    total = sum(prior.values())
    return {k: v / total for k, v in prior.items()}


@pytest.mark.parametrize("n_bins", [2, 3, 4])
def test_brief_a_1_separating_experiment_is_informative(n_bins: int) -> None:
    bins = [f"b{i}" for i in range(n_bins)]
    a = standin.likelihood_from_samples({b: (8 if b == "b0" else 0) for b in bins}, n_bins)
    b_ = standin.likelihood_from_samples({b: (8 if b == "b1" else 0) for b in bins}, n_bins)
    lik = {"a": a, "b": b_, "q.h0": {b: 1 / n_bins for b in bins}}
    prior = _renormalised({"a": 0.5, "b": 0.5, "q.h0": 0.15})
    assert standin.expected_info_gain(prior, lik) > 0.5


def test_brief_a_2_agreeing_experiment_is_not_informative() -> None:
    same = standin.likelihood_from_samples({"x": 8, "y": 0}, 2)
    lik = {"a": same, "b": same, "h0": {"x": 0.5, "y": 0.5}}
    assert standin.expected_info_gain({"a": 0.425, "b": 0.425, "h0": 0.15}, lik) < 0.01
    assert standin.expected_info_gain({"a": 0.5, "b": 0.5}, {"a": same, "b": same}) < 0.01


def test_brief_a_3_update_matches_hand_computation() -> None:
    prior = {"a": 0.6, "b": 0.3, "h0": 0.1}
    lik = {"a": {"x": 0.9, "y": 0.1}, "b": {"x": 0.2, "y": 0.8}, "h0": {"x": 0.5, "y": 0.5}}
    post = standin.update(prior, lik, "x")
    z = 0.6 * 0.9 + 0.3 * 0.2 + 0.1 * 0.5
    expected = {"a": 0.54 / z, "b": 0.06 / z, "h0": 0.05 / z}
    assert post.keys() == expected.keys()
    assert all(math.isclose(post[k], expected[k], rel_tol=1e-12) for k in expected)


def test_brief_a_4_shared_answer_key_is_robust() -> None:
    lik = {"a": {"x": 0.9, "y": 0.1}, "b": {"x": 0.1, "y": 0.9}}
    prior = {"a": 0.5, "b": 0.5}
    assert standin.expected_info_gain(prior, lik, {"a": "k", "b": "k"}) < 1e-9
    assert standin.expected_info_gain(prior, lik) > 0.4
    hyps = [
        Hypothesis(
            id=h,
            question_id="q",
            statement=h,
            author="p",
            rationale="r",
            answer_key="k",
            prior=0.5,
            posterior=0.5,
        )
        for h in ("a", "b")
    ]
    assert standin.robust_answer(hyps) == "k"
    q = Question(id="q", text="t", kind="categorical", raised_by="p")
    exp = Experiment(id="e", question_ids=["q"], kind="compute", goal="g", likelihood=lik)
    assert standin.rank(q, hyps, [exp])[0].eig_bits < 1e-9
    assert standin.robust_answer({"a": "k", "b": "j"}) is None


def test_brief_a_5_unpredicted_bin_is_a_surprise() -> None:
    lik = {"a": {"x": 0.96, "y": 0.04}, "b": {"x": 0.97, "y": 0.03}, "q.h0": {"x": 0.5, "y": 0.5}}
    prior = {"a": 0.425, "b": 0.425, "q.h0": 0.15}
    assert standin.is_surprise(prior, lik, "y")
    assert standin.update(prior, lik, "y")["q.h0"] > prior["q.h0"]
    assert not standin.is_surprise(prior, lik, "x")


def test_brief_a_6_e_value() -> None:
    assert math.isclose(standin.e_value(1e-3), 15.811, rel_tol=1e-4)
    assert standin.e_value(1.0) == 0.5


def test_standin_score_rank_and_samples() -> None:
    assert math.isclose(standin.score(1.0, 60), 1 / 1.2)
    assert standin.likelihood_from_samples([8, 0], 2) == pytest.approx([0.9, 0.1])
    with pytest.raises(ValueError):
        standin.likelihood_from_samples({"x": 8}, 2)
    q = Question(id="q", text="t", kind="categorical", impact=0.5, raised_by="p")
    hyps = [
        Hypothesis(
            id=h, question_id="q", statement=h, author="p", rationale="r", prior=0.5, posterior=0.5
        )
        for h in ("a", "b")
    ]
    good = Experiment(
        id="good",
        question_ids=["q"],
        kind="compute",
        goal="g",
        est_cost_s=60,
        likelihood={"a": {"x": 0.9, "y": 0.1}, "b": {"x": 0.1, "y": 0.9}},
    )
    flat = Experiment(
        id="flat",
        question_ids=["q"],
        kind="compute",
        goal="g",
        likelihood={"a": {"x": 0.5, "y": 0.5}, "b": {"x": 0.5, "y": 0.5}},
    )
    ranked = standin.rank(q, hyps, [flat, good])
    assert [e.id for e in ranked] == ["good", "flat"]
    assert math.isclose(ranked[0].score, 0.5 * ranked[0].eig_bits / 1.2)
