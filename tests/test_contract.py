import asyncio
import hashlib
import re
from pathlib import Path

import pytest

from core.events import EventBus
from core.fakes import FakeLab, FakeMemory
from core.schemas import (
    Artifact,
    CriticFinding,
    Evidence,
    Experiment,
    Hypothesis,
    Observation,
    Question,
    Report,
    RunEvent,
)
from scripts.make_demo_fixture import generate


def read_events(path):
    return [
        RunEvent.model_validate_json(line) for line in path.read_bytes().splitlines()
    ]


def test_live_delivery_matches_disk_and_cleans_up(tmp_path):
    async def scenario():
        bus = EventBus("live", tmp_path, clock=lambda: 1.0)
        async with bus.subscribe() as first, bus.subscribe() as second:
            await bus.emit("run.started", {})
            await asyncio.gather(
                *(bus.emit("experiment.log", {"line": str(i)}) for i in range(10))
            )
            await bus.emit("run.finished", {})
            recorded = read_events(bus.path)
            assert [e.seq for e in recorded] == list(range(12))
            assert [e async for e in first] == recorded
            assert [e async for e in second] == recorded
        assert not bus._subscribers
        with pytest.raises(ValueError):
            await bus.emit("run.error", {})
        with pytest.raises(FileExistsError):
            await EventBus("live", tmp_path).emit("run.started", {})
        async with bus.subscribe() as late:
            assert [e async for e in late] == []
        other = EventBus("other", tmp_path)
        assert (await other.emit("run.started", {})).seq == 0
        with pytest.raises(ValueError):
            EventBus("../other", tmp_path)

    asyncio.run(scenario())


def test_fake_memory_is_scoped_and_demo_inventory(tmp_path):
    async def scenario():
        lab = FakeLab(tmp_path)
        assert len(lab.questions) == 2
        assert len(lab.experiments) == 4
        for options in lab.hypotheses.values():
            assert len(options) == 3
            assert sum(h.is_null for h in options) == 1
            assert sum(h.prior for h in options) == pytest.approx(1)
        memory = FakeMemory()
        [profile] = await lab.profile()
        for _ in range(2):
            await memory.write(profile, "one", profile.author_role)
        assert await memory.recall("sensor", "one", "critic") == [profile]
        assert await memory.recall("sensor", "two", "critic") == []

    asyncio.run(scenario())


def test_fixtures_reproduce_and_cover_published_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    types = set(
        re.findall(
            r"^\| `([\w.]+)` \|",
            (root / "docs/EVENTS.md").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    asyncio.run(generate(tmp_path))
    covered = set()
    for name, terminal in [("demo", "run.finished"), ("demo-error", "run.error")]:
        path = tmp_path / f"{name}.jsonl"
        assert path.read_bytes() == (root / "web/fixtures" / path.name).read_bytes()
        events = read_events(path)
        assert all(e.run_id == name for e in events)
        assert [e.seq for e in events] == list(range(len(events)))
        assert events[0].type == "run.started"
        assert [e.type for e in events if e.type in {"run.finished", "run.error"}] == [
            terminal
        ]
        assert events[-1].type == terminal
        covered.update(e.type for e in events)
    assert covered == types


def test_demo_references_and_payloads():
    root = Path(__file__).resolve().parents[1]
    questions, hypotheses, experiments, valid = {}, {}, {}, {}
    beliefs = {}
    artifacts = []
    for event in read_events(root / "web/fixtures/demo.jsonl"):
        p = event.payload
        match event.type:
            case "profile.done":
                assert all(Evidence.model_validate(ev) for ev in p["evidence"])
            case "question.opened":
                q = Question.model_validate(p["question"])
                questions[q.id] = q
            case "hypothesis.proposed":
                h = Hypothesis.model_validate(p["hypothesis"])
                assert h.question_id in questions
                hypotheses[h.id] = h
            case "hypothesis.merged":
                assert all(
                    hypotheses[key].question_id == p["question_id"]
                    for key in p["priors"]
                )
                beliefs[p["question_id"]] = p["priors"]
            case "experiment.proposed":
                exp = Experiment.model_validate(p["experiment"])
                assert all(q in questions for q in exp.question_ids)
                experiments[exp.id] = exp
            case "experiment.scored":
                assert p["chosen"] == p["ranking"][0]["experiment_id"]
                assert all(row["experiment_id"] in experiments for row in p["ranking"])
            case "experiment.finished":
                obs = Observation.model_validate(p["observation"])
                assert obs.experiment_id in experiments
                valid[obs.experiment_id] = obs.valid
            case "belief.updated":
                assert valid[p["experiment_id"]]
                assert beliefs[p["question_id"]] == p["prior"]
                assert sum(p["posterior"].values()) == pytest.approx(1)
                beliefs[p["question_id"]] = p["posterior"]
            case "artifact.written":
                artifacts.append(Artifact.model_validate(p["artifact"]))
            case "critic.finding":
                finding = CriticFinding.model_validate(p["finding"])
                assert finding.artifact in [a.name for a in artifacts]
            case "run.finished":
                report = Report.model_validate(p["report"])
                assert report.artifacts == artifacts
                assert (
                    artifacts[0].sha256
                    == hashlib.sha256(report.methods_md.encode()).hexdigest()
                )
    assert set(valid.values()) == {True, False}
