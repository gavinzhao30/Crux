import asyncio
import json
from pathlib import Path

import httpx
import pytest

from core.config import RunConfig
from core.schemas import Evidence
from eval.lesson_comparison import (
    CaseResult,
    HeldOutCase,
    RunConditions,
    _FixtureMemory,
    compare,
    write_report,
)
from memory.lessons import (
    Applicability,
    DevelopmentEvidence,
    FrozenLessons,
    LessonMemory,
    ReviewedLesson,
)

FIXTURE = Path(__file__).resolve().parents[1] / "eval/fixtures/reviewed_lessons.json"


def corpus():
    return FrozenLessons.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def cases():
    return [
        HeldOutCase(
            case_id=name,
            split="test",
            input_hash="fixture-input",
            public_context={"analysis": "de_from_counts"},
            lesson_query="preflight checks",
        )
        for name in corpus().held_out_case_ids
    ]


def conditions():
    return RunConditions(
        model_ids={"designer": "fixture-model"},
        sandbox_image_id="fixture-image",
        budget_usd=1,
        seed=23,
    )


class RecordingMemory(_FixtureMemory):
    instances = []

    def __init__(self, context, *_args, **_kwargs):
        super().__init__()
        self.context = context
        self.calls = []
        self.instances.append(self)

    async def recall(self, query, run_id, role, k=8):
        self.calls.append((query, run_id, role))
        return await super().recall(query, run_id, role, k)


@pytest.mark.parametrize("split", ["test", "train", "unknown"])
def test_only_development_sources_can_form_lessons(split):
    raw = corpus().lessons[0].sources[0].model_dump()
    raw["split"] = split
    with pytest.raises(ValueError):
        DevelopmentEvidence.model_validate(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("approved", False),
        ("reviewed_by", " "),
        ("diagnostic", " "),
        ("mistake", "grade.json says 0.9"),
    ],
)
def test_unreviewed_or_answer_bearing_lessons_are_rejected(field, value):
    raw = corpus().lessons[0].model_dump()
    raw[field] = value
    with pytest.raises(ValueError):
        ReviewedLesson.model_validate(raw)


def test_grade_fields_and_grading_sources_are_not_in_the_contract():
    raw = corpus().lessons[0].model_dump()
    raw["official_score"] = 0.9
    with pytest.raises(ValueError):
        ReviewedLesson.model_validate(raw)
    raw = corpus().lessons[0].sources[0].model_dump()
    raw["kind"] = "grade"
    with pytest.raises(ValueError):
        DevelopmentEvidence.model_validate(raw)


def test_held_out_case_cannot_contribute_to_its_own_corpus():
    raw = corpus().model_dump(mode="json")
    raw["lessons"][0]["sources"][0]["case_id"] = raw["held_out_case_ids"][0]
    with pytest.raises(ValueError, match="Held-out cases"):
        FrozenLessons.model_validate(raw)


def test_review_must_precede_freeze():
    raw = corpus().model_dump(mode="json")
    raw["lessons"][0]["reviewed_at"] = "2026-11-15T20:00:00Z"
    with pytest.raises(ValueError, match="before the freeze"):
        FrozenLessons.model_validate(raw)


def test_freeze_reload_and_tamper_detection(tmp_path):
    frozen = corpus()
    path = frozen.save(tmp_path)
    assert frozen.save(tmp_path) == path
    assert FrozenLessons.load(path) == frozen
    raw = json.loads(path.read_text())
    raw["manifest"]["lessons"][0]["repair"] = "Changed after comparison"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        FrozenLessons.load(path)
    with pytest.raises(ValueError, match="has changed"):
        frozen.save(tmp_path)


def test_applicability_uses_exact_types_and_exclusions():
    assert Applicability(key="paired", value=True).matches({"paired": True})
    assert not Applicability(key="paired", value=True).matches({"paired": 1})
    lesson = corpus().lessons[0]
    assert lesson.applies({"analysis": "de_from_counts"})
    assert not lesson.applies({"analysis": "other"})
    excluded = lesson.model_copy(update={"excludes": (Applicability(key="unsafe", value=True),)})
    assert not excluded.applies({"analysis": "de_from_counts", "unsafe": True})


def test_lessons_off_constructs_no_provider_or_requires_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PAST_API_KEY", raising=False)

    def forbidden_factory(*_args, **_kwargs):
        raise AssertionError("No provider should be created")

    async def scenario():
        async with LessonMemory(
            corpus(), RunConfig(lessons=False), tmp_path, memory_factory=forbidden_factory
        ) as memory:
            await memory.prepare()
            assert (
                await memory.recall(
                    "checks", case_id=cases()[0].case_id, role="designer", context={}
                )
                == []
            )
            assert memory.memory is None
        assert not list(tmp_path.iterdir())

    asyncio.run(scenario())


def test_indexing_barrier_scope_and_applicability(tmp_path):
    async def scenario():
        async with LessonMemory(
            corpus(),
            RunConfig(lessons=True),
            tmp_path,
            allow_synthetic=True,
            memory_factory=RecordingMemory,
        ) as memory:
            with pytest.raises(RuntimeError, match="prepare"):
                await memory.recall(
                    "checks", case_id=cases()[0].case_id, role="designer", context={}
                )
            await memory.prepare()
            found = await memory.recall(
                "checks",
                case_id=cases()[0].case_id,
                role="designer",
                context={"analysis": "de_from_counts"},
            )
            assert [ev.kind for ev in found] == ["lesson"]
            assert memory.memory.context.run_id.startswith("lessons-")
            assert memory.memory.context.run_id != corpus().lessons[0].sources[0].run_id
            assert not await memory.recall(
                "checks", case_id=cases()[0].case_id, role="designer", context={"analysis": "other"}
            )
            with pytest.raises(ValueError, match="cohort"):
                await memory.recall("checks", case_id="unknown", role="designer", context={})
            with pytest.raises(ValueError, match="proposers/designers"):
                await memory.recall("checks", case_id=cases()[0].case_id, role="critic", context={})

    asyncio.run(scenario())


def test_remote_results_must_resolve_to_frozen_canonical_lessons(tmp_path):
    class AlteredRemote(_FixtureMemory):
        async def recall(self, *_args, **_kwargs):
            known = next(iter(self.records.values())).model_copy(update={"claim": "REMOTE TAMPER"})
            return [
                Evidence(id="foreign", kind="lesson", author_role="reporter", claim="foreign"),
                known,
                known,
            ]

    async def scenario():
        async with LessonMemory(
            corpus(),
            RunConfig(lessons=True),
            tmp_path,
            allow_synthetic=True,
            memory_factory=AlteredRemote,
        ) as memory:
            await memory.prepare()
            result = await memory.recall(
                "checks",
                case_id=cases()[0].case_id,
                role="designer",
                context={"analysis": "de_from_counts"},
            )
            assert len(result) == 1 and "REMOTE TAMPER" not in result[0].claim
            assert result[0] == corpus().evidence(corpus().lessons[0])

    asyncio.run(scenario())


def test_live_adapter_contract_with_http_mock_does_not_upload_test_case_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("PAST_API_KEY", "fake-key")

    async def scenario():
        records = []

        def handle(request):
            body = json.loads(request.content) if request.content else {}
            if "/audiences/" in request.url.path:
                assert len(body["identities"]) == 6
                return httpx.Response(201, json={"ok": True})
            if request.url.path.endswith("/ingest"):
                assert "synthetic-heldout" not in body["content"]
                records.append(body)
                return httpx.Response(202, json={"ingestionId": "ingest-1", "status": "queued"})
            if "/ingest/" in request.url.path:
                return httpx.Response(200, json={"status": "completed"})
            return httpx.Response(
                200, json={"results": [{"sources": [{"sourceId": r["id"]}]} for r in records]}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            async with LessonMemory(
                corpus(),
                RunConfig(lessons=True),
                tmp_path,
                allow_synthetic=True,
                client=client,
                background=False,
            ) as memory:
                await memory.prepare()
                for role in ("proposer", "designer"):
                    assert await memory.recall(
                        "checks",
                        case_id=cases()[0].case_id,
                        role=role,
                        context={"analysis": "de_from_counts"},
                    )
                assert len(records) == 1
                assert (
                    memory.memory.store.db.execute(
                        "SELECT count(*) FROM audit WHERE operation='lesson.written'"
                    ).fetchone()[0]
                    == 1
                )

    asyncio.run(scenario())


def test_pair_protocol_keeps_models_inputs_seeds_and_budgets_fixed(tmp_path):
    async def scenario():
        calls = []

        async def runner(case, config, fixed, contexts):
            calls.append((case.model_dump(), config.model_dump(), fixed.model_dump(), contexts))
            assert bool(contexts["designer"]) == config.lessons
            return CaseResult(
                status="ok",
                result_kind="synthetic",
                cost_usd=0.1,
                diagnostics_before_fault=2 if config.lessons else 4,
            )

        result = await compare(
            corpus(),
            cases(),
            conditions(),
            runner,
            tmp_path,
            allow_synthetic=True,
            live_memory=False,
            memory_factory=_FixtureMemory,
        )
        assert len(calls) == 4 and result["memory_service"] == "fixture_memory"
        for case_id in corpus().held_out_case_ids:
            pair = [c for c in calls if c[0]["case_id"] == case_id]
            assert pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
            left, right = dict(pair[0][1]), dict(pair[1][1])
            left.pop("lessons")
            right.pop("lessons")
            assert left == right
        assert all(p["official_score_delta_on_minus_off"] is None for p in result["paired_deltas"])
        assert all(
            p["diagnostics_before_fault_delta_on_minus_off"] == -2 for p in result["paired_deltas"]
        )
        write_report(result, tmp_path / "report")
        assert "not measured" in (tmp_path / "report/comparison.md").read_text()

    asyncio.run(scenario())


def test_changed_runner_settings_are_not_a_valid_comparison(tmp_path):
    async def scenario():
        async def mutating_runner(_case, _config, fixed, _contexts):
            fixed.model_ids["designer"] = "different-model"
            return CaseResult(status="ok", result_kind="synthetic")

        result = await compare(
            corpus(),
            cases(),
            conditions(),
            mutating_runner,
            tmp_path,
            allow_synthetic=True,
            live_memory=False,
            memory_factory=_FixtureMemory,
        )
        assert all(
            r["status"] == "failed" and r["error_class"] == "ValueError" for r in result["rows"]
        )
        assert conditions().model_ids["designer"] == "fixture-model"

    asyncio.run(scenario())


def test_provider_preparation_failure_does_not_masquerade_as_empty_lessons(tmp_path):
    class FailedMemory(_FixtureMemory):
        async def flush(self):
            raise TimeoutError("unavailable")

    async def scenario():
        async def runner(_case, _config, _fixed, _contexts):
            return CaseResult(status="ok", result_kind="synthetic")

        result = await compare(
            corpus(),
            cases(),
            conditions(),
            runner,
            tmp_path,
            allow_synthetic=True,
            live_memory=False,
            memory_factory=FailedMemory,
        )
        assert result["corpus_preparation_error_class"] == "TimeoutError"
        assert all(r["status"] == ("failed" if r["lessons"] else "ok") for r in result["rows"])
        assert all(p["official_score_delta_on_minus_off"] is None for p in result["paired_deltas"])

    asyncio.run(scenario())


def test_synthetic_corpus_is_not_implicitly_enabled(tmp_path):
    with pytest.raises(ValueError, match="allow_synthetic"):
        LessonMemory(corpus(), RunConfig(lessons=True), tmp_path)


def test_comparison_cannot_silently_drop_a_frozen_case(tmp_path):
    async def scenario():
        async def runner(*_args):
            raise AssertionError("Invalid cohorts must fail before invoking a runner")

        with pytest.raises(ValueError, match="exactly match"):
            await compare(
                corpus(),
                cases()[:1],
                conditions(),
                runner,
                tmp_path,
                allow_synthetic=True,
                live_memory=False,
                memory_factory=_FixtureMemory,
            )

    asyncio.run(scenario())
