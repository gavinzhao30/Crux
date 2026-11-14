import asyncio
import json
from pathlib import Path

import httpx
import pytest

from core.config import RunConfig
from core.schemas import Evidence
from eval.memory_replay import FrozenReplay, run_replay, write_report
from memory.base import MemoryRecord, RunContext
from memory.factory import create_memory
from memory.instrumentation import MemoryTelemetry, format_memory, percentile
from memory.local import LocalMemory
from memory.pastdev import PastMemory

FIXTURE = Path(__file__).resolve().parents[1] / "eval/fixtures/memory_replay.json"


def snapshot():
    return FrozenReplay.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def observation(id="own", role="experimenter", kind="observation"):
    return Evidence(id=id, kind=kind, author_role=role, claim=f"Evidence {id}")


class MemoryService:
    """HTTP-contract fake; reports using it must be labeled mock_http."""

    def __init__(self):
        self.audiences = {}
        self.records = {}
        self.requests = []
        self.failed = False

    def handle(self, request):
        self.requests.append(request.url.path)
        if self.failed:
            return httpx.Response(503, json={"error": "unavailable"})
        body = json.loads(request.content) if request.content else {}
        path = request.url.path.removeprefix("/api/v1")
        if path.startswith("/audiences/"):
            self.audiences[path.split("/")[-1]] = body["identities"]
            return httpx.Response(201, json={"ok": True})
        if path == "/ingest":
            ingestion_id = f"ingest-{len(self.records)}"
            self.records[ingestion_id] = body
            return httpx.Response(202, json={"ingestionId": ingestion_id, "status": "queued"})
        if path.startswith("/ingest/"):
            return httpx.Response(200, json={"status": "completed"})
        if path == "/recall":
            visible = [
                r
                for r in self.records.values()
                if body["identity"] in self.audiences[r["audience"]]
            ]
            return httpx.Response(
                200, json={"results": [{"sources": [{"sourceId": r["id"]}]} for r in visible]}
            )
        raise AssertionError(path)


def test_role_only_filters_cross_role_objections_before_k_and_has_no_outbox(tmp_path, monkeypatch):
    monkeypatch.delenv("PAST_API_KEY", raising=False)

    async def scenario():
        ctx = RunContext(run_id="r1", task_id="demo")
        async with LocalMemory(ctx, tmp_path) as memory:
            await memory.write(observation(), "r1", "experimenter")
            for n in range(8):
                await memory.write(
                    observation(f"critic-{n}", "critic", "objection"), "r1", "critic"
                )
            result = await memory.recall("history", "r1", "experimenter", k=1)
            assert result == [observation()]
            assert not memory.store.pending()
            assert memory.store.db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
            assert not await memory.recall("history", "r1", "designer")
            assert all(a == "experimenter" for a in memory.telemetry.events[-2][1]["authors"])

    asyncio.run(scenario())


def test_local_restart_deduplication_and_copies(tmp_path):
    async def scenario():
        ctx = RunContext(run_id="r1", task_id="demo")
        async with LocalMemory(ctx, tmp_path) as memory:
            await memory.write(observation(), "r1", "experimenter")
        async with LocalMemory(ctx, tmp_path) as restored:
            await restored.write(observation(), "r1", "experimenter")
            result = await restored.recall("history", "r1", "experimenter")
            result[0].claim = "mutated externally"
            assert (await restored.recall("history", "r1", "experimenter")) == [observation()]
            assert restored.store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 1
            changed = observation().model_copy(update={"claim": "different"})
            with pytest.raises(ValueError, match="immutable"):
                await restored.write(changed, "r1", "experimenter")

    asyncio.run(scenario())


def test_canonical_supersession_filters_old_records_without_leaking_review(tmp_path):
    async def scenario():
        ctx = RunContext(run_id="r1", task_id="demo")
        async with LocalMemory(ctx, tmp_path) as memory:
            await memory.write(observation(), "r1", "experimenter")
            await memory.write_record(
                MemoryRecord(
                    event_id="review",
                    run_id="r1",
                    task_id="demo",
                    evidence=observation("review", "critic", "objection"),
                    supersedes=["own"],
                )
            )
            assert await memory.recall("history", "r1", "experimenter") == []
            result = await memory.recall("history", "r1", "critic")
            assert [e.id for e in result] == ["review"]

    asyncio.run(scenario())


def test_local_scope_author_and_role_guards(tmp_path):
    async def scenario():
        async with LocalMemory(RunContext(run_id="r1", task_id="demo"), tmp_path) as memory:
            with pytest.raises(ValueError, match="another run"):
                await memory.write(observation(), "r2", "experimenter")
            with pytest.raises(ValueError, match="author"):
                await memory.write(observation(), "r1", "critic")
            with pytest.raises(ValueError, match="Unknown role"):
                await memory.recall("history", "r1", "intruder")
            with pytest.raises(ValueError, match="k between"):
                await memory.recall("history", "r1", "critic", k=0)
        with pytest.raises(RuntimeError, match="closed"):
            await memory.recall("history", "r1", "critic")

    asyncio.run(scenario())


def test_shared_formatter_exact_counter_and_event_contract(tmp_path):
    async def scenario():
        events = []
        telemetry = MemoryTelemetry(
            "local_role_only",
            token_counter=len,
            token_method="fixture_character_counter",
            on_event=lambda *v: events.append(v),
        )
        async with LocalMemory(
            RunContext(run_id="r1", task_id="demo"), tmp_path, telemetry=telemetry
        ) as memory:
            await memory.write(observation(), "r1", "experimenter")
            result = await memory.recall("history", "r1", "experimenter")
            event, payload = events[-1]
            assert event == "memory.recalled"
            assert {
                "reader_role",
                "query",
                "evidence_ids",
                "authors",
                "backend",
                "latency_ms",
            } <= payload.keys()
            assert payload["prompt_tokens"] == len(format_memory(result))
            assert payload["authors"] == ["experimenter"]
            assert payload["token_method"] == "fixture_character_counter"
            assert telemetry.summary()["write_calls"] == 1

    asyncio.run(scenario())


def test_event_sink_failure_preserves_memory(tmp_path):
    def failing_sink(*_):
        raise RuntimeError("sink unavailable")

    async def scenario():
        telemetry = MemoryTelemetry("local_role_only", on_event=failing_sink)
        async with LocalMemory(
            RunContext(run_id="r1", task_id="demo"), tmp_path, telemetry=telemetry
        ) as memory:
            await memory.write(observation(), "r1", "experimenter")
            assert await memory.recall("history", "r1", "experimenter") == [observation()]
            assert (
                memory.store.db.execute(
                    "SELECT count(*) FROM audit WHERE operation='memory.event_sink_error'"
                ).fetchone()[0]
                == 2
            )

    asyncio.run(scenario())


def test_backend_factory_needs_no_key_for_role_only(tmp_path, monkeypatch):
    monkeypatch.delenv("PAST_API_KEY", raising=False)

    async def scenario():
        ctx = RunContext(run_id="r1", task_id="demo")
        async with create_memory(ctx, RunConfig(memory="local_role_only"), tmp_path) as memory:
            assert isinstance(memory, LocalMemory)
        with pytest.raises(ValueError, match="PAST_API_KEY"):
            create_memory(ctx, RunConfig(memory="pastdev"), tmp_path)

    asyncio.run(scenario())


def test_frozen_replay_rejects_future_labels_and_reversed_cutoffs():
    raw = json.loads(FIXTURE.read_text())
    raw["probes"][0]["decisive_evidence_ids"] = ["independence-objection"]
    with pytest.raises(ValueError, match="future"):
        FrozenReplay.model_validate(raw)
    raw = json.loads(FIXTURE.read_text())
    raw["probes"][2]["history_count"] = 1
    with pytest.raises(ValueError, match="cutoffs"):
        FrozenReplay.model_validate(raw)


def test_no_key_report_has_measured_local_and_unrun_provider_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("PAST_API_KEY", raising=False)

    async def scenario():
        result = await run_replay(snapshot(), tmp_path / "runs")
        local, remote = result["arms"]
        assert local["status"] == "completed"
        assert local["summary"]["decisive_recall_hits"] == 1
        assert local["summary"]["scored_probes"] == 4
        assert local["summary"]["decisions_changed_by_another_role"] == 0
        assert local["summary"]["repeated_rejected_options"] == 2
        assert local["summary"]["prompt_tokens"] > 0
        assert remote["status"] == "skipped_missing_key"
        assert remote["summary"] is None and remote["provider_mode"] == "unrun"
        write_report(result, tmp_path / "report")
        csv = (tmp_path / "report/comparison.csv").read_text()
        assert "skipped_missing_key,unrun" in csv
        assert "estimate_utf8_bytes_div4" in csv

    asyncio.run(scenario())


def test_two_arm_replay_with_explicit_mock_provider_and_no_future_leakage(tmp_path, monkeypatch):
    monkeypatch.setenv("PAST_API_KEY", "fake-project-key")

    async def scenario():
        service = MemoryService()
        async with httpx.AsyncClient(transport=httpx.MockTransport(service.handle)) as client:

            def factory(context, root, telemetry):
                return PastMemory(
                    context,
                    root,
                    telemetry=telemetry,
                    client=client,
                    background=False,
                    poll_interval_s=0,
                )

            frozen = snapshot()
            before = frozen.model_dump_json()
            report = await run_replay(
                frozen, tmp_path, provider_factory=factory, provider_mode="mock_http"
            )
            local, shared = report["arms"]
            assert local["status"] == shared["status"] == "completed"
            assert shared["provider_mode"] == "mock_http"
            assert shared["summary"]["decisive_recall_hits"] == 4
            assert shared["summary"]["decisions_changed_by_another_role"] == 3
            assert "independence-objection" not in shared["decisions"][0]["evidence_ids"]
            assert not local["decisions"][0]["evidence_ids"]
            assert frozen.model_dump_json() == before
            assert local["fixture_sha256"] == shared["fixture_sha256"] == frozen.fingerprint
            assert shared["summary"]["token_method"] == local["summary"]["token_method"]
            assert service.requests
            with pytest.raises(ValueError, match="mock_http"):
                await run_replay(frozen, tmp_path, provider_factory=factory)

    asyncio.run(scenario())


def test_provider_failures_are_failed_not_successful_empty_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("PAST_API_KEY", "fake-project-key")

    async def scenario():
        service = MemoryService()
        service.failed = True
        async with httpx.AsyncClient(transport=httpx.MockTransport(service.handle)) as client:

            def factory(context, root, telemetry):
                return PastMemory(
                    context, root, telemetry=telemetry, client=client, background=False
                )

            report = await run_replay(
                snapshot(), tmp_path, provider_factory=factory, provider_mode="mock_http"
            )
            shared = report["arms"][1]
            assert shared["status"] == "failed"
            assert shared["error_class"] == "HTTPStatusError"
            assert not shared["decisions"]
            assert shared["summary"]["recall_latency_p50_ms"] is None
            write_report(report, tmp_path / "report")
            assert "| - / - |" in (tmp_path / "report/comparison.md").read_text()

    asyncio.run(scenario())


def test_metric_percentiles_and_counter_label_guards():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert percentile([], 0.95) is None
    assert format_memory([]) == ""
    with pytest.raises(ValueError, match="Name the tokenizer"):
        MemoryTelemetry("local_role_only", token_counter=len)
