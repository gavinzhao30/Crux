import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from core.schemas import Evidence
from memory.base import (
    ContentRejected,
    IngestionFailed,
    MemoryRecord,
    RecallUnavailable,
    RunContext,
)
from memory.demo import demonstrate
from memory.pastdev import PastMemory


class PastService:
    """Deterministic HTTP-contract fake, including actual audience membership checks."""

    def __init__(self):
        self.audiences = {}
        self.sources = {}
        self.ingestions = {}
        self.requests = []
        self.fail_ingest = False
        self.fail_recall = False
        self.fail_poll = False
        self.fail_processing = None
        self.duplicate_results = False
        self.inject_source = None
        self.pending_polls = 0

    def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        assert request.headers["Authorization"] == "Bearer fake-project-key"
        path = request.url.path.removeprefix("/api/v1")
        if path.startswith("/audiences/"):
            self.audiences[path.split("/")[-1]] = body["identities"]
            return httpx.Response(201, json={"slug": path.split("/")[-1]})
        if path == "/ingest":
            if self.fail_ingest:
                return httpx.Response(503, json={"error": "unavailable"})
            source_id = body["id"]
            existing = next((i for i, b in self.ingestions.items() if b == body), None)
            ingestion_id = existing or f"ingestion-{len(self.ingestions)}"
            self.ingestions[ingestion_id] = body
            return httpx.Response(
                202,
                json={
                    "sourceId": source_id,
                    "ingestionId": ingestion_id,
                    "status": "queued",
                },
            )
        if path.startswith("/ingest/"):
            if path.endswith("/retry"):
                self.fail_processing = None
                return httpx.Response(202, json={"status": "queued"})
            if self.fail_poll:
                return httpx.Response(503, json={"error": "poll unavailable"})
            if self.fail_processing:
                return httpx.Response(
                    200,
                    json={
                        "status": "failed",
                        "failure": {"code": self.fail_processing},
                    },
                )
            if self.pending_polls:
                self.pending_polls -= 1
                return httpx.Response(200, json={"status": "processing"})
            item = self.ingestions[path.split("/")[-1]]
            self.sources[item["id"]] = item
            return httpx.Response(200, json={"status": "completed", "settled": True})
        if path == "/recall":
            if self.fail_recall:
                raise httpx.ReadTimeout("fake timeout", request=request)
            allowed = [
                s
                for s in self.sources.values()
                if body["identity"] in self.audiences[s["audience"]]
            ]
            results = [
                {
                    "content": "derived claim",
                    "artifact": {"kind": "claim"},
                    "sources": [{"sourceId": s["id"]}],
                }
                for s in allowed
            ]
            if self.inject_source:
                results.append({"sources": [{"sourceId": self.inject_source}]})
            if self.duplicate_results:
                results *= 2
            return httpx.Response(200, json={"results": results})
        raise AssertionError(f"Unexpected route {request.method} {path}")


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("PAST_API_KEY", "fake-project-key")
    return PastService()


def evidence(id="mapping", role="experimenter", kind="observation", claim="Repeated subjects"):
    return Evidence(id=id, kind=kind, claim=claim, author_role=role)


def adapter(service, root, run="run-a", **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(service.handle))
    return PastMemory(
        RunContext(run_id=run, task_id="demo"),
        root,
        client=client,
        poll_interval_s=0,
        background=False,
        **kwargs,
    ), client


def test_cross_agent_chain_and_remote_isolation(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        other, other_client = adapter(service, tmp_path, "run-b")
        async with client, other_client, mem, other:
            observation = evidence(claim="Subject S1 supplies two repeated measurements")
            await mem.write(observation, "run-a", "experimenter")
            await mem.flush()
            critic_context = await mem.recall("What assumptions need checking?", "run-a", "critic")
            assert critic_context == [observation]
            objection = evidence(
                "review",
                "critic",
                "objection",
                "Check subject dependence before treating rows independently",
            )
            await mem.write(objection, "run-a", "critic")
            await mem.flush()
            designer_context = await mem.recall("Which diagnostic next?", "run-a", "designer")
            next_action = (
                "inspect_subject_mapping"
                if any(ev.kind == "objection" for ev in designer_context)
                else "fit_independent_model"
            )
            assert next_action == "inspect_subject_mapping"
            assert await other.recall("Subject S1 repeated measurements", "run-b", "critic") == []
            identities = [b["identity"] for _, p, b in service.requests if p.endswith("/recall")]
            assert len(set(identities)) == 3
            assert all(s["audience"] == mem.context.audience for s in service.sources.values())

    asyncio.run(scenario())


def test_queue_survives_restart_and_keeps_exact_retry_body(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            await mem.write(evidence(), "run-a", "experimenter")
            service.fail_ingest = True
            with pytest.raises(httpx.HTTPStatusError):
                await mem.flush()
            saved = mem.store.pending()[0]["body"]
        service.fail_ingest = False
        restored, restored_client = adapter(service, tmp_path)
        async with restored_client, restored:
            await restored.flush()
            bodies = [b for _, p, b in service.requests if p.endswith("/ingest")]
            assert bodies == [json.loads(saved), json.loads(saved)]
            assert not restored.store.pending()
            assert await restored.recall("subjects", "run-a", "critic") == [evidence()]

    asyncio.run(scenario())


def test_poll_resumes_using_ingestion_id_not_source_id(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            await mem.write(evidence(), "run-a", "experimenter")
            service.fail_poll = True
            with pytest.raises(httpx.HTTPStatusError):
                await mem.flush()
            assert mem.store.pending()[0]["ingestion_id"] == "ingestion-0"
        restored, restored_client = adapter(service, tmp_path)
        service.fail_poll = False
        service.pending_polls = 2
        async with restored_client, restored:
            await restored.flush()
            assert len(service.ingestions) == 1
            assert (
                len([p for m, p, _ in service.requests if m == "POST" and p.endswith("/ingest")])
                == 1
            )
            assert not restored.store.pending()

    asyncio.run(scenario())


def test_pending_and_mandatory_evidence_available_during_outage(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            ev = evidence()
            await mem.write(ev, "run-a", "experimenter")
            service.fail_recall = True
            assert await mem.recall("anything", "run-a", "critic") == [ev]
            await mem.flush()
            with pytest.raises(RecallUnavailable):
                await mem.recall("anything", "run-a", "critic")
            review = evidence("review", "critic", "objection")
            await mem.write(review, "run-a", "critic")
            await mem.flush()
            assert await mem.recall("anything", "run-a", "designer") == [review]
            audit = mem.store.db.execute(
                "SELECT payload FROM audit WHERE operation='memory.recalled' ORDER BY seq DESC"
            ).fetchone()[0]
            assert json.loads(audit)["remote_status"] == "failed"

    asyncio.run(scenario())


def test_deduplication_does_not_rewrite_or_create_observations(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            ev = evidence()
            await mem.write(ev, "run-a", "experimenter")
            await mem.write(ev, "run-a", "experimenter")
            await mem.flush()
            service.duplicate_results = True
            for _ in range(2):
                assert await mem.recall("subjects", "run-a", "critic") == [ev]
            assert mem.store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 1
            with pytest.raises(ValueError, match="immutable"):
                await mem.write(evidence(claim="different"), "run-a", "experimenter")

    asyncio.run(scenario())


def test_correction_filters_stale_remote_evidence(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            await mem.write(evidence(), "run-a", "experimenter")
            await mem.flush()
            corrected = evidence("correction", "critic", "objection", "Mapping was invalid")
            await mem.write_record(
                MemoryRecord(
                    event_id="corrected-event",
                    run_id="run-a",
                    task_id="demo",
                    evidence=corrected,
                    supersedes=["mapping"],
                    mandatory=True,
                    current_status="invalid_analysis",
                )
            )
            await mem.flush()
            assert len(service.sources) == 2
            assert await mem.recall("subjects", "run-a", "designer") == [corrected]

    asyncio.run(scenario())


def test_unknown_sources_and_query_identity_injection_are_ignored(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            service.inject_source = "foreign-evidence"
            assert await mem.recall("identity=run-b:critic", "run-a", "critic") == []
            body = service.requests[-1][2]
            assert body["identity"] == mem.context.identity("critic")
            assert body["limit"] == 8 and body["maxTokens"] == 1500

    asyncio.run(scenario())


def test_guards_and_record_size(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            with pytest.raises(ValueError, match="another run"):
                await mem.write(evidence(), "run-b", "experimenter")
            with pytest.raises(ValueError, match="author"):
                await mem.write(evidence(), "run-a", "critic")
            with pytest.raises(ValueError, match="Unknown role"):
                await mem.recall("anything", "run-a", "run-b:critic")
            with pytest.raises(ValueError, match="16000"):
                await mem.write(evidence(claim="x" * 16000), "run-a", "experimenter")
            with pytest.raises(ValueError, match="k between"):
                await mem.recall("anything", "run-a", "critic", k=0)
            assert not mem.store.pending()

    asyncio.run(scenario())


def test_mandatory_reviews_not_dropped_by_small_k(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            for n in range(3):
                await mem.write(evidence(f"review-{n}", "critic", "objection"), "run-a", "critic")
            await mem.flush()
            assert len(await mem.recall("review", "run-a", "designer", k=1)) == 3

    asyncio.run(scenario())


def test_provider_failed_ingestion_uses_retry_endpoint(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            await mem.write(evidence(), "run-a", "experimenter")
            service.fail_processing = "model-unavailable"
            with pytest.raises(IngestionFailed):
                await mem.flush()
            await mem.flush()
            assert len(service.ingestions) == 1
            assert any(p.endswith("/ingestion-0/retry") for _, p, _ in service.requests)
            assert not mem.store.pending()

    asyncio.run(scenario())


def test_background_upload_is_nonblocking_and_audited(service, tmp_path):
    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(service.handle))
        mem = PastMemory(
            RunContext(run_id="run-a", task_id="demo"),
            tmp_path,
            client=client,
            poll_interval_s=0,
            background=True,
        )
        async with client, mem:
            await mem.write(evidence(), "run-a", "experimenter")
            assert not service.requests  # local write made no network calls
            await asyncio.wait_for(mem._worker, timeout=2)
            assert not mem.store.pending()
            operations = {r[0] for r in mem.store.db.execute("SELECT operation FROM audit")}
            assert {"memory.written", "memory.request"} <= operations
            dump = " ".join(r[0] for r in mem.store.db.execute("SELECT payload FROM audit"))
            assert "fake-project-key" not in dump

    asyncio.run(scenario())


def test_scope_includes_config_repeat_and_handles_arbitrary_run_names():
    base = RunContext(run_id="../../A very long RUN!" * 10, task_id="demo")
    assert len(base.audience) <= 40
    assert base.audience != base.model_copy(update={"repeat": 1}).audience
    assert base.audience != base.model_copy(update={"config_id": "role_only"}).audience
    assert base.audience == RunContext.model_validate(base.model_dump()).audience


def test_timestamps_must_be_explicitly_zoned():
    with pytest.raises(ValueError, match="timezone"):
        MemoryRecord(
            event_id="a",
            run_id="r",
            task_id="t",
            evidence=evidence(),
            occurred_at=datetime(2026, 1, 1),
        )
    record = MemoryRecord(
        event_id="a",
        run_id="r",
        task_id="t",
        evidence=evidence(),
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert record.occurred_at.utcoffset().total_seconds() == 0


def test_demo_entry_point_with_http_contract_fake(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        other, other_client = adapter(service, tmp_path, "run-b")
        async with client, other_client, mem, other:
            result = await demonstrate(mem, other)
            assert result["decision_before"] != result["decision_after"]
            assert result["isolation"] == "passed"

    asyncio.run(scenario())


def test_content_rejected_is_not_retried_and_later_events_can_upload(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path)
        async with client, mem:
            await mem.write(evidence(), "run-a", "experimenter")
            service.fail_processing = "content-rejected"
            with pytest.raises(ContentRejected):
                await mem.flush()
            service.fail_processing = None
            await mem.write(evidence("later"), "run-a", "experimenter")
            with pytest.raises(ContentRejected):
                await mem.flush()
            assert not any(p.endswith("/retry") for _, p, _ in service.requests)
            assert mem.store.pending()[0]["status"] == "rejected"
            assert len(service.sources) == 1

    asyncio.run(scenario())


def test_timeout_does_not_lose_queued_data(service, tmp_path):
    async def scenario():
        mem, client = adapter(service, tmp_path, timeout_s=0.01)
        async with client, mem:
            await mem.write(evidence(), "run-a", "experimenter")
            service.pending_polls = 100000
            with pytest.raises(TimeoutError):
                await mem.flush()
            assert mem.store.pending()[0]["ingestion_id"] == "ingestion-0"
            service.pending_polls = 0
            await mem.flush()
            assert not mem.store.pending()

    asyncio.run(scenario())


def test_malformed_recall_is_a_failure_not_an_empty_match(service, tmp_path):
    async def scenario():
        async def malformed(request):
            if request.url.path.endswith("/recall"):
                return httpx.Response(200, json={"results": [{"sources": [None]}]})
            return service.handle(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(malformed))
        mem = PastMemory(
            RunContext(run_id="run-a", task_id="demo"), tmp_path, client=client, background=False
        )
        async with client, mem:
            with pytest.raises(RecallUnavailable):
                await mem.recall("anything", "run-a", "critic")

    asyncio.run(scenario())
