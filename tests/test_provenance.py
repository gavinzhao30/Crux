"""DU-45: durable provenance, isolated telemetry and independent trajectory replay."""

import asyncio
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agents.llm import LLM, LLMConfig
from api.server import fold_events
from core.config import load_config
from core.fakes_engine import FakeClock, FakeLab, FakeNumericLab, fake_deps
from core.orchestrator import run, spec_hash
from core.replay import main as replay_main
from core.replay import replay_check
from core.tracing import TraceSession, _Exporters, current_run


def execute(tmp_path, lab=None, run_id="hero"):
    lab = lab or FakeLab()
    cfg = load_config(runs_dir=str(tmp_path))
    deps = fake_deps(lab, clock=FakeClock())
    report = asyncio.run(run(lab, cfg, deps, run_id=run_id))
    folder = tmp_path / run_id
    manifest = json.loads((folder / "manifest.json").read_text())
    events = [
        json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()
    ]
    return report, manifest, events


def test_manifest_is_durable_before_each_diagnostic_and_hashes_actual_artifacts(
    tmp_path,
):
    class CheckingLab(FakeLab):
        async def run(self, exp):
            manifest = json.loads((tmp_path / "hero/manifest.json").read_text())
            record = manifest["diagnostics"][-1]
            assert record["experiment_id"] == exp.id
            assert record["spec_hash"] == spec_hash(exp)
            assert manifest["status"] == "running" and manifest["ended_at"] is None
            return await super().run(exp)

        async def deliver(self, state):
            artifacts = await super().deliver(state)
            for artifact in artifacts:
                artifact.sha256 = "incorrect adapter digest"
            return artifacts

    report, manifest, events = execute(tmp_path, CheckingLab())
    assert manifest["status"] == "ok" and manifest["ended_at"]
    assert len(manifest["git_commit"]) == 40
    assert manifest["grader_commit"]
    assert manifest["prompt_hashes"]
    for path, digest in manifest["prompt_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for artifact in report.artifacts:
        assert (
            artifact.sha256
            == hashlib.sha256(Path(artifact.path).read_bytes()).hexdigest()
        )
    assert len(manifest["diagnostics"]) == sum(
        e["type"] == "experiment.started" for e in events
    )
    assert manifest["usage"]["calls"] > 0
    assert manifest["observed_model_ids"]["proposer"] == "fake-model"
    assert manifest["seeds"]["orchestrator"] == manifest["config"]["seed"]
    assert replay_check(tmp_path / "hero").ok


@pytest.mark.parametrize("cancelled", [False, True])
def test_manifest_finalises_and_flushes_on_error_or_cancel(
    tmp_path, monkeypatch, cancelled
):
    flushed = []
    original = TraceSession.flush

    async def flush(self):
        flushed.append(self.run_id)
        await original(self)

    class Broken(FakeLab):
        async def profile(self):
            raise (
                asyncio.CancelledError()
                if cancelled
                else RuntimeError("profile failed")
            )

    monkeypatch.setattr(TraceSession, "flush", flush)
    with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
        execute(tmp_path, Broken())
    manifest = json.loads((tmp_path / "hero/manifest.json").read_text())
    assert manifest["status"] == ("stopped" if cancelled else "error")
    assert manifest["ended_at"] and flushed == ["hero"]
    assert current_run.get() is None


def test_existing_run_cannot_overwrite_manifest(tmp_path):
    execute(tmp_path)
    before = (tmp_path / "hero/manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        execute(tmp_path)
    assert (tmp_path / "hero/manifest.json").read_bytes() == before


def test_numeric_replay_recomputes_and_rejects_normalised_corruption(tmp_path):
    _, _, events = execute(tmp_path, FakeNumericLab())
    path = tmp_path / "hero/events.jsonl"
    replay = replay_check(path)
    assert replay.ok and replay.n_checked == replay.n_updates > 0
    update = next(e["payload"] for e in events if e["type"] == "belief.updated")
    keys = list(update["posterior"])
    update["posterior"] = {k: 1 / len(keys) for k in keys}
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    assert "numeric posterior differs" in replay_check(path).summary()


def test_legacy_numeric_logs_cannot_claim_verified_replay(tmp_path):
    _, _, events = execute(tmp_path, FakeNumericLab())
    for event in events:
        event["payload"].pop("log_likelihood", None)
    path = tmp_path / "legacy.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    assert "lacks replay log_likelihood" in replay_check(path).summary()


def test_replay_cli_resolves_run_id(tmp_path, monkeypatch):
    execute(tmp_path / "runs")
    monkeypatch.chdir(tmp_path)
    assert replay_main(["hero"]) == 0
    assert replay_main(["runs/hero"]) == 0
    assert replay_main(["runs/hero/events.jsonl"]) == 0


class Item(BaseModel):
    value: int


class Client:
    def __init__(self):
        self.calls = 0
        self.responses = SimpleNamespace(parse=self.parse)

    async def parse(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)
        return SimpleNamespace(
            output_parsed=Item(value=7),
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )


def test_concurrent_calls_have_scoped_tags_and_usage_hooks(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "project-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    client = Client()
    llm = LLM(LLMConfig(cache=False), client=client)

    class Sink:
        def export(self, item):
            pass

        def flush(self):
            pass

    async def task(run_id):
        seen = []
        session = TraceSession(run_id, lambda: "design:q1", exporter=Sink())
        token = current_run.set(session)
        restore = llm.bind_usage(seen.append)
        try:
            assert (
                await llm.structured(
                    "designer", Item, [{"role": "user", "content": run_id}]
                )
            )[0].value == 7
            await session.flush()
            return session, seen
        finally:
            restore()
            current_run.reset(token)

    async def both():
        return await asyncio.gather(task("a"), task("b"))

    results = asyncio.run(both())
    assert client.calls == 2
    for run_id, (session, seen) in zip(("a", "b"), results, strict=True):
        assert len(seen) == 1
        ref = session.refs[0]
        assert (ref["run_id"], ref["role"], ref["step"]) == (
            run_id,
            "designer",
            "design:q1",
        )
        assert ref["url"].endswith(f"/project/project-test/traces/{ref['trace_id']}")
        assert len(ref["prompt_sha256"]) == 64


def test_export_errors_and_bounded_flush_never_duplicate_inference(monkeypatch):
    gate = threading.Event()

    class BrokenExporter:
        def export(self, item):
            gate.wait(2)
            raise RuntimeError("telemetry down")

        def flush(self):
            raise RuntimeError("flush down")

    client = Client()
    llm = LLM(LLMConfig(cache=False), client=client)
    session = TraceSession("r", lambda: "propose", exporter=BrokenExporter())

    async def check():
        token = current_run.set(session)
        try:
            assert (await llm.structured("proposer", Item, []))[0].value == 7
            await session.flush(timeout=0.02)
            assert "FlushTimeout" in session.errors
        finally:
            gate.set()
            current_run.reset(token)

    asyncio.run(check())
    session._thread.join(timeout=2)
    assert client.calls == 1
    assert session.errors.count("RuntimeError") == 2


def test_local_record_failure_keeps_successful_inference(monkeypatch):
    session = TraceSession("r", lambda: "propose")

    def broken(**kwargs):
        raise RuntimeError("local record error")

    monkeypatch.setattr(session, "record", broken)
    client = Client()

    async def check():
        token = current_run.set(session)
        try:
            return await LLM(LLMConfig(cache=False), client=client).structured(
                "proposer", Item, []
            )
        finally:
            current_run.reset(token)

    assert asyncio.run(check())[0].value == 7
    assert client.calls == 1


def test_real_langfuse_sdk_exports_tags_usage_output_and_matching_trace_id():
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider()
    sink = InMemorySpanExporter()
    client = Langfuse(public_key="pk-unit-sdk-du45", secret_key="sk-unit-sdk-du45",
                      tracer_provider=provider, span_exporter=sink)
    exporter = _Exporters.__new__(_Exporters)
    exporter.langfuse = client
    exporter.phoenix = None
    exporter.provider = None
    session = TraceSession("real-sdk-run", lambda: "design:q", exporter=exporter)

    async def check():
        token = current_run.set(session)
        try:
            await LLM(LLMConfig(cache=False), client=Client()).structured("designer", Item, [])
            await session.flush()
        finally:
            current_run.reset(token)

    try:
        asyncio.run(check())
        assert not session.errors
        spans = sink.get_finished_spans()
        assert len(spans) == 1
        assert f"{spans[0].context.trace_id:032x}" == session.refs[0]["trace_id"]
        attrs = spans[0].attributes
        assert attrs["session.id"] == "real-sdk-run"
        assert json.loads(attrs["langfuse.observation.metadata.role"]) == "designer"
        assert json.loads(attrs["langfuse.observation.output"])["value"] == 7
        assert json.loads(attrs["langfuse.observation.usage_details"])["input"] == 3
    finally:
        client.shutdown()


def test_phoenix_receives_same_trace_when_langfuse_fails(monkeypatch):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from phoenix.otel import register

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(ExportTraceServiceRequest.FromString(data))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    provider = register(
        endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
        set_global_tracer_provider=False,
        batch=True,
        verbose=False,
    )

    class Down:
        def start_observation(self, **kwargs):
            raise RuntimeError("Langfuse unavailable")

        def flush(self):
            raise RuntimeError("Langfuse unavailable")

    exporter = _Exporters.__new__(_Exporters)
    exporter.langfuse = Down()
    exporter.provider = provider
    exporter.phoenix = provider.get_tracer("test")
    session = TraceSession("phoenix-run", lambda: "design:q", exporter=exporter)

    async def check():
        token = current_run.set(session)
        try:
            await LLM(LLMConfig(cache=False), client=Client()).structured(
                "designer", Item, []
            )
            await session.flush()
        finally:
            current_run.reset(token)

    try:
        asyncio.run(check())
        spans = [
            s
            for r in received
            for resource in r.resource_spans
            for scope in resource.scope_spans
            for s in scope.spans
        ]
        assert len(spans) == 1
        assert spans[0].trace_id.hex() == session.refs[0]["trace_id"]
        attrs = {a.key: a.value.string_value for a in spans[0].attributes}
        assert json.loads(attrs["metadata"])["run_id"] == "phoenix-run"
        assert attrs["openinference.span.kind"] == "LLM"
    finally:
        provider.shutdown()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_api_snapshot_exposes_decision_trace():
    ref = {
        "trace_id": "trace",
        "run_id": "r",
        "role": "designer",
        "step": "design:q",
        "url": "https://cloud.langfuse.com/project/p/traces/trace",
    }
    snapshot = fold_events(
        [
            {
                "seq": 0,
                "type": "experiment.scored",
                "payload": {
                    "question_id": "q",
                    "ranking": [],
                    "chosen": "exp",
                    "traces": [ref],
                },
            }
        ]
    )
    assert snapshot["decision_traces"]["q"] == [ref]
