import asyncio
import json
from types import SimpleNamespace

import pytest

from core.config import load_config
from core.events import EventBus
from core.manifest import RunManifest
from core.tracing import TraceSession, current_run
from scripts import run_proposers


def test_manifest_retries_only_transient_windows_replacement_errors(tmp_path, monkeypatch):
    from core import manifest as module

    manifest = RunManifest(tmp_path / "run", "run", load_config(),
        SimpleNamespace(name="boxing"), SimpleNamespace(executor=None, clock=lambda: 1000))
    replace = module.os.replace
    attempts = []

    def locked_once(source, target):
        attempts.append(target)
        if len(attempts) == 1:
            error = PermissionError("sharing violation")
            error.winerror = 32
            raise error
        replace(source, target)

    monkeypatch.setattr(module.os, "replace", locked_once)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    manifest.write()
    assert len(attempts) == 2
    assert json.loads(manifest.path.read_text())["run_id"] == "run"

    def denied(source, target):
        raise PermissionError("permanent denial")

    monkeypatch.setattr(module.os, "replace", denied)
    with pytest.raises(PermissionError, match="permanent denial"):
        manifest.write()
    assert not list(manifest.path.parent.glob(".manifest-*"))


def test_late_open_records_resolved_execution_provenance(tmp_path):
    cfg = load_config()
    executor = SimpleNamespace(inputs=SimpleNamespace(files={"counts": "a" * 64}, bundle_id="bundle"),
                               image_ids={})
    lab = SimpleNamespace(name="bixbench3", executor=executor, sandbox_id=None)
    deps = SimpleNamespace(executor=None, llm=None, clock=lambda: 1000)
    manifest = RunManifest(tmp_path / "run", "run", cfg, lab, deps)
    manifest.start()
    lab.sandbox_id = "sandbox-1"
    executor.image_ids["sandbox-1"] = "image-1"
    manifest.refresh_execution(lab, deps)
    recorded = json.loads(manifest.path.read_text())
    assert recorded["sandbox_image_id"] == "image-1"
    assert recorded["sandbox_id"] == "sandbox-1"
    assert recorded["input_hashes"] == {"counts": "a" * 64}
    from eval.grade_bixbench3 import GRADER_COMMIT

    assert recorded["grader_commit"] == GRADER_COMMIT


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_standalone_session_flushes_and_resets(tmp_path, monkeypatch, failure):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"brief": {"goal": "test"}, "profile": []}))
    flushed = []
    original = TraceSession.flush

    async def flush(self):
        flushed.append(self.run_id)
        await original(self)

    async def propose(*args, **kwargs):
        session = current_run.get()
        assert session is not None and session.run_id == "standalone"
        session.record(role="proposer", model="test", prompt_sha256="a" * 64,
                       messages=[], output={}, usage={})
        if failure:
            raise failure()
        return SimpleNamespace(model_dump_json=lambda **kw: "{}")

    monkeypatch.setattr(run_proposers, "EventBus", lambda rid: EventBus(rid, tmp_path))
    monkeypatch.setattr(run_proposers, "LLM", lambda cfg: None)
    monkeypatch.setattr(run_proposers, "propose", propose)
    monkeypatch.setattr(TraceSession, "flush", flush)
    args = SimpleNamespace(input=source, run_id="standalone")
    if failure:
        with pytest.raises(failure):
            asyncio.run(run_proposers.run(args))
    else:
        asyncio.run(run_proposers.run(args))
    assert flushed == ["standalone"]
    assert current_run.get() is None
    assert len(json.loads((tmp_path / "standalone/traces.json").read_text())["traces"]) == 1
