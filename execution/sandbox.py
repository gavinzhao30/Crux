"""Run Task 9 analyses in a network-blocked Modal sandbox."""

import asyncio
import json
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from time import monotonic
from uuid import uuid4

import modal

from execution.image import task9_image
from execution.staging import MANIFEST, VOLUME_NAME, Task9Inputs

_CONDA_BIN = "/opt/conda/bin"
_PYTHON = f"{_CONDA_BIN}/python"
_EXEC_ENV = {"PATH": f"{_CONDA_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
_COMMANDS = {
    "python": (_PYTHON, "-u"),
    "r": (f"{_CONDA_BIN}/Rscript", "--vanilla"),
    "bash": ("/bin/bash",),
}
_REFERENCE_PROBE = "https://storage.googleapis.com/bixbench3-ground-truth/"
_VERIFY_INPUTS = """
import hashlib, json, pathlib, sys
root = pathlib.Path('/data')
expected = json.loads(sys.argv[1])
assert {p.name for p in root.iterdir()} == set(expected) | {'input_manifest.json'}
for name, digest in expected.items():
    path = root / name
    assert path.is_file() and not path.is_symlink(), name
    with path.open('rb') as file:
        assert hashlib.file_digest(file, 'sha256').hexdigest() == digest, name
"""
_LIST_OUTPUTS = """
import json, pathlib
root = pathlib.Path('/outputs')
files = {}
if root.is_dir() and not root.is_symlink():
    for path in root.rglob('*'):
        if path.is_file() and path.resolve() == path:
            files[path.relative_to(root).as_posix()] = path.stat().st_size
        if len(files) > 200:
            raise ValueError('Too many output files')
print(json.dumps(files))
"""


@dataclass
class _Run:
    sandbox: modal.Sandbox
    directory: Path
    run_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closing: bool = False


class _Output:
    """Keep a bounded preview while processing every RESULT marker."""

    def __init__(self):
        self.head: list[str] = []
        self.tail: deque[str] = deque(maxlen=12)
        self.results = 0
        self.result: dict | None = None

    def add(self, line: str, *, parse_result: bool) -> str:
        if parse_result and line.startswith("RESULT:"):
            self.results += 1
            if self.results == 1 and len(line) <= 256_000:
                try:
                    payload = json.loads(line.removeprefix("RESULT:"))
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    self.result = payload
        preview = line.rstrip("\n")
        if len(preview) > 2000:
            preview = f"{preview[:1000]} … {preview[-1000:]}"
        if len(self.head) < 12:
            self.head.append(preview)
        else:
            self.tail.append(preview)
        return preview

    def text(self) -> str:
        return "\n".join([*self.head, *self.tail])


MAX_DIAGNOSTIC_TIMEOUT_S = 120


class ModalExecutor:
    def __init__(
        self,
        inputs: Task9Inputs,
        runs_dir: Path = Path("runs"),
        *,
        app_name: str = "crux-task9",
        minimal_image: bool = False,
        diagnostic_timeout_s: int = MAX_DIAGNOSTIC_TIMEOUT_S,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        if not 1 <= diagnostic_timeout_s <= MAX_DIAGNOSTIC_TIMEOUT_S:
            raise ValueError("Diagnostic timeout must be between 1 and 120 seconds")
        self.diagnostic_timeout_s = diagnostic_timeout_s
        self.inputs = inputs
        self.runs_dir = runs_dir
        self.app_name = app_name
        self.minimal_image = minimal_image
        self.on_event = on_event
        self._runs: dict[str, _Run] = {}
        self._active_run_ids: set[str] = set()

    def _event(self, run: _Run, event_type: str, payload: dict) -> None:
        event = {"type": event_type, "run_id": run.run_id, **payload}
        with (run.directory / "execution.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(event) + "\n")
        if self.on_event is not None:
            self.on_event(event_type, {"run_id": run.run_id, **payload})

    async def open(self, lab_name: str, run_id: str) -> str:
        if lab_name != "bixbench3" or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("Expected the bixbench3 lab and a simple run ID")
        if run_id in self._active_run_ids:
            raise ValueError(f"Sandbox already open for {run_id}")
        self._active_run_ids.add(run_id)
        sandbox = None
        started = monotonic()
        try:
            directory = self.runs_dir / run_id
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / MANIFEST).exists():
                raise ValueError(f"Run ID {run_id} already has a staged input manifest")
            app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
            volume = modal.Volume.from_name(VOLUME_NAME)
            sandbox = await modal.Sandbox.create.aio(
                app=app,
                image=task9_image(minimal=self.minimal_image),
                volumes={
                    "/data": volume.with_mount_options(
                        read_only=True, sub_path=f"/{self.inputs.bundle_id}"
                    )
                },
                block_network=True,
                cpu=4,
                memory=16384,
                timeout=3600,
                idle_timeout=600,
            )
            check = await sandbox.exec.aio(
                _PYTHON, "-c", _VERIFY_INPUTS, json.dumps(self.inputs.files), timeout=120
            )
            if await check.wait.aio() != 0:
                details = (await check.stderr.read.aio()).strip()
                raise RuntimeError(f"Task 9 staged inputs failed verification: {details}")
            (directory / MANIFEST).write_text(self.inputs.json(), encoding="utf-8")
            run = _Run(sandbox, directory, run_id)
            self._event(run, "sandbox.ready", {"startup_seconds": monotonic() - started})
            self._runs[sandbox.object_id] = run
            return sandbox.object_id
        except BaseException:
            try:
                if sandbox is not None:
                    await sandbox.terminate.aio()
            finally:
                self._active_run_ids.discard(run_id)
            raise

    async def exec(self, sandbox_id: str, language: str, code: str, timeout_s: int = 120) -> dict:
        return await self.run_code(sandbox_id, language, code, timeout_s=timeout_s)

    async def run_code(
        self,
        sandbox_id: str,
        language: str,
        code: str,
        *,
        timeout_s: int = 120,
        experiment_id: str | None = None,
        require_result: bool = True,
    ) -> dict:
        if language not in _COMMANDS or timeout_s <= 0:
            raise ValueError("Expected python, r or bash and a positive timeout")
        run = self._runs[sandbox_id]
        started = monotonic()
        stdout, stderr = _Output(), _Output()
        error = None
        returncode = None
        command_id = uuid4().hex
        log_ref = f"execution.jsonl#{command_id}"

        async def drain(stream, name, output):
            async for line in stream:
                preview = output.add(line, parse_result=name == "stdout")
                self._event(
                    run,
                    "experiment.log",
                    {
                        "line": preview,
                        "stream": name,
                        "command_id": command_id,
                        "experiment_id": experiment_id,
                    },
                )

        async with run.lock:
            if run.closing:
                raise RuntimeError("Cannot execute in a closing sandbox")
            filename = f"/work/{command_id}.{language}"
            try:
                await run.sandbox.filesystem.write_text.aio(code, filename)
                process = await run.sandbox.exec.aio(
                    *_COMMANDS[language],
                    filename,
                    timeout=timeout_s,
                    workdir="/work",
                    bufsize=1,
                    env=_EXEC_ENV,
                )
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(drain(process.stdout, "stdout", stdout))
                    tasks.create_task(drain(process.stderr, "stderr", stderr))
                    wait = tasks.create_task(process.wait.aio())
                returncode = wait.result()
            except (modal.exception.Error, TimeoutError, ExceptionGroup) as exc:
                error = f"Command execution failed: {type(exc).__name__}"
            if error is None and returncode != 0:
                error = f"Command exited with status {returncode}"
            if error is None and require_result and (stdout.results != 1 or stdout.result is None):
                error = "Expected exactly one RESULT: line containing a JSON object"
            if error is not None:
                self._event(
                    run,
                    "execution.failed",
                    {
                        "command_id": command_id,
                        "experiment_id": experiment_id,
                        "error": error,
                    },
                )
        return {
            "valid": error is None,
            "result": stdout.result if error is None else None,
            "stdout": stdout.text(),
            "stderr": stderr.text(),
            "returncode": returncode,
            "seconds": monotonic() - started,
            "error": error,
            "log_ref": log_ref,
        }

    async def _copy_outputs(self, run: _Run) -> list[dict]:
        process = await run.sandbox.exec.aio(_PYTHON, "-c", _LIST_OUTPUTS, timeout=30)
        raw = await process.stdout.read.aio()
        if await process.wait.aio() != 0:
            raise RuntimeError("Sandbox output listing failed")
        paths = json.loads(raw)
        if len(paths) > 200 or sum(paths.values()) > 128 * 1024**2:
            raise ValueError("Sandbox output exceeds the export limit")
        destination = run.directory / "artifacts"
        artifacts = []
        for name, size in sorted(paths.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Invalid sandbox output path")
            if not 0 <= size <= 32 * 1024**2:
                raise ValueError("Invalid sandbox output size")
            data = await run.sandbox.filesystem.read_bytes.aio(f"/outputs/{relative}")
            if len(data) != size:
                raise ValueError("Sandbox output changed during export")
            path = destination / relative
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Output destination escapes the artifact directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{uuid4().hex}.tmp")
            try:
                temp.write_bytes(data)
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
            artifacts.append({"name": name, "path": str(path), "sha256": sha256(data).hexdigest()})
        return artifacts

    async def collect_outputs(self, sandbox_id: str) -> list[dict]:
        run = self._runs[sandbox_id]
        async with run.lock:
            if run.closing:
                raise RuntimeError("Cannot export from a closing sandbox")
            return await self._copy_outputs(run)

    async def verify_isolation(self, sandbox_id: str) -> dict:
        code = f"""
import errno, json, pathlib, urllib.error, urllib.request
read_only = False
try:
    pathlib.Path('/data/.write-probe').touch(exist_ok=False)
except OSError as exc:
    read_only = exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
print('Probe outbound connectivity for ' + {_REFERENCE_PROBE!r}, flush=True)
blocked = False
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
    with opener.open({_REFERENCE_PROBE!r}, timeout=3):
        pass
except urllib.error.HTTPError:
    pass  # An HTTP response proves the request escaped.
except (urllib.error.URLError, TimeoutError):
    blocked = True
print('RESULT: ' + json.dumps({{'read_only': read_only, 'network_blocked': blocked}}))
"""
        result = await self.exec(sandbox_id, "python", code, timeout_s=10)
        data = result["result"] or {}
        self._event(self._runs[sandbox_id], "egress.probe", {"url": _REFERENCE_PROBE, **data})
        if not (data.get("read_only") and data.get("network_blocked")):
            raise RuntimeError("Task 9 sandbox isolation probe failed")
        return data

    async def close(self, sandbox_id: str) -> None:
        run = self._runs.get(sandbox_id)
        if run is None:
            return
        async with run.lock:
            if run.closing:
                return
            run.closing = True
            try:
                await self._copy_outputs(run)
            finally:
                try:
                    await run.sandbox.terminate.aio()
                finally:
                    self._runs.pop(sandbox_id, None)
                    self._active_run_ids.discard(run.run_id)
