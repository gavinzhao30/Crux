"""Run Task 9 code on this machine, with the same interface as ``ModalExecutor``.

For development and tests only. It is NOT a sandbox: there is no network block and no
read-only mount, so never use it for graded runs. Code written for the Modal image reads
``/data`` and writes ``/outputs``; quoted path literals that start with ``/data``,
``/outputs`` or ``/work`` are rewritten to this run's local folders, and the same
folders are exported as ``CRUX_DATA_DIR``, ``CRUX_OUTPUT_DIR`` and ``CRUX_WORK_DIR``.
R code needs ``Rscript`` on PATH.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from time import monotonic
from uuid import uuid4

from execution.staging import COUNTS, METADATA, SAMPLES, Task9Inputs

_SUFFIX = {"python": "py", "r": "R", "bash": "sh"}
_PATH_LITERAL = re.compile(
    r"""(?P<q>['"])/(?P<root>data|outputs|work)(?P<rest>(?:/[^'"\n]*)?)(?P=q)"""
)


def local_inputs(data_dir: Path) -> Task9Inputs:
    """Hash the case files the way staging does, so evidence names the same bundle."""
    files = {}
    for name in (COUNTS, SAMPLES, METADATA):
        path = data_dir / name
        if path.is_file():
            with path.open("rb") as fh:
                files[name] = sha256(fh.read()).hexdigest()
    bundle = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return Task9Inputs(bundle_id=bundle, files=files)


def rewrite_paths(code: str, roots: dict[str, Path]) -> str:
    """Point quoted ``/data``, ``/outputs`` and ``/work`` literals at local folders."""

    def swap(match: re.Match) -> str:
        base = roots[match["root"]].as_posix()
        return f"{match['q']}{base}{match['rest']}{match['q']}"

    return _PATH_LITERAL.sub(swap, code)


class LocalExecutor:
    """``open``, ``exec``, ``run_code``, ``collect_outputs`` and ``close`` like ModalExecutor."""

    def __init__(
        self,
        data_dir: Path,
        runs_dir: Path = Path("runs"),
        *,
        python: str | None = None,
        rscript: str | None = None,
        diagnostic_timeout_s: int = 120,
    ) -> None:
        # Same cap as ModalExecutor; Task9Lab.run passes it to every diagnostic.
        if not 1 <= diagnostic_timeout_s <= 120:
            raise ValueError("Diagnostic timeout must be between 1 and 120 seconds")
        self.diagnostic_timeout_s = diagnostic_timeout_s
        self.data_dir = Path(data_dir).resolve()
        if (
            not (self.data_dir / COUNTS).is_file()
            or not (self.data_dir / SAMPLES).is_file()
        ):
            raise FileNotFoundError(f"{self.data_dir} needs {COUNTS} and {SAMPLES}")
        self.inputs = local_inputs(self.data_dir)
        self.runs_dir = Path(runs_dir)
        self.python = python or sys.executable
        self.rscript = rscript or shutil.which("Rscript")
        self._runs: dict[str, Path] = {}

    async def open(self, lab_name: str, run_id: str) -> str:
        if lab_name != "bixbench3" or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("Expected the bixbench3 lab and a simple run ID")
        if run_id in self._runs:
            raise ValueError(f"Sandbox already open for {run_id}")
        directory = (self.runs_dir / run_id).resolve()
        for sub in ("work", "outputs", "logs"):
            (directory / sub).mkdir(parents=True, exist_ok=True)
        (directory / "input_manifest.json").write_text(
            self.inputs.json(), encoding="utf-8"
        )
        self._runs[run_id] = directory
        return run_id

    def _roots(self, directory: Path) -> dict[str, Path]:
        return {
            "data": self.data_dir,
            "outputs": directory / "outputs",
            "work": directory / "work",
        }

    async def exec(
        self, sandbox_id: str, language: str, code: str, timeout_s: int = 120
    ) -> dict:
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
        if language not in _SUFFIX or timeout_s <= 0:
            raise ValueError("Expected python, r or bash and a positive timeout")
        directory = self._runs[sandbox_id]
        roots = self._roots(directory)
        command_id = uuid4().hex
        script = roots["work"] / f"{command_id}.{_SUFFIX[language]}"
        script.write_text(rewrite_paths(code, roots), encoding="utf-8")
        log_rel = f"logs/{command_id}.txt"
        if language == "r" and not self.rscript:
            return self._failed(
                "Rscript is not installed on this machine", log_rel, 0.0
            )
        command = {
            "python": [self.python, "-u", str(script)],
            "r": [str(self.rscript), "--vanilla", str(script)],
            "bash": ["bash", str(script)],
        }[language]
        env = {
            **os.environ,
            "CRUX_DATA_DIR": str(roots["data"]),
            "CRUX_OUTPUT_DIR": str(roots["outputs"]),
            "CRUX_WORK_DIR": str(roots["work"]),
            "PYTHONIOENCODING": "utf-8",
        }
        started = monotonic()
        error = None
        returncode = None
        stdout = stderr = b""
        try:
            # A worker thread works under any event loop (Windows selector loops cannot
            # spawn subprocesses through asyncio).
            done = await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=roots["work"],
                env=env,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            stdout, stderr, returncode = done.stdout, done.stderr, done.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            error = "Command execution failed: TimeoutError"
        except OSError as exc:
            error = f"Command execution failed: {type(exc).__name__}"
        out_lines = stdout.decode("utf-8", "replace").splitlines()
        err_lines = stderr.decode("utf-8", "replace").splitlines()
        (directory / log_rel).write_text(
            "\n".join([*out_lines, *(f"[stderr] {line}" for line in err_lines)]) + "\n",
            encoding="utf-8",
        )
        results = [line for line in out_lines if line.startswith("RESULT:")]
        result = None
        if results and len(results[0]) <= 256_000:
            try:
                payload = json.loads(results[0].removeprefix("RESULT:"))
            except ValueError:
                payload = None
            result = payload if isinstance(payload, dict) else None
        if error is None and returncode != 0:
            error = f"Command exited with status {returncode}"
        if error is None and require_result and (len(results) != 1 or result is None):
            error = "Expected exactly one RESULT: line containing a JSON object"
        return {
            "valid": error is None,
            "result": result if error is None else None,
            "stdout": "\n".join(out_lines[:12] + out_lines[12:][-12:]),
            "stderr": "\n".join(err_lines[:12] + err_lines[12:][-12:]),
            "returncode": returncode,
            "seconds": monotonic() - started,
            "error": error,
            "log_ref": log_rel,
        }

    @staticmethod
    def _failed(error: str, log_ref: str, seconds: float) -> dict:
        return {
            "valid": False,
            "result": None,
            "stdout": "",
            "stderr": error,
            "returncode": None,
            "seconds": seconds,
            "error": error,
            "log_ref": log_ref,
        }

    async def collect_outputs(self, sandbox_id: str) -> list[dict]:
        directory = self._runs[sandbox_id]
        source, destination = directory / "outputs", directory / "artifacts"
        # Like ModalExecutor, the artifact folder mirrors the outputs exactly: files the
        # code deleted do not linger from an earlier export.
        if destination.exists():
            shutil.rmtree(destination)
        artifacts = []
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            name = path.relative_to(source).as_posix()
            data = path.read_bytes()
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            artifacts.append(
                {"name": name, "path": str(target), "sha256": sha256(data).hexdigest()}
            )
        return artifacts

    async def close(self, sandbox_id: str) -> None:
        if sandbox_id in self._runs:
            await self.collect_outputs(sandbox_id)
            self._runs.pop(sandbox_id, None)
