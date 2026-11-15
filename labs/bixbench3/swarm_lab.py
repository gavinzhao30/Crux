"""Task 9 lab as the orchestrator runs it: sandbox lifecycle, logs and delivery retries.

Wraps ``Task9Lab`` (DU-13) without changing it:

* ``bind``/``close`` (``core.interfaces.LabHooks``) own the sandbox. It opens lazily on
  the first async call, since ``brief()`` runs before ``bind``.
* ``ModalExecutor`` logs commands to ``execution.jsonl``; ``Observation.log_ref`` comes back
  as ``execution.jsonl#<command>``. That command's lines are copied to
  ``logs/<command>.txt`` so the orchestrator can stream them as ``experiment.log``.
* ``deliver`` asks the delivery planner for code, runs it, and retries invalid plans
  or failed executions up to ``max_attempts`` times per delivery round. Generated code
  is kept under ``delivery/`` in the run folder.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from labs.bixbench3.task9 import Task9Lab, validate_de_table
from labs.bixbench3.task9_contract import load_contract, load_prompt

if TYPE_CHECKING:
    from core.schemas import Artifact, EngineState, Evidence, Experiment, Observation

Planner = Callable[..., Awaitable[list[tuple[str, str]]]]


async def _no_planner(state: object) -> list[tuple[str, str]]:  # pragma: no cover
    raise RuntimeError("SwarmTask9Lab.deliver plans delivery itself")


class SwarmTask9Lab:
    name = "bixbench3"

    def __init__(
        self,
        executor: Any,
        *,
        llm: Any,
        planner: Planner | None = None,
        contract: dict | None = None,
        task_prompt: str | None = None,
        search: Any = None,
        summarize_search: Any = None,
        max_attempts: int = 3,
        delivery_timeout_s: int = 600,
    ) -> None:
        if planner is None:
            from agents.deliverer import plan_delivery

            planner = plan_delivery
        self.executor = executor
        self.llm = llm
        self.planner = planner
        self.contract = contract if contract is not None else load_contract()
        self.max_attempts = max_attempts
        self.delivery_timeout_s = delivery_timeout_s
        self.de_tables = frozenset(
            item["path"] for item in self.contract.values() if item.get("kind") == "de"
        )
        self.inner = Task9Lab(
            executor,
            "",
            task_prompt=task_prompt if task_prompt is not None else load_prompt()[0],
            contract=self.contract,
            plan_delivery=_no_planner,
            search=search,
            summarize_search=summarize_search,
            de_tables=self.de_tables,
        )
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.sandbox_id: str | None = None
        self.profile_facts: list[dict] = []

    # --- Lab -------------------------------------------------------------------------

    def brief(self) -> dict:
        return self.inner.brief()

    def candidate_designs(self) -> None:
        return None

    async def profile(self) -> list[Evidence]:
        await self._open()
        evidence = await self.inner.profile()
        for ev in evidence:
            try:
                self.profile_facts.append(json.loads(ev.claim))
            except ValueError:
                pass
        return evidence

    async def run(self, exp: Experiment) -> Observation:
        await self._open()
        obs = await self.inner.run(exp)
        obs.log_ref = self._text_log(obs.log_ref)
        return obs

    async def deliver(self, state: EngineState) -> list[Artifact]:
        from agents.llm import LLMOutputError
        from core.schemas import Artifact

        await self._open()
        error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                steps = await self.planner(
                    self.llm, state, last_error=error, profile=self.profile_facts
                )
            except LLMOutputError:
                if attempt == self.max_attempts:
                    raise
                error = (
                    "The previous plan was incomplete or invalid. Return concise code, "
                    "reuse existing scratch files, and return a complete plan."
                )
                continue
            error = None
            for i, (language, code) in enumerate(steps, start=1):
                self._keep_code(state.delivery_round, attempt, i, language, code)
                result = await self.executor.run_code(
                    self.sandbox_id,
                    language,
                    code,
                    require_result=False,
                    timeout_s=self.delivery_timeout_s,
                )
                if not result["valid"]:
                    error = f"step {i} ({language}): {result['error']}\n{result.get('stderr', '')}"
                    break
            if error is None:
                break
        artifacts = []
        for export in await self.executor.collect_outputs(self.sandbox_id):
            name = export["name"]
            findings = (
                validate_de_table(Path(export["path"]))
                if name in self.de_tables
                else []
            )
            artifacts.append(
                Artifact(**export, contract_ok=not findings, findings=findings)
            )
        return artifacts

    # --- LabHooks --------------------------------------------------------------------

    def bind(self, run_id: str, run_dir: Path, executor: Any = None) -> None:
        self.run_id, self.run_dir = run_id, Path(run_dir)
        if self.executor is None and executor is not None:
            self.executor = self.inner.executor = executor

    async def close(self) -> None:
        if self.sandbox_id is not None and self.executor is not None:
            sandbox, self.sandbox_id = self.sandbox_id, None
            await self.executor.close(sandbox)

    # --- helpers ---------------------------------------------------------------------

    async def _open(self) -> None:
        if self.sandbox_id is None:
            if self.run_id is None:
                raise RuntimeError("bind() must run before the lab is used")
            self.sandbox_id = await self.executor.open("bixbench3", self.run_id)
            self.inner.sandbox_id = self.sandbox_id

    def _text_log(self, log_ref: str) -> str:
        """``execution.jsonl#<command>`` -> ``logs/<command>.txt`` with that command's lines."""
        if "#" not in (log_ref or "") or self.run_dir is None:
            return log_ref
        name, command = log_ref.split("#", 1)
        source = self.run_dir / name
        if not source.is_file():
            return log_ref
        lines = []
        for raw in source.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if (
                event.get("type") == "experiment.log"
                and event.get("command_id") == command
            ):
                prefix = "[stderr] " if event.get("stream") == "stderr" else ""
                lines.append(prefix + str(event.get("line", "")))
        target = self.run_dir / "logs" / f"{command}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"logs/{command}.txt"

    def _keep_code(
        self, round_: int, attempt: int, step: int, language: str, code: str
    ) -> None:
        if self.run_dir is None:
            return
        folder = self.run_dir / "delivery"
        folder.mkdir(parents=True, exist_ok=True)
        suffix = {"r": "R", "python": "py"}.get(language, "txt")
        (folder / f"round{round_}-attempt{attempt}-step{step}.{suffix}").write_text(
            code, encoding="utf-8"
        )


def build_task9_lab(
    *,
    llm: Any,
    runs_dir: Path,
    case_dir: Path | None = None,
    manifest: Path | None = None,
    search: Any = None,
) -> SwarmTask9Lab:
    """A Task 9 lab on Modal (``manifest`` from ``execution.staging``) or, for development,
    on this machine (``case_dir`` holding the counts and sample sheet)."""
    if (case_dir is None) == (manifest is None):
        raise ValueError("Pass exactly one of case_dir (local) or manifest (Modal)")
    if manifest is not None:
        from execution.sandbox import ModalExecutor
        from execution.staging import Task9Inputs

        executor: Any = ModalExecutor(Task9Inputs.load(Path(manifest)), Path(runs_dir))
    else:
        from execution.local import LocalExecutor

        executor = LocalExecutor(Path(case_dir), Path(runs_dir))
    return SwarmTask9Lab(executor, llm=llm, search=search)


def build_task9_run(
    cfg: Any,
    *,
    case_dir: Path | None = None,
    manifest: Path | None = None,
    llm: Any = None,
    search: Any = None,
    memory: Any = None,
) -> tuple[SwarmTask9Lab, Any]:
    """The lab and ``Deps`` for one real Task 9 run with the swarm's agents."""
    from agents.llm import LLM, STRONG_MODEL, LLMConfig
    from agents.predictor import SampledPredictor
    from agents.swarm import SwarmRoles
    from core.orchestrator import Deps
    from execution.staging import COUNTS

    if llm is None:
        config = LLMConfig()
        # The delivery code is the deliverable, so it gets the strong model.
        config.role_models.setdefault("deliverer", STRONG_MODEL)
        llm = LLM(config)
    lab = build_task9_lab(
        llm=llm,
        runs_dir=Path(cfg.runs_dir),
        case_dir=case_dir,
        manifest=manifest,
        search=search,
    )
    counts = Path(case_dir) / COUNTS if case_dir is not None else None
    # The referee reads log files relative to the run folder, which the lab learns in bind().
    roles = SwarmRoles(
        llm, cfg, memory=memory, counts_path=counts, run_dir=lambda: lab.run_dir
    )
    deps = Deps(
        agents=roles,
        predictor=SampledPredictor(llm, n=cfg.n_predictor_samples, eps=cfg.eps),
        memory=memory,
        search=search,
        llm=llm,
    )
    return lab, deps


def discover_cases(root: Path | None) -> dict[str, dict]:
    """Cases under ``root``: a folder with the counts and sample sheet runs locally; a
    staging manifest (``*.json`` with ``bundle_id``) runs on Modal. Nothing here reads an
    evaluator truth key, so splits are not known to the engine."""
    from execution.staging import COUNTS, SAMPLES

    cases: dict[str, dict] = {}
    if root is None or not Path(root).is_dir():
        return cases
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_dir() and (path / COUNTS).is_file() and (path / SAMPLES).is_file():
            cases[path.relative_to(root).as_posix()] = {
                "kind": "local",
                "path": str(path),
            }
        elif (
            path.suffix == ".json" and path.is_file() and "evaluator" not in path.parts
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and "bundle_id" in data and "files" in data:
                cases[path.relative_to(root).with_suffix("").as_posix()] = {
                    "kind": "modal",
                    "path": str(path),
                }
    return cases
