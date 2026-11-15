"""The real agent roles behind ``core.interfaces.AgentRoles``.

* ``propose``: Gavin's independent proposers plus the DU-42 fork library
  (``agents.proposer.propose``). The orchestrator owns events and the null option, so the
  proposer runs with a silent bus and ``null_hypothesis`` off; question ids it assigns are
  kept by the orchestrator.
* ``design``: stand-in until DU-10's designer lands. It returns the library diagnostics
  the proposer attached to the question. A question with no library diagnostic gets none
  and ends unresolved.
* ``referee``/``critic``/``report``: DU-11 (``agents.referee``, ``agents.critic``,
  ``agents.reporter``), imported when first used so this module loads before they exist.
* ``regenerate``: stand-in, no new options (a surprise is still logged and the question is
  left to the null option).

Every collaborator can be injected, which is how the tests run without a model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.schemas import (
    Artifact,
    CriticFinding,
    EngineState,
    Evidence,
    Experiment,
    Hypothesis,
    Observation,
    Question,
    Report,
    Trial,
)


class SilentBus:
    """What ``agents.proposer.propose`` needs from an EventBus, minus the events."""

    def __init__(self, run_id: str = "swarm") -> None:
        self.run_id = run_id
        self.path = Path("__no_events__.jsonl")
        self.events: list[tuple[str, dict]] = []

    async def emit(self, type_: str, payload: dict) -> None:
        self.events.append((type_, payload))


class NullMemory:
    async def write(self, ev: Evidence, run_id: str, author_role: str) -> None:
        return None

    async def recall(
        self, query: str, run_id: str, reader_role: str, k: int = 8
    ) -> list:
        return []


class SwarmRoles:
    def __init__(
        self,
        llm: Any,
        config: Any,
        *,
        memory: Any = None,
        library: Callable[[dict], list] | None = None,
        propose_fn: Callable | None = None,
        referee_fn: Callable | None = None,
        critic_fn: Callable | None = None,
        report_fn: Callable | None = None,
        counts_path: Path | None = None,
        run_dir: Callable[[], Path | None] | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.memory = memory or NullMemory()
        self.library = library
        self.propose_fn = propose_fn
        self.referee_fn = referee_fn
        self.critic_fn = critic_fn
        self.report_fn = report_fn
        self.counts_path = counts_path
        self.run_dir = run_dir
        self.brief: dict = {}
        self.diagnostics: dict[str, list[Experiment]] = {}
        self.rounds = 0

    # --- proposers -------------------------------------------------------------------

    async def propose(
        self, brief: dict, evidence: list[Evidence]
    ) -> tuple[list[Question], list[Hypothesis]]:
        propose_fn = self.propose_fn
        if propose_fn is None:
            from agents.proposer import propose as propose_fn
        library = self.library
        if library is None:
            try:
                from diagnostics.de_from_counts.library import (
                    library_questions as library,
                )
            except ImportError:
                library = None
        self.brief = dict(brief)
        entries = library(brief) if library is not None else []
        self.rounds += 1
        cfg = self.config.model_copy(update={"null_hypothesis": False})
        result = await propose_fn(
            brief,
            list(evidence),
            llm=self.llm,
            memory=self.memory,
            bus=SilentBus(f"round{self.rounds}"),
            config=cfg,
            library=entries,
        )
        for exp in result.diagnostics:
            for qid in exp.question_ids:
                self.diagnostics.setdefault(qid, []).append(exp)
        hypotheses = [h for h in result.hypotheses if not h.is_null]
        return list(result.questions), hypotheses

    async def regenerate(
        self, question: Question, hypotheses: list[Hypothesis], history: list[Trial]
    ) -> list[Hypothesis]:
        return []

    # --- designer (stand-in for DU-10) -----------------------------------------------

    async def design(
        self,
        question: Question,
        hypotheses: list[Hypothesis],
        evidence: list[Evidence],
        history: list[Trial],
    ) -> list[Experiment]:
        return [
            exp.model_copy(deep=True) for exp in self.diagnostics.get(question.id, [])
        ]

    # --- DU-11 -----------------------------------------------------------------------

    async def referee(
        self, experiment: Experiment, observation: Observation
    ) -> tuple[bool, str]:
        fn = self.referee_fn
        if fn is None:
            from agents.referee import referee as fn
        run_dir = self.run_dir() if self.run_dir is not None else None
        return await fn(self.llm, experiment, observation, run_dir=run_dir)

    async def critic(
        self, artifacts: list[Artifact], contract: dict
    ) -> list[CriticFinding]:
        fn = self.critic_fn
        if fn is None:
            from agents.critic import critique as fn
        return await fn(
            self.llm,
            artifacts,
            contract,
            counts_path=self.counts_path,
            brief=self.brief,
        )

    async def report(self, state: EngineState, artifacts: list[Artifact]) -> Report:
        fn = self.report_fn
        if fn is None:
            from agents.reporter import report as fn
        return await fn(self.llm, state, artifacts)
