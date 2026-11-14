"""Structural interfaces for the v2 engine and interchangeable adapters."""

from typing import Protocol

from pydantic import BaseModel

from core.schemas import Artifact, Evidence, Experiment, Hypothesis, Observation


class RunState(Protocol):
    """Opaque orchestrator state passed through to the lab for delivery."""


class Lab(Protocol):
    name: str

    def brief(self) -> dict: ...

    async def profile(self) -> list[Evidence]: ...

    def candidate_designs(self) -> list[dict] | None: ...

    async def run(self, exp: Experiment) -> Observation: ...

    async def deliver(self, state: "RunState") -> list[Artifact]: ...


class Executor(Protocol):
    async def open(self, lab_name: str, run_id: str) -> str: ...

    async def exec(
        self, sandbox_id: str, language: str, code: str, timeout_s: int = 120
    ) -> dict: ...

    async def close(self, sandbox_id: str) -> None: ...


class LLM(Protocol):
    async def structured(
        self, role: str, schema: type[BaseModel], messages: list[dict], n: int = 1
    ) -> list[BaseModel]: ...


class Predictor(Protocol):
    async def predict(
        self, exp: Experiment, hypotheses: list[Hypothesis], context: str
    ) -> dict[str, dict[str, float]]: ...


class Memory(Protocol):
    async def write(self, ev: Evidence, run_id: str, author_role: str) -> None: ...

    async def recall(
        self, query: str, run_id: str, reader_role: str, k: int = 8
    ) -> list[Evidence]: ...


class Search(Protocol):
    async def find(self, query: str, k: int = 5) -> list[Evidence]: ...
