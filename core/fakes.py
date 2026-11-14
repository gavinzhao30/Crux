"""Deterministic, credential-free adapters for offline integration tests."""

import hashlib
from pathlib import Path

from pydantic import BaseModel

from core.interfaces import RunState
from core.schemas import (
    Artifact,
    Evidence,
    Experiment,
    Hypothesis,
    Observation,
    Question,
)


class FakeExecutor:
    def __init__(self):
        self._open: set[str] = set()

    async def open(self, lab_name: str, run_id: str) -> str:
        sandbox_id = f"{lab_name}/{run_id}"
        self._open.add(sandbox_id)
        return sandbox_id

    async def exec(
        self, sandbox_id: str, language: str, code: str, timeout_s: int = 120
    ) -> dict:
        if sandbox_id not in self._open:
            raise ValueError("sandbox is closed")
        return {"stdout": "RESULT: " + code, "exit_code": 0}

    async def close(self, sandbox_id: str) -> None:
        self._open.discard(sandbox_id)


class FakeLLM:
    """Return independent validated copies of caller-supplied canned responses."""

    def __init__(self, responses: dict[type[BaseModel], dict] | None = None):
        self.responses = responses or {}

    async def structured(
        self, role: str, schema: type[BaseModel], messages: list[dict], n: int = 1
    ) -> list[BaseModel]:
        if n < 1:
            raise ValueError("n must be positive")
        return [schema.model_validate(self.responses.get(schema, {})) for _ in range(n)]


class FakePredictor:
    async def predict(
        self, exp: Experiment, hypotheses: list[Hypothesis], context: str
    ) -> dict[str, dict[str, float]]:
        bins = [b.id for b in exp.bins]
        if not bins:
            return {}
        return {h.id: {b: 1 / len(bins) for b in bins} for h in hypotheses}


class FakeMemory:
    def __init__(self):
        self._runs: dict[str, dict[str, Evidence]] = {}

    async def write(self, ev: Evidence, run_id: str, author_role: str) -> None:
        self._runs.setdefault(run_id, {})[ev.id] = ev.model_copy(deep=True)

    async def recall(
        self, query: str, run_id: str, reader_role: str, k: int = 8
    ) -> list[Evidence]:
        matches = [
            ev.model_copy(deep=True)
            for ev in self._runs.get(run_id, {}).values()
            if query.casefold() in ev.claim.casefold()
        ]
        return matches[: max(0, k)]


class FakeSearch:
    async def find(self, query: str, k: int = 5) -> list[Evidence]:
        result = Evidence(
            id="docs-units",
            kind="search",
            author_role="search",
            claim="Both labels denote the same unit in the synthetic manual.",
            url="https://example.invalid/manual/units",
        )
        return [result] if k > 0 else []


class FakeLab:
    name = "demo"
    methods = "Synthetic calibration and unit-label checks.\n"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.questions = [
            Question(
                id="curve",
                text="Which calibration fits the readings?",
                kind="model",
                raised_by="statistician",
            ),
            Question(
                id="units",
                text="Which unit label should appear in the output?",
                kind="categorical",
                raised_by="critic",
            ),
        ]
        self.hypotheses = {
            q.id: [
                Hypothesis(
                    id=f"{q.id}-{i}",
                    question_id=q.id,
                    statement=label,
                    author="proposer",
                    rationale="Synthetic calibration case",
                    is_null=i == 2,
                    prior=p,
                    posterior=p,
                )
                for i, (label, p) in enumerate(
                    zip(labels, (0.425, 0.425, 0.15), strict=True)
                )
            ]
            for q, labels in zip(
                self.questions,
                [
                    ("Linear", "Quadratic", "Neither family"),
                    ("Millivolts", "mV", "Unlisted label"),
                ],
                strict=True,
            )
        }
        self.experiments = [
            Experiment(
                id="sensor",
                kind="measure",
                question_ids=["curve"],
                goal="Read the sensor at x=2",
                design={"x": 2},
            ),
            Experiment(
                id="repeat",
                kind="measure",
                question_ids=["curve"],
                goal="Repeat calibration",
                design={"x": 1},
            ),
            Experiment(
                id="unit-check",
                kind="compute",
                question_ids=["units"],
                goal="Inspect unit metadata",
            ),
            Experiment(
                id="manual",
                kind="search",
                question_ids=["units"],
                goal="Look up unit aliases",
                query="unit aliases",
            ),
        ]

    def brief(self) -> dict:
        return {
            "goal": "Calibrate a synthetic sensor and label its output.",
            "contract": ["METHODS.md"],
        }

    async def profile(self) -> list[Evidence]:
        return [
            Evidence(
                id="profile",
                kind="profile",
                claim="Synthetic sensor with two calibration families.",
                author_role="experimenter",
            )
        ]

    def candidate_designs(self) -> list[dict]:
        return [e.design.copy() for e in self.experiments if e.design is not None]

    async def run(self, exp: Experiment) -> Observation:
        if exp.id == "unit-check":
            return Observation(
                experiment_id=exp.id, valid=False, referee_note="Unit metadata missing"
            )
        if exp.id == "manual":
            return Observation(
                experiment_id=exp.id,
                valid=True,
                value={"same_unit": True},
                bin_id="aliases",
            )
        readings = {"sensor": 4.0, "repeat": 1.0}
        return Observation(experiment_id=exp.id, valid=True, value=readings[exp.id])

    async def deliver(self, state: RunState) -> list[Artifact]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = self.methods.encode()
        (self.output_dir / "METHODS.md").write_bytes(content)
        return [
            Artifact(
                name="METHODS.md",
                path="artifacts/METHODS.md",
                sha256=hashlib.sha256(content).hexdigest(),
                contract_ok=True,
            )
        ]
