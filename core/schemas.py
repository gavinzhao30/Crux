"""Wire models shared by Crux labs, agents, and event consumers."""

from typing import Literal

from pydantic import BaseModel


class Question(BaseModel):
    id: str
    text: str
    kind: Literal["categorical", "model"]
    impact: float = 1.0
    status: Literal["open", "settled", "robust", "unresolved"] = "open"
    raised_by: str


class Hypothesis(BaseModel):
    id: str
    question_id: str
    statement: str
    author: str
    rationale: str
    answer_key: str | None = None
    model_code: str | None = None
    is_null: bool = False
    prior: float
    posterior: float
    status: Literal["active", "rejected", "accepted"] = "active"
    rejected_by: list[str] = []


class OutcomeBin(BaseModel):
    id: str
    description: str
    rule: str


class Experiment(BaseModel):
    id: str
    question_ids: list[str]
    kind: Literal["compute", "search", "multiverse", "measure"]
    goal: str
    language: Literal["python", "r", "bash"] | None = None
    code: str | None = None
    query: str | None = None
    design: dict | None = None
    bins: list[OutcomeBin] = []
    likelihood: dict[str, dict[str, float]] = {}
    likelihood_source: Literal["exact", "sampled", "jev", "stated"] = "sampled"
    est_cost_s: float = 30
    est_cost_usd: float = 0.0
    eig_bits: float | None = None
    score: float | None = None
    spec_hash: str | None = None
    status: Literal["proposed", "running", "done", "invalid", "spent"] = "proposed"


class Observation(BaseModel):
    experiment_id: str
    valid: bool
    value: float | dict | None = None
    bin_id: str | None = None
    surprise: float = 0.0
    referee_note: str = ""
    log_ref: str = ""
    seconds: float = 0
    cost_usd: float = 0


class BeliefUpdate(BaseModel):
    question_id: str
    experiment_id: str
    prior: dict[str, float]
    posterior: dict[str, float]
    e_values: dict[str, float] | None = None


class Evidence(BaseModel):
    id: str
    kind: Literal["profile", "observation", "search", "objection", "lesson"]
    claim: str
    author_role: str
    source_ref: str | None = None
    url: str | None = None
    conditions: dict = {}


class Artifact(BaseModel):
    name: str
    path: str
    sha256: str
    contract_ok: bool = False
    findings: list[str] = []


class CriticFinding(BaseModel):
    id: str
    artifact: str
    severity: Literal["block", "warn"]
    rule: str
    message: str
    reopens_question: str | None = None


class Report(BaseModel):
    run_id: str
    artifacts: list[Artifact]
    methods_md: str
    rejected: list[dict]
    unresolved: list[str]
    cost_usd: float
    seconds: float


class RunEvent(BaseModel):
    run_id: str
    seq: int
    ts: float
    type: str
    payload: dict
