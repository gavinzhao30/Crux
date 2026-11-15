"""Wire models shared by Crux labs, agents, and event consumers."""

import ast
import math
import operator
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    likelihood_source: Literal["exact", "sampled", "jev", "stated", "tinker"] = "sampled"
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


# --- engine additions: run-time state and outcome bins computed by code ------------------
# Appended for the orchestrator; nothing above this line changed. ``EngineState`` is the
# concrete state the orchestrator keeps, and ``core.interfaces.RunState`` stays the opaque
# protocol labs see. ``RunState`` below aliases ``EngineState`` for engine-side code.


class Trial(BaseModel):
    """One experiment and what it produced (the history agents reason over)."""

    experiment: Experiment
    observation: Observation


class EngineState(BaseModel):
    """Everything the orchestrator knows during a run.

    ``Lab.deliver`` reads it to build artifacts and the reporter reads it to write the
    methods. ``run_dir`` is ``<runs_dir>/<run_id>``; labs write artifacts under
    ``<run_dir>/artifacts``.
    """

    run_id: str
    run_dir: str
    lab: str
    brief: dict = Field(default_factory=dict)
    questions: dict[str, Question] = Field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = Field(default_factory=dict)
    experiments: dict[str, Experiment] = Field(default_factory=dict)
    trials: list[Trial] = Field(default_factory=list)
    updates: list[BeliefUpdate] = Field(default_factory=list)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    findings: list[CriticFinding] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    numeric: dict = Field(default_factory=dict)  # numeric labs: points, posterior
    budget: dict = Field(default_factory=dict)  # latest budget.tick payload
    delivery_round: int = 0

    def options(
        self, question_id: str, *, active_only: bool = False, include_null: bool = True
    ) -> list[Hypothesis]:
        """The question's options in proposal order."""
        return [
            h
            for h in self.hypotheses.values()
            if h.question_id == question_id
            and (include_null or not h.is_null)
            and (not active_only or h.status != "rejected")
        ]

    def best(self, question_id: str) -> Hypothesis | None:
        """The accepted option, else the real option with the highest posterior."""
        live = self.options(question_id, active_only=True, include_null=False)
        for h in live:
            if h.status == "accepted":
                return h
        return max(live, key=lambda h: h.posterior, default=None)

    def answers(self) -> dict[str, Hypothesis]:
        """Question id -> the option the deliverable should follow (``best``)."""
        out = {}
        for qid in self.questions:
            h = self.best(qid)
            if h is not None:
                out[qid] = h
        return out

    def trials_for(self, question_id: str, *, valid_only: bool = True) -> list[Trial]:
        mine = [t for t in self.trials if question_id in t.experiment.question_ids]
        return [t for t in mine if t.observation.valid or not valid_only]


RunState = EngineState  # engine-side name; labs type against core.interfaces.RunState

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_FUNCS: dict[str, Any] = {
    "abs": abs,
    "len": len,
    "min": min,
    "max": max,
    "round": round,
    "float": float,
    "int": int,
    "lower": lambda s: str(s).lower(),
}
_CONSTS = {"True": True, "False": False, "None": None}
_MAX_RULE_LEN = 2000
_MAX_POW = 1000
_CALL_ERROR = "only abs, len, min, max, round, float, int and lower may be called"


class _RuleError(ValueError):
    pass


def _eval_node(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float, str, bool, type(None))
    ):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise _RuleError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for v in node.values:
                result = _eval_node(v, names)
                if not result:
                    return result
            return result
        result = False
        for v in node.values:
            result = _eval_node(v, names)
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, names)
        if isinstance(node.op, ast.Not):
            return not val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise _RuleError("unsupported unary operator")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left, names), _eval_node(node.right, names)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
            raise _RuleError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, names)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _eval_node(comp, names)
            if type(op) not in _CMP_OPS or not _CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e, names) for e in node.elts]
    if isinstance(node, ast.Subscript):
        container = _eval_node(node.value, names)
        key = _eval_node(node.slice, names)
        if not isinstance(key, (str, int)):
            raise _RuleError("subscripts must be a string or an integer")
        return container[key]
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name not in _FUNCS or node.keywords:
            raise _RuleError(_CALL_ERROR)
        return _FUNCS[name](*[_eval_node(a, names) for a in node.args])
    raise _RuleError(f"unsupported syntax: {type(node).__name__}")


def eval_rule(rule: str, value: Any) -> bool:
    """Evaluate one bin rule against an experiment's JSON result. Never raises.

    The rule is a Python expression over the result's keys (``frac_sense > 0.85``,
    ``'reverse' in text``). A scalar result is available as ``value``. Only literals,
    names, arithmetic, comparisons, ``and``/``or``/``not``, subscripts and a few pure
    functions are allowed. A malformed rule, a missing key or a failed evaluation is false,
    so a model never decides which bin was observed (ERD section 7).
    """
    if not isinstance(rule, str) or not rule.strip() or len(rule) > _MAX_RULE_LEN:
        return False
    names: dict[str, Any] = {"value": value}
    if isinstance(value, dict):
        names.update({str(k): v for k, v in value.items() if str(k).isidentifier()})
    try:
        tree = ast.parse(rule.strip(), mode="eval")
        result = _eval_node(tree, names)
    except (SyntaxError, _RuleError, TypeError, ValueError, KeyError, IndexError):
        return False
    except (ZeroDivisionError, OverflowError, RecursionError):
        return False
    if isinstance(result, float) and math.isnan(result):
        return False
    return bool(result)


def bin_for(bins: list[OutcomeBin], value: Any) -> str | None:
    """The id of the first bin whose rule holds for ``value``, or ``None`` if none does."""
    if value is None:
        return None
    for b in bins:
        if eval_rule(b.rule, value):
            return b.id
    return None
