"""Execute precommitted search experiments; code maps JSON results to outcome bins.

Rules are a small expression language over `text` and `n_sources`, not executable
Python. No function calls, attributes, indexing, comprehensions or arithmetic.
The orchestrator owns ranking, posterior updates and the referee verdict.
"""

import ast
import hashlib
import json
import math
import operator
import sqlite3
import time

from core.schemas import Experiment, Observation
from search.exa import ExaSearch, SearchUnavailable
from search.policy import PolicyDenied

OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
ENGINE_VERSION = "documentation-rules-v1"


def _evaluate(node, values):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    if isinstance(node, ast.Name) and node.id in {"text", "n_sources"}:
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, values)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        outcomes = [_evaluate(child, values) for child in node.values]
        return all(outcomes) if isinstance(node.op, ast.And) else any(outcomes)
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, values)
        outcomes = []
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comparator, values)
            if type(op) not in OPERATORS:
                raise ValueError("Unsupported rule comparison")
            outcomes.append(OPERATORS[type(op)](left, right))
            left = right
        return all(outcomes)
    raise ValueError("Unsupported rule expression")


def evaluate_rule(rule: str, values: dict) -> bool:
    if not rule or len(rule) > 2000:
        raise ValueError("Rules must contain 1 to 2000 characters")
    try:
        expression = ast.parse(rule, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid rule syntax") from exc
    if sum(1 for _ in ast.walk(expression)) > 100:
        raise ValueError("Rule is too complex")
    try:
        outcome = _evaluate(expression, values)
    except TypeError as exc:
        raise ValueError("Invalid rule operand types") from exc
    if not isinstance(outcome, bool):
        raise ValueError("Outcome rules must evaluate to booleans")  # noqa: TRY004
    return outcome


class SearchExperiments:
    def __init__(self, search: ExaSearch):
        self.search = search

    def _spec(self, exp: Experiment) -> tuple[str, str]:
        if exp.kind != "search" or not exp.query or not exp.question_ids:
            raise ValueError("Expected a search experiment with query and question IDs")
        if not exp.id.strip() or exp.code is not None or exp.language is not None:
            raise ValueError("Search experiments do not execute agent-authored code")
        if (
            not exp.bins
            or any(not b.id.strip() for b in exp.bins)
            or len({b.id for b in exp.bins}) != len(exp.bins)
        ):
            raise ValueError("Search experiments need uniquely named outcome bins")
        for outcome in exp.bins:
            evaluate_rule(outcome.rule, {"text": "", "n_sources": 0})
        # Both fields and rules, together with the built-in rule-engine version,
        # are fixed durably BEFORE the first external request.
        spec = json.dumps(
            {
                "engine": ENGINE_VERSION,
                "query": exp.query,
                "questions": exp.question_ids,
                "bins": [b.model_dump() for b in exp.bins],
                "policy": self.search.policy.fingerprint,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(spec.encode()).hexdigest()
        if exp.spec_hash is not None and exp.spec_hash != digest:
            raise ValueError("Experiment spec hash does not match its fixed definition")
        return spec, digest

    async def run(self, exp: Experiment) -> Observation:
        if not self.search.config.search:
            self.search.audit("search.disabled", {"experiment_id": exp.id, "query": exp.query})
            return Observation(experiment_id=exp.id, valid=False, referee_note="search is off")
        spec, digest = self._spec(exp)
        try:
            with self.search.db:
                self.search.db.execute(
                    "INSERT INTO experiments(id,spec_hash,spec) VALUES(?,?,?)",
                    (exp.id, digest, spec),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Experiment ID or spec already spent; reuse its recorded observation"
            ) from exc
        self.search.audit(
            "experiment.started",
            {
                "experiment_id": exp.id,
                "question_ids": exp.question_ids,
                "kind": "search",
                "spec_hash": digest,
                "spec": json.loads(spec),
            },
        )
        started = time.monotonic()
        observation = Observation(experiment_id=exp.id, valid=False)
        try:
            result = await self.search.lookup(exp.query)
            observation.cost_usd = result.cost_usd
            values = {
                "text": "\n".join(ev.claim for ev in result.evidence),
                "n_sources": len(result.evidence),
            }
            observation.value = {
                **values,
                "evidence": [ev.model_dump() for ev in result.evidence],
                "spec_hash": digest,
                "cost_basis": result.cost_basis,
            }
            matched = [b.id for b in exp.bins if evaluate_rule(b.rule, values)]
            if not result.evidence:
                observation.referee_note = "No usable documentation evidence; do not update beliefs"
            elif len(matched) != 1:
                observation.referee_note = "Outcome rules matched zero or multiple bins"
            else:
                observation.valid = True
                observation.bin_id = matched[0]
        except SearchUnavailable as exc:
            observation.cost_usd = exc.cost_reservation_usd
            observation.value = {
                "cost_basis": "reservation_after_error",
                "actual_cost_known": False,
                "spec_hash": digest,
            }
            observation.referee_note = "Search unavailable; do not update beliefs"
        except (PolicyDenied, ValueError, TypeError) as exc:
            observation.referee_note = f"Search result invalid: {type(exc).__name__}"
        finally:
            observation.seconds = time.monotonic() - started
            observation.log_ref = f"search.sqlite3:experiment:{exp.id}"
            with self.search.db:
                self.search.db.execute(
                    "UPDATE experiments SET observation=? WHERE id=?",
                    (observation.model_dump_json(), exp.id),
                )
            self.search.audit(
                "experiment.finished",
                {
                    "experiment_id": exp.id,
                    "question_ids": exp.question_ids,
                    "kind": "search",
                    "observation": observation.model_dump(),
                },
            )
        return observation

    def record_belief_effect(self, experiment_id: str, question_id: str, bits: float) -> None:
        """Record an orchestrator-computed change; never compute or apply a posterior here."""
        if not math.isfinite(bits):
            raise ValueError("Information change must be finite")
        row = self.search.db.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not row or not row["observation"]:
            raise ValueError("No recorded observation for this experiment")
        observation = Observation.model_validate_json(row["observation"])
        if not observation.valid or question_id not in json.loads(row["spec"])["questions"]:
            raise ValueError("Cannot attach a belief effect to an invalid or unrelated observation")
        self.search.audit(
            "search.belief_effect",
            {
                "experiment_id": experiment_id,
                "question_id": question_id,
                "information_change_bits": bits,
                "bin_id": observation.bin_id,
            },
        )
