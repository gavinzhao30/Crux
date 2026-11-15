"""Reporter: the methods write-up for a finished run.

``await report(llm, state, artifacts)`` backs ``AgentRoles.report``. Code assembles every fact
(ids, numbers, outcomes) from ``EngineState`` with ``report_facts``; the model (role
``"reporter"``) only turns them into methods prose. The prose is used only if it cites every
question and rejected option by id and every number in it comes from the facts. Otherwise, or
if the model call fails, ``render_template`` writes the methods from the same facts.

The orchestrator overwrites the run id, artifact list, rejected ledger, unresolved list, cost
and time on the returned ``Report``; ``methods_md`` is what this module is for.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from core.budget import BudgetExhausted
from core.interfaces import LLM
from core.schemas import Artifact, EngineState, Hypothesis, Report, Trial

PROMPT = Path(__file__).parent / "prompts" / "reporter.md"
log = logging.getLogger(__name__)

SMALL_INTEGERS = 10  # counts and list numbers up to this may appear without being facts
_FACT_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TEXT_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)(?!\w)")
_CITATION = re.compile(r"\[[\w.:\-]+\]|`[^`\s]+`")  # ids and file names, never prose


class MethodsText(BaseModel):
    methods_md: str


async def report(llm: LLM, state: EngineState, artifacts: list[Artifact]) -> Report:
    facts = report_facts(state, artifacts)
    methods = await write_methods(llm, facts)
    return Report(
        run_id=state.run_id,
        artifacts=list(artifacts),
        methods_md=methods,
        rejected=facts["rejected"],
        unresolved=[u["question"] for u in facts["unresolved"]],
        cost_usd=float(state.budget.get("cost_usd", 0.0) or 0.0),
        seconds=float(state.budget.get("seconds", 0.0) or 0.0),
    )


async def write_methods(llm: LLM, facts: dict) -> str:
    """The model's methods prose if it passes the checks, else the template."""
    messages = [
        {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(facts, default=str)},
    ]
    try:
        [draft] = await llm.structured("reporter", MethodsText, messages)
        text = MethodsText.model_validate(draft).methods_md.strip()
    except BudgetExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed call falls back, never crashes
        log.warning("reporter model call failed, using the template: %s", exc)
        return render_template(facts)
    problems = check_methods(text, facts)
    if problems:
        log.warning(
            "reporter prose rejected, using the template: %s", "; ".join(problems)
        )
        return render_template(facts)
    return text


# --- facts ----------------------------------------------------------------------------------


def report_facts(state: EngineState, artifacts: list[Artifact]) -> dict:
    """Everything the methods may state, assembled from the state. Floats are rounded."""
    questions, robust, unresolved = [], [], []
    for q in state.questions.values():
        entry: dict = {
            "id": q.id,
            "text": q.text,
            "kind": q.kind,
            "status": q.status,
            "raised_by": q.raised_by,
            "options": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "author": h.author,
                    "status": h.status,
                    "prior": _r(h.prior),
                    "posterior": _r(h.posterior),
                    **({"null": True} if h.is_null else {}),
                }
                for h in state.options(q.id)
            ],
            "diagnostics": [
                _trial(state, t, q.id) for t in state.trials_for(q.id, valid_only=False)
            ],
        }
        live = state.options(q.id, active_only=True, include_null=False)
        if q.status == "settled" and (best := state.best(q.id)) is not None:
            entry["settled_on"] = _option(best)
        elif q.status == "robust":
            entry["robust_answer"] = live[0].answer_key if live else None
            robust.append(
                {
                    "question": q.id,
                    "text": q.text,
                    "answer_key": entry["robust_answer"],
                    "options": [h.id for h in live],
                }
            )
        elif q.status in ("open", "unresolved"):
            top = sorted(live, key=lambda h: -h.posterior)[:2]
            entry["top_options"] = [_option(h) for h in top]
            unresolved.append(
                {"question": q.id, "text": q.text, "top_options": entry["top_options"]}
            )
        questions.append(entry)

    rejected = []
    for r in state.rejected:
        h = state.hypotheses.get(str(r.get("hypothesis")))
        ev = state.evidence.get(str(r.get("evidence_id") or ""))
        rejected.append(
            {
                **r,
                "statement": h.statement if h else None,
                "evidence": ev.claim if ev else None,
                "experiment": ev.source_ref if ev else None,
            }
        )

    facts: dict = {
        "run_id": state.run_id,
        "lab": state.lab,
        "goal": str(state.brief.get("goal", "")),
        "questions": questions,
        "rejected": rejected,
        "robust": robust,
        "unresolved": unresolved,
        "deliverables": [
            {"name": a.name, "contract_ok": a.contract_ok, "findings": list(a.findings)}
            for a in artifacts
        ],
        "delivery_rounds": state.delivery_round,
        "critic_findings": [
            {
                "id": f.id,
                "artifact": f.artifact,
                "severity": f.severity,
                "message": f.message,
            }
            for f in state.findings
        ],
    }
    if state.numeric.get("points"):
        facts["measurements"] = len(state.numeric["points"])
    return facts


def _trial(state: EngineState, t: Trial, qid: str) -> dict:
    exp, obs = t.experiment, t.observation
    label = next((b.description for b in exp.bins if b.id == obs.bin_id), None)
    out = {
        "experiment": exp.id,
        "kind": exp.kind,
        "goal": exp.goal,
        "outcome_bin": obs.bin_id,
        "outcome": label,
        "valid": obs.valid,
        "referee_note": obs.referee_note,
        "est_cost_s": _r(exp.est_cost_s),
    }
    if exp.eig_bits is not None:
        out["eig_bits"] = _r(exp.eig_bits)
    update = next(
        (
            u
            for u in state.updates
            if u.question_id == qid and u.experiment_id == exp.id
        ),
        None,
    )
    if update is not None:
        out["posterior_after"] = {h: _r(p) for h, p in update.posterior.items()}
    return out


def _option(h: Hypothesis) -> dict:
    return {"option": h.id, "statement": h.statement, "posterior": _r(h.posterior)}


def _r(x: float) -> float:
    return round(float(x), 3)


# --- checks on the model's prose ------------------------------------------------------------


def check_methods(text: str, facts: dict) -> list[str]:
    """Why the prose cannot be used (empty if it can)."""
    if not text:
        return ["the prose is empty"]
    problems = []
    ids = [q["id"] for q in facts["questions"]] + [
        str(r.get("hypothesis")) for r in facts["rejected"]
    ]
    missing = [
        i for i in ids if not re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", text)
    ]
    if missing:
        problems.append(f"it does not cite {missing}")
    stray = stray_numbers(text, facts)
    if stray:
        problems.append(f"numbers not in the facts: {stray}")
    return problems


def stray_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in ``text`` that do not come from ``facts`` (as given or rounded)."""
    blob = json.dumps(facts, default=str)
    exact = set(_FACT_NUMBER.findall(blob))
    values = {float(n) for n in exact}
    stray = []
    for match in _TEXT_NUMBER.finditer(_CITATION.sub(" ", text)):
        token, percent = match.group(1), match.group(2)
        decimals = len(token.split(".")[1]) if "." in token else 0
        number = float(token)
        if percent:
            number, decimals = number / 100, decimals + 2
        elif token in exact or (decimals == 0 and number <= SMALL_INTEGERS):
            continue
        if decimals and any(
            round(v, decimals) == round(number, decimals) for v in values
        ):
            continue
        stray.append(match.group(0))
    return stray


# --- the template ---------------------------------------------------------------------------


def render_template(facts: dict) -> str:
    """Plain methods text from the facts alone (the fallback, and a readable default)."""
    lines = ["# Methods", ""]
    if facts.get("goal"):
        lines += ["## Objective", "", facts["goal"], ""]

    lines += ["## Questions and diagnostics", ""]
    if not facts["questions"]:
        lines += ["none", ""]
    for q in facts["questions"]:
        options = "; ".join(
            f"[{o['id']}] {o['statement']}" for o in q["options"] if not o.get("null")
        )
        lines += [f"### [{q['id']}] {q['text']}", ""]
        lines.append(f"Raised by {q['raised_by']}. Options: {options or 'none'}.")
        lines.append("")
        for d in q["diagnostics"]:
            seen = f"observed `{d['outcome_bin']}`"
            if d["outcome"]:
                seen += f" ({d['outcome']})"
            verdict = "valid" if d["valid"] else "invalid"
            note = f": {d['referee_note']}" if d["referee_note"] else ""
            lines.append(
                f"- [{d['experiment']}] {d['goal']}: {seen}; referee: {verdict}{note}."
            )
        if q["diagnostics"]:
            lines.append("")
        lines += [_outcome(q), ""]

    lines += ["## Rejected options", ""]
    if not facts["rejected"]:
        lines.append("none")
    for r in facts["rejected"]:
        statement = f" {r['statement']}" if r.get("statement") else ""
        line = f"- [{r.get('hypothesis')}]{statement} (question [{r.get('question')}]): {r.get('reason')}."
        if r.get("evidence"):
            source = f" from [{r['experiment']}]" if r.get("experiment") else ""
            line += f" Evidence [{r.get('evidence_id')}]{source}: {r['evidence']}."
        lines.append(line)
    lines.append("")

    lines += ["## Robust choices", ""]
    if not facts["robust"]:
        lines.append("none")
    for r in facts["robust"]:
        lines.append(
            f"- [{r['question']}] {r['text']}: every remaining option "
            f"({', '.join(r['options'])}) gives the same deliverable (`{r['answer_key']}`)."
        )
    lines.append("")

    lines += ["## Unresolved", ""]
    if not facts["unresolved"]:
        lines.append("none")
    for u in facts["unresolved"]:
        lines.append(f"- [{u['question']}] {u['text']}: {_top(u['top_options'])}")
    lines.append("")

    lines += ["## Deliverables", ""]
    if not facts["deliverables"]:
        lines.append("none")
    for d in facts["deliverables"]:
        if d["contract_ok"]:
            status = "passed the contract checks"
        elif d["findings"]:
            status = "did not pass: " + "; ".join(d["findings"])
        else:
            status = "has no contract verdict"
        lines.append(f"- `{d['name']}` {status}.")
    lines.append("")
    return "\n".join(lines)


def _outcome(q: dict) -> str:
    if q.get("settled_on"):
        s = q["settled_on"]
        return (
            f"Settled on [{s['option']}] {s['statement']} (posterior {s['posterior']})."
        )
    if q["status"] == "robust":
        return f"Robust: every remaining option gives the same deliverable (`{q['robust_answer']}`)."
    if "top_options" in q:
        return f"Unresolved: {_top(q['top_options'])}"
    return f"Status: {q['status']}."


def _top(top: list[dict]) -> str:
    if not top:
        return "no option survived."
    named = " and ".join(
        f"[{o['option']}] {o['statement']} ({o['posterior']})" for o in top
    )
    if len(top) == 1:
        return f"only {named} remained; it is reported."
    return f"top options {named}; both are reported."
