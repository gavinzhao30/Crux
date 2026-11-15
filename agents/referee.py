"""Blind referee: did a diagnostic run actually test what it claimed?

``await referee(llm, experiment, observation)`` returns ``(valid, note)`` and backs
``AgentRoles.referee``. Code checks come first and need no model call: an invalid observation,
an empty or non-dict result, an observed bin that is not one of the experiment's bins, or a log
that shows an error. Only then does the model (role ``"referee"``) judge three things: did it
run, does the code do what the goal says, and is the result derived from the data.

The model is blind. Its message is built from an explicit allowlist (``SPEC_FIELDS``,
``BIN_FIELDS``, the observed bin, the raw result and a log tail), so it never sees hypothesis
statements, option ids, likelihood tables, priors or posteriors.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from core.budget import BudgetExhausted
from core.interfaces import LLM
from core.schemas import Experiment, Observation

PROMPT = Path(__file__).parent / "prompts" / "referee.md"
LOG_TAIL_LINES = 40
MAX_LINE_CHARS = 300
MAX_CODE_CHARS = 12_000
MAX_RESULT_CHARS = 6_000
MAX_NOTE_CHARS = 300

# The only experiment fields the model sees. Everything else (likelihood keyed by option ids,
# question ids, design, EIG and score) could tell it which option a result favours.
SPEC_FIELDS = ("kind", "goal", "language", "code", "query")
BIN_FIELDS = ("id", "rule")

_ERROR = re.compile(
    r"Traceback \(most recent call last\)"
    r"|^(?:\[stderr\] )?(?:[A-Za-z_][\w.]*)?(?:Error|Exception)(?::| in )"
    r"|^(?:\[stderr\] )?Execution halted"
    r"|^(?:\[stderr\] )?\[execution failed\]"
    r"|Segmentation fault",
    re.MULTILINE,
)

log = logging.getLogger(__name__)


class RefereeVerdict(BaseModel):
    valid: bool
    note: str


async def referee(
    llm: LLM,
    experiment: Experiment,
    observation: Observation,
    *,
    run_dir: Path | str | None = None,
) -> tuple[bool, str]:
    """``(valid, note)`` for one run. ``run_dir`` resolves a relative ``log_ref``."""
    lines = read_log(observation.log_ref, run_dir)
    rejected = code_check(experiment, observation, lines)
    if rejected is not None:
        return False, rejected
    messages = [
        {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
        {
            "role": "user",
            "content": json.dumps(blind_view(experiment, observation, lines)),
        },
    ]
    try:
        [verdict] = await llm.structured("referee", RefereeVerdict, messages)
    except BudgetExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - a referee that cannot judge rejects
        log.warning("referee model call failed for %s: %s", experiment.id, exc)
        return False, f"the referee could not judge this run ({type(exc).__name__})"
    verdict = RefereeVerdict.model_validate(verdict)
    return bool(verdict.valid), _one_line(verdict.note) or "no note given"


def code_check(
    experiment: Experiment, observation: Observation, lines: list[str]
) -> str | None:
    """The note for a run that code can reject on its own, or ``None`` if it passes."""
    if not observation.valid:
        reason = _one_line(observation.referee_note)
        return (
            f"the run reported itself invalid: {reason}"
            if reason
            else "the run was invalid"
        )
    value = observation.value
    if not isinstance(value, dict):
        kind = (
            "no result"
            if value is None
            else f"a {type(value).__name__}, not a JSON object"
        )
        return f"the run returned {kind}"
    if not value:
        return "the run returned an empty result"
    bins = [b.id for b in experiment.bins]
    if observation.bin_id not in bins:
        return f"observed bin {observation.bin_id!r} is not one of the bins {bins}"
    for line in lines:
        if _ERROR.search(line):
            return f"the log shows an error: {_one_line(line)}"
    return None


def blind_view(
    experiment: Experiment, observation: Observation, lines: list[str]
) -> dict:
    """What the model sees, built field by field from the allowlist."""
    spec = {}
    for field in SPEC_FIELDS:
        value = getattr(experiment, field)
        if value is not None:
            spec[field] = _clip(value, MAX_CODE_CHARS) if field == "code" else value
    return {
        "spec": spec,
        "bins": [{f: getattr(b, f) for f in BIN_FIELDS} for b in experiment.bins],
        "observed_bin": observation.bin_id,
        "result": _result(observation.value),
        "log_tail": [line[:MAX_LINE_CHARS] for line in lines[-LOG_TAIL_LINES:]],
    }


def read_log(log_ref: str, run_dir: Path | str | None = None) -> list[str]:
    """The run's log lines, or ``[]`` if there is no readable log.

    ``log_ref`` is a text file (absolute, or relative to ``run_dir``) or ``<file>.jsonl#<id>``,
    the sandbox's event log filtered to one command: ``experiment.log`` lines (stderr prefixed
    with ``[stderr] ``) and ``execution.failed`` errors.
    """
    if not log_ref:
        return []
    ref, _, anchor = log_ref.partition("#")
    path = Path(ref)
    if not path.is_absolute() and run_dir is not None:
        path = Path(run_dir) / path
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not anchor:
        return text.splitlines()
    lines = []
    for raw in text.splitlines():
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("command_id") != anchor:
            continue
        if event.get("type") == "experiment.log":
            prefix = "[stderr] " if event.get("stream") == "stderr" else ""
            lines.append(f"{prefix}{event.get('line', '')}")
        elif event.get("type") == "execution.failed":
            lines.append(f"[execution failed] {event.get('error', '')}")
    return lines


def _result(value: object) -> object:
    text = json.dumps(value, default=str)
    return value if len(text) <= MAX_RESULT_CHARS else _clip(text, MAX_RESULT_CHARS)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]} ...[truncated]"


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())[:MAX_NOTE_CHARS]
