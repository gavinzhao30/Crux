"""Check an event file on its own: envelope rules and the categorical belief trajectory.

``replay_check(path)`` rebuilds every categorical question's belief from the event file alone
and compares it with the recorded ``belief.updated`` events:

* a question's prior is the ``priors`` of its latest ``hypothesis.merged`` event;
* each ``belief.updated`` must start from the rebuilt belief (its recorded ``prior``) and end at
  Bayes' rule applied with the experiment's likelihood table (the latest ``experiment.proposed``
  for that experiment) and the observed bin, within ``tol``;
* ``hypothesis.rejected`` removes the option and renormalises the rest.

The Bayes update here is written independently of ``core/eig.py``. Numeric questions
(particle beliefs) are checked only for finite posteriors that sum to one.

It also checks the envelope: ``seq`` starts at 0 with no gaps, one ``run_id``, ``run.started``
first, exactly one ``run.finished`` or ``run.error`` and it is last, and no ``belief.updated``
for an experiment the referee ruled invalid.

``python -m core.replay <events.jsonl>`` prints a short summary and exits 0 when every check
passes, 1 otherwise.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import events as E

__all__ = ["ReplayResult", "replay_check"]


@dataclass
class ReplayResult:
    path: str
    n_events: int = 0
    n_updates: int = 0
    n_checked: int = 0  # categorical updates recomputed and compared
    errors: list[str] = field(default_factory=list)
    final_beliefs: dict[str, dict[str, float]] = field(default_factory=dict)
    status: str | None = None  # run.finished status, or "error"

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        head = "OK" if self.ok else "FAILED"
        text = (
            f"{head}: {self.path}: {self.n_events} events, {self.n_updates} belief updates, "
            f"{self.n_checked} recomputed, status={self.status}"
        )
        if self.errors:
            shown = self.errors[:10]
            text += "\n" + "\n".join(f"  - {e}" for e in shown)
            if len(self.errors) > len(shown):
                text += f"\n  ... and {len(self.errors) - len(shown)} more"
        return text


def _bayes(prior: dict[str, float], table: dict[str, dict[str, float]], observed: str) -> dict:
    joint = {}
    for hid, p in prior.items():
        row = table.get(hid)
        if row is None:
            raise KeyError(hid)
        total = sum(v for v in row.values() if v > 0)
        joint[hid] = p * (row.get(observed, 0.0) / total if total > 0 else 0.0)
    z = sum(joint.values())
    if z <= 0:
        raise ZeroDivisionError("every option gives the observed bin zero probability")
    return {hid: v / z for hid, v in joint.items()}


def _close(a: dict[str, float], b: dict[str, float], tol: float) -> bool:
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= tol for k in a)


def _renormalise(belief: dict[str, float]) -> dict[str, float]:
    z = sum(belief.values())
    return {k: v / z for k, v in belief.items()} if z > 0 else belief


def replay_check(path: str | Path, tol: float = 1e-9) -> ReplayResult:
    """Validate the event file at ``path``; see the module docstring."""
    path = Path(path)
    result = ReplayResult(path=path.as_posix())
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result.errors.append(f"cannot read the file: {exc}")
        return result
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            result.errors.append(f"line {n}: not JSON ({exc.msg})")
            continue
        missing = {"run_id", "seq", "ts", "type", "payload"} - set(obj)
        if missing or not isinstance(obj.get("payload"), dict):
            result.errors.append(f"line {n}: bad envelope (missing {sorted(missing)})")
            continue
        events.append(obj)
    result.n_events = len(events)
    if not events:
        result.errors.append("no events")
        return result

    run_ids = {e["run_id"] for e in events}
    if len(run_ids) > 1:
        result.errors.append(f"more than one run_id: {sorted(run_ids)}")
    for i, e in enumerate(events):
        if e["seq"] != i:
            result.errors.append(f"seq {e['seq']} at position {i} (expected {i})")
            break
    if events[0]["type"] != E.RUN_STARTED:
        result.errors.append(f"first event is {events[0]['type']}, not {E.RUN_STARTED}")
    if sum(e["type"] == E.RUN_STARTED for e in events) != 1:
        result.errors.append("run.started must appear exactly once")
    terminal = [i for i, e in enumerate(events) if e["type"] in E.TERMINAL_EVENTS]
    if len(terminal) != 1 or terminal[0] != len(events) - 1:
        result.errors.append("exactly one run.finished or run.error must be the last event")
    last = events[-1]
    if last["type"] == E.RUN_FINISHED:
        result.status = last["payload"].get("status")
    elif last["type"] == E.RUN_ERROR:
        result.status = "error"

    kinds: dict[str, str] = {}
    tables: dict[str, dict[str, dict[str, float]]] = {}
    beliefs: dict[str, dict[str, float]] = {}
    invalid: set[str] = set()
    for e in events:
        etype, p, seq = e["type"], e["payload"], e["seq"]
        try:
            if etype == E.QUESTION_OPENED:
                q = p["question"]
                kinds[q["id"]] = q["kind"]
            elif etype == E.HYPOTHESIS_MERGED:
                beliefs[p["question_id"]] = {k: float(v) for k, v in p["priors"].items()}
            elif etype == E.EXPERIMENT_PROPOSED:
                exp = p["experiment"]
                tables[exp["id"]] = exp.get("likelihood") or {}
            elif etype == E.REFEREE_VERDICT and not p["valid"]:
                invalid.add(p["experiment_id"])
            elif etype == E.HYPOTHESIS_REJECTED:
                qid = p["question_id"]
                if kinds.get(qid) == "categorical" and qid in beliefs:
                    beliefs[qid].pop(p["hypothesis_id"], None)
                    beliefs[qid] = _renormalise(beliefs[qid])
            elif etype == E.BELIEF_UPDATED:
                result.n_updates += 1
                _check_update(result, seq, p, kinds, tables, beliefs, invalid, tol)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            result.errors.append(f"seq {seq} ({etype}): malformed payload ({exc!r})")
    result.final_beliefs = {qid: b for qid, b in beliefs.items() if kinds.get(qid) == "categorical"}
    return result


def _check_update(
    result: ReplayResult,
    seq: int,
    p: dict[str, Any],
    kinds: dict[str, str],
    tables: dict[str, dict[str, dict[str, float]]],
    beliefs: dict[str, dict[str, float]],
    invalid: set[str],
    tol: float,
) -> None:
    qid, eid = p["question_id"], p["experiment_id"]
    posterior = {k: float(v) for k, v in p["posterior"].items()}
    if eid in invalid:
        result.errors.append(
            f"seq {seq}: belief.updated for {eid}, which the referee ruled invalid"
        )
    if kinds.get(qid) != "categorical":
        values = list(posterior.values())
        if not values or not all(math.isfinite(v) and v >= 0 for v in values):
            result.errors.append(f"seq {seq}: non-finite posterior for {qid}")
        elif abs(sum(values) - 1.0) > 1e-6:
            result.errors.append(f"seq {seq}: posterior for {qid} sums to {sum(values)}")
        return
    if qid not in beliefs:
        result.errors.append(f"seq {seq}: belief.updated for {qid} before its priors")
        return
    prior = {k: float(v) for k, v in p["prior"].items()}
    if not _close(prior, beliefs[qid], tol):
        result.errors.append(f"seq {seq}: recorded prior for {qid} does not match the replay")
    if eid not in tables:
        result.errors.append(f"seq {seq}: no experiment.proposed with a table for {eid}")
        return
    try:
        expected = _bayes(beliefs[qid], tables[eid], str(p["observed"]))
    except KeyError as exc:
        result.errors.append(f"seq {seq}: {eid} has no likelihood row for {exc.args[0]}")
        return
    except ZeroDivisionError as exc:
        result.errors.append(f"seq {seq}: {exc}")
        return
    result.n_checked += 1
    if not _close(expected, posterior, tol):
        worst = max(
            (abs(expected.get(k, 0.0) - posterior.get(k, 0.0)), k)
            for k in set(expected) | set(posterior)
        )
        result.errors.append(
            f"seq {seq}: posterior for {qid} after {eid} differs from the replay "
            f"(worst {worst[1]}: off by {worst[0]:.3g})"
        )
    beliefs[qid] = expected


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m core.replay <events.jsonl>", file=sys.stderr)
        return 2
    result = replay_check(args[0])
    print(result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
