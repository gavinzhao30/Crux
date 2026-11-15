"""Shared helpers for the predictor dataset: the library index, loading splits and prompts.

Records are written by ``eval/predictor_data/make_examples.py`` to
``runs/predictor_data/<split>.jsonl`` (one per case and library diagnostic). Prompts are always
built with ``agents.tinker_predictor.build_prompt`` from the record's raw profile facts, so the
training text is exactly what ``TinkerPredictor`` builds at inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from agents.tinker_predictor import bin_completion, build_prompt
from core.schemas import Experiment, Hypothesis
from diagnostics.de_from_counts import library_questions

DATA_DIR = Path("runs/predictor_data")
SPLITS = ("train", "val", "test", "shift")
SPEND_LEDGER = "spend.json"


@dataclass(frozen=True)
class LibraryDiagnostic:
    fork: str
    question_text: str
    experiment: Experiment
    options: dict[str, Hypothesis]  # answer_key -> option
    null_key: str
    fault_key: str


@cache
def library_index() -> dict[str, LibraryDiagnostic]:
    """Diagnostic script name -> its fork, experiment and options (as the orchestrator has them)."""
    index: dict[str, LibraryDiagnostic] = {}
    for question in library_questions():
        options = {
            o.answer_key: Hypothesis(
                id=f"{question.key}.{o.answer_key}",
                question_id=question.key,
                statement=o.statement,
                author="library",
                rationale=o.rationale,
                answer_key=o.answer_key,
                prior=0.5,
                posterior=0.5,
            )
            for o in question.proposal.options
            if o.answer_key
        }
        keys = [o.answer_key for o in question.proposal.options if o.answer_key]
        for exp in question.diagnostics:
            index[exp.design["diagnostic"]] = LibraryDiagnostic(
                fork=question.key,
                question_text=question.proposal.text,
                experiment=exp,
                options=options,
                null_key=keys[0],
                fault_key=keys[1],
            )
    return index


def load_split(
    name: str, root: Path = DATA_DIR, *, valid_only: bool = True
) -> list[dict]:
    path = Path(root) / f"{name}.jsonl"
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not valid_only or record.get("observed_bin") is not None:
                    records.append(record)
    return records


def record_prompt(record: dict, answer_key: str | None = None) -> str:
    """The prompt for a record's diagnostic under ``answer_key`` (default: the true option)."""
    entry = library_index()[record["diagnostic"]]
    option = entry.options[answer_key or record["answer_key"]]
    return build_prompt(entry.experiment, option, [record["profile"]])


def record_example(record: dict) -> tuple[str, str]:
    """(prompt, completion) for supervised training: the true option and the observed bin."""
    return record_prompt(record), bin_completion(record["observed_bin"])


def read_ledger(root: Path = DATA_DIR) -> dict:
    path = Path(root) / SPEND_LEDGER
    if not path.is_file():
        return {"entries": [], "total_usd": 0.0}
    return json.loads(path.read_text(encoding="utf-8"))


def add_to_ledger(
    root: Path, label: str, usd: float, detail: dict | None = None
) -> dict:
    """Append an (estimated) Tinker/OpenAI spend entry; returns the updated ledger."""
    ledger = read_ledger(root)
    ledger["entries"].append(
        {"label": label, "usd": round(float(usd), 4), **(detail or {})}
    )
    ledger["total_usd"] = round(sum(e["usd"] for e in ledger["entries"]), 4)
    path = Path(root) / SPEND_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return ledger
