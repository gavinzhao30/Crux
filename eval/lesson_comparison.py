"""Paired lessons-on/off evaluation; grades stay here and never enter lesson memory."""

import argparse
import asyncio
import csv
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.config import RunConfig
from core.schemas import Evidence
from memory.instrumentation import format_memory
from memory.lessons import FrozenLessons, LessonMemory


class HeldOutCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    split: Literal["test"]
    input_hash: str = Field(min_length=1)
    public_context: dict[str, str | bool | int | float] = Field(default_factory=dict)
    lesson_query: str = Field(min_length=1)


class RunConditions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model_ids: dict[str, str]
    sandbox_image_id: str
    seed: int = 0
    budget_usd: float = Field(ge=0)

    @property
    def fingerprint(self) -> str:
        body = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: Literal["ok", "failed"]
    result_kind: Literal["measured", "synthetic"] = "measured"
    diagnostics_before_fault: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    official_score: float | None = Field(default=None, ge=0, le=1)
    note: str = ""


Runner = Callable[
    [HeldOutCase, RunConfig, RunConditions, dict[str, list[Evidence]]], Awaitable[CaseResult]
]


async def compare(
    corpus: FrozenLessons,
    cases: list[HeldOutCase],
    conditions: RunConditions,
    runner: Runner,
    root: Path,
    *,
    memory_factory=None,
    allow_synthetic: bool = False,
    live_memory: bool = True,
) -> dict:
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Choose a nonempty set of unique held-out cases")
    if {case.case_id for case in cases} != set(corpus.held_out_case_ids):
        raise ValueError("Comparison cases must exactly match the frozen cohort")
    if corpus.synthetic and not allow_synthetic:
        raise ValueError("Synthetic comparisons need explicit opt-in")
    if not live_memory and not corpus.synthetic:
        raise ValueError("The fixture memory service is for synthetic demonstrations only")
    if (memory_factory is not None) != (not live_memory):
        raise ValueError("Injected memory factories require the fixture-memory label")
    evaluation_started_at = datetime.now(UTC)
    if corpus.frozen_at > evaluation_started_at:
        raise ValueError("Corpus must be frozen before the evaluation starts")
    options = {"memory_factory": memory_factory} if memory_factory else {}
    rows = []
    started = time.perf_counter()
    async with LessonMemory(
        corpus,
        RunConfig(lessons=True),
        root,
        allow_synthetic=allow_synthetic,
        **options,
    ) as lesson_memory:
        prepare_started = time.perf_counter()
        preparation_error = None
        try:
            await lesson_memory.prepare()
        except Exception as exc:
            preparation_error = type(exc).__name__
        preparation_seconds = time.perf_counter() - prepare_started
        for case in cases:
            # Alternate order to avoid always warming the same arm first. The case input,
            # model IDs, image, seed and budget are cloned unchanged for each runner call.
            order = (False, True) if len(rows) // 2 % 2 == 0 else (True, False)
            for enabled in order:
                config = RunConfig(lessons=enabled)
                row = {
                    "case_id": case.case_id,
                    "lessons": enabled,
                    "corpus_sha256": corpus.digest,
                    "conditions_sha256": conditions.fingerprint,
                    "input_hash": case.input_hash,
                    "status": "failed",
                }
                begun = time.perf_counter()
                if enabled and preparation_error:
                    row["error_class"] = preparation_error
                    row["seconds"] = time.perf_counter() - begun
                    rows.append(row)
                    continue
                try:
                    contexts = {}
                    if enabled:
                        for role in ("proposer", "designer"):
                            contexts[role] = await lesson_memory.recall(
                                case.lesson_query,
                                case_id=case.case_id,
                                role=role,
                                context=case.public_context,
                            )
                    else:
                        contexts = {"proposer": [], "designer": []}
                    row["lesson_ids"] = {
                        role: [ev.id for ev in evidence] for role, evidence in contexts.items()
                    }
                    row["memory_prompt_tokens_estimate"] = sum(
                        (len(format_memory(evidence).encode("utf-8")) + 3) // 4
                        for evidence in contexts.values()
                    )
                    local_conditions = conditions.model_copy(deep=True)
                    local_case = case.model_copy(deep=True)
                    original_case = local_case.model_dump_json()
                    result = await runner(local_case, config, local_conditions, contexts)
                    if not isinstance(result, CaseResult):
                        raise TypeError("Runners must return a validated CaseResult")
                    if (
                        local_conditions.fingerprint != conditions.fingerprint
                        or local_case.model_dump_json() != original_case
                        or config.lessons != enabled
                    ):
                        raise ValueError("Runner changed the fixed comparison inputs")
                    if corpus.synthetic and result.result_kind != "synthetic":
                        raise ValueError(
                            "Synthetic fixtures cannot report measured scientific results"
                        )
                    row.update(result.model_dump())
                except Exception as exc:
                    row["error_class"] = type(exc).__name__
                row["seconds"] = time.perf_counter() - begun
                rows.append(row)
    pairs = []
    for case in cases:
        pair = {row["lessons"]: row for row in rows if row["case_id"] == case.case_id}
        on, off = pair[True], pair[False]
        delta = {}
        for field in ("diagnostics_before_fault", "cost_usd", "official_score"):
            comparable = (
                on["status"] == off["status"] == "ok"
                and on.get(field) is not None
                and off.get(field) is not None
            )
            delta[field + "_delta_on_minus_off"] = on[field] - off[field] if comparable else None
        pairs.append({"case_id": case.case_id, **delta})
    return {
        "synthetic": corpus.synthetic,
        "memory_service": "pastdev_live" if live_memory else "fixture_memory",
        "corpus_sha256": corpus.digest,
        "frozen_at": corpus.frozen_at.isoformat(),
        "evaluation_started_at": evaluation_started_at.isoformat(),
        "lesson_events": lesson_memory.events,
        "conditions": conditions.model_dump(),
        "conditions_sha256": conditions.fingerprint,
        "corpus_preparation_seconds": preparation_seconds,
        "corpus_preparation_error_class": preparation_error,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
        "paired_deltas": pairs,
        "cost_scope": "case-runner cost only; corpus preparation/retrieval billing is separate",
        "prompt_count_method": "estimate_utf8_bytes_div4; memory fragments only",
    }


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = [
        "case_id",
        "lessons",
        "status",
        "result_kind",
        "diagnostics_before_fault",
        "cost_usd",
        "official_score",
        "memory_prompt_tokens_estimate",
        "seconds",
        "corpus_sha256",
        "conditions_sha256",
        "error_class",
    ]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    lines = [
        "# Lessons-on/off comparison",
        "",
        f"Synthetic: {report['synthetic']}. Memory service: {report['memory_service']}.",
        "",
        "| Case | Lessons | Status | Diagnostics to fault | Runner cost | Official score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        cells = [row["case_id"], "on" if row["lessons"] else "off", row["status"]]
        cells += [
            str(row.get(field)) if row.get(field) is not None else "not measured"
            for field in ("diagnostics_before_fault", "cost_usd", "official_score")
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Missing scores and diagnostics remain unmeasured. Synthetic context checks do not",
        "establish scientific improvement. Corpus preparation/retrieval costs are separate",
        "from the case-runner cost. Full records and paired deltas are in report.json.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FixtureMemory:
    """Explicit context-plumbing fake, never called or labeled a live memory service."""

    def __init__(self, *_args, **_kwargs):
        self.records = {}

    async def write_record(self, record):
        self.records[record.evidence.id] = record.evidence

    async def flush(self):
        return None

    async def recall(self, _query, _run_id, _role, k=8):
        return list(self.records.values())[:k]

    async def aclose(self):
        return None


async def _context_demo(case, config, _conditions, contexts):
    ids = {role: [ev.id for ev in evidence] for role, evidence in contexts.items()}
    valid = all(contexts.values()) if config.lessons else not any(contexts.values())
    return CaseResult(
        status="ok" if valid else "failed",
        result_kind="synthetic",
        note=f"Context-only check for {case.case_id}, lessons={config.lessons}, records={ids}; "
        "no model, pipeline, diagnostic-count or official-grade measurement",
    )


async def main(args):
    corpus = FrozenLessons.model_validate_json(args.fixture.read_text(encoding="utf-8"))
    if not corpus.synthetic:
        raise ValueError(
            "The built-in CLI is a synthetic context demo; use compare() for real runners"
        )
    frozen_path = corpus.save(args.output_dir / "frozen")
    cases = [
        HeldOutCase(
            case_id=case,
            split="test",
            input_hash="synthetic-input-v1",
            public_context={"analysis": "de_from_counts"},
            lesson_query="Which preflight checks should precede differential expression?",
        )
        for case in corpus.held_out_case_ids
    ]
    report = await compare(
        FrozenLessons.load(frozen_path),
        cases,
        RunConditions(model_ids={}, sandbox_image_id="no-pipeline", budget_usd=0),
        _context_demo,
        args.root,
        allow_synthetic=True,
        live_memory=args.live,
        memory_factory=None if args.live else _FixtureMemory,
    )
    write_report(report, args.output_dir)
    print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))
    return int(any(row["status"] != "ok" for row in report["rows"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("eval/fixtures/reviewed_lessons.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results/lessons"))
    parser.add_argument("--root", type=Path, default=Path("runs/lesson-comparisons"))
    parser.add_argument(
        "--live", action="store_true", help="Use real past.dev with the synthetic corpus"
    )
    raise SystemExit(asyncio.run(main(parser.parse_args())))
