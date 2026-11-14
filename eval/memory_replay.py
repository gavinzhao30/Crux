"""Replay frozen evidence and decisions through role-local and past.dev memory.

This measures retrieval and a deterministic decision proxy, not scientific accuracy.
No model calls or analysis pipelines are executed. Missing provider credentials yield
an explicit skipped row, never a simulated past.dev measurement.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config import RunConfig
from core.schemas import Evidence
from memory.base import MemoryRecord, RecallUnavailable, RunContext
from memory.factory import create_memory
from memory.instrumentation import MemoryTelemetry, format_memory

ARMS = ("local_role_only", "pastdev")


class ReplayProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    reader_role: str
    query: str = Field(min_length=1)
    history_count: int = Field(ge=0)
    default_action: str = Field(min_length=1)
    kind: Literal["diagnostic", "analysis"] = "diagnostic"
    conditions_hash: str = Field(min_length=1)
    # Evaluation annotations: never provided to memory or the decision policy.
    decisive_evidence_ids: list[str] = Field(default_factory=list)


class FrozenReplay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    synthetic: bool = True
    context: RunContext
    records: list[MemoryRecord]
    probes: list[ReplayProbe] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_replay(self):
        seen_ids, seen_events = set(), set()
        previous_time = None
        for record in self.records:
            if record.run_id != self.context.run_id or record.task_id != self.context.task_id:
                raise ValueError("Frozen records must belong to the declared run/task")
            self.context.identity(record.evidence.author_role)
            if record.evidence.id in seen_ids or record.event_id in seen_events:
                raise ValueError("Frozen record IDs must be unique")
            if not set(record.supersedes) <= seen_ids:
                raise ValueError("Corrections may supersede only earlier evidence")
            if previous_time is not None and record.occurred_at < previous_time:
                raise ValueError("Frozen evidence must be in chronological order")
            previous_time = record.occurred_at
            seen_ids.add(record.evidence.id)
            seen_events.add(record.event_id)
        cutoff, probe_ids = 0, set()
        for probe in self.probes:
            self.context.identity(probe.reader_role)
            if probe.id in probe_ids:
                raise ValueError("Probe IDs must be unique")
            probe_ids.add(probe.id)
            if not cutoff <= probe.history_count <= len(self.records):
                raise ValueError("Probe history cutoffs must increase and stay within history")
            available = {r.evidence.id for r in self.records[: probe.history_count]}
            if not set(probe.decisive_evidence_ids) <= available:
                raise ValueError("A decisive evidence annotation cannot name future evidence")
            cutoff = probe.history_count
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def decide(evidence: list[Evidence], reader_role: str, default_action: str) -> str:
    """Fixed, declared proxy: use an applicable recorded action hint or keep the plan.

    Role-specific hints are part of the permitted source evidence, not evaluation labels.
    Conflicting suggestions require review. Action strings are never executed.
    """
    suggestions = set()
    for ev in evidence:
        by_role = ev.conditions.get("next_action_by_role", {})
        if isinstance(by_role, dict):
            suggestion = by_role.get(reader_role)
            if isinstance(suggestion, str) and suggestion:
                suggestions.add(suggestion)
    if not suggestions:
        return default_action
    return next(iter(suggestions)) if len(suggestions) == 1 else "review_conflicting_evidence"


async def replay_arm(
    snapshot: FrozenReplay,
    arm: str,
    root: Path,
    session_id: str,
    *,
    provider_factory: Callable | None = None,
    provider_mode: Literal["live", "mock_http"] = "live",
    telemetry_factory: Callable[[str], MemoryTelemetry] = MemoryTelemetry,
) -> dict:
    result = {
        "backend": arm,
        "status": "completed",
        "synthetic": snapshot.synthetic,
        "provider_mode": provider_mode if arm == "pastdev" else "local",
        "fixture_sha256": snapshot.fingerprint,
        "decisions": [],
        "summary": None,
    }
    if arm == "pastdev" and not provider_factory and not os.environ.get("PAST_API_KEY"):
        result["status"] = "skipped_missing_key"
        result["provider_mode"] = "unrun"
        return result
    context = snapshot.context.model_copy(
        update={
            "run_id": f"memory-replay-{session_id}-{arm}",
            "config_id": arm,
        }
    )
    backend_label = "pastdev_mock" if arm == "pastdev" and provider_mode == "mock_http" else arm
    telemetry = telemetry_factory(backend_label)
    memory = (
        provider_factory(context, root, telemetry)
        if provider_factory and arm == "pastdev"
        else create_memory(context, RunConfig(memory=arm), root, telemetry=telemetry)
    )
    started = time.perf_counter()
    wait_seconds = 0.0
    applied = 0
    seen_diagnostics = set()
    counters = {
        "duplicate_diagnostics": 0,
        "repeated_rejected_options": 0,
        "decisions_changed_by_another_role": 0,
        "action_changes": 0,
        "decisive_recall_hits": 0,
        "scored_probes": 0,
        "recall_failures": 0,
    }
    try:
        async with memory:
            for probe in snapshot.probes:
                for record in snapshot.records[applied : probe.history_count]:
                    await memory.write_record(
                        record.model_copy(
                            update={
                                "run_id": context.run_id,
                                "task_id": context.task_id,
                            },
                            deep=True,
                        )
                    )
                applied = probe.history_count
                wait_start = time.perf_counter()
                await memory.flush()  # Static replay compares indexed, current records.
                wait_seconds += time.perf_counter() - wait_start
                decision = {
                    "probe_id": probe.id,
                    "reader_role": probe.reader_role,
                    "history_count": probe.history_count,
                    "status": "ok",
                }
                try:
                    evidence = await memory.recall(probe.query, context.run_id, probe.reader_role)
                except RecallUnavailable:
                    decision["status"] = "recall_failed"
                    counters["recall_failures"] += 1
                    result["decisions"].append(decision)
                    continue
                _, metrics = telemetry.events[-1]
                if metrics.get("remote_status") == "failed":
                    # A pending/mandatory fallback is not successful semantic retrieval.
                    result["status"] = "degraded"
                    decision["status"] = "fallback"
                action = decide(evidence, probe.reader_role, probe.default_action)
                own = [ev for ev in evidence if ev.author_role == probe.reader_role]
                own_action = decide(own, probe.reader_role, probe.default_action)
                changed_by_foreign = action != own_action
                ids = {ev.id for ev in evidence}
                decisive = (
                    bool(set(probe.decisive_evidence_ids) <= ids)
                    if probe.decisive_evidence_ids
                    else None
                )
                if decisive is not None:
                    counters["scored_probes"] += 1
                    counters["decisive_recall_hits"] += int(decisive)
                counters["action_changes"] += int(action != probe.default_action)
                counters["decisions_changed_by_another_role"] += int(changed_by_foreign)
                key = (probe.conditions_hash, action)
                if probe.kind == "diagnostic":
                    counters["duplicate_diagnostics"] += int(key in seen_diagnostics)
                    seen_diagnostics.add(key)
                visible_history = snapshot.records[: probe.history_count]
                superseded = {eid for r in visible_history for eid in r.supersedes}
                rejected = {
                    r.evidence.conditions.get("rejects_action")
                    for r in visible_history
                    if r.evidence.id not in superseded
                    if isinstance(r.evidence.conditions.get("rejects_action"), str)
                }
                counters["repeated_rejected_options"] += int(action in rejected)
                decision.update(
                    {
                        "evidence_ids": [ev.id for ev in evidence],
                        "authors": [ev.author_role for ev in evidence],
                        "action": action,
                        "default_action": probe.default_action,
                        "changed_by_foreign_evidence": changed_by_foreign,
                        "decisive_evidence_retrieved": decisive,
                        "prompt": format_memory(evidence),
                        "prompt_tokens": metrics["prompt_tokens"],
                        "token_method": metrics["token_method"],
                        "latency_ms": metrics["latency_ms"],
                    }
                )
                result["decisions"].append(decision)
    except Exception as exc:
        result["status"] = "failed"
        result["error_class"] = type(exc).__name__
    result["summary"] = {
        **telemetry.summary(),
        **counters,
        "ingestion_wait_seconds": wait_seconds,
        "total_wall_seconds": time.perf_counter() - started,
    }
    if counters["recall_failures"]:
        result["status"] = "failed"
    return result


async def run_replay(
    snapshot: FrozenReplay,
    root: Path,
    *,
    arms: tuple[str, ...] = ARMS,
    provider_factory: Callable | None = None,
    provider_mode: Literal["live", "mock_http"] = "live",
    telemetry_factory: Callable[[str], MemoryTelemetry] = MemoryTelemetry,
) -> dict:
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("Choose unique arms from local_role_only,pastdev")
    if (provider_factory is not None) != (provider_mode == "mock_http"):
        raise ValueError("Injected provider factories require the explicit mock_http label")
    session = uuid.uuid4().hex
    results = []
    for arm in arms:
        results.append(
            await replay_arm(
                snapshot,
                arm,
                root,
                session,
                provider_factory=provider_factory,
                provider_mode=provider_mode,
                telemetry_factory=telemetry_factory,
            )
        )
    return {
        "fixture_name": snapshot.name,
        "fixture_sha256": snapshot.fingerprint,
        "synthetic": snapshot.synthetic,
        "decision_policy": "recorded_role_action_hints_v1",
        "measurement": "deterministic context/decision replay; no LLM or pipeline runs",
        "arms": results,
    }


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = [
        "backend",
        "status",
        "provider_mode",
        "scored_probes",
        "decisive_recall_hits",
        "action_changes",
        "decisions_changed_by_another_role",
        "duplicate_diagnostics",
        "repeated_rejected_options",
        "recall_latency_p50_ms",
        "recall_latency_p95_ms",
        "prompt_tokens",
        "token_method",
        "ingestion_wait_seconds",
        "total_wall_seconds",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for arm in report["arms"]:
            writer.writerow({**arm, **(arm["summary"] or {})})
    lines = [
        "# Memory decision replay",
        "",
        report["measurement"],
        "",
        f"Synthetic fixture: {report['synthetic']}. Fixture SHA-256: `{report['fixture_sha256']}`.",
        "",
        "| Backend | Status | Provider | Decisive hits | Cross-role action changes |"
        " p50 / p95 ms | Prompt tokens |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        s = arm["summary"]
        details = (
            [
                f"{s['decisive_recall_hits']}/{s['scored_probes']}",
                str(s["decisions_changed_by_another_role"]),
                " / ".join(
                    f"{value:.3f}" if value is not None else "-"
                    for value in [s["recall_latency_p50_ms"], s["recall_latency_p95_ms"]]
                ),
                str(s["prompt_tokens"]),
            ]
            if s
            else ["-"] * 4
        )
        lines.append(
            "| "
            + " | ".join([arm["backend"], arm["status"], arm["provider_mode"], *details])
            + " |"
        )
    lines += [
        "",
        "Missing credentials produce skipped rows, not shared-memory results. Mock HTTP",
        "results are labeled mock_http. Default prompt counts estimate UTF-8 bytes / 4;",
        "supply a named tokenizer for actual memory-fragment token counts. Full prompts",
        "and per-probe actions are in report.json. No scientific gain is claimed.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main(args) -> int:
    snapshot = FrozenReplay.model_validate_json(args.fixture.read_text(encoding="utf-8"))
    report = await run_replay(snapshot, args.root, arms=tuple(args.arms.split(",")))
    write_report(report, args.output_dir)
    print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))
    return 1 if any(a["status"] == "failed" for a in report["arms"]) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("eval/fixtures/memory_replay.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results/memory-replay"))
    parser.add_argument("--root", type=Path, default=Path("runs/memory-replays"))
    parser.add_argument("--arms", default=",".join(ARMS))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
