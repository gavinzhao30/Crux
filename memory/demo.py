"""Live synthetic integration check, not a scientific benchmark or an LLM run.

Run: uv run python -m memory.demo
Requires PAST_API_KEY in the environment and spends a small amount of provider credit.
"""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from core.schemas import Evidence
from memory.base import RunContext
from memory.pastdev import PastMemory


async def demonstrate(memory: PastMemory, isolated: PastMemory) -> dict:
    """Two writers and two readers, a changed action, and a negative scope check."""
    run_id = memory.context.run_id
    observation = Evidence(
        id="sample-mapping",
        kind="observation",
        author_role="experimenter",
        claim="Samples S1 and S2 are repeated measurements from the same subject.",
        conditions={"input_hash": "synthetic-demo-v1"},
    )
    await memory.write(observation, run_id, "experimenter")
    await memory.flush()
    critic_evidence = await memory.recall("Which samples share a subject?", run_id, "critic")
    if observation.id not in {ev.id for ev in critic_evidence}:
        raise RuntimeError("Critic did not retrieve the experimenter's observation")
    objection = Evidence(
        id="independence-objection",
        kind="objection",
        author_role="critic",
        claim="S1 and S2 share a subject: inspect the subject mapping before an independent fit.",
        source_ref=observation.id,
    )
    await memory.write(objection, run_id, "critic")
    await memory.flush()
    designer_evidence = await memory.recall(
        "What should I check before fitting?", run_id, "designer"
    )
    before = "fit_independent_model"
    after = (
        "inspect_subject_mapping"
        if any(ev.id == objection.id for ev in designer_evidence)
        else before
    )
    if after == before:
        raise RuntimeError("Recalled objection did not change the next action")

    sentinel = Evidence(
        id="isolation-sentinel",
        kind="observation",
        author_role="experimenter",
        claim=f"Private isolation sentinel {uuid.uuid4().hex}",
    )
    await memory.write(sentinel, run_id, "experimenter")
    await memory.flush()
    foreign = await isolated.recall(sentinel.claim, isolated.context.run_id, "critic")
    last_audit = isolated.store.db.execute(
        "SELECT payload FROM audit WHERE operation='memory.recalled' ORDER BY seq DESC LIMIT 1"
    ).fetchone()[0]
    isolation_audit = json.loads(last_audit)
    private_sources = {row["source_id"] for row in memory.store.eligible()}
    if foreign or private_sources.intersection(isolation_audit["source_ids"]):
        raise RuntimeError("Run isolation failed")
    if isolation_audit["remote_status"] != "ok":
        raise RuntimeError("Isolation check did not get a successful remote response")
    return {
        "run_id": run_id,
        "critic_evidence_ids": [ev.id for ev in critic_evidence],
        "designer_evidence_ids": [ev.id for ev in designer_evidence],
        "decision_before": before,
        "decision_after": after,
        "isolation": "passed",
        "mode": "synthetic integration; no scientific result or score gain claimed",
    }


async def main(root: Path) -> None:
    run = "memory-demo-" + uuid.uuid4().hex
    async with (
        PastMemory(RunContext(run_id=run, task_id="synthetic-memory-demo"), root) as memory,
        PastMemory(
            RunContext(run_id=run + "-isolated", task_id="synthetic-memory-demo"), root
        ) as isolated,
    ):
        print(json.dumps(await demonstrate(memory, isolated), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    asyncio.run(main(parser.parse_args().root))
