"""Shared memory in real runs: the run-scoped adapter, the Task 9 wiring and the challenge.

No network: past.dev is an HTTP-contract fake with audience-scoped readers. The checks
are the past.dev challenge's: two or more roles write, two or more read, a role's prompt
gets a record another role wrote, and recalled records never update beliefs twice.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from agents.swarm import NullMemory, SwarmRoles
from core.config import load_config
from core.fakes_engine import FakeClock, FakeLab, FakeMemory, fake_deps
from core.orchestrator import run
from core.schemas import CriticFinding, Evidence
from labs.bixbench3.swarm_lab import TASK_ID, build_task9_run
from memory.base import RunContext
from memory.pastdev import PastMemory
from memory.run import MAX_CLAIM_CHARS, RunMemory, memory_from_config
from tests.test_task9_wiring import (
    FakeLLM,
    Planner,
    make_case,
    referee_ok,
    report_plain,
    scripted_propose,
)

KEY = "fake-project-key"


class PastService:
    """past.dev's ingest/poll/recall contract with server-side audience membership."""

    def __init__(self) -> None:
        self.audiences: dict[str, list[str]] = {}
        self.ingestions: dict[str, dict] = {}
        self.sources: dict[str, dict] = {}
        self.recalls: list[tuple[str, list[str]]] = []
        self.log: list[str] = []
        self.polls: dict[str, int] = {}
        self.polls_before_done = 0
        self.fail_recall = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        body = json.loads(request.content) if request.content else None
        path = request.url.path.removeprefix("/api/v1")
        self.log.append(f"{request.method} {path}")
        if path.startswith("/audiences/"):
            self.audiences[path.split("/")[-1]] = body["identities"]
            return httpx.Response(201, json={"slug": path.split("/")[-1]})
        if path == "/ingest":
            assert len(body["content"]) < 16000
            ingestion = f"ingestion-{len(self.ingestions)}"
            self.ingestions[ingestion] = body
            return httpx.Response(202, json={"ingestionId": ingestion, "status": "queued"})
        if path.startswith("/ingest/"):
            ingestion = path.split("/")[-1]
            self.polls[ingestion] = self.polls.get(ingestion, 0) + 1
            if self.polls[ingestion] <= self.polls_before_done:
                return httpx.Response(200, json={"status": "processing"})
            item = self.ingestions[ingestion]
            self.sources[item["id"]] = item
            return httpx.Response(200, json={"status": "completed"})
        if path == "/recall":
            if self.fail_recall:
                return httpx.Response(503, json={"error": "unavailable"})
            words = set(re.findall(r"\w+", body["query"].lower()))
            allowed = [
                s
                for s in self.sources.values()
                if body["identity"] in self.audiences.get(s["audience"], [])
            ]
            allowed.sort(key=lambda s: -len(words & set(re.findall(r"\w+", s["content"].lower()))))
            hits = [s["id"] for s in allowed[: body["limit"]]]
            self.recalls.append((body["identity"], hits))
            return httpx.Response(
                200, json={"results": [{"sources": [{"sourceId": h}]} for h in hits]}
            )
        raise AssertionError(f"unexpected route {request.method} {path}")

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


def of_type(path: Path, etype: str) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [e["payload"] for e in map(json.loads, lines) if e["type"] == etype]


def cross_role(recalls: list[dict]) -> list[dict]:
    return [p for p in recalls if any(a != p["reader_role"] for a in p["authors"])]


# --- choosing the backend -----------------------------------------------------------------


def test_memory_from_config_picks_the_backend_and_falls_back_without_a_key(tmp_path):
    cfg = load_config("default", runs_dir=str(tmp_path))
    assert memory_from_config(cfg, task_id="t", env={}) is None  # no key, no memory
    shared = memory_from_config(cfg, task_id="t", env={"PAST_API_KEY": KEY})
    assert isinstance(shared, RunMemory) and shared.backend == "pastdev"
    assert shared.options["timeout_s"] == 15.0 and shared.runs_dir == tmp_path
    local = memory_from_config(
        load_config("crux_role_only_memory", runs_dir=str(tmp_path)), task_id="t", env={}
    )
    assert isinstance(local, RunMemory) and local.backend == "local_role_only"
    assert local.config_id == "crux_role_only_memory"
    vector = load_config("default", runs_dir=str(tmp_path), memory="vector")
    assert memory_from_config(vector, task_id="t", env={"PAST_API_KEY": KEY}) is None


def test_build_task9_run_builds_memory_from_the_config(tmp_path, monkeypatch):
    case = make_case(tmp_path / "case")
    cfg = load_config("default", runs_dir=str(tmp_path / "runs"))
    monkeypatch.delenv("EXA_API_KEY", raising=False)  # no live search in tests
    monkeypatch.delenv("PAST_API_KEY", raising=False)
    lab, deps = build_task9_run(cfg, case_dir=case, llm=FakeLLM(["none"]))
    assert deps.memory is None and lab.memory is None
    assert isinstance(deps.agents.memory, NullMemory)

    monkeypatch.setenv("PAST_API_KEY", KEY)
    lab, deps = build_task9_run(cfg, case_dir=case, llm=FakeLLM(["none"]))
    assert isinstance(deps.memory, RunMemory) and deps.memory.backend == "pastdev"
    assert deps.agents.memory is deps.memory and lab.memory is deps.memory
    assert deps.memory.task_id == TASK_ID and deps.memory.memory is None  # opened lazily
    lab.bind("r-bound", tmp_path / "runs" / "r-bound")
    assert deps.agents.run_id() == "r-bound"

    monkeypatch.delenv("PAST_API_KEY")
    local_cfg = load_config("crux_role_only_memory", runs_dir=str(tmp_path / "runs"))
    _, deps = build_task9_run(local_cfg, case_dir=case, llm=FakeLLM(["none"]))
    assert deps.memory.backend == "local_role_only"

    given = FakeMemory()
    lab, deps = build_task9_run(cfg, case_dir=case, llm=FakeLLM(["none"]), memory=given)
    assert deps.memory is given and deps.agents.memory is given
    assert lab.memory is None  # the caller owns a memory it passed in


# --- the run-scoped adapter ---------------------------------------------------------------


def test_run_memory_binds_to_one_run_and_never_raises_into_it(tmp_path):
    async def scenario():
        memory = RunMemory("local_role_only", tmp_path, task_id="t")
        big = Evidence(id="p1", kind="profile", claim="x" * 20000, author_role="experimenter")
        await memory.write(big, "r1", "experimenter")
        [kept] = await memory.recall("profile", "r1", "experimenter")
        assert kept.id == "p1" and len(kept.claim) <= MAX_CLAIM_CHARS
        assert kept.conditions["claim_chars"] == 20000 and "full record p1" in kept.claim
        assert (tmp_path / "r1" / "memory" / "local_role_only").is_dir()

        # another run, an undeclared role: logged and audited, never raised
        assert await memory.recall("profile", "r2", "experimenter") == []
        odd = Evidence(id="o1", kind="observation", claim="c", author_role="intern")
        await memory.write(odd, "r1", "intern")
        assert [e["operation"] for e in memory.errors] == ["bind", "write"]
        audit = memory.memory.store.db.execute(
            "SELECT count(*) FROM audit WHERE operation='memory.degraded'"
        ).fetchone()[0]
        assert audit == 2
        summary = memory.summary()
        assert summary["write_calls"] == 1 and summary["errors"] == 2
        await memory.aclose()
        await memory.aclose()
        assert await memory.recall("profile", "r1", "experimenter") == []

    asyncio.run(scenario())


def test_failed_past_recall_gives_no_context_instead_of_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PAST_API_KEY", KEY)
    service = PastService()
    service.fail_recall = True

    async def scenario():
        async with service.client() as client:
            memory = RunMemory(
                "pastdev", tmp_path, task_id="t", client=client, poll_interval_s=0
            )
            assert await memory.recall("anything", "r1", "critic") == []
            assert memory.errors[0]["operation"] == "recall"
            assert "RecallUnavailable" in memory.errors[0]["error"]
            await memory.aclose()

    asyncio.run(scenario())


def test_records_are_sent_before_any_poll_and_close_sends_without_waiting(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PAST_API_KEY", KEY)
    service = PastService()
    service.polls_before_done = 2  # indexing takes a while

    def record(i: int) -> Evidence:
        return Evidence(id=f"e{i}", kind="observation", claim=f"claim {i}", author_role="referee")

    async def scenario():
        async with service.client() as client:
            options = {"client": client, "poll_interval_s": 0, "background": False}
            memory = PastMemory(RunContext(run_id="r1", task_id="t"), tmp_path, **options)
            for i in range(3):
                await memory.write(record(i), "r1", "referee")
            await memory.flush()
            ingest = [op for op in service.log if "/ingest" in op]
            first_poll = next(i for i, op in enumerate(ingest) if op.startswith("GET"))
            assert ingest[:first_poll] == ["POST /ingest"] * 3
            await memory.aclose()

            run_memory = RunMemory("pastdev", tmp_path, task_id="t", **options)
            for i in range(3, 5):
                await run_memory.write(record(i), "r2", "referee")
            before = list(service.log)
            await run_memory.aclose()
            sent = service.log[len(before) :]
            assert sent.count("POST /ingest") == 2 and not any(op.startswith("GET") for op in sent)
            assert len(service.ingestions) == 5 and not run_memory.errors

    asyncio.run(scenario())


# --- the orchestrator on the offline fakes --------------------------------------------------


def test_fake_run_shares_records_across_roles_and_never_counts_them_twice(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PAST_API_KEY", KEY)
    service = PastService()
    cfg = load_config("default", runs_dir=str(tmp_path))

    async def go(run_id: str, memory):
        lab = FakeLab()
        deps = fake_deps(lab, clock=FakeClock())
        deps.memory = memory
        await run(lab, cfg, deps, run_id=run_id)
        return deps

    async def scenario():
        async with service.client() as client:
            memory = RunMemory("pastdev", tmp_path, task_id="fake", client=client)
            deps = await go("r-mem", memory)
            await memory.aclose()
            await go("r-none", None)
        return deps

    deps = asyncio.run(scenario())
    events = tmp_path / "r-mem" / "events.jsonl"
    writers = {p["author_role"] for p in of_type(events, "memory.written")}
    assert {"lab", "referee", "critic"} <= writers
    recalls = of_type(events, "memory.recalled")
    assert {"designer", "critic"} <= {p["reader_role"] for p in recalls}
    assert all(p["backend"] == "pastdev" for p in recalls)
    assert any(p["reader_role"] == "designer" for p in cross_role(recalls))
    assert any(p["reader_role"] == "critic" for p in cross_role(recalls))
    # the critic's round-1 block is its own objection, recalled again in round 2
    critic_reads = [p for p in recalls if p["reader_role"] == "critic"]
    assert "critic" not in critic_reads[0]["authors"] and "critic" in critic_reads[1]["authors"]
    # recalled records reach prompts only: the beliefs match a run without memory
    baseline = tmp_path / "r-none" / "events.jsonl"
    assert of_type(events, "belief.updated") == of_type(baseline, "belief.updated")
    assert not of_type(baseline, "memory.recalled")
    # every record reached this run's audience exactly once, and readers got them back
    assert len(service.ingestions) == len(of_type(events, "memory.written"))
    assert service.sources and len(service.audiences) == 1
    assert not deps.memory.errors


def test_role_only_memory_hides_every_other_roles_record(tmp_path):
    cfg = load_config("crux_role_only_memory", runs_dir=str(tmp_path))
    lab = FakeLab()
    deps = fake_deps(lab, clock=FakeClock())
    deps.memory = RunMemory("local_role_only", tmp_path, task_id="fake")

    async def scenario():
        await run(lab, cfg, deps, run_id="r-local")
        await deps.memory.aclose()

    asyncio.run(scenario())
    recalls = of_type(tmp_path / "r-local" / "events.jsonl", "memory.recalled")
    assert recalls and not cross_role(recalls)
    assert any(p["reader_role"] == "critic" and p["authors"] for p in recalls)


# --- the Task 9 run, end to end ----------------------------------------------------------


@pytest.mark.parametrize("lessons", [False, True])
def test_task9_run_meets_the_memory_challenge(tmp_path, monkeypatch, lessons):
    monkeypatch.delenv("EXA_API_KEY", raising=False)  # no live search in tests
    monkeypatch.setenv("PAST_API_KEY", KEY)
    service = PastService()
    case = make_case(tmp_path / "case")
    cfg = load_config("default", runs_dir=str(tmp_path / "runs"), lessons=lessons)
    llm = FakeLLM(["none"])
    lab, deps = build_task9_run(cfg, case_dir=case, llm=llm)
    roles: SwarmRoles = deps.agents
    proposer_reads: list[tuple[str, list[Evidence], list[str]]] = []
    contracts: list[dict] = []

    async def propose(brief, evidence, *, llm, memory, bus, config, library):
        # what agents.proposer.propose does for each proposer role
        found = await memory.recall(brief.get("goal", ""), bus.run_id, "statistician")
        proposer_reads.append((bus.run_id, found, [e.id for e in evidence]))
        return await scripted_propose(
            brief, evidence, llm=llm, memory=memory, bus=bus, config=config, library=library
        )

    async def critic(llm, artifacts, contract, *, counts_path=None, brief=None):
        contracts.append(contract)
        if len(contracts) > 1:
            return []
        return [
            CriticFinding(
                id="methods-versioned-ids",
                artifact="METHODS.md",
                severity="block",
                rule="versioned_gene_ids",
                message="Gene IDs keep their version suffix",
            )
        ]

    roles.propose_fn, roles.library = propose, (lambda brief: [])
    roles.referee_fn, roles.critic_fn, roles.report_fn = referee_ok, critic, report_plain
    lab.planner = Planner()

    async def scenario():
        async with service.client() as client:
            deps.memory.options.update(client=client, poll_interval_s=0)
            await run(lab, cfg, deps, run_id="r-task9")

    asyncio.run(scenario())
    events = tmp_path / "runs" / "r-task9" / "events.jsonl"
    assert of_type(events, "run.finished")[0]["status"] == "ok"

    # the proposers recall under the real run id; the profile they already hold is dropped
    [(run_id, found, profile_ids)] = proposer_reads
    assert run_id == "r-task9" and profile_ids
    assert not {e.id for e in found} & set(profile_ids)

    # two or more roles write
    writers = {p["author_role"] for p in of_type(events, "memory.written")}
    assert {"experimenter", "referee", "critic"} <= writers
    assert ("reporter" in writers) == lessons
    # two or more read, and the critic's prompt input holds records other roles wrote
    recalls = of_type(events, "memory.recalled")
    assert {"designer", "critic"} <= {p["reader_role"] for p in recalls}
    assert any(p["reader_role"] == "critic" for p in cross_role(recalls))
    first, second = contracts
    authors = {e["author_role"] for e in first["recalled_evidence"]}
    assert authors - {"critic"} and {"experimenter", "referee"} & authors
    assert any(
        e["author_role"] == "critic" and e["kind"] == "objection"
        for e in second["recalled_evidence"]
    )
    # the journal lives with the run, everything reached this run's audience, and the
    # memory closed with the lab
    assert list((tmp_path / "runs" / "r-task9" / "memory").glob("crux-*/memory.sqlite3"))
    assert len(service.ingestions) == len(of_type(events, "memory.written"))
    assert len(service.audiences) == 1
    assert lab.memory is None and deps.memory._closed and not deps.memory.errors
