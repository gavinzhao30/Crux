"""One Letter engine: EIG over strip windows, surprise, replicate, regeneration, lessons,
verdict fallback, the reader's /show, the event contract, the launcher and the CLI.

Every run here uses the fake reader (labs/oneletter/fake_reader.py), served in process
through httpx's ASGI transport, except the CLI test, which serves it over real HTTP.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from core import scoring
from core.replay import replay_check
from core.schemas import Experiment, Hypothesis, Observation, Question, Report
from labs.oneletter import variants as V
from labs.oneletter.fake_reader import FakeReader, Misread, create_app
from labs.oneletter.lab import (
    EVENT_TYPES,
    NULL_ID,
    HttpReader,
    OneLetterConfig,
    run_oneletter,
    window_table,
)

DEMO = Path(__file__).resolve().parents[1] / "runs" / "demo" / "oneletter_fake.jsonl"

DNA_KEYS = {
    "dna.request": {"window", "reason", "eig_bits", "replicate"},
    "dna.read": {
        "experiment_id",
        "window",
        "bases",
        "expected",
        "mismatch_positions",
        "confidence",
    },
    "dna.variant": {
        "hypothesis_id",
        "name",
        "hgvs",
        "legacy",
        "rsid",
        "codon",
        "amino_acid",
    },
    "dna.verdict": {
        "hypothesis_id",
        "name",
        "posterior",
        "reads",
        "text",
        "author",
        "latency_ms",
        "pdb_id",
        "residue",
    },
    "dna.lesson": {"added", "lessons_total"},
}


def run_fake(
    tmp_path: Path,
    fake: FakeReader,
    *,
    run_id: str = "r-ol",
    lessons: Path | None = None,
    on_event=None,
    **kw,
) -> tuple[list, list[dict]]:
    cfg = OneLetterConfig(
        reader_url="http://fake-reader",
        poll_s=0.0,
        timeout_s=kw.pop("timeout_s", 5.0),
        replicate_gap_s=0.0,
        lessons_path=str(lessons or tmp_path / "lessons.jsonl"),
        fake_reader=True,
        **kw,
    )
    reader = HttpReader("http://fake-reader", transport=httpx.ASGITransport(app=create_app(fake)))
    results = asyncio.run(
        run_oneletter(cfg, run_id=run_id, runs_dir=tmp_path, reader=reader, on_event=on_event)
    )
    path = tmp_path / run_id / "events.jsonl"
    return results, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def of(events: list[dict], type_: str) -> list[dict]:
    return [e["payload"] for e in events if e["type"] == type_]


def index(events: list[dict], type_: str, pred=lambda p: True, start: int = 0) -> int:
    for i, e in enumerate(events[start:], start):
        if e["type"] == type_ and pred(e["payload"]):
            return i
    raise AssertionError(f"no {type_} after {start}")


# --- the data ------------------------------------------------------------------------------


def test_reference_windows_and_variants_match_the_contract():
    assert V.REFERENCE == "ATGGTGCATCTGACTCCTGAGGAGAAG"
    assert [V.reference_window(w) for w in (1, 2, 3)] == [
        "ATGGTGCAT",
        "CTGACTCCT",
        "GAGGAGAAG",
    ]
    assert V.HBS.window(3) == "GTGGAGAAG" and V.HBC.window(3) == "AAGGAGAAG"
    for v in (V.HBS, V.HBC):
        assert v.window(1) == "ATGGTGCAT" and v.window(2) == "CTGACTCCT"
    assert (V.HBS.rsid, V.HBS.legacy, V.HBS.codon, V.HBS.amino_acid) == (
        "rs334",
        "Glu6Val",
        "GTG",
        "Val",
    )
    assert V.HBS.hgvs == "NM_000518.5:c.20A>T (p.Glu7Val)"
    assert V.HBC.hgvs == "NM_000518.5:c.19G>A (p.Glu7Lys)" and V.HBC.rsid == "rs33930165"
    assert [V.translate(c) for c in ("GAG", "GTG", "AAG", "ATG", "TAA")] == [
        "Glu",
        "Val",
        "Lys",
        "Met",
        "Ter",
    ]
    assert V.HBS.pdb_id == "2HBS" and V.HBA.pdb_id == V.HBC.pdb_id == "1A3N"
    novel = V.novel_variant(3, "GCGGAGAAG")
    assert novel is not None and novel.legacy == "Glu6Ala" and novel.rsid == ""
    assert V.novel_variant(3, "TTTTTTTTT") is None  # too far from HBB to name


def test_windows_1_and_2_are_worth_zero_bits_between_hba_and_hbs():
    for real in ([V.HBA, V.HBS], [V.HBA, V.HBS, V.HBC]):
        prior = {v.hypothesis_id: 0.9 / len(real) for v in real} | {NULL_ID: 0.1}
        bits = {}
        for w in (1, 2, 3):
            _, lik = window_table(w, real, 0.02)
            bits[w] = scoring.expected_info_gain(prior, lik, null_ids={NULL_ID})
        assert bits[1] == pytest.approx(0.0, abs=1e-12)
        assert bits[2] == pytest.approx(0.0, abs=1e-12)
        assert bits[3] > 0.5


def test_eig_picks_window_3_first(tmp_path):
    _, events = run_fake(tmp_path, FakeReader("hbs"))
    first = of(events, "experiment.scored")[0]
    assert first["chosen"] == "window-3"
    bits = {r["experiment_id"]: r["eig_bits"] for r in first["ranking"]}
    assert bits["window-1"] == pytest.approx(0.0, abs=1e-12)
    assert bits["window-2"] == pytest.approx(0.0, abs=1e-12)
    assert bits["window-3"] > 0.5
    assert of(events, "dna.request")[0]["window"] == 3


# --- runs ----------------------------------------------------------------------------------


def test_hbs_settles_in_one_or_two_reads(tmp_path):
    results, events = run_fake(tmp_path, FakeReader("hbs"))
    assert [r.outcome for r in results] == ["settled"]
    assert results[0].hypothesis_id == "h-hbs" and results[0].posterior >= 0.95
    reads = of(events, "dna.read")
    assert 1 <= len(reads) <= 2 and all(r["bases"] == "GTGGAGAAG" for r in reads)
    assert not of(events, "surprise.raised")
    verdict = of(events, "dna.verdict")[0]
    assert verdict["pdb_id"] == "2HBS" and verdict["residue"] == 6
    assert of(events, "question.settled") == [
        {
            "question_id": "q-variant",
            "hypothesis_id": "h-hbs",
            "posterior": verdict["posterior"],
        }
    ]
    assert events[-1]["type"] == "run.finished" and events[-1]["payload"]["status"] == "ok"


def test_hbc_swap_raises_a_surprise_then_replicates_regenerates_and_settles(tmp_path):
    fake = FakeReader("hbs", swaps=["hbc"])
    lessons = tmp_path / "lessons.jsonl"
    results, events = run_fake(tmp_path, fake, lessons=lessons, watch=True, watch_timeout_s=0.3)
    assert [(r.outcome, r.hypothesis_id) for r in results] == [
        ("settled", "h-hbs"),
        ("settled", "h-hbc"),
    ]
    first_settle = index(events, "question.settled")
    reopen = index(events, "question.opened", lambda p: p.get("reopened"), first_settle)
    surprise = index(events, "surprise.raised", start=reopen)
    replicate = index(events, "dna.request", lambda p: p["replicate"], surprise)
    regen = index(events, "hypothesis.proposed", lambda p: p["hypothesis"]["id"] == "h-hbc")
    settled = index(events, "question.settled", lambda p: p["hypothesis_id"] == "h-hbc")
    assert first_settle < reopen < surprise < replicate < regen < settled
    assert events[regen + 1]["type"] == "dna.variant"
    assert events[regen + 1]["payload"]["rsid"] == "rs33930165"
    assert of(events, "surprise.raised")[0]["bases"] == "AAGGAGAAG"
    assert of(events, "surprise.raised")[0]["mismatch_positions"] == [0]
    verdicts = of(events, "dna.verdict")
    assert [(v["hypothesis_id"], v["pdb_id"]) for v in verdicts] == [
        ("h-hbs", "2HBS"),
        ("h-hbc", "1A3N"),
    ]
    assert of(events, "dna.lesson") == [{"added": ["h-hbc"], "lessons_total": 1}]
    assert [v.key for v in V.load_lessons(lessons)] == ["hbc"]
    # red on the surprise, green on each settle, cleared when the new sample was seen
    colors = [o.get("color", "clear" if o.get("clear") else None) for o in fake.overlays]
    assert colors.index("red") < len(colors) - 1 - colors[::-1].index("green")
    assert {"clear", "red", "green"} <= set(colors)
    assert replay_check(tmp_path / "r-ol" / "events.jsonl").ok


def test_a_single_misread_corrected_by_the_replicate_does_not_regenerate(tmp_path):
    fake = FakeReader("hbs", misreads=[Misread.parse("hbs:3:1:GTGGAGAAC")])
    lessons = tmp_path / "lessons.jsonl"
    results, events = run_fake(tmp_path, fake, lessons=lessons)
    assert results[0].outcome == "settled" and results[0].hypothesis_id == "h-hbs"
    assert len(of(events, "surprise.raised")) == 1
    second = of(events, "dna.request")[1]
    assert second["replicate"] is True and "misread" in second["reason"]
    assert [p["hypothesis"]["id"] for p in of(events, "hypothesis.proposed")] == [
        "h-hba",
        "h-hbs",
        "h0",
    ]
    assert not of(events, "dna.lesson") and not lessons.exists()
    assert [o.get("color") for o in fake.overlays if o.get("color")][:2] == [
        "red",
        "yellow",
    ]
    assert replay_check(tmp_path / "r-ol" / "events.jsonl").ok


def test_the_lesson_makes_the_next_run_start_with_hbc(tmp_path):
    lessons = tmp_path / "lessons.jsonl"
    _, first = run_fake(tmp_path, FakeReader("hbc"), run_id="r-1", lessons=lessons)
    assert of(first, "dna.lesson") == [{"added": ["h-hbc"], "lessons_total": 1}]
    results, events = run_fake(tmp_path, FakeReader("hbc"), run_id="r-2", lessons=lessons)
    assert events[0]["payload"]["config"]["lessons_loaded"] == ["h-hbc"]
    first_scored = index(events, "experiment.scored")
    proposed = [
        e["payload"]["hypothesis"]["id"]
        for e in events[:first_scored]
        if e["type"] == "hypothesis.proposed"
    ]
    assert proposed == ["h-hba", "h-hbs", "h-hbc", "h0"]
    assert of(events, "hypothesis.proposed")[2]["hypothesis"]["author"] == "lesson"
    assert of(events, "experiment.scored")[0]["chosen"] == "window-3"
    assert not of(events, "surprise.raised") and not of(events, "dna.lesson")
    assert results[0].hypothesis_id == "h-hbc" and len(results[0].reads) <= 2
    assert len(V.load_lessons(lessons)) == 1


def test_the_verdict_falls_back_to_the_template_when_verdict_fails(tmp_path):
    fake = FakeReader("hbs", verdict_mode="down")
    _, events = run_fake(tmp_path, fake, run_id="r-down")
    v = of(events, "dna.verdict")[0]
    assert v["author"] == "template" and v["latency_ms"] is None
    assert "GTG (Val)" in v["text"] and "rs334" in v["text"]
    assert "GTGGAGAAG" in fake.prompts[0] and "rs334" in fake.prompts[0]

    ok = FakeReader("hbs", verdict_mode="ok")
    _, events = run_fake(tmp_path, ok, run_id="r-ok")
    v = of(events, "dna.verdict")[0]
    assert v["author"] == "jetson/fake-model" and v["latency_ms"] == 1.0
    assert v["text"].startswith("Fake verdict") and "rs334" in v["text"]
    assert [c for c in ok.calls if c[0] == "verdict"] == [("verdict", "ok")]


def test_a_model_verdict_with_a_wrong_fact_is_retried_then_replaced_by_the_template(tmp_path):
    wrong = FakeReader("hbs", verdict_mode="wrong")
    _, events = run_fake(tmp_path, wrong, run_id="r-wrong")
    v = of(events, "dna.verdict")[0]
    assert v["author"] == "template" and v["latency_ms"] is None
    assert "GTG (Val)" in v["text"] and "GAG (Val)" not in v["text"]
    assert [c for c in wrong.calls if c[0] == "verdict"] == [("verdict", "wrong")] * 2
    lines = [p["line"] for p in of(events, "experiment.log") if p["experiment_id"] == "verdict"]
    assert sum("rejected by the fact check" in ln and "GAG with Val" in ln for ln in lines) == 2
    assert lines[-1].startswith("verdict from the template")


def test_the_engine_posts_show_before_polling_for_each_request(tmp_path):
    fake = FakeReader("hbs", misreads=[Misread.parse("hbs:3:1:GTGGAGAAC")])

    def mark(event):
        if event.type == "dna.request":
            fake.calls.append(("request", event.payload["window"]))

    run_fake(tmp_path, fake, on_event=mark)
    calls = [c for c in fake.calls if c[0] in ("request", "show", "read")]
    requests = [i for i, c in enumerate(calls) if c[0] == "request"]
    assert len(requests) >= 2
    for i in requests:
        assert calls[i + 1] == ("show", calls[i][1])  # /show before any /read
    assert sum(c[0] == "show" for c in calls) == len(requests)


def test_a_novel_change_is_named_from_the_genetic_code_and_reported_unknown(tmp_path):
    lessons = tmp_path / "lessons.jsonl"
    results, events = run_fake(
        tmp_path, FakeReader(V.strip_sequence(3, "GCGGAGAAG")), lessons=lessons
    )
    assert results[0].outcome == "settled" and results[0].hypothesis_id == "h-glu6ala"
    variant = next(p for p in of(events, "dna.variant") if p["hypothesis_id"] == "h-glu6ala")
    assert variant["name"] == "Unknown variant" and variant["legacy"] == "Glu6Ala"
    assert variant["rsid"] == "" and variant["codon"] == "GCG" and variant["amino_acid"] == "Ala"
    assert "unknown variant" in of(events, "dna.verdict")[0]["text"]
    assert [(v.key, v.known) for v in V.load_lessons(lessons)] == [("novel-glu6ala", False)]
    assert replay_check(tmp_path / "r-ol" / "events.jsonl").ok


def test_letters_far_from_hbb_end_unresolved_rather_than_invented(tmp_path):
    results, events = run_fake(tmp_path, FakeReader(V.strip_sequence(3, "TTTTTTTTT")))
    assert results[0].outcome == "unresolved"
    assert "not an HBB variant" in of(events, "question.unresolved")[0]["reason"]
    assert not of(events, "question.settled") and not of(events, "dna.verdict")


def test_a_window_that_never_appears_times_out(tmp_path):
    results, events = run_fake(tmp_path, FakeReader("hbs", frozen=True), timeout_s=0.2)
    assert results[0].outcome == "timeout"
    assert "no stable read of window 3" in of(events, "question.unresolved")[0]["reason"]
    assert any("timed out" in p["line"] for p in of(events, "experiment.log"))
    assert events[-1]["payload"]["status"] == "stopped"


# --- the contract --------------------------------------------------------------------------


def check_contract(events: list[dict]) -> None:
    assert [e["seq"] for e in events] == list(range(len(events)))
    started = events[0]["payload"]
    assert events[0]["type"] == "run.started" and started["lab"] == "oneletter"
    cfg = started["config"]
    assert cfg["annotated_url"] == cfg["reader_url"].rstrip("/") + "/annotated_stream"
    for e in events:
        t, p = e["type"], e["payload"]
        assert t in EVENT_TYPES, t
        if t in DNA_KEYS:
            assert DNA_KEYS[t] <= set(p), (t, DNA_KEYS[t] - set(p))
        if t == "question.opened":
            Question.model_validate(p["question"])
        elif t == "hypothesis.proposed":
            Hypothesis.model_validate(p["hypothesis"])
        elif t == "experiment.proposed":
            Experiment.model_validate(p["experiment"])
        elif t == "experiment.finished":
            Observation.model_validate(p["observation"])
        elif t == "run.finished":
            Report.model_validate(p["report"])
        elif t == "experiment.scored":
            assert p["chosen"] in {r["experiment_id"] for r in p["ranking"]}
            assert all(
                re.match(r"^window-[123](-r\d+)?(-s\d+)?$", r["experiment_id"])
                for r in p["ranking"]
            )
        elif t == "dna.read":
            assert V.is_bases(p["bases"]) and all(0 <= i < 9 for i in p["mismatch_positions"])
        elif t == "dna.verdict":
            assert p["author"] == "template" or p["author"].startswith("jetson/")
            assert p["residue"] == 6 and p["pdb_id"] in ("2HBS", "1A3N")


def test_events_follow_the_contract_and_replay(tmp_path):
    from api.server import fold_events

    _, events = run_fake(tmp_path, FakeReader("hbs", swaps=["hbc"]), strips=2)
    check_contract(events)
    assert events[-1]["type"] == "run.finished"
    result = replay_check(tmp_path / "r-ol" / "events.jsonl")
    assert result.ok, result.summary()
    assert result.n_checked == len(of(events, "belief.updated"))
    snap = fold_events(events)
    (q,) = snap["questions"]
    assert q["id"] == "q-variant" and q["status"] == "settled"
    assert q["settled_hypothesis"] == "h-hbc"


@pytest.mark.skipif(not DEMO.exists(), reason="recorded demo not generated")
def test_the_recorded_demo_is_a_valid_fake_reader_run():
    events = [json.loads(line) for line in DEMO.read_text(encoding="utf-8").splitlines()]
    check_contract(events)
    assert events[0]["payload"]["config"]["fake_reader"] is True
    assert replay_check(DEMO).ok
    settled = [p["hypothesis_id"] for p in of(events, "question.settled")]
    assert settled == ["h-hbs", "h-hbc"]


# --- launcher and CLI ----------------------------------------------------------------------


def test_launcher_registers_oneletter_and_tasks_lists_it(tmp_path, monkeypatch):
    from api.launcher import known_labs
    from api.server import create_app as create_api

    assert "oneletter" in known_labs()
    monkeypatch.delenv("CRUX_ONELETTER_READER", raising=False)

    async def go():
        transport = httpx.ASGITransport(app=create_api(tmp_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            labs = (await c.get("/tasks")).json()["labs"]
            bad = await c.post("/runs", json={"lab": "oneletter", "overrides": {"nope": 1}})
            no_reader = await c.post("/runs", json={"lab": "oneletter"})
            r = await c.post(
                "/runs",
                json={
                    "lab": "oneletter",
                    "overrides": {
                        "fake": ["hbs"],
                        "fake_show_delay_s": 0,
                        "poll_s": 0.01,
                    },
                },
            )
            run_id = r.json()["run_id"]
            stream = await asyncio.wait_for(c.get(f"/runs/{run_id}/events"), timeout=30)
            snap = (await c.get(f"/runs/{run_id}")).json()
        return labs, bad, no_reader, r, stream, snap, run_id

    labs, bad, no_reader, r, stream, snap, run_id = asyncio.run(go())
    assert "oneletter" in labs
    assert bad.status_code == 400 and "nope" in bad.json()["detail"]
    assert no_reader.status_code == 400 and "CRUX_ONELETTER_READER" in no_reader.json()["detail"]
    assert r.status_code == 202
    assert "event: run.finished" in stream.text
    assert snap["status"] == "finished"
    assert snap["questions"][0]["settled_hypothesis"] == "h-hbs"
    events = [json.loads(x) for x in (tmp_path / run_id / "events.jsonl").read_text().splitlines()]
    assert events[0]["payload"]["config"]["fake_reader"] is True
    assert events[0]["payload"]["config"]["lessons_path"].endswith("oneletter/lessons.jsonl")


def test_launcher_uses_the_reader_env_var(tmp_path, monkeypatch):
    from api.launcher import DefaultLauncher

    monkeypatch.setenv("CRUX_ONELETTER_READER", "http://jetson.local:8770")
    coro = DefaultLauncher(tmp_path)._engine_coroutine("oneletter", "default", 0, {}, "r-x")
    frame = coro.cr_frame
    assert frame.f_locals["cfg"].reader_url == "http://jetson.local:8770"
    coro.close()


def test_cli_runs_against_the_fake_reader_over_http(tmp_path, capsys):
    from labs.oneletter.run import main

    events_path = tmp_path / "out" / "events.jsonl"
    code = main(
        [
            "--fake",
            "hbs",
            "--fake-show-delay",
            "0",
            "--poll",
            "0.01",
            "--events",
            str(events_path),
            "--lessons",
            str(tmp_path / "l.jsonl"),
            "--runs-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert "SETTLED h-hbs" in capsys.readouterr().out
    events = [json.loads(x) for x in events_path.read_text().splitlines()]
    check_contract(events)
    assert events[0]["payload"]["config"]["fake_reader"] is True
    assert events[0]["payload"]["config"]["reader_url"].startswith("http://127.0.0.1:")
    assert replay_check(events_path).ok
