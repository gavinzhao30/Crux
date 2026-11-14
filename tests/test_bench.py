"""Tests for the bench demo: the fake node, the EIG loop and the events it writes."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import threading
from pathlib import Path

import pytest

from labs.bench.lab import BenchClient, BenchLab
from labs.bench.run_bench import compare, run_loop

ROOT = Path(__file__).resolve().parent.parent
NODE_PATH = ROOT / "labs" / "bench" / "node.py"


def _load_node():
    spec = importlib.util.spec_from_file_location("bench_node", NODE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def node_url():
    node = _load_node()
    server = node.build_server(
        "127.0.0.1", 0, backend=node.FakeBackend(), settle=0.0, discard=5, frames=8, roi=80
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_node_endpoints(node_url):
    client = BenchClient(node_url)
    health = client.health()
    assert health["fake"] is True and "python" in health and "camera" in health
    values = []
    for level in range(0, 256, 17):
        assert client.patch(level) == {"ok": True, "level": level}
        m = client.measure()
        assert set(m) == {"mean", "std", "n"} and m["n"] == 8
        values.append(m["mean"])
    assert values[-1] > values[0] + 150
    # monotone up to noise (sd 1.5)
    assert all(b > a - 6.0 for a, b in itertools.pairwise(values))


def test_lab_run_matches_client(node_url):
    import asyncio

    lab = BenchLab(BenchClient(node_url))
    assert len(lab.candidate_designs()) == 16 and lab.candidate_designs()[-1] == 255.0
    assert {f.name for f in lab.families()} == {"linear", "power", "log", "saturating"}
    assert 100 < asyncio.run(lab.run(128.0)) < 200


@pytest.mark.parametrize("seed", range(5))
def test_eig_reaches_power(node_url, seed):
    r = run_loop(node_url, "eig", max_n=10, seed=seed, verbose=False)
    assert r["top"] == "power"
    assert r["reached"] and r["n_measured"] <= 10, r


def test_compare_eig_not_worse_than_random(node_url):
    res = compare(node_url, seeds=range(5), max=12)
    print(f"EIG mean={res['eig']:.2f} random mean={res['random']:.2f}")
    assert res["eig"] <= res["random"]


REQUIRED = {
    "run.started": {"lab", "goal", "config", "started_at"},
    "question.opened": {"question"},
    "hypothesis.proposed": {"hypothesis"},
    "hypothesis.merged": {"question_id", "kept", "merged", "priors"},
    "experiment.scored": {"question_id", "ranking", "chosen"},
    "experiment.started": {"experiment_id", "question_ids", "kind"},
    "experiment.finished": {"observation"},
    "belief.updated": {"question_id", "experiment_id", "observed", "prior", "posterior"},
    "model.curves": {"question_id", "x", "families", "points"},
    "surprise.raised": {"question_id", "experiment_id", "reason", "z_score", "null_mass"},
    "question.settled": {"question_id", "hypothesis_id", "posterior"},
    "question.unresolved": {"question_id", "reason", "top"},
    "run.finished": {"report", "status"},
}


def test_tint_run_events(node_url, tmp_path):
    path = tmp_path / "runs" / "t" / "events.jsonl"
    r = run_loop(
        node_url, "eig", max_n=10, seed=1, events_path=path, tint_after=6, tint=0.5, verbose=False
    )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert r["surprises"] >= 1

    # envelope
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert len({e["run_id"] for e in events}) == 1
    assert all({"run_id", "seq", "ts", "type", "payload"} <= set(e) for e in events)
    assert events[0]["type"] == "run.started" and events[0]["payload"]["lab"] == "bench"
    assert events[-1]["type"] == "run.finished"
    for e in events:
        need = REQUIRED.get(e["type"])
        assert need is not None, f"unexpected event type {e['type']}"
        assert need <= set(e["payload"]), e["type"]

    # the surprise comes after the tint, i.e. after 6 finished experiments
    finished = 0
    surprise_after = []
    for e in events:
        if e["type"] == "experiment.finished":
            finished += 1
        if e["type"] == "surprise.raised":
            surprise_after.append(finished)
    assert surprise_after and max(surprise_after) > 6

    hyps = [e["payload"]["hypothesis"] for e in events if e["type"] == "hypothesis.proposed"]
    assert len(hyps) == 5 and any(h["id"] == "h0" and h["is_null"] for h in hyps)
    scored = next(e for e in events if e["type"] == "experiment.scored")["payload"]
    assert scored["chosen"].startswith("level=") and len(scored["ranking"]) == 16
    curves = next(e for e in events if e["type"] == "model.curves")["payload"]
    assert len(curves["x"]) == 32 and set(curves["families"]) >= {"h-power", "h0"}


def test_node_is_python36_syntax():
    source = NODE_PATH.read_text()
    ast.parse(source, feature_version=(3, 6))
    assert "dataclass" not in source and ":=" not in source
    assert "from __future__" not in source
