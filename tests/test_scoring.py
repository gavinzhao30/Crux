"""The scoring adapter (core/scoring.py) over DU-6's array-based ``core.eig`` must agree with the
dict-based stand-in the orchestrator was built on, so merging DU-6 changes no run.

DU-6's module is taken from ``core/eig.py`` when it is already in the tree, else read from the
``origin/du-6-eig-scorer`` blob with a local ``git show`` (no network) and loaded into a
temporary module. The tests skip when neither is available.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from core import _eig_standin as standin
from core import fakes_engine, orchestrator, scoring
from core.config import load_config
from core.fakes_engine import (
    FakeClock,
    FakeLab,
    FakePredictor,
    FakeSurpriseLab,
    fake_deps,
)
from core.schemas import Experiment, Hypothesis, Question

ROOT = Path(__file__).resolve().parents[1]
DU6_REF = "origin/du-6-eig-scorer"
TOL = 1e-9


def _load_du6() -> types.ModuleType | None:
    if importlib.util.find_spec("core.eig") is not None:
        return importlib.import_module("core.eig")
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(  # fixed argv, read-only, local repository
            [git, "show", f"{DU6_REF}:core/eig.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    module = types.ModuleType("du6_eig")
    module.__file__ = f"{DU6_REF}:core/eig.py"
    sys.modules["du6_eig"] = module  # dataclass/typing helpers look modules up by name
    try:
        exec(compile(out.stdout, module.__file__, "exec"), module.__dict__)  # noqa: S102
    finally:
        sys.modules.pop("du6_eig", None)
    return module


@pytest.fixture(scope="module")
def du6() -> types.ModuleType:
    module = _load_du6()
    if module is None:
        pytest.skip(f"DU-6's core/eig.py is not available (no core/eig.py and no {DU6_REF})")
    return module


@pytest.fixture(scope="module")
def adapter(du6: types.ModuleType) -> scoring.Scorer:
    assert scoring.is_compatible(du6), "DU-6's core.eig no longer matches the array API"
    return scoring.Scorer(du6)


def close(a: object, b: object, tol: float = TOL) -> bool:
    """Structural equality with floats compared to ``tol``."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)
    return a == b


def hyp(hid: str, p: float, *, question: str = "q", null: bool = False, **kw) -> Hypothesis:
    return Hypothesis(
        id=hid,
        question_id=question,
        statement=hid,
        author="p",
        rationale="r",
        is_null=null,
        prior=p,
        posterior=p,
        **kw,
    )


SEP = {"a": {"x": 0.9, "y": 0.1}, "b": {"x": 0.1, "y": 0.9}, "q.h0": {"x": 0.5, "y": 0.5}}
CASES = [
    # (prior, likelihood, observed bin)
    ({"a": 0.425, "b": 0.425, "q.h0": 0.15}, SEP, "x"),
    ({"a": 0.6, "b": 0.3, "h0": 0.1}, {k.replace("q.", ""): v for k, v in SEP.items()}, "y"),
    (  # unnormalised prior, a zero entry, a row with a missing bin, an all-zero row
        {"a": 2.0, "b": 1.0, "c": 0.0, "d": 1.0, "q.h0": 0.5},
        {
            "a": {"x": 3.0, "y": 1.0, "z": 0.0},
            "b": {"y": 1.0},
            "d": {"x": 0.0, "y": 0.0},
            "q.h0": {"x": 1, "y": 1, "z": 1},
        },
        "z",
    ),
    (  # an observed bin no row names, and the near-certain surprise case
        {"a": 0.425, "b": 0.425, "q.h0": 0.15},
        {"a": {"x": 0.96, "y": 0.04}, "b": {"x": 0.97, "y": 0.03}, "q.h0": {"x": 0.5, "y": 0.5}},
        "y",
    ),
    ({"a": 0.5, "b": 0.5}, {"a": {"x": 1.0, "y": 0.0}, "b": {"x": 1.0, "y": 0.0}}, "y"),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_update_predictive_and_surprise_match_the_standin(
    adapter: scoring.Scorer, case: int
) -> None:
    prior, lik, observed = CASES[case]
    assert close(adapter.update(prior, lik, observed), standin.update(prior, lik, observed))
    assert close(adapter.predictive(prior, lik, observed), standin.predictive(prior, lik, observed))
    assert adapter.is_surprise(prior, lik, observed) == standin.is_surprise(prior, lik, observed)
    for null_in_eig in (None, True, False):
        got = adapter.expected_info_gain(prior, lik, null_in_eig=null_in_eig)
        want = standin.expected_info_gain(prior, lik, null_in_eig=null_in_eig)
        assert close(got, want), (null_in_eig, got, want)


def test_answer_keys_and_several_nulls_match_the_standin(adapter: scoring.Scorer) -> None:
    lik = {
        "a": {"x": 0.8, "y": 0.1, "z": 0.1},
        "b": {"x": 0.1, "y": 0.8, "z": 0.1},
        "c": {"x": 0.1, "y": 0.1, "z": 0.8},
        "n1": {"x": 1 / 3, "y": 1 / 3, "z": 1 / 3},
        "n2": {"x": 0.2, "y": 0.2, "z": 0.6},
    }
    prior = {"a": 0.3, "b": 0.3, "c": 0.2, "n1": 0.1, "n2": 0.1}
    nulls = {"n1", "n2"}
    for keys in (None, {"a": "k", "b": "k", "c": None}, {"a": "k", "b": "k", "c": "k"}):
        got = adapter.expected_info_gain(prior, lik, keys, null_ids=nulls)
        want = standin.expected_info_gain(prior, lik, keys, null_ids=nulls)
        assert close(got, want), keys
    for observed in ("x", "y", "z"):
        got = adapter.is_surprise(prior, lik, observed, null_ids=nulls)
        assert got == standin.is_surprise(prior, lik, observed, null_ids=nulls)
    assert adapter.is_surprise({"a": 0.5, "n1": 0.5}, lik, "z", null_ids=nulls)


def test_rankings_match_the_standin_on_the_fake_lab(adapter: scoring.Scorer) -> None:
    lab = FakeLab()

    async def tables(question: str) -> tuple[Question, list[Hypothesis], list[Experiment]]:
        agents = fakes_engine.FakeAgents(lab)
        questions, proposals = await agents.propose({}, [])
        q = next(x for x in questions if x.id == question)
        real = [h for h in proposals if h.question_id == question and ".p2" not in h.id]
        weight = 0.85 / len(real)
        options = [h.model_copy(update={"prior": weight, "posterior": weight}) for h in real]
        options.append(hyp(f"{question}.h0", 0.15, question=question, null=True))
        exps = await agents.design(q, options, [], [])
        predictor = FakePredictor(lab)
        for e in exps:
            table = await predictor.predict(e, [h for h in options if not h.is_null], "")
            n = len(e.bins)
            e.likelihood = {**table, f"{question}.h0": {b.id: 1 / n for b in e.bins}}
        return q, options, exps

    for question in ("q1", "q2"):
        q, options, exps = asyncio.run(tables(question))
        for lam in (0.0, 0.2, 1.5):
            got = adapter.rank(q, options, exps, lam)
            want = standin.rank(q, options, exps, lam)
            assert [e.id for e in got] == [e.id for e in want]
            for g, w in zip(got, want, strict=True):
                assert close([g.eig_bits, g.score], [w.eig_bits, w.score])
        # the adapter scores copies and leaves the inputs alone (DU-6's rank mutates them)
        assert all(e.eig_bits is None and e.score is None for e in exps)


def test_rank_ties_keep_input_order_and_skip_rejected(adapter: scoring.Scorer) -> None:
    q = Question(id="q", text="t", kind="categorical", impact=0.5, raised_by="p")
    options = [
        hyp("b", 0.25),
        hyp("a", 0.6, status="accepted"),
        hyp("gone", 0.0, status="rejected"),
        hyp("q.h0", 0.15, null=True),
    ]
    lik = {"a": {"yes": 1.0, "no": 0.0}, "b": {"yes": 0.0, "no": 1.0}, "gone": {"yes": 1.0}}
    exps = [
        Experiment(id=i, question_ids=["q"], kind="compute", goal="g", likelihood=lik, est_cost_s=c)
        for i, c in (("slow", 120), ("zz-tie", 30), ("fast", 30))
    ]
    got = adapter.rank(q, options, exps)
    want = standin.rank(q, options, exps)
    assert [e.id for e in got] == [e.id for e in want] == ["zz-tie", "fast", "slow"]
    assert close([e.eig_bits for e in got], [e.eig_bits for e in want])


def test_scalar_helpers_match_the_standin(adapter: scoring.Scorer) -> None:
    for args in [(0.6, 60, 0.5, 0.2), (1.0, -5.0, 1.0, 0.2), (-1e-12, 30, 1.0, 0.2)]:
        assert close(adapter.score(*args), standin.score(*args))
    for p in (1e-3, 0.5, 1.0):
        assert close(adapter.e_value(p), standin.e_value(p))
    with pytest.raises(ValueError):
        adapter.e_value(0.0)  # the stand-in's domain, not DU-6's inf
    counts = {"x": 8, "y": 0, "z": 3}
    assert close(
        adapter.likelihood_from_samples(counts, 3), standin.likelihood_from_samples(counts, 3)
    )
    assert close(adapter.likelihood_from_samples([8, 0], 2), [0.9, 0.1])
    assert close(
        adapter.likelihood_from_samples([1.5, 0.5], 2),
        standin.likelihood_from_samples([1.5, 0.5], 2),
    )
    with pytest.raises(ValueError):
        adapter.likelihood_from_samples({"x": 8}, 2)


def test_whole_runs_are_the_same_on_either_backend(
    adapter: scoring.Scorer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_events(lab, run_id: str) -> list[dict]:
        cfg = load_config(runs_dir=str(tmp_path), lessons=True)
        asyncio.run(orchestrator.run(lab, cfg, fake_deps(lab, clock=FakeClock()), run_id=run_id))
        text = (tmp_path / run_id / "events.jsonl").read_text(encoding="utf-8")
        return [json.loads(line.replace(run_id, "RUN")) for line in text.splitlines()]

    for make in (FakeLab, FakeSurpriseLab):
        baseline = run_events(make(), f"r-standin-{make.__name__}")
        with monkeypatch.context() as m:
            m.setattr(orchestrator, "scoring", adapter)
            m.setattr(fakes_engine, "scoring", adapter)
            ported = run_events(make(), f"r-du6-{make.__name__}")
        assert [e["type"] for e in ported] == [e["type"] for e in baseline]
        assert close(ported, baseline), make.__name__


def test_default_backend_is_reported_and_used() -> None:
    expected = "core.eig" if importlib.util.find_spec("core.eig") else "core._eig_standin"
    assert scoring.BACKEND == expected
    assert orchestrator.scoring is scoring
    assert not scoring.is_compatible(standin)  # dict-based, so never mistaken for DU-6's API
