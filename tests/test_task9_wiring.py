"""The Task 9 swarm wiring end to end on this machine, with scripted model roles.

Covers the local executor, the lab wrapper (sandbox lifecycle, delivery retry, artifact
export), the swarm roles (library diagnostics reach the designer), the predictor (exact
rows by answer key, sampled rows otherwise), the launcher and CLI registration, and one
full orchestrator run that settles a question and delivers the contract's artifacts.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.predictor import SampledPredictor, exact_row
from agents.proposer import Proposals
from agents.swarm import SwarmRoles
from core.config import load_config
from core.orchestrator import Deps, run
from core.schemas import (
    CriticFinding,
    Experiment,
    Hypothesis,
    OutcomeBin,
    Question,
    Report,
)
from execution.local import LocalExecutor, rewrite_paths
from execution.staging import COUNTS, SAMPLES
from labs.bixbench3.swarm_lab import SwarmTask9Lab, discover_cases
from labs.bixbench3.task9_contract import load_contract

GROUPS = {"T98G": ("Ctrl", "C10", "F7", "G2", "D4"), "HEK293": ("Ctrl", "D10", "E6")}


def make_case(root: Path, *, duplicate: bool = False) -> Path:
    rng = np.random.default_rng(3)
    rows = [(c, k) for c, ks in GROUPS.items() for k in ks for _ in range(2)]
    ids = [f"S{i:02d}" for i in range(1, 17)]
    samples = pd.DataFrame(rows, columns=["cell_line", "condition"])
    samples.insert(0, "sample_id", ids)
    counts = pd.DataFrame(rng.integers(50, 5000, size=(200, 16)), columns=ids)
    if duplicate:
        counts["S04"] = counts["S03"]
    counts.insert(0, "Geneid", [f"ENSG{i:011d}.{1 + i % 3}" for i in range(200)])
    root.mkdir(parents=True, exist_ok=True)
    samples.to_csv(root / SAMPLES, sep="\t", index=False)
    counts.to_csv(root / COUNTS, sep="\t", index=False)
    return root


# A diagnostic in the shape DU-42's library produces: python, one RESULT line, rule bins,
# and an exact table keyed by answer key.
DUPLICATE_CODE = """
import json, os, pathlib
import pandas as pd
root = pathlib.Path(os.environ.get('CRUX_DATA_DIR', '/data'))
counts = pd.read_csv(root / 'tagseq_featurecounts_matrix.tsv', sep='\\t', comment='#')
values = counts.drop(columns=[counts.columns[0]])
pairs = [
    [a, b] for i, a in enumerate(values.columns) for b in values.columns[i + 1:]
    if (values[a] == values[b]).all()
]
print('RESULT: ' + json.dumps({'identical_pairs': len(pairs), 'pairs': pairs}))
"""


CORRELATION_CODE = """
import json, os, pathlib
import numpy as np
import pandas as pd
root = pathlib.Path(os.environ.get('CRUX_DATA_DIR', '/data'))
counts = pd.read_csv(root / 'tagseq_featurecounts_matrix.tsv', sep='\\t', comment='#')
values = np.log1p(counts.drop(columns=[counts.columns[0]]).to_numpy(dtype=float))
corr = np.corrcoef(values, rowvar=False)
np.fill_diagonal(corr, 0)
print('RESULT: ' + json.dumps({'max_correlation': float(corr.max())}))
"""


def correlation_diagnostic() -> Experiment:
    return Experiment(
        id="corr",
        question_ids=[],
        kind="compute",
        goal="Find libraries whose log counts are almost perfectly correlated",
        language="python",
        code=CORRELATION_CODE,
        bins=[
            OutcomeBin(
                id="none",
                description="no pair above 0.9999",
                rule="max_correlation <= 0.9999",
            ),
            OutcomeBin(
                id="pair",
                description="a pair above 0.9999",
                rule="max_correlation > 0.9999",
            ),
        ],
        design={
            "library": "duplicates",
            "diagnostic": "max_correlation",
            "exact": {
                "none": {"none": 1.0, "pair": 0.0},
                "dup": {"none": 0.0, "pair": 1.0},
            },
        },
        est_cost_s=5,
    )


def duplicate_diagnostic() -> Experiment:
    return Experiment(
        id="dup",
        question_ids=[],
        kind="compute",
        goal="Find libraries with identical counts",
        language="python",
        code=DUPLICATE_CODE,
        bins=[
            OutcomeBin(
                id="none", description="no identical pair", rule="identical_pairs == 0"
            ),
            OutcomeBin(
                id="pair", description="an identical pair", rule="identical_pairs > 0"
            ),
        ],
        design={
            "library": "duplicates",
            "diagnostic": "identical_columns",
            "exact": {
                "none": {"none": 1.0, "pair": 0.0},
                "dup": {"none": 0.0, "pair": 1.0},
            },
        },
        est_cost_s=5,
    )


async def scripted_propose(brief, evidence, *, llm, memory, bus, config, library):
    assert config.null_hypothesis is False  # the orchestrator adds the null option
    q = Question(
        id="q-dup", text="Is a library duplicated?", kind="categorical", raised_by="lib"
    )
    hyps = [
        Hypothesis(
            id="q-dup-h0",
            question_id=q.id,
            statement="No duplicate",
            author="lib",
            rationale="-",
            answer_key="none",
            prior=0.5,
            posterior=0.5,
        ),
        Hypothesis(
            id="q-dup-h1",
            question_id=q.id,
            statement="One library under two names",
            author="lib",
            rationale="-",
            answer_key="dup",
            prior=0.5,
            posterior=0.5,
        ),
    ]
    # Two independent diagnostics: with the 0.15 null option and a 0.9 settle bar, one
    # perfect two-bin diagnostic stops at 0.85.
    diags = [
        make().model_copy(update={"id": f"q-dup-d{i}", "question_ids": [q.id]})
        for i, make in enumerate((duplicate_diagnostic, correlation_diagnostic))
    ]
    return Proposals(questions=[q], hypotheses=hyps, diagnostics=diags)


async def referee_ok(llm, exp, obs, *, run_dir=None):
    assert run_dir is None or Path(run_dir).is_dir()
    return obs.valid, "ok"


async def critic_from_lab(llm, artifacts, contract, *, counts_path=None, brief=None):
    assert brief and "contract" in brief
    names = {a.name for a in artifacts}
    return [
        CriticFinding(
            id=f"{item['path']}:missing",
            artifact=item["path"],
            severity="block",
            rule="missing",
            message="missing deliverable",
        )
        for item in contract.values()
        if isinstance(item, dict) and "path" in item and item["path"] not in names
    ]


async def report_plain(llm, state, artifacts):
    return Report(
        run_id=state.run_id,
        artifacts=artifacts,
        methods_md="# Methods",
        rejected=[],
        unresolved=[],
        cost_usd=0.0,
        seconds=0.0,
    )


def writer_code(fail: bool = False) -> str:
    """A delivery step that writes every contract file (placeholder values, test only)."""
    contract = load_contract()
    return f"""
import json, pathlib
import pandas as pd
contract = json.loads({json.dumps(json.dumps(contract))})
out = pathlib.Path('/outputs')
out.mkdir(parents=True, exist_ok=True)
{"raise RuntimeError('first attempt fails')" if fail else ""}
for item in contract.values():
    cols = item['required_columns']
    row = {{c: ('ENSG00000000001' if c == 'gene_id' else 'up' if c == 'regulation'
            else 'GENE1' if c == 'gene_symbol' else 0.01) for c in cols}}
    pd.DataFrame([row], columns=cols).to_csv(out / item['path'], index=False)
(out / 'METHODS.md').write_text('# Methods')
"""


class Planner:
    def __init__(self) -> None:
        self.errors: list[str | None] = []

    async def __call__(self, llm, state, *, last_error=None, profile=None):
        self.errors.append(last_error)
        return [("python", writer_code(fail=len(self.errors) == 1))]


class FakeLLM:
    on_usage = None

    def __init__(self, picks: list[str]) -> None:
        self.picks = picks
        self.calls: list[str] = []

    async def structured(self, role, schema, messages, n=1, **_):
        self.calls.append(role)
        return [schema(bin_id=self.picks[i % len(self.picks)]) for i in range(n)]


def test_rewrite_paths_only_touches_quoted_roots(tmp_path):
    roots = {"data": tmp_path / "d", "outputs": tmp_path / "o", "work": tmp_path / "w"}
    code = "a='/data/x.tsv'; b=\"/outputs\"; c='/database'; d='/work/a/b'; e=/data"
    out = rewrite_paths(code, roots)
    assert f"'{(tmp_path / 'd').as_posix()}/x.tsv'" in out
    assert f'"{(tmp_path / "o").as_posix()}"' in out
    assert "'/database'" in out and "e=/data" in out
    assert f"'{(tmp_path / 'w').as_posix()}/a/b'" in out


def test_local_executor_runs_python_and_exports(tmp_path):
    case = make_case(tmp_path / "case")
    ex = LocalExecutor(case, tmp_path / "runs")

    async def go():
        sid = await ex.open("bixbench3", "r-1")
        ok = await ex.run_code(sid, "python", DUPLICATE_CODE)
        bad = await ex.run_code(sid, "python", "print('RESULT: 1')")
        await ex.run_code(
            sid,
            "python",
            "import pathlib; pathlib.Path('/outputs/a.csv').write_text('x')",
            require_result=False,
        )
        exports = await ex.collect_outputs(sid)
        await ex.close(sid)
        return ok, bad, exports

    ok, bad, exports = asyncio.run(go())
    assert ok["valid"] and ok["result"]["identical_pairs"] == 0
    assert (tmp_path / "runs" / "r-1" / ok["log_ref"]).is_file()
    assert not bad["valid"] and "RESULT" in bad["error"]
    assert [e["name"] for e in exports] == ["a.csv"]
    assert Path(exports[0]["path"]).read_text() == "x"


def test_predictor_uses_exact_rows_and_samples_the_rest():
    exp = duplicate_diagnostic()
    exact = Hypothesis(
        id="h1",
        question_id="q",
        statement="dup",
        author="a",
        rationale="-",
        answer_key="dup",
        prior=0.5,
        posterior=0.5,
    )
    other = exact.model_copy(update={"id": "h2", "answer_key": "something-else"})
    assert exact_row(exp, exact) == {"none": 0.0, "pair": 1.0}
    llm = FakeLLM(["pair", "pair", "none", "bogus"])
    table = asyncio.run(SampledPredictor(llm, n=4).predict(exp, [exact, other], "ctx"))
    assert table["h1"] == {"none": 0.0, "pair": 1.0}
    assert (
        set(table["h2"]) == {"none", "pair"}
        and table["h2"]["pair"] > table["h2"]["none"]
    )
    assert abs(sum(table["h2"].values()) - 1) < 1e-9
    assert llm.calls == ["predictor"]  # one batched call for the sampled option only


@pytest.mark.parametrize("duplicate", [False, True])
def test_full_run_settles_and_delivers(tmp_path, duplicate):
    case = make_case(tmp_path / "case", duplicate=duplicate)
    cfg = load_config("default", runs_dir=str(tmp_path / "runs"))
    llm = FakeLLM(["none"])
    planner = Planner()
    lab = SwarmTask9Lab(
        LocalExecutor(case, tmp_path / "runs"), llm=llm, planner=planner
    )
    roles = SwarmRoles(
        llm,
        cfg,
        propose_fn=scripted_propose,
        library=lambda brief: [],
        referee_fn=referee_ok,
        critic_fn=critic_from_lab,
        report_fn=report_plain,
    )
    deps = Deps(agents=roles, predictor=SampledPredictor(llm), llm=llm)
    report = asyncio.run(run(lab, cfg, deps, run_id="r-wiring"))

    run_dir = tmp_path / "runs" / "r-wiring"
    events = [
        json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    types = [e["type"] for e in events]
    assert types[0] == "run.started" and types[-1] == "run.finished", types[-3:]
    settled = [e["payload"] for e in events if e["type"] == "question.settled"]
    assert [s["hypothesis_id"] for s in settled] == [
        "q-dup-h1" if duplicate else "q-dup-h0"
    ]
    logs = [e for e in events if e["type"] == "experiment.log"]
    assert logs and "RESULT" in logs[-1]["payload"]["line"]
    # delivery: the first attempt fails, the retry gets the error and writes every file
    assert planner.errors[0] is None and "first attempt fails" in (
        planner.errors[1] or ""
    )
    expected = {item["path"] for item in load_contract().values()} | {"METHODS.md"}
    assert {a.name for a in report.artifacts} == expected
    assert sorted(p.name for p in (run_dir / "delivery").iterdir()) == [
        "round1-attempt1-step1.py",
        "round1-attempt2-step1.py",
    ]
    assert lab.sandbox_id is None  # closed


def test_discover_cases_skips_the_evaluator(tmp_path):
    make_case(tmp_path / "suite" / "cases" / "case-01")
    (tmp_path / "suite" / "evaluator").mkdir()
    (tmp_path / "suite" / "evaluator" / "truth.json").write_text(
        '{"bundle_id": "x", "files": {}}'
    )
    (tmp_path / "suite" / "m.json").write_text('{"bundle_id": "b", "files": {}}')
    cases = discover_cases(tmp_path / "suite")
    assert cases.keys() == {"cases/case-01", "m"}
    assert cases["cases/case-01"]["kind"] == "local" and cases["m"]["kind"] == "modal"


def test_tasks_endpoint_and_launcher(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.launcher import DefaultLauncher, InvalidRunRequest
    from api.server import create_app

    make_case(tmp_path / "cases" / "clean")
    monkeypatch.setenv("CRUX_TASK9_CASES", str(tmp_path / "cases"))
    client = TestClient(create_app(tmp_path / "runs"))
    body = client.get("/tasks").json()
    assert "bixbench3" in body["labs"]
    assert body["cases"] == [{"lab": "bixbench3", "case": "clean", "kind": "local"}]
    launcher = DefaultLauncher(tmp_path / "runs")
    with pytest.raises(InvalidRunRequest, match="unknown case"):
        launcher._engine_coroutine("bixbench3", "default", 0, {"case": "nope"}, "r-x")
