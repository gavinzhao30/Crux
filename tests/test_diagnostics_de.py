"""DU-42 diagnostic library: contract checks and planted-fault runs on synthetic counts.

The synthetic data follow the Task 9 design (T98G Ctrl, C10, F7, G2, D4 and HEK293 Ctrl,
D10, E6, two replicates each) with negative-binomial counts, a large cell line effect, a
knockout signature partly shared by both lines (including a target transcript lost in every
knockout clone), and clone-specific effects. The generator and the fault appliers live in
``diagnostics.de_from_counts.synth`` (shared with the predictor training data); faults are
applied there, not by ``eval.suites.task9.make_cases``, so a swap really exchanges two
columns.
"""

import ast
import asyncio
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from agents.proposer import LibraryQuestion, MergePlan, ProposalBatch, propose
from core.config import RunConfig
from core.events import EventBus
from core.fakes import FakeMemory
from diagnostics.de_from_counts import library_questions
from diagnostics.de_from_counts.synth import (
    FAULT_FORK,
    GROUPS,
    featurecounts_case,
    plant,
    simulate,
    suite_case,
    write_case,
)
from labs.bixbench3.bins import choose_bin

ALLOWED_IMPORTS = {"json", "math", "os", "pathlib", "re", "numpy", "pandas"}


# --- running diagnostics ----------------------------------------------------------------


def experiments():
    return [(q, e) for q in library_questions() for e in q.diagnostics]


def run_code(code: str, data: Path, scratch: Path, name: str) -> dict:
    script = scratch / f"{name}.py"
    if not script.exists():
        script.write_text(code, encoding="utf-8")
    threads = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    env = {**os.environ, **threads, "CRUX_DATA_DIR": str(data)}
    done = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT:")]
    assert len(lines) == 1, done.stdout[-2000:]
    return json.loads(lines[0].removeprefix("RESULT:"))


def bin_of(experiment, result: dict) -> str:
    return choose_bin(((b.id, b.rule) for b in experiment.bins), result)


def run_all(cases: dict[str, Path], scratch: Path) -> dict[tuple[str, str], dict]:
    jobs = [
        (case, e.design["diagnostic"], e.code, folder)
        for case, folder in cases.items()
        for _, e in experiments()
    ]
    for _, name, code, _ in jobs:
        (scratch / f"{name}.py").write_text(code, encoding="utf-8")
    with ThreadPoolExecutor(max_workers=max(2, min(12, os.cpu_count() or 2))) as pool:
        results = pool.map(lambda job: run_code(job[2], job[3], scratch, job[1]), jobs)
        return {(case, name): result for (case, name, _, _), result in zip(jobs, results)}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("de_cases")
    scratch = tmp_path_factory.mktemp("de_code")
    counts, genes, design = simulate(seed=42)
    sheet, frame = suite_case(counts, genes, design, seed=42)
    cases, truth = {}, {}
    write_case(root / "clean", sheet, frame)
    cases["clean"], truth["clean"] = root / "clean", []
    published = featurecounts_case(counts, genes, design)
    write_case(root / "clean_featurecounts", *published, comment=True)
    cases["clean_featurecounts"], truth["clean_featurecounts"] = root / "clean_featurecounts", []
    rng = np.random.default_rng(7)
    for fault in FAULT_FORK:
        lines = ["T98G"] if fault in {"cell_line_swap", "gene_version"} else list(GROUPS)
        for line in lines:
            case = f"{fault}-{line}"
            faulty_sheet, faulty_frame, implicated = plant(fault, sheet, frame, line, rng)
            write_case(root / case, faulty_sheet, faulty_frame)
            cases[case], truth[case] = root / case, implicated
    return run_all(cases, scratch), truth


def null_and_fault_bins(question: LibraryQuestion, experiment) -> tuple[str, str]:
    """The most likely bin under the fork's first (null) and second (fault) options."""
    rows = experiment.design.get("exact") or experiment.design["stated"]
    null_key, fault_key = (o.answer_key for o in question.proposal.options[:2])
    return max(rows[null_key], key=rows[null_key].get), max(
        rows[fault_key], key=rows[fault_key].get
    )


# --- contract ---------------------------------------------------------------------------


def test_library_models_validate_and_follow_the_contract():
    questions = library_questions()
    assert [q.key for q in questions] == [
        "sample_labels",
        "duplicates",
        "sample_quality",
        "cell_line",
        "gene_ids",
        "group_direction",
    ]
    impacts = [q.proposal.impact for q in questions]
    assert impacts == sorted(impacts, reverse=True) and all(0 < i <= 1 for i in impacts)
    ids, codes = set(), set()
    for question in questions:
        assert LibraryQuestion.model_validate(question.model_dump()) == question
        assert question.proposal.kind == "categorical"
        keys = [o.answer_key for o in question.proposal.options]
        assert all(keys) and len(set(keys)) == len(keys)
        for experiment in question.diagnostics:
            assert experiment.kind == "compute" and experiment.language == "python"
            assert experiment.question_ids == []
            assert experiment.design["library"] == question.key
            assert experiment.est_cost_s > 0 and experiment.goal
            assert experiment.id not in ids
            ids.add(experiment.id)
            codes.add(experiment.code)
    assert len(codes) == len(ids)  # every diagnostic has its own code
    assert library_questions() == questions  # stable across calls


def test_every_fork_can_settle_past_the_null_option():
    for question in library_questions():
        diagnostics = question.diagnostics
        independent = len({e.code for e in diagnostics})
        assert independent >= 2 or any(len(e.bins) >= 4 for e in diagnostics), question.key


def test_likelihood_rows_use_real_answer_keys_and_bins():
    for question in library_questions():
        keys = {o.answer_key for o in question.proposal.options}
        for experiment in question.diagnostics:
            bins = {b.id for b in experiment.bins}
            assert len(bins) == len(experiment.bins) >= 2
            tables = {
                k: experiment.design[k] for k in ("exact", "stated") if k in experiment.design
            }
            assert len(tables) == 1, experiment.id  # exact or stated, never both or neither
            for rows in tables.values():
                assert set(rows) == keys
                for row in rows.values():
                    assert set(row) == bins
                    assert all(0 <= p <= 1 for p in row.values())
                    assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


def _number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return None if inner is None else -inner
    return None


def _cuts_by_name(rules: list[str]) -> tuple[dict[str, set[float]], set[str]]:
    """Thresholds each name is compared against, and names used bare (booleans)."""
    cuts: dict[str, set[float]] = {}
    bare: set[str] = set()
    for rule in rules:
        tree = ast.parse(rule, mode="eval")
        compared = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                numbers = {v for v in map(_number, operands) if v is not None}
                for operand in operands:
                    if isinstance(operand, ast.Name):
                        cuts.setdefault(operand.id, set()).update(numbers)
                        compared.add(operand.id)
        bare |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - compared
    return cuts, bare - set(cuts)


def test_bins_are_exhaustive_and_exclusive_over_a_value_grid():
    """Every combination of values around each threshold selects exactly one bin."""
    for _, experiment in experiments():
        cuts, booleans = _cuts_by_name([b.rule for b in experiment.bins])
        grids = {}
        for name, values in cuts.items():
            grid = values | {v + d for v in values for d in (-0.01, 0.01)} | {-1e3, 1e3}
            if name.startswith("n_"):  # counts are non-negative integers
                grid = {max(0, round(v)) for v in grid} | {0, 1, 2, 5}
            grids[name] = sorted(grid)
        grids.update({name: [True, False] for name in booleans})
        names = sorted(grids)
        for combo in itertools.product(*(grids[n] for n in names)):
            bin_of(experiment, dict(zip(names, combo)))  # raises unless exactly one matches


def test_diagnostic_code_is_self_contained():
    for _, experiment in experiments():
        tree = ast.parse(experiment.code)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert imported <= ALLOWED_IMPORTS, (experiment.id, imported)
        assert "truth" not in experiment.code.lower()
        assert "CRUX_DATA_DIR" in experiment.code


# --- planted faults ---------------------------------------------------------------------


@pytest.mark.parametrize("case", ["clean", "clean_featurecounts"])
def test_clean_data_lands_in_the_null_bin_everywhere(runs, case):
    results, _ = runs
    for question, experiment in experiments():
        null_bin, _ = null_and_fault_bins(question, experiment)
        result = results[case, experiment.design["diagnostic"]]
        assert bin_of(experiment, result) == null_bin, (experiment.id, result)
        assert result["flagged_samples"] == []


def _fault_cases():
    return [
        (fault, line)
        for fault in FAULT_FORK
        for line in (["T98G"] if fault in {"cell_line_swap", "gene_version"} else list(GROUPS))
    ]


@pytest.mark.parametrize(("fault", "line"), _fault_cases())
def test_each_fault_lands_in_its_option_bin_and_nowhere_else(runs, fault, line):
    results, truth = runs
    case = f"{fault}-{line}"
    for question, experiment in experiments():
        _, fault_bin = null_and_fault_bins(question, experiment)
        observed = bin_of(experiment, results[case, experiment.design["diagnostic"]])
        if question.key == FAULT_FORK[fault]:
            assert observed == fault_bin, (
                case,
                experiment.id,
                results[case, experiment.design["diagnostic"]],
            )
        else:  # specificity: other forks never report their own fault
            assert observed != fault_bin, (case, experiment.id)
    implicated = set(truth[case])
    flagged = {
        name: set(results[case, name]["flagged_samples"])
        for name in (e.design["diagnostic"] for q, e in experiments() if q.key == FAULT_FORK[fault])
    }
    if fault == "sample_swap":
        # Exchanging the pair or their two partners explains the counts equally well.
        assert all(implicated < samples and len(samples) == 4 for samples in flagged.values())
        swap = results[case, "labels_swap_search"]
        assert implicated in (set(swap["candidate_pair"]), set(swap["alternative_pair"]))
        assert set(swap["preferred_pair"]) in (set(), implicated)
    elif fault == "gene_version":
        assert all(samples == set() for samples in flagged.values())
        assert results[case, "gene_versions"]["n_unversioned"] >= 140
    else:
        assert all(samples == implicated for samples in flagged.values()), flagged


def test_swap_search_prefers_the_true_pair_when_the_other_line_agrees(runs):
    results, truth = runs
    for line in GROUPS:
        case = f"sample_swap-{line}"
        assert set(results[case, "labels_swap_search"]["preferred_pair"]) == set(truth[case])


# --- scale and integration --------------------------------------------------------------


def test_every_diagnostic_runs_in_under_a_minute_on_a_full_size_matrix(tmp_path):
    counts, genes, design = simulate(seed=3, n_genes=60_000, depth=20e6)
    write_case(tmp_path / "full", *featurecounts_case(counts, genes, design), comment=True)
    scratch = tmp_path / "code"
    scratch.mkdir()

    def timed(experiment):
        started = time.monotonic()
        result = run_code(
            experiment.code, tmp_path / "full", scratch, experiment.design["diagnostic"]
        )
        return time.monotonic() - started, bin_of(experiment, result)

    with ThreadPoolExecutor(max_workers=4) as pool:
        timings = list(pool.map(timed, [e for _, e in experiments()]))
    assert max(seconds for seconds, _ in timings) < 60, timings


class _LibraryOnlyLLM:
    """Proposers raise nothing; the merge keeps every library question and option apart."""

    async def structured(self, role, schema, messages, n=1):
        if schema is ProposalBatch:
            return [ProposalBatch(questions=[])]
        assert schema is MergePlan
        data = json.loads(messages[1]["content"])
        return [
            MergePlan.model_validate(
                {
                    "questions": [
                        {
                            "source_ids": [source],
                            "options": [{"source_ids": [o["source_id"]]} for o in entry["options"]],
                        }
                        for source, entry in data.items()
                    ]
                }
            )
        ]


def test_proposer_merges_the_library_and_reids_its_diagnostics(tmp_path):
    async def scenario():
        bus = EventBus("library-de", tmp_path)
        await bus.emit("run.started", {})
        library = library_questions()
        result = await propose(
            {},
            [],
            llm=_LibraryOnlyLLM(),
            memory=FakeMemory(),
            bus=bus,
            config=RunConfig(),
            library=library,
        )
        assert len(result.questions) == len(library)
        assert len(result.diagnostics) == sum(len(q.diagnostics) for q in library)
        by_key = {q.key: q for q in library}
        for diagnostic in result.diagnostics:
            [qid] = diagnostic.question_ids
            source = result.provenance[qid][0]
            question = by_key[source.removeprefix("library-")]
            assert diagnostic.id.startswith(f"{qid}-{source}-diagnostic-")
            assert diagnostic.design["library"] == question.key
            keys = {
                h.answer_key for h in result.hypotheses if h.question_id == qid and not h.is_null
            }
            rows = diagnostic.design.get("exact") or diagnostic.design["stated"]
            assert set(rows) == keys

    asyncio.run(scenario())
