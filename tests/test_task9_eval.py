import ast
import asyncio
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd

from eval.baseline_agent import Action, run_baseline
from eval.suites.task9.make_cases import GROUPS, make_cases
from execution.staging import COUNTS, SAMPLES
from labs.bixbench3.task9_contract import load_contract, load_prompt


def test_public_contract_and_fallback():
    contract = load_contract()
    assert len(contract) == 9
    assert len(load_contract(include_go=True)) == 13
    assert sum(item["kind"] == "de" for item in contract.values()) == 6
    assert all(
        "padj" in item["required_columns"] for item in contract.values() if item["kind"] == "de"
    )
    assert "fallback" in load_prompt()[1]
    assert "tagseq_featurecounts_matrix" not in contract


def test_suite_is_deterministic_and_contains_only_the_declared_mutations(tmp_path):
    design = [
        (cell, condition)
        for cell, conditions in GROUPS.items()
        for condition in conditions
        for _ in range(2)
    ]
    ids = [f"original_{i}" for i in range(16)]
    samples = pd.DataFrame(design, columns=["cell_line", "condition"]).assign(sample_id=ids)
    rng = np.random.default_rng(7)
    counts = pd.DataFrame(rng.integers(100, 50_000, size=(20, 16)), columns=ids)
    counts.insert(0, "gene_id", [f"ENSG{i:011d}.1" for i in range(20)])
    samples.to_csv(tmp_path / SAMPLES, sep="\t", index=False)
    counts.to_csv(tmp_path / COUNTS, sep="\t", index=False)
    first, second = tmp_path / "one", tmp_path / "two"
    for output in (first, second):
        make_cases(tmp_path / COUNTS, tmp_path / SAMPLES, output)
    files = {path.relative_to(first) for path in first.rglob("*") if path.is_file()}
    assert len(files) == 27
    for path in files:
        assert (first / path).read_bytes() == (second / path).read_bytes()
    truth = json.loads((first / "evaluator/truth.json").read_bytes())["cases"]
    assert sum(info["split"] == "dev" for info in truth.values()) == 4
    clean = next(case for case, info in truth.items() if info["fault"] == "clean")
    clean_dir = first / "cases" / clean
    original_counts = pd.read_csv(clean_dir / COUNTS, sep="\t")
    original_samples = pd.read_csv(clean_dir / SAMPLES, sep="\t")
    for case, info in truth.items():
        folder = first / "cases" / case
        assert {path.name for path in folder.iterdir()} == {COUNTS, SAMPLES}
        assert "original_" not in (folder / SAMPLES).read_text()
        case_counts = pd.read_csv(folder / COUNTS, sep="\t")
        case_samples = pd.read_csv(folder / SAMPLES, sep="\t")
        changed_columns = set(case_counts.columns[(case_counts != original_counts).any(axis=0)])
        changed_rows = set(
            case_samples.loc[(case_samples != original_samples).any(axis=1), "sample_id"]
        )
        expected = set(info["affected_samples"])
        assert changed_columns - {"gene_id"} | changed_rows == expected
        assert (
            list(case_counts.index[case_counts.gene_id != original_counts.gene_id])
            == (info["affected_gene_rows"])
        )


def test_grader_cannot_be_imported_by_agent_packages():
    root = Path(__file__).resolve().parents[1]
    for package in ("core", "agents", "memory"):
        for file in (root / package).rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [
                        node.module or "",
                        *(f"{node.module}.{alias.name}" for alias in node.names),
                    ]
                else:
                    continue
                assert not any(name.startswith("eval.") or name == "eval" for name in names), file


def test_baseline_shares_executor_and_records_usage(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    executor = SimpleNamespace(
        runs_dir=tmp_path,
        open=AsyncMock(return_value="sandbox"),
        run_code=AsyncMock(
            return_value={
                "seconds": 0.2,
                "log_ref": "trace",
                "valid": True,
                "stdout": "files",
            }
        ),
        collect_outputs=AsyncMock(return_value=[]),
        close=AsyncMock(),
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, model_dump=lambda **_: {})
    actions = [Action(action="python", content="print(1)"), Action(action="finish", content="done")]
    parse = AsyncMock(
        side_effect=[
            SimpleNamespace(
                usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(parsed=action))]
            )
            for action in actions
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    report = asyncio.run(
        run_baseline(
            executor,
            client,
            model="test-model",
            run_id="run",
            prompt="study",
            prompt_source="test",
            input_usd_per_million=Decimal("1"),
            output_usd_per_million=Decimal("1"),
            contract=load_contract(),
            max_calls=2,
        )
    )
    assert report["status"] == "finished"
    assert report["calls"] == 2
    assert Decimal(report["usd_unadjusted_for_cache"]) == Decimal("0.000024")
    executor.collect_outputs.assert_awaited_once_with("sandbox")
    executor.close.assert_awaited_once_with("sandbox")
    assert (run / "baseline.json").exists()
