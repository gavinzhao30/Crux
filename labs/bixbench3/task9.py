"""BixBench3 Task 9 downstream lab, using published counts as input."""

import csv
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

from execution.sandbox import ModalExecutor
from labs.bixbench3.bins import choose_bin

if TYPE_CHECKING:
    from core.interfaces import Search
    from core.schemas import Artifact, Evidence, Experiment, Observation

_PROFILE = """
import json, pathlib
import pandas as pd
root = pathlib.Path('/data')
samples = pd.read_csv(root / 'samples.tsv', sep='\\t', dtype=str, keep_default_na=False)
counts = pd.read_csv(root / 'tagseq_featurecounts_matrix.tsv', sep='\\t', comment='#')
column_field = 'count_column' if 'count_column' in samples else 'sample_id'
columns = samples[column_field].tolist()
if len(columns) != len(set(columns)) or not set(columns) <= set(counts.columns):
    raise ValueError('Sample sheet count columns must be unique and present in the count matrix')
genes = counts['Geneid'] if 'Geneid' in counts else counts.iloc[:, 0]
result = {
    'files': sorted(p.name for p in root.iterdir()),
    'shape': [len(genes), len(columns)],
    'example_gene_ids': genes.head(3).astype(str).tolist(),
    'versioned_gene_ids': int(genes.astype(str).str.match(r'ENSG\\d+\\.\\d+$').sum()),
    'library_sizes': counts[columns].sum().to_dict(),
    'samples': samples.to_dict(orient='records'),
}
print('RESULT: ' + json.dumps(result, allow_nan=False))
"""
_DE_COLUMNS = (
    "gene_id",
    "gene_symbol",
    "base_mean",
    "log2_fold_change",
    "lfc_se",
    "stat",
    "p_value",
    "padj",
)


def validate_de_table(path: Path) -> list[str]:
    """Check the Task 9 DE table schema and row identity."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != _DE_COLUMNS:
            return ["DE table header does not match the required columns"]
        seen = set()
        for row in reader:
            gene_id = row["gene_id"] or ""
            if None in row or any(value is None for value in row.values()):
                return ["DE table has rows with the wrong number of fields"]
            if not gene_id or "." in gene_id or gene_id in seen:
                return ["DE table must have unique, unversioned gene IDs"]
            seen.add(gene_id)
            if row["padj"] not in ("", "NA"):
                try:
                    adjusted = float(row["padj"])
                except ValueError:
                    return ["padj must be numeric or missing"]
                if not 0 <= adjusted <= 1:
                    return ["padj must be between zero and one"]
        return [] if seen else ["DE table has no rows"]


class Task9Lab:
    name = "bixbench3"

    def __init__(
        self,
        executor: ModalExecutor,
        sandbox_id: str,
        *,
        task_prompt: str,
        contract: dict,
        plan_delivery: Callable[[object], Awaitable[Sequence[tuple[str, str]]]],
        search: "Search | None" = None,
        summarize_search: Callable[["Experiment", list["Evidence"]], Awaitable[dict]] | None = None,
        de_tables: frozenset[str] = frozenset(),
    ):
        self.executor = executor
        self.sandbox_id = sandbox_id
        self.task_prompt = task_prompt
        self.contract = contract
        self.plan_delivery = plan_delivery
        self.search = search
        self.summarize_search = summarize_search
        self.de_tables = de_tables

    def brief(self) -> dict:
        return {
            "goal": self.task_prompt,
            "contract": self.contract,
            "protocol": "BixBench3 task 9, downstream pilot (published counts as input)",
            "budget": {"usd": 2, "model_calls": 40},
            "search_policy": "Task 9 documentation policy" if self.search else "disabled",
        }

    def candidate_designs(self) -> None:
        return None

    async def profile(self) -> list["Evidence"]:
        from core.schemas import Evidence

        result = await self.executor.exec(self.sandbox_id, "python", _PROFILE)
        if not result["valid"]:
            raise RuntimeError(f"Task 9 profiling failed; see {result['log_ref']}")
        return [
            Evidence(
                id=uuid4().hex,
                kind="profile",
                claim=json.dumps(result["result"], sort_keys=True),
                author_role="experimenter",
                source_ref=result["log_ref"],
                conditions={"input_bundle": self.executor.inputs.bundle_id},
            )
        ]

    async def run(self, exp: "Experiment") -> "Observation":
        from core.schemas import Observation

        started = monotonic()
        log_ref = ""
        try:
            if exp.kind in {"compute", "multiverse"} and exp.code and exp.language:
                executed = await self.executor.run_code(
                    self.sandbox_id,
                    exp.language,
                    exp.code,
                    experiment_id=exp.id,
                    timeout_s=self.executor.diagnostic_timeout_s,
                )
                log_ref = executed["log_ref"]
                if not executed["valid"]:
                    raise ValueError(executed["error"])
                result = executed["result"]
            elif exp.kind == "search" and exp.query and self.search is not None:
                from search.task9 import search_value

                evidence = await self.search.find(exp.query)
                # Without an injected mapper the bins see the same code-computed fields as
                # the orchestrator's search path: n_results, text (lower-cased) and urls.
                result = (
                    await self.summarize_search(exp, evidence)
                    if self.summarize_search is not None
                    else {"hit_count": len(evidence), **search_value(evidence)}
                )
            else:
                raise ValueError("Unsupported or incomplete Task 9 experiment")
            bin_id = choose_bin(((item.id, item.rule) for item in exp.bins), result)
        except (ValueError, KeyError, SyntaxError, TypeError, PermissionError) as exc:
            return Observation(
                experiment_id=exp.id,
                valid=False,
                referee_note=str(exc),
                log_ref=log_ref,
                seconds=monotonic() - started,
            )
        return Observation(
            experiment_id=exp.id,
            valid=True,
            value=result,
            bin_id=bin_id,
            log_ref=log_ref,
            seconds=monotonic() - started,
        )

    async def deliver(self, state: object) -> list["Artifact"]:
        from core.schemas import Artifact

        exports = await self.executor.collect_outputs(self.sandbox_id)
        for language, code in await self.plan_delivery(state):
            result = await self.executor.run_code(
                self.sandbox_id,
                language,
                code,
                require_result=False,
                timeout_s=600,
            )
            exports = await self.executor.collect_outputs(self.sandbox_id)
            if not result["valid"]:
                break
        artifacts = []
        for export in exports:
            name = export["name"]
            findings = validate_de_table(Path(export["path"])) if name in self.de_tables else []
            artifacts.append(
                Artifact(
                    **export,
                    contract_ok=name in self.de_tables and not findings,
                    findings=findings,
                )
            )
        return artifacts
