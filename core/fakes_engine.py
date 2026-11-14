"""Deterministic fakes for the engine, so the whole run loop works offline and in tests.

These drive ``core.orchestrator``; ``core/fakes.py`` holds the simpler adapters behind the
scripted UI replay (``scripts/demo_loop.py``) and the contract tests.

* ``FakeLab``: a categorical lab with two questions (library strandedness, batch covariate), a
  small diagnostic library with outcome bins, and a hidden truth that decides the observed bins.
  ``deliver`` writes two CSV artifacts; on the first round it forgets the ``padj`` column, so
  the critic blocks it and the repair round fixes it. ``FakeLab(surprise=True)`` hides a truth
  that no proposed option predicts ("libraries were made with two kits").
* ``FakeNumericLab``: two model families, ``y = a*x`` and ``y = a*x**2`` with
  ``a ~ Normal(1, 0.2)``, designs on 17 points in [0, 2]; the truth is quadratic with a = 1.1
  and noise sd 0.1.
* ``FakeAgents`` (all roles), ``FakePredictor``, ``FakeExecutor``, ``FakeMemory``,
  ``FakeSearch``, ``FakeLLM`` (usage accounting only) and ``FakeClock``.
* ``fake_deps(lab)`` wires them into ``core.orchestrator.Deps``.

Nothing here touches the network. Randomness comes from explicit seeds (string seeds for
``random.Random`` are stable across processes), so two runs with the same seed and a
``FakeClock`` write byte-identical event files apart from ``run_id``.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from agents.llm import Usage
from core import scoring
from core.eig_exec import ModelFamily
from core.interfaces import Executor
from core.schemas import (
    Artifact,
    CriticFinding,
    EngineState,
    Evidence,
    Experiment,
    Hypothesis,
    Observation,
    OutcomeBin,
    Question,
    Report,
    Trial,
    bin_for,
)

__all__ = [
    "FakeAgents",
    "FakeClock",
    "FakeExecutor",
    "FakeLLM",
    "FakeLab",
    "FakeMemory",
    "FakeNumericLab",
    "FakePredictor",
    "FakeSearch",
    "FakeSurpriseLab",
    "fake_deps",
]

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 3}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _pause(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)


# --- clock and LLM accounting ------------------------------------------------------------


class FakeClock:
    """A deterministic clock: each call returns the previous time plus ``step`` seconds."""

    def __init__(self, start: float = 1_794_700_000.0, step: float = 0.05) -> None:
        self.t = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        now = self.t
        self.t += self.step
        return now


class FakeLLM:
    """Stands in for ``agents.llm.LLM``: counts calls and reports ``Usage`` to ``on_usage``.

    ``structured`` returns canned objects from ``responses`` (schema name -> instance, or a
    callable taking the messages) and charges one call per sample.
    """

    model = "fake-model"

    def __init__(
        self,
        cost_per_call: float = 0.002,
        input_tokens: int = 400,
        output_tokens: int = 120,
        responses: dict[str, Any] | None = None,
    ) -> None:
        self.cost_per_call = cost_per_call
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.responses = dict(responses or {})
        self.on_usage: Callable[[Usage], None] | None = None
        self.calls: list[str] = []

    def charge(self, role: str, n: int = 1) -> None:
        for _ in range(n):
            self.calls.append(role)
            usage = Usage(
                role=role,
                model=self.model,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost_usd=self.cost_per_call,
                cached=False,
                seconds=0.0,
            )
            if self.on_usage is not None:
                self.on_usage(usage)

    async def structured(
        self, role: str, schema: type[BaseModel], messages: list[dict], n: int = 1
    ) -> list[BaseModel]:
        self.charge(role, n)
        canned = self.responses.get(schema.__name__)
        if canned is None:
            raise KeyError(f"FakeLLM has no canned response for {schema.__name__}")
        out = canned(messages) if callable(canned) else canned
        return [schema.model_validate(out.model_dump() if isinstance(out, BaseModel) else out)] * n


# --- the categorical scenario --------------------------------------------------------------


@dataclass(frozen=True)
class FakeOption:
    key: str
    statement: str
    rationale: str


@dataclass(frozen=True)
class FakeDiagnostic:
    """One library diagnostic: its bins, what each option predicts, what each truth yields."""

    name: str
    question: str
    kind: str
    goal: str
    bins: tuple[tuple[str, str, str], ...]  # (id, description, rule)
    predicts: dict[str, str | dict[str, float]]  # option key -> bin id (exact) or distribution
    values: dict[str, dict[str, Any]]  # truth key -> the diagnostic's JSON result
    cost_s: float
    code: str | None = None
    query: str | None = None

    def outcome_bins(self) -> list[OutcomeBin]:
        return [OutcomeBin(id=i, description=d, rule=r) for i, d, r in self.bins]


@dataclass(frozen=True)
class FakeQuestion:
    id: str
    text: str
    impact: float
    options: tuple[FakeOption, ...]
    regenerated: tuple[FakeOption, ...] = ()
    duplicate: str | None = None  # an option key that a second proposer proposes again


@dataclass(frozen=True)
class FakeScenario:
    questions: tuple[FakeQuestion, ...]
    diagnostics: tuple[FakeDiagnostic, ...]
    docs: tuple[tuple[str, str], ...] = ()  # (url, text) for FakeSearch
    contract: dict[str, Any] = field(default_factory=dict)

    def diagnostic(self, name: str | None) -> FakeDiagnostic | None:
        return next((d for d in self.diagnostics if d.name == name), None)

    def question(self, qid: str) -> FakeQuestion | None:
        return next((q for q in self.questions if q.id == qid), None)


_STRAND = FakeQuestion(
    id="q1",
    text="Which strand protocol was used to make the libraries?",
    impact=0.9,
    options=(
        FakeOption(
            "unstranded",
            "The libraries are unstranded; count reads on both strands.",
            "No kit is named in the metadata, and unstranded prep is the common default.",
        ),
        FakeOption(
            "forward",
            "The libraries are forward-stranded; count sense reads only.",
            "Ligation-based kits give reads on the transcript strand.",
        ),
        FakeOption(
            "reverse",
            "The libraries are reverse-stranded (dUTP); count antisense reads only.",
            "Most current Illumina kits mark the second strand with dUTP.",
        ),
    ),
    regenerated=(
        FakeOption(
            "mixed_kits",
            "The libraries were made with two kits; set the strand per library.",
            "A sense fraction near 0.3 is what a mix of stranded and unstranded libraries gives.",
        ),
        FakeOption(
            "gdna_contamination",
            "Reverse-stranded libraries contaminated with genomic DNA; count antisense reads.",
            "Genomic DNA adds reads on both strands and raises the intronic fraction.",
        ),
    ),
    duplicate="reverse",
)

_BATCH = FakeQuestion(
    id="q2",
    text="Should the cell line enter the design formula as a covariate?",
    impact=0.7,
    options=(
        FakeOption(
            "no_batch",
            "Use ~ condition; the cell lines need no covariate.",
            "Knockouts are compared within each cell line already.",
        ),
        FakeOption(
            "batch",
            "Use ~ cell_line + condition.",
            "T98G and HEK293 differ far more than any knockout does.",
        ),
    ),
)

_Q1_SCRIPT = (
    "import pandas as pd\n"
    "counts = pd.read_csv('counts_by_strand.csv')\n"
    "frac = counts['sense'].sum() / (counts['sense'].sum() + counts['antisense'].sum())\n"
    "print('RESULT: {\"frac_sense\": %.3f}' % frac)\n"
)

CATEGORICAL_SCENARIO = FakeScenario(
    questions=(_STRAND, _BATCH),
    diagnostics=(
        FakeDiagnostic(
            name="strand_fraction",
            question="q1",
            kind="compute",
            goal="Fraction of reads on the sense strand of annotated genes",
            bins=(
                ("sense_high", "most reads on the sense strand", "frac_sense > 0.85"),
                ("balanced", "reads split evenly", "0.35 <= frac_sense <= 0.65"),
                ("antisense_high", "most reads on the antisense strand", "frac_sense < 0.15"),
                ("skewed", "an uneven split no single protocol gives", "True"),
            ),
            predicts={
                "unstranded": "balanced",
                "forward": "sense_high",
                "reverse": "antisense_high",
                "mixed_kits": "skewed",
                "gdna_contamination": "skewed",
            },
            values={
                "unstranded": {"frac_sense": 0.502},
                "forward": {"frac_sense": 0.941},
                "reverse": {"frac_sense": 0.061},
                "mixed_kits": {"frac_sense": 0.268},
                "gdna_contamination": {"frac_sense": 0.297},
            },
            cost_s=45,
            code=_Q1_SCRIPT,
        ),
        FakeDiagnostic(
            name="strand_count_ratio",
            question="q1",
            kind="multiverse",
            goal="Count genes both ways and compare forward to reverse totals",
            bins=(
                ("fwd_gt_rev", "forward counts dominate", "ratio > 1.5"),
                ("similar", "forward and reverse counts similar", "0.67 <= ratio <= 1.5"),
                ("rev_gt_fwd", "reverse counts dominate", "ratio < 0.67"),
            ),
            predicts={
                "unstranded": "similar",
                "forward": "fwd_gt_rev",
                "reverse": "rev_gt_fwd",
                "mixed_kits": "rev_gt_fwd",
                "gdna_contamination": "similar",
            },
            values={
                "unstranded": {"ratio": 1.02},
                "forward": {"ratio": 14.8},
                "reverse": {"ratio": 0.07},
                "mixed_kits": {"ratio": 0.41},
                "gdna_contamination": {"ratio": 1.1},
            },
            cost_s=120,
            code="# count with strand=1 and strand=2, then\nprint('RESULT: {\"ratio\": ...}')\n",
        ),
        FakeDiagnostic(
            name="intronic_rate",
            question="q1",
            kind="compute",
            goal="Fraction of reads that fall in introns",
            bins=(
                ("intronic_high", "many intronic reads", "intronic_frac > 0.3"),
                ("intronic_low", "few intronic reads", "intronic_frac <= 0.3"),
            ),
            predicts={
                "unstranded": "intronic_low",
                "forward": "intronic_low",
                "reverse": "intronic_low",
                "mixed_kits": "intronic_low",
                "gdna_contamination": "intronic_high",
            },
            values={
                "unstranded": {"intronic_frac": 0.12},
                "forward": {"intronic_frac": 0.11},
                "reverse": {"intronic_frac": 0.12},
                "mixed_kits": {"intronic_frac": 0.13},
                "gdna_contamination": {"intronic_frac": 0.46},
            },
            cost_s=60,
            code="print('RESULT: {\"intronic_frac\": ...}')\n",
        ),
        FakeDiagnostic(
            name="kit_search",
            question="q1",
            kind="search",
            goal="Search protocol notes for the library prep kit",
            bins=(
                ("reverse_mentioned", "sources describe a dUTP kit", "'dutp' in text"),
                ("no_mention", "no source names the kit", "not ('dutp' in text)"),
            ),
            predicts={
                "unstranded": {"reverse_mentioned": 0.25, "no_mention": 0.75},
                "forward": {"reverse_mentioned": 0.2, "no_mention": 0.8},
                "reverse": {"reverse_mentioned": 0.8, "no_mention": 0.2},
                "mixed_kits": {"reverse_mentioned": 0.6, "no_mention": 0.4},
                "gdna_contamination": {"reverse_mentioned": 0.6, "no_mention": 0.4},
            },
            values={},
            cost_s=8,
            query="library prep kit strandedness dutp",
        ),
        FakeDiagnostic(
            name="batch_search",
            question="q2",
            kind="search",
            goal="Search the dataset record for how the cell lines were sequenced",
            bins=(
                ("separate_runs", "cell lines sequenced separately", "'separate' in text"),
                ("no_mention", "nothing about sequencing runs", "not ('separate' in text)"),
            ),
            predicts={
                "no_batch": {"separate_runs": 0.25, "no_mention": 0.75},
                "batch": {"separate_runs": 0.85, "no_mention": 0.15},
            },
            values={},
            cost_s=5,
            query="cell line libraries sequenced separate batches",
        ),
        FakeDiagnostic(
            name="pca_cell_line",
            question="q2",
            kind="compute",
            goal="Share of PC1 variance explained by cell line",
            bins=(
                ("cell_line_dominates", "cell line drives PC1", "pc1_cell_line_r2 > 0.5"),
                ("condition_dominates", "condition drives PC1", "pc1_cell_line_r2 <= 0.5"),
            ),
            predicts={"no_batch": "condition_dominates", "batch": "cell_line_dominates"},
            values={"no_batch": {"pc1_cell_line_r2": 0.21}, "batch": {"pc1_cell_line_r2": 0.83}},
            cost_s=540,
            code="print('RESULT: {\"pc1_cell_line_r2\": ...}')\n",
        ),
        FakeDiagnostic(
            name="residual_clusters",
            question="q2",
            kind="compute",
            goal="Do residuals under ~ condition cluster by cell line?",
            bins=(
                ("clustered", "residuals cluster by cell line", "silhouette > 0.25"),
                ("unclustered", "no structure in residuals", "silhouette <= 0.25"),
            ),
            predicts={"no_batch": "unclustered", "batch": "clustered"},
            values={"no_batch": {"silhouette": 0.04}, "batch": {"silhouette": 0.41}},
            cost_s=720,
            code="print('RESULT: {\"silhouette\": ...}')\n",
        ),
        FakeDiagnostic(
            name="design_comparison",
            question="q2",
            kind="multiverse",
            goal="Fit both designs and compare the mean dispersion",
            bins=(
                ("batch_better", "the covariate lowers dispersion", "delta_dispersion > 0.05"),
                ("no_difference", "dispersion unchanged", "delta_dispersion <= 0.05"),
            ),
            predicts={
                "no_batch": {"batch_better": 0.2, "no_difference": 0.8},
                "batch": {"batch_better": 0.85, "no_difference": 0.15},
            },
            values={
                "no_batch": {"delta_dispersion": 0.01},
                "batch": {"delta_dispersion": 0.12},
            },
            cost_s=150,
            code="print('RESULT: {\"delta_dispersion\": ...}')\n",
        ),
        FakeDiagnostic(
            name="library_sizes",
            question="q2",
            kind="compute",
            goal="Smallest library size",
            bins=(
                ("ok", "every library above 1M reads", "min_lib > 1e6"),
                ("low", "a library below 1M reads", "min_lib <= 1e6"),
            ),
            predicts={"no_batch": "ok", "batch": "ok"},  # separates nothing: dropped
            values={"no_batch": {"min_lib": 4.2e6}, "batch": {"min_lib": 4.2e6}},
            cost_s=10,
            code="print('RESULT: {\"min_lib\": ...}')\n",
        ),
    ),
    docs=(
        (
            "https://protocols.example.org/kits/dutp-stranded",
            (
                "Kit notes: the prep kit marks the second strand with dUTP, so reads are "
                "reverse-stranded."
            ),
        ),
        (
            "https://forum.example.org/threads/strandedness-check",
            "To check library strandedness, compare sense and antisense read counts per gene.",
        ),
        (
            "https://geo.example.org/series/toy-knockouts",
            (
                "Series record: T98G and HEK293 libraries were sequenced in separate runs, "
                "months apart, as separate batches."
            ),
        ),
        (
            "https://bixbench3-ground-truth.example.org/task9/answers",
            "Answer key for task 9: strandedness, batches and the design formula.",
        ),
    ),
    contract={
        "artifacts": {
            "de_table.csv": {"columns": ["gene", "log2fc", "pvalue", "padj"]},
            "qc_summary.csv": {"columns": ["sample", "library_size", "strand", "design"]},
        }
    },
)


def _check_columns(artifacts: list[Artifact], contract: dict) -> list[CriticFinding]:
    """Block every artifact whose CSV header misses a column the contract requires."""
    spec = contract.get("artifacts", {}) if isinstance(contract, dict) else {}
    findings = []
    for a in artifacts:
        required = (spec.get(a.name) or {}).get("columns", [])
        try:
            with Path(a.path).open(newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
        except OSError:
            header = []
        missing = [c for c in required if c not in header]
        if missing:
            findings.append(
                CriticFinding(
                    id=f"f-{a.name}-columns",
                    artifact=a.name,
                    severity="block",
                    rule="required_columns",
                    message=f"{a.name} is missing required column(s): {', '.join(missing)}",
                )
            )
    return findings


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _artifact(path: Path) -> Artifact:
    return Artifact(name=path.name, path=path.as_posix(), sha256=_sha256(path))


# --- labs -----------------------------------------------------------------------------------


class FakeLab:
    """Categorical lab over ``CATEGORICAL_SCENARIO`` with a hidden truth per question.

    ``seed`` is accepted so every fake lab has the same constructor; this lab is not random.
    """

    def __init__(self, surprise: bool = False, latency_s: float = 0.0, seed: int = 0) -> None:
        self.seed = seed
        self.name = "fake-surprise" if surprise else "fake"
        self.scenario = CATEGORICAL_SCENARIO
        self.surprise = surprise
        self.truth = {"q1": "mixed_kits" if surprise else "reverse", "q2": "batch"}
        self.latency_s = latency_s
        self.run_id = ""
        self.run_dir: Path | None = None
        self.executor: Executor | None = None
        self.sandbox: str | None = None

    def brief(self) -> dict:
        return {
            "goal": "Reproduce the knockout-vs-control differential expression table",
            "deliverable": "de_table.csv and qc_summary.csv",
            "contract": self.scenario.contract,
            "budget": {"experiments": 8},
            "search_policy": {"deny": ["ground-truth"]},
        }

    def bind(self, run_id: str, run_dir: Path, executor: Executor | None) -> None:
        self.run_id, self.run_dir, self.executor = run_id, Path(run_dir), executor

    async def profile(self) -> list[Evidence]:
        await _pause(self.latency_s)
        return [
            Evidence(
                id="profile-1",
                kind="profile",
                claim=(
                    "Count matrix: 16 libraries x 2,000 genes; cell lines T98G and HEK293; "
                    "control and knockout libraries, two replicates each."
                ),
                author_role="lab",
                source_ref="counts.csv",
            ),
            Evidence(
                id="profile-2",
                kind="profile",
                claim=(
                    "Library sizes range from 4.2M to 9.8M reads; the sample sheet does not "
                    "record the strand protocol or the sequencing batch."
                ),
                author_role="lab",
                source_ref="samples.csv",
            ),
        ]

    def candidate_designs(self) -> list[dict] | None:
        return None

    async def run(self, exp: Experiment) -> Observation:
        await _pause(self.latency_s)
        diag = self.scenario.diagnostic((exp.design or {}).get("diagnostic"))
        if diag is None or diag.question not in self.truth:
            return Observation(experiment_id=exp.id, valid=False, referee_note="unknown diagnostic")
        value = diag.values.get(self.truth[diag.question])
        if value is None:
            return Observation(experiment_id=exp.id, valid=False, referee_note="no result")
        lines = [f"$ {exp.language or 'python'} {diag.name}.py"]
        if self.executor is not None:
            if self.sandbox is None:
                self.sandbox = await self.executor.open(self.name, self.run_id)
            out = await self.executor.exec(self.sandbox, exp.language or "python", exp.code or "")
            lines += [ln for ln in str(out.get("stdout", "")).splitlines() if ln]
            lines += [f"[stderr] {ln}" for ln in str(out.get("stderr", "")).splitlines() if ln]
        lines += ["loaded counts: 16 libraries x 2000 genes", f"RESULT: {json.dumps(value)}"]
        log_ref = ""
        if self.run_dir is not None:
            log_path = self.run_dir / "logs" / f"{exp.id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            log_ref = f"logs/{exp.id}.log"
        return Observation(
            experiment_id=exp.id,
            valid=True,
            value=dict(value),
            bin_id=bin_for(exp.bins, value),
            log_ref=log_ref,
            seconds=diag.cost_s,
            cost_usd=0.0,
        )

    async def deliver(self, state: EngineState) -> list[Artifact]:
        await _pause(self.latency_s)
        out_dir = Path(state.run_dir) / "artifacts"
        answers = {qid: h.answer_key for qid, h in state.answers().items()}
        strand = answers.get("q1") or "unknown"
        design = "~ cell_line + condition" if answers.get("q2") == "batch" else "~ condition"
        # The analyst forgets padj until the critic blocks the table once.
        fixed = any(f.artifact == "de_table.csv" and f.severity == "block" for f in state.findings)
        genes = [("TP53", 2.31, 1e-8), ("EGFR", -1.42, 3e-5), ("MYC", 0.88, 0.004)]
        genes += [("GAPDH", 0.02, 0.91), ("PTEN", -0.65, 0.03)]
        header = ["gene", "log2fc", "pvalue"] + (["padj"] if fixed else [])
        rows = []
        for gene, lfc, p in genes:
            row: list[Any] = [gene, lfc, p]
            if fixed:
                row.append(min(p * len(genes), 1.0))
            rows.append(row)
        de_path = out_dir / "de_table.csv"
        _write_csv(de_path, header, rows)
        qc_path = out_dir / "qc_summary.csv"
        samples = [
            f"{line}_{cond}_{rep}"
            for line in ("T98G", "HEK293")
            for cond in ("ctrl", "ko")
            for rep in (1, 2)
        ]
        sizes = [4.2e6 + 0.7e6 * i for i in range(len(samples))]
        _write_csv(
            qc_path,
            ["sample", "library_size", "strand", "design"],
            [[s, int(n), strand, design] for s, n in zip(samples, sizes, strict=True)],
        )
        return [_artifact(de_path), _artifact(qc_path)]

    async def close(self) -> None:
        if self.executor is not None and self.sandbox is not None:
            await self.executor.close(self.sandbox)
            self.sandbox = None


class FakeSurpriseLab(FakeLab):
    """``FakeLab(surprise=True)``: the hidden strand protocol is one no proposer suggests."""

    def __init__(self, latency_s: float = 0.0, seed: int = 0) -> None:
        super().__init__(surprise=True, latency_s=latency_s, seed=seed)


def _linear(designs: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[:, :1] * designs[:, 0][None, :]


def _quadratic(designs: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[:, :1] * designs[:, 0][None, :] ** 2


def _a_prior(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(1.0, 0.2, size=(n, 1))


_LINEAR_SRC = """import numpy as np


def predict(designs, params):
    return params[:, :1] * designs[:, 0][None, :]


def sample_prior(rng, n):
    return rng.normal(1.0, 0.2, size=(n, 1))
"""

_QUADRATIC_SRC = _LINEAR_SRC.replace("designs[:, 0][None, :]", "designs[:, 0][None, :] ** 2")


class FakeNumericLab:
    """Numeric lab: which law, ``y = a*x`` or ``y = a*x**2``, produced the measurements?"""

    name = "fake-numeric"
    sigma = 0.1
    max_measurements = 10
    measure_cost_s = 1.0

    def __init__(
        self, seed: int = 0, truth: str = "quadratic", a: float = 1.1, latency_s: float = 0.0
    ) -> None:
        if truth not in ("linear", "quadratic"):
            raise ValueError("truth must be 'linear' or 'quadratic'")
        self.truth, self.a, self.latency_s = truth, a, latency_s
        self._rng = np.random.default_rng(seed)
        self.scenario = None

    def brief(self) -> dict:
        return {
            "goal": "Find the law that relates the response y to the setting x",
            "deliverable": "measurements.csv and posterior.csv",
            "contract": {
                "artifacts": {
                    "measurements.csv": {"columns": ["x", "y"]},
                    "posterior.csv": {"columns": ["family", "posterior"]},
                }
            },
            "budget": {"measurements": self.max_measurements},
        }

    async def profile(self) -> list[Evidence]:
        await _pause(self.latency_s)
        return [
            Evidence(
                id="profile-1",
                kind="profile",
                claim="The instrument reads y with noise sd 0.1; x can be set anywhere in [0, 2].",
                author_role="lab",
            )
        ]

    def candidate_designs(self) -> list[dict] | None:
        return [{"x": float(round(v, 6))} for v in np.linspace(0.0, 2.0, 17)]

    def model_families(self) -> list[ModelFamily]:
        return [
            ModelFamily("linear", _linear, _a_prior),
            ModelFamily("quadratic", _quadratic, _a_prior),
        ]

    async def run(self, exp: Experiment) -> Observation:
        await _pause(self.latency_s)
        x = float((exp.design or {}).get("x", 0.0))
        mean = self.a * (x**2 if self.truth == "quadratic" else x)
        y = mean + float(self._rng.normal(0.0, self.sigma))
        return Observation(
            experiment_id=exp.id, valid=True, value=round(y, 6), seconds=self.measure_cost_s
        )

    async def deliver(self, state: EngineState) -> list[Artifact]:
        await _pause(self.latency_s)
        out_dir = Path(state.run_dir) / "artifacts"
        points = state.numeric.get("points", [])
        m_path = out_dir / "measurements.csv"
        _write_csv(m_path, ["x", "y"], [[p["x"], p["y"]] for p in points])
        p_path = out_dir / "posterior.csv"
        post = state.numeric.get("family_posterior", {})
        _write_csv(p_path, ["family", "posterior"], [[k, round(v, 6)] for k, v in post.items()])
        return [_artifact(m_path), _artifact(p_path)]


# --- agents, predictor and services --------------------------------------------------------


class FakeAgents:
    """Every agent role, canned and deterministic. Charges ``FakeLLM`` like real calls would."""

    def __init__(
        self,
        lab: FakeLab | FakeNumericLab,
        llm: FakeLLM | None = None,
        n_proposers: int = 2,
        reject_experiments: tuple[str, ...] = (),
        latency_s: float = 0.0,
    ) -> None:
        self.lab = lab
        self.llm = llm
        self.n_proposers = n_proposers
        self.reject_experiments = set(reject_experiments)
        self.latency_s = latency_s
        self.critic_rounds = 0

    async def _call(self, role: str, n: int = 1) -> None:
        await _pause(self.latency_s)
        if self.llm is not None:
            self.llm.charge(role, n)

    async def propose(
        self, brief: dict, evidence: list[Evidence]
    ) -> tuple[list[Question], list[Hypothesis]]:
        await self._call("proposer", self.n_proposers)
        if isinstance(self.lab, FakeNumericLab):
            q = Question(
                id="q1",
                text="Which law relates y to x?",
                kind="model",
                raised_by="proposer-1",
            )
            hyps = [
                Hypothesis(
                    id=f"q1.{key}",
                    question_id="q1",
                    statement=text,
                    author=f"proposer-{i + 1}",
                    rationale="Fits the first look at the setup.",
                    answer_key=key,
                    model_code=src,
                    prior=1.0,
                    posterior=1.0,
                )
                for i, (key, text, src) in enumerate(
                    [
                        ("linear", "y grows linearly with x: y = a*x", _LINEAR_SRC),
                        ("quadratic", "y grows with the square of x: y = a*x^2", _QUADRATIC_SRC),
                    ]
                )
            ]
            return [q], hyps
        questions, hyps = [], []
        for fq in self.lab.scenario.questions:
            questions.append(
                Question(
                    id=fq.id,
                    text=fq.text,
                    kind="categorical",
                    impact=fq.impact,
                    raised_by="proposer-1",
                )
            )
            for opt in fq.options:
                hyps.append(self._hypothesis(fq.id, opt, "proposer-1"))
            if fq.duplicate and self.n_proposers > 1:
                opt = next(o for o in fq.options if o.key == fq.duplicate)
                dup = self._hypothesis(fq.id, opt, "proposer-2")
                hyps.append(dup.model_copy(update={"id": f"{dup.id}.p2"}))
        return questions, hyps

    @staticmethod
    def _hypothesis(qid: str, opt: FakeOption, author: str) -> Hypothesis:
        return Hypothesis(
            id=f"{qid}.{opt.key}",
            question_id=qid,
            statement=opt.statement,
            author=author,
            rationale=opt.rationale,
            answer_key=opt.key,
            prior=1.0,
            posterior=1.0,
        )

    async def design(
        self,
        question: Question,
        hypotheses: list[Hypothesis],
        evidence: list[Evidence],
        history: list[Trial],
    ) -> list[Experiment]:
        await self._call("designer")
        if question.kind == "model":
            return [
                Experiment(
                    id="midpoint",
                    question_ids=[question.id],
                    kind="measure",
                    goal="measure in the middle of the range",
                    design={"x": 1.0},
                )
            ]
        scenario = self.lab.scenario
        assert scenario is not None
        out = []
        for d in scenario.diagnostics:
            if d.question != question.id:
                continue
            out.append(
                Experiment(
                    id=d.name,
                    question_ids=[question.id],
                    kind=d.kind,  # type: ignore[arg-type]
                    goal=d.goal,
                    language="python" if d.code else None,
                    code=d.code,
                    query=d.query,
                    design={"diagnostic": d.name},
                    bins=d.outcome_bins(),
                    est_cost_s=d.cost_s,
                )
            )
        return out

    async def referee(self, experiment: Experiment, observation: Observation) -> tuple[bool, str]:
        await self._call("referee")
        name = (experiment.design or {}).get("diagnostic", experiment.id)
        if experiment.id in self.reject_experiments or name in self.reject_experiments:
            return False, "the script read the wrong column, so the result does not test the claim"
        return True, f"bin '{observation.bin_id}' follows from the RESULT line"

    async def critic(self, artifacts: list[Artifact], contract: dict) -> list[CriticFinding]:
        await self._call("critic")
        self.critic_rounds += 1
        return _check_columns(artifacts, contract)

    async def report(self, state: EngineState, artifacts: list[Artifact]) -> Report:
        await self._call("reporter")
        lines = ["# Methods", ""]
        for q in state.questions.values():
            best = state.best(q.id)
            runs = [t.experiment.id for t in state.trials_for(q.id)]
            answer = f"'{best.statement}' (posterior {best.posterior:.2f})" if best else "none"
            lines.append(f"- **{q.text}** {q.status}: {answer}; checks: {', '.join(runs) or '-'}")
        if state.rejected:
            lines += ["", "## Rejected options", ""]
            lines += [f"- {r['hypothesis']}: {r['reason']}" for r in state.rejected]
        lines += ["", "## Artifacts", ""] + [f"- {a.name} ({a.sha256[:12]})" for a in artifacts]
        return Report(
            run_id=state.run_id,
            artifacts=artifacts,
            methods_md="\n".join(lines),
            rejected=[],
            unresolved=[],
            cost_usd=0.0,
            seconds=0.0,
        )

    async def regenerate(
        self, question: Question, hypotheses: list[Hypothesis], history: list[Trial]
    ) -> list[Hypothesis]:
        await self._call("proposer")
        scenario = self.lab.scenario
        fq = scenario.question(question.id) if scenario else None
        if fq is None:
            return []
        fresh = [self._hypothesis(fq.id, o, "proposer-1") for o in fq.regenerated]
        # Re-proposing a known option is harmless: the orchestrator drops it.
        return fresh + [self._hypothesis(fq.id, fq.options[-1], "proposer-2")]


class FakePredictor:
    """Likelihood tables from the scenario.

    An option that fully determines the bin gets an exact table (with the ``eps`` noise
    allowance); otherwise ``n_samples`` predictions are drawn from the option's distribution
    with a seeded generator and smoothed with ``likelihood_from_samples``.
    """

    def __init__(
        self,
        lab: FakeLab | FakeNumericLab,
        seed: int = 0,
        n_samples: int = 8,
        eps: float = 0.1,
        llm: FakeLLM | None = None,
    ) -> None:
        self.scenario = lab.scenario
        self.seed, self.n_samples, self.eps, self.llm = seed, n_samples, eps, llm

    async def predict(
        self, exp: Experiment, hypotheses: list[Hypothesis], context: str
    ) -> dict[str, dict[str, float]]:
        diag = (
            self.scenario.diagnostic((exp.design or {}).get("diagnostic"))
            if self.scenario
            else None
        )
        if diag is None:
            return {}
        bins = [b.id for b in exp.bins]
        n_bins = len(bins)
        exact_only = exp.likelihood_source == "exact"
        out: dict[str, dict[str, float]] = {}
        sampled = False
        for h in hypotheses:
            pred = diag.predicts.get(h.answer_key or "")
            if isinstance(pred, str) and pred in bins:
                out[h.id] = {b: (1.0 - self.eps) * (b == pred) + self.eps / n_bins for b in bins}
                continue
            if exact_only:
                return {}
            dist = pred if isinstance(pred, dict) else {b: 1.0 for b in bins}
            weights = [max(float(dist.get(b, 0.0)), 0.0) for b in bins]
            rng = random.Random(f"{self.seed}:{diag.name}:{h.answer_key}")
            counts = dict.fromkeys(bins, 0)
            for b in rng.choices(bins, weights=weights, k=self.n_samples):
                counts[b] += 1
            out[h.id] = dict(scoring.likelihood_from_samples(counts, n_bins, self.eps))
            sampled = True
        if sampled and self.llm is not None:
            self.llm.charge("predictor")
        exp.likelihood_source = "sampled" if sampled else "exact"
        return out


class FakeExecutor:
    """Records calls and returns scripted output; never executes anything."""

    def __init__(self, script: Callable[[str, str], str] | None = None) -> None:
        self.script = script
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.calls: list[tuple[str, str, str]] = []

    async def open(self, lab_name: str, run_id: str) -> str:
        sid = f"fake-sandbox-{len(self.opened) + 1}"
        self.opened.append(sid)
        return sid

    async def exec(self, sandbox_id: str, language: str, code: str, timeout_s: int = 120) -> dict:
        self.calls.append((sandbox_id, language, code))
        if self.script is not None:
            stdout = self.script(language, code)
        else:
            stdout = f"[sandbox {sandbox_id}] {language}: {len(code.splitlines())} line(s), net off"
        return {"stdout": stdout, "stderr": "", "exit_code": 0, "seconds": 0.0}

    async def close(self, sandbox_id: str) -> None:
        self.closed.append(sandbox_id)


class FakeMemory:
    """An in-process evidence store that remembers who wrote each record. Emits nothing.

    ``recall`` ranks records by word overlap with the query (ties keep write order). Role
    filtering for the ``role_only`` ablation is done by the orchestrator, not here.
    """

    backend = "fake"

    def __init__(self) -> None:
        self.records: list[tuple[str, Evidence]] = []

    async def write(self, ev: Evidence, run_id: str, author_role: str) -> None:
        if any(r == run_id and e.id == ev.id for r, e in self.records):
            return
        self.records.append((run_id, ev.model_copy(update={"author_role": author_role})))

    async def recall(self, query: str, run_id: str, reader_role: str, k: int = 8) -> list[Evidence]:
        words = _tokens(query)
        scored = []
        for i, (_, ev) in enumerate(self.records):
            overlap = len(words & _tokens(ev.claim))
            if overlap:
                scored.append((-overlap, i, ev))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [ev for _, _, ev in scored[:k]]


class FakeSearch:
    """Keyword search over canned documents with a deny-list leakage policy."""

    def __init__(
        self, docs: tuple[tuple[str, str], ...] = (), deny: tuple[str, ...] = ("ground-truth",)
    ) -> None:
        self.docs = tuple(docs)
        self.deny = tuple(deny)
        self.last_blocked: list[str] = []
        self.queries: list[str] = []

    async def find(self, query: str, k: int = 5) -> list[Evidence]:
        self.queries.append(query)
        words = _tokens(query)
        hits = []
        for i, (url, text) in enumerate(self.docs):
            overlap = len(words & _tokens(text))
            if overlap:
                hits.append((-overlap, i, url, text))
        hits.sort(key=lambda t: (t[0], t[1]))
        self.last_blocked = [u for _, _, u, _ in hits if any(d in u for d in self.deny)]
        allowed = [(u, t) for _, _, u, t in hits if u not in self.last_blocked]
        return [
            Evidence(
                id=f"search-{hashlib.sha256(url.encode()).hexdigest()[:8]}",
                kind="search",
                claim=text,
                author_role="search",
                source_ref=query,
                url=url,
            )
            for url, text in allowed[:k]
        ]


def fake_deps(
    lab: FakeLab | FakeNumericLab,
    *,
    seed: int = 0,
    clock: Callable[[], float] | None = None,
    latency_s: float = 0.0,
    reject_experiments: tuple[str, ...] = (),
    n_proposers: int = 2,
) -> Any:
    """``core.orchestrator.Deps`` wired with fakes.

    ``clock`` defaults to wall time (``time.time``); pass a ``FakeClock`` for event files that
    are identical across runs.
    """
    from core.orchestrator import (
        Deps,  # local import: the orchestrator imports nothing from here
    )

    llm = FakeLLM()
    return Deps(
        agents=FakeAgents(
            lab,
            llm=llm,
            n_proposers=n_proposers,
            reject_experiments=reject_experiments,
            latency_s=latency_s,
        ),
        predictor=FakePredictor(lab, seed=seed, llm=llm),
        executor=FakeExecutor(),
        memory=FakeMemory(),
        search=FakeSearch(lab.scenario.docs if lab.scenario else ()),
        llm=llm,
        clock=clock if clock is not None else time.time,
    )
