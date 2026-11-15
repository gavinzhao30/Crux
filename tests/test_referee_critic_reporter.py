"""The blind referee, the end critic and the reporter, on a scripted model (no network)."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from agents.critic import RULES, critique, deliverable_specs
from agents.referee import BIN_FIELDS, SPEC_FIELDS, referee
from agents.reporter import render_template, report, report_facts, stray_numbers
from core.budget import BudgetExhausted
from core.config import load_config
from core.fakes_engine import FakeAgents, FakeClock, FakeLab, fake_deps
from core.orchestrator import run
from core.schemas import (
    Artifact,
    BeliefUpdate,
    EngineState,
    Evidence,
    Experiment,
    Hypothesis,
    Observation,
    OutcomeBin,
    Question,
    Trial,
)
from labs.bixbench3.task9 import _DE_COLUMNS
from labs.bixbench3.task9_contract import load_contract


class ScriptedLLM:
    """Replies per role: a dict, a list (used in order), a callable or an exception."""

    def __init__(self, **script: Any) -> None:
        self.script = {
            k: list(v) if isinstance(v, list) else v for k, v in script.items()
        }
        self.calls: list[dict] = []

    async def structured(
        self, role: str, schema: type[BaseModel], messages: list[dict], n: int = 1
    ) -> list[BaseModel]:
        self.calls.append(
            {"role": role, "schema": schema.__name__, "messages": messages}
        )
        reply = self.script[role]
        if isinstance(reply, list):
            reply = reply.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            reply = reply(messages)
        return [schema.model_validate(reply)] * n


def user_payload(call: dict) -> dict:
    return json.loads(
        next(m["content"] for m in call["messages"] if m["role"] == "user")
    )


def all_text(call: dict) -> str:
    return "\n".join(m["content"] for m in call["messages"])


# --- referee -------------------------------------------------------------------------------

HYPOTHESES = {
    "q1.reverse": "The libraries were made with a reverse-stranded dUTP kit",
    "q1.unstranded": "The libraries are unstranded, so counts ignore strand",
}


def strand_experiment() -> Experiment:
    return Experiment(
        id="strand_ratio",
        question_ids=["q1"],
        kind="compute",
        goal="Count genes both ways and compare forward to reverse totals",
        language="python",
        code=(
            "import pandas as pd\n"
            "c = pd.read_csv('/data/counts.tsv', sep='\\t')\n"
            "print('RESULT: ' + json.dumps({'ratio': c.fwd.sum() / c.rev.sum()}))\n"
        ),
        design={"diagnostic": "strand_ratio", "favours": HYPOTHESES["q1.reverse"]},
        bins=[
            OutcomeBin(
                id="rev_gt_fwd", description="reverse dominates", rule="ratio < 0.67"
            ),
            OutcomeBin(
                id="similar", description="similar", rule="0.67 <= ratio <= 1.5"
            ),
        ],
        likelihood={
            "q1.reverse": {"rev_gt_fwd": 0.9, "similar": 0.1},
            "q1.unstranded": {"rev_gt_fwd": 0.1, "similar": 0.9},
        },
        eig_bits=0.53,
        score=0.0044,
        spec_hash="abc123",
    )


def good_observation(**update: Any) -> Observation:
    obs = Observation(
        experiment_id="strand_ratio",
        valid=True,
        value={"ratio": 0.07},
        bin_id="rev_gt_fwd",
        surprise=0.11,
    )
    return obs.model_copy(update=update)


VERDICT = {
    "valid": True,
    "note": "reads the counts and divides forward by reverse totals",
}
TRACEBACK = ["loading", "Traceback (most recent call last):", "KeyError: 'fwd'"]


@pytest.mark.parametrize(
    ("update", "log_lines", "expected"),
    [
        ({"valid": False, "referee_note": "timed out"}, None, "invalid"),
        ({"value": None}, None, "no result"),
        ({"value": 0.07}, None, "not a JSON object"),
        ({"value": [0.07]}, None, "not a JSON object"),
        ({"value": {}}, None, "empty result"),
        ({"bin_id": "made_up"}, None, "not one of the bins"),
        ({"bin_id": None}, None, "not one of the bins"),
        ({}, TRACEBACK, "error"),
        ({}, ["[stderr] Error in DESeq(dds) : all genes have zero counts"], "error"),
        ({}, ["[stderr] Execution halted"], "error"),
    ],
)
def test_referee_rejects_in_code_without_calling_the_model(
    tmp_path: Path, update: dict, log_lines: list[str] | None, expected: str
) -> None:
    if log_lines is not None:
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "strand_ratio.log").write_text(
            "\n".join(log_lines), "utf-8"
        )
        update = {**update, "log_ref": "logs/strand_ratio.log"}
    llm = ScriptedLLM(referee=VERDICT)
    obs = good_observation(**update)
    valid, note = asyncio.run(referee(llm, strand_experiment(), obs, run_dir=tmp_path))
    assert valid is False
    assert expected in note and "\n" not in note
    assert llm.calls == []


def test_referee_reads_the_sandbox_event_log_for_one_command(tmp_path: Path) -> None:
    events = [
        {
            "type": "experiment.log",
            "command_id": "other",
            "line": "Traceback (most recent",
        },
        {
            "type": "experiment.log",
            "command_id": "abc",
            "line": "loaded",
            "stream": "stdout",
        },
        {
            "type": "execution.failed",
            "command_id": "abc",
            "error": "Command exited with 1",
        },
    ]
    log = tmp_path / "execution.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    llm = ScriptedLLM(referee=VERDICT)
    obs = good_observation(log_ref="execution.jsonl#abc")
    valid, note = asyncio.run(referee(llm, strand_experiment(), obs, run_dir=tmp_path))
    assert (valid, llm.calls) == (False, [])
    assert "Command exited with 1" in note

    clean = good_observation(log_ref="execution.jsonl#nothing-failed")
    assert (
        asyncio.run(referee(llm, strand_experiment(), clean, run_dir=tmp_path))[0]
        is True
    )


def test_referee_prompt_is_blind_to_hypotheses(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        'loaded counts: 16 x 2000\nRESULT: {"ratio": 0.07}\n', encoding="utf-8"
    )
    llm = ScriptedLLM(referee=VERDICT)
    obs = good_observation(log_ref=str(log), referee_note="favours q1.reverse")
    valid, note = asyncio.run(referee(llm, strand_experiment(), obs))
    assert (valid, note) == (True, VERDICT["note"])

    [call] = llm.calls
    assert call["role"] == "referee"
    payload = user_payload(call)
    assert set(payload) == {"spec", "bins", "observed_bin", "result", "log_tail"}
    assert set(payload["spec"]) <= set(SPEC_FIELDS)
    assert all(set(b) == set(BIN_FIELDS) for b in payload["bins"])
    assert payload["result"] == {"ratio": 0.07}
    assert payload["log_tail"][-1].startswith("RESULT:")
    text = all_text(call)
    for hid, statement in HYPOTHESES.items():
        assert hid not in text
        assert statement not in text
    for leak in (
        "likelihood",
        "posterior",
        "eig_bits",
        "favours",
        "surprise",
        "abc123",
    ):
        assert leak not in json.dumps(payload)


def test_referee_rejects_when_the_model_call_fails_but_not_on_budget() -> None:
    llm = ScriptedLLM(referee=RuntimeError("schema violation"))
    valid, note = asyncio.run(referee(llm, strand_experiment(), good_observation()))
    assert valid is False and "could not judge" in note

    broke = ScriptedLLM(referee=BudgetExhausted("cap"))
    with pytest.raises(BudgetExhausted):
        asyncio.run(referee(broke, strand_experiment(), good_observation()))


# --- critic --------------------------------------------------------------------------------

DE_KEY = "t98g_c10_vs_control_degs"
DE_FILE = f"{DE_KEY}.csv"
DE = list(_DE_COLUMNS)


def task9_contract() -> dict:
    """One real Task 9 DE spec, plus a gene list whose regulation labels are defined."""
    contract = load_contract()
    shared = {**contract["t98g_slow_shared_degs"], "regulation_labels": ["up", "down"]}
    return {DE_KEY: contract[DE_KEY], "t98g_slow_shared_degs": shared}


def write_rows(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def artifact(path: Path, sha256: str | None = None) -> Artifact:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Artifact(name=path.name, path=str(path), sha256=sha256 or digest)


def de_rows(n: int = 6) -> list[list[Any]]:
    rows = []
    for i in range(n):
        p = 10.0 ** -(i + 1)
        gene = f"ENSG{i + 1:011d}"
        rows.append(
            [gene, f"GENE{i}", 100.0 + 10 * i, 1.5 - i, 0.2, 7.5, p, min(p * n, 1.0)]
        )
    return rows


def cell(row_index: int, column: str, value: Any) -> list[list[Any]]:
    rows = de_rows()
    rows[row_index][DE.index(column)] = value
    return rows


def gene_list(tmp_path: Path, labels: list[str]) -> Artifact:
    path = tmp_path / "t98g_slow_shared_degs.csv"
    rows = [[f"GENE{i}", label] for i, label in enumerate(labels)] or [["GENE0", "up"]]
    return artifact(write_rows(path, ["gene_symbol", "regulation"], rows))


def good_artifacts(tmp_path: Path) -> list[Artifact]:
    de = write_rows(tmp_path / DE_FILE, DE, de_rows())
    return [artifact(de), gene_list(tmp_path, ["up", "down", "up"])]


def with_de(tmp_path: Path, header: list[str], rows: list[list[Any]]) -> list[Artifact]:
    de = write_rows(tmp_path / DE_FILE, header, rows)
    return [artifact(de), gene_list(tmp_path, ["up", "down"])]


def raw_de(tmp_path: Path, data: bytes) -> list[Artifact]:
    path = tmp_path / DE_FILE
    path.write_bytes(data)
    return [artifact(path), gene_list(tmp_path, ["up"])]


def wrong_hash(tmp_path: Path) -> list[Artifact]:
    de, shared = good_artifacts(tmp_path)
    return [de.model_copy(update={"sha256": "0" * 64}), shared]


HEADER = ",".join(_DE_COLUMNS).encode()
VIOLATIONS = {  # case -> (the one rule it must trip, the artifacts)
    "missing": ("missing_deliverable", lambda tmp: good_artifacts(tmp)[1:]),
    "ragged": (
        "unparseable",
        lambda tmp: raw_de(tmp, HEADER + b"\nENSG00000000001,A,1\n"),
    ),
    "binary": ("unparseable", lambda tmp: raw_de(tmp, HEADER + b"\n\xff\xfe\x00bad\n")),
    "no_padj": (
        "missing_columns",
        lambda tmp: with_de(tmp, DE[:-1], [r[:-1] for r in de_rows()]),
    ),
    "header_only": ("empty_table", lambda tmp: raw_de(tmp, HEADER + b"\n")),
    "zero_bytes": ("empty_table", lambda tmp: raw_de(tmp, b"")),
    "symbol_key": (
        "row_key_pattern",
        lambda tmp: with_de(tmp, DE, cell(2, "gene_id", "TP53")),
    ),
    "dup_key": (
        "duplicate_keys",
        lambda tmp: with_de(tmp, DE, de_rows() + de_rows()[:1]),
    ),
    "padj_text": ("padj_invalid", lambda tmp: with_de(tmp, DE, cell(1, "padj", "low"))),
    "padj_range": ("padj_invalid", lambda tmp: with_de(tmp, DE, cell(1, "padj", 1.7))),
    "versioned": (
        "versioned_ids",
        lambda tmp: with_de(tmp, DE, cell(0, "gene_id", "ENSG00000000001.14")),
    ),
    "labels": (
        "invalid_labels",
        lambda tmp: [good_artifacts(tmp)[0], gene_list(tmp, ["up", "sideways"])],
    ),
    "sha256": ("sha256_mismatch", wrong_hash),
}


@pytest.mark.parametrize("case", sorted(VIOLATIONS))
def test_critic_blocks_each_contract_violation(tmp_path: Path, case: str) -> None:
    rule, build = VIOLATIONS[case]
    llm = ScriptedLLM()
    findings = asyncio.run(critique(llm, build(tmp_path), task9_contract()))
    assert [f.rule for f in findings] == [rule]
    [finding] = findings
    assert finding.severity == "block" and finding.rule in RULES
    expected = "t98g_slow_shared_degs.csv" if rule == "invalid_labels" else DE_FILE
    assert finding.artifact == expected
    assert finding.id == f"{expected}:{rule}"
    assert llm.calls == []


def test_critic_passes_a_good_table(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    assert asyncio.run(critique(llm, good_artifacts(tmp_path), task9_contract())) == []
    # padj NA (DESeq2's independent filtering) is allowed, as in validate_de_table
    with_na = with_de(tmp_path, DE, cell(3, "padj", "NA"))
    assert asyncio.run(critique(llm, with_na, task9_contract())) == []


def test_critic_reads_the_whole_task9_contract_and_the_fake_shape() -> None:
    specs = deliverable_specs(
        {**load_contract(), "recalled_evidence": [{"claim": "x"}]}
    )
    assert [s.key for s in specs] == list(load_contract())
    de = next(s for s in specs if s.key == DE_KEY)
    assert (de.path, de.row_key, de.unversioned, de.delimiter) == (
        DE_FILE,
        "gene_id",
        True,
        ",",
    )
    fake = deliverable_specs(FakeLab().brief()["contract"])
    assert [(s.path, s.required_columns[-1]) for s in fake] == [
        ("de_table.csv", "padj"),
        ("qc_summary.csv", "design"),
    ]


def write_counts(tmp_path: Path, n_genes: int = 40) -> tuple[Path, dict[str, float]]:
    """A featureCounts matrix with versioned ids, and DESeq2's base mean worked out by hand."""
    rng = np.random.default_rng(7)
    means = np.exp(rng.normal(4.0, 2.2, size=n_genes))  # a realistic spread
    depth = np.array([0.8, 1.0, 1.25, 0.9, 1.1, 1.0])
    counts = rng.poisson(means[:, None] * depth[None, :])
    counts[0, 2] = 0  # a gene with a zero count is left out of the size factors
    ids = [f"ENSG{i + 1:011d}.{i % 5 + 1}" for i in range(n_genes)]
    samples = "\t".join(f"S{j}" for j in range(depth.size))
    lines = [
        "# Program:featureCounts v2.0.6",
        f"Geneid\tChr\tStart\tEnd\tStrand\tLength\t{samples}",
    ]
    for gene, row in zip(ids, counts, strict=True):
        lines.append(f"{gene}\tchr1\t1\t100\t+\t100\t" + "\t".join(map(str, row)))
    path = tmp_path / "tagseq_featurecounts_matrix.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    positive = counts[(counts > 0).all(axis=1)].astype(float)
    log_geo = np.log(positive).mean(axis=1)
    size_factors = np.exp(np.median(np.log(positive) - log_geo[:, None], axis=0))
    base_mean = (counts / size_factors).mean(axis=1)
    return path, {
        g.split(".")[0]: float(b) for g, b in zip(ids, base_mean, strict=True)
    }


def de_with_base_mean(
    tmp_path: Path, genes: list[str], values: list[float]
) -> Artifact:
    rows = [
        [gene, f"G{i}", round(v, 4), 0.5, 0.1, 5.0, 0.01, 0.02]
        for i, (gene, v) in enumerate(zip(genes, values, strict=True))
    ]
    return artifact(write_rows(tmp_path / DE_FILE, DE, rows))


def test_base_mean_spot_check_catches_constant_placeholders(tmp_path: Path) -> None:
    counts_path, base_mean = write_counts(tmp_path)
    genes, truth = list(base_mean), list(base_mean.values())
    contract = {DE_KEY: load_contract()[DE_KEY]}
    llm = ScriptedLLM()

    def check(values: list[float], counts: Path | None = counts_path) -> list:
        table = de_with_base_mean(tmp_path, genes, values)
        return asyncio.run(critique(llm, [table], contract, counts_path=counts))

    assert check(truth) == []

    constant = check([100.0] * len(genes))
    assert [(f.rule, f.severity) for f in constant] == [("base_mean_mismatch", "block")]
    assert "constant" in constant[0].message

    logged = list(np.log2(np.array(truth) + 1.0))
    assert [f.rule for f in check(logged)] == ["base_mean_mismatch"]
    shuffled = list(np.random.default_rng(1).permutation(truth))
    assert [f.rule for f in check(shuffled)] == ["base_mean_mismatch"]
    # without a count matrix the spot check is skipped
    assert check([100.0] * len(genes), counts=None) == []


METHODS_TEXT = (
    "# Methods\nWe ran DESeq2 with design ~ condition. The F7 comparison was skipped "
    "because it timed out. Gene ids keep their version suffix.\n"
)
GUIDANCE = (
    "Use biological replicates. Use the supplied artifact names and column names."
)
RECALLED = {
    "id": "e1",
    "kind": "lesson",
    "claim": "Earlier runs forgot the HEK293 shared set.",
}


def issue(kind: str, message: str, quote: str = "") -> dict:
    return {"kind": kind, "message": message, "contract_quote": quote}


def methods_review(messages: list[dict]) -> dict:
    return {
        "issues": [
            issue("stopped_early", "The F7 comparison was skipped."),
            issue("omission", "The shared gene set is not described."),
            issue(
                "contract_violation",
                "Versioned ids break the contract.",
                "Use the supplied artifact names and column names",
            ),
            issue(
                "contract_violation", "Uses shrinkage.", "lfcShrink must never be used"
            ),
        ]
    }


def methods_artifact(tmp_path: Path) -> Artifact:
    path = tmp_path / "METHODS.md"
    path.write_text(METHODS_TEXT, encoding="utf-8")
    return artifact(path)


def test_critic_reviews_methods_with_stable_ids(tmp_path: Path) -> None:
    contract = {
        **task9_contract(),
        "method_guidance": GUIDANCE,
        "recalled_evidence": [RECALLED],
    }
    methods = methods_artifact(tmp_path)
    bad = with_de(tmp_path, DE, cell(0, "gene_id", "ENSG00000000001.2"))
    llm = ScriptedLLM(critic=methods_review)

    first = asyncio.run(critique(llm, [*bad, methods], contract))
    second = asyncio.run(critique(llm, [*bad, methods], contract))
    assert [f.model_dump() for f in first] == [f.model_dump() for f in second]
    assert [f.id for f in first] == [
        f"{DE_FILE}:versioned_ids",
        "METHODS.md:methods_stopped_early",
        "METHODS.md:methods_omission",
        "METHODS.md:methods_contract_violation",
    ]
    severity = {f.rule: f.severity for f in first}
    assert severity["methods_stopped_early"] == "warn"
    assert severity["methods_omission"] == "warn"
    # one of the two quotes is verbatim in the guidance, so the merged finding blocks
    assert severity["methods_contract_violation"] == "block"

    call = llm.calls[0]
    payload = user_payload(call)
    assert call["role"] == "critic"
    assert payload["methods_md"] == METHODS_TEXT
    assert GUIDANCE in payload["method_guidance"]
    assert payload["recalled_evidence"] == [
        {"kind": "lesson", "claim": RECALLED["claim"]}
    ]
    assert f"{DE_FILE}: versioned_ids" in payload["code_check_failures"]
    assert not any(RECALLED["claim"] in f.message for f in first)

    # the repaired table keeps every rule name, so its old id simply disappears
    later = asyncio.run(critique(llm, [*good_artifacts(tmp_path), methods], contract))
    assert {f.id for f in later} == {f.id for f in first} - {f"{DE_FILE}:versioned_ids"}


def test_critic_model_findings_without_a_verified_quote_only_warn(
    tmp_path: Path,
) -> None:
    artifacts = [*good_artifacts(tmp_path), methods_artifact(tmp_path)]
    reply = {
        "issues": [
            issue("contract_violation", "Uses shrinkage.", "never use lfcShrink")
        ]
    }
    findings = asyncio.run(
        critique(ScriptedLLM(critic=reply), artifacts, task9_contract())
    )
    assert [(f.id, f.severity) for f in findings] == [
        ("METHODS.md:methods_contract_violation", "warn")
    ]

    failing = ScriptedLLM(critic=RuntimeError("refused"))
    findings = asyncio.run(critique(failing, artifacts, task9_contract()))
    assert [(f.id, f.severity) for f in findings] == [
        ("METHODS.md:methods_unreviewed", "warn")
    ]


# --- reporter ------------------------------------------------------------------------------

REJECTED_STATEMENT = "The libraries are unstranded, so counts ignore strand"
EVIDENCE_CLAIM = "Count genes both ways: observed 'rev_gt_fwd' (reverse dominates)"


def hypothesis(hid: str, statement: str, posterior: float, **kw: Any) -> Hypothesis:
    return Hypothesis(
        id=hid,
        question_id=hid.split(".")[0],
        statement=statement,
        author="statistician",
        rationale="from the profile",
        prior=0.5,
        posterior=posterior,
        **kw,
    )


def question(qid: str, text: str, status: str) -> Question:
    return Question(
        id=qid, text=text, kind="categorical", status=status, raised_by="sceptic"
    )


def engine_state(tmp_path: Path) -> tuple[EngineState, list[Artifact]]:
    questions = [
        question("q1", "Which strand protocol was used?", "settled"),
        question("q2", "Should the design include batch?", "robust"),
        question("q3", "Which outlier rule applies?", "unresolved"),
    ]
    hyps = [
        hypothesis(
            "q1.reverse",
            "The libraries were made with a reverse-stranded kit",
            0.97,
            status="accepted",
        ),
        hypothesis("q1.unstranded", REJECTED_STATEMENT, 0.0, status="rejected"),
        hypothesis(
            "q2.batch", "Include sequencing batch", 0.5, answer_key="same_table"
        ),
        hypothesis("q2.none", "Leave batch out", 0.5, answer_key="same_table"),
        hypothesis("q3.cooks", "Use Cook's distance filtering", 0.55),
        hypothesis("q3.none", "Keep every sample", 0.45),
    ]
    exp = strand_experiment().model_copy(update={"status": "done"})
    obs = good_observation(referee_note="reads the counts and divides the totals")
    de = write_rows(tmp_path / DE_FILE, DE, de_rows())
    artifacts = [artifact(de).model_copy(update={"contract_ok": True})]
    evidence = Evidence(
        id="ev-1",
        kind="observation",
        claim=EVIDENCE_CLAIM,
        author_role="referee",
        source_ref=exp.id,
    )
    state = EngineState(
        run_id="r-1",
        run_dir=str(tmp_path),
        lab="bixbench3",
        brief={"goal": "Reproduce the DESeq2 comparisons"},
        questions={q.id: q for q in questions},
        hypotheses={h.id: h for h in hyps},
        experiments={exp.id: exp},
        trials=[Trial(experiment=exp, observation=obs)],
        updates=[
            BeliefUpdate(
                question_id="q1",
                experiment_id=exp.id,
                prior={"q1.reverse": 0.5, "q1.unstranded": 0.5},
                posterior={"q1.reverse": 0.97, "q1.unstranded": 0.03},
            )
        ],
        evidence={evidence.id: evidence},
        artifacts=artifacts,
        rejected=[
            {
                "hypothesis": "q1.unstranded",
                "question": "q1",
                "evidence_id": "ev-1",
                "reason": "posterior 0.0300 fell below 0.05 after strand_ratio",
            }
        ],
        budget={"cost_usd": 0.42, "llm_calls": 9, "seconds": 61.5},
        delivery_round=1,
    )
    return state, artifacts


def faithful_prose(messages: list[dict]) -> dict:
    """A model that writes from the facts it was given, citing ids and copying numbers."""
    facts = json.loads(messages[-1]["content"])
    lines = ["# Methods", "", "## Questions and diagnostics", ""]
    for q in facts["questions"]:
        lines.append(f"Question [{q['id']}] ended {q['status']}.")
        if q.get("settled_on"):
            s = q["settled_on"]
            lines.append(
                f"It settled on [{s['option']}] with posterior {s['posterior']}."
            )
    lines += ["", "## Rejected options", ""]
    for r in facts["rejected"]:
        lines.append(
            f"We rejected [{r['hypothesis']}] ({r['statement']}) because {r['reason']}; "
            f"the evidence [{r['evidence_id']}] was: {r['evidence']}."
        )
    return {"methods_md": "\n".join(lines)}


def test_reporter_text_includes_a_rejected_option_and_its_evidence(
    tmp_path: Path,
) -> None:
    state, artifacts = engine_state(tmp_path)
    llm = ScriptedLLM(reporter=faithful_prose)
    result = asyncio.run(report(llm, state, artifacts))

    assert REJECTED_STATEMENT in result.methods_md
    assert EVIDENCE_CLAIM in result.methods_md
    # the model's prose was kept, not replaced by the template
    assert "We rejected [q1.unstranded]" in result.methods_md
    call = llm.calls[0]
    facts = user_payload(call)
    assert call["role"] == "reporter"
    assert facts["rejected"][0]["statement"] == REJECTED_STATEMENT
    assert facts["rejected"][0]["evidence"] == EVIDENCE_CLAIM
    assert facts["rejected"][0]["experiment"] == "strand_ratio"
    assert facts["unresolved"][0]["question"] == "q3"
    assert [o["option"] for o in facts["unresolved"][0]["top_options"]] == [
        "q3.cooks",
        "q3.none",
    ]
    assert facts["robust"] == [
        {
            "question": "q2",
            "text": "Should the design include batch?",
            "answer_key": "same_table",
            "options": ["q2.batch", "q2.none"],
        }
    ]
    assert result.unresolved == ["q3"]
    assert (result.run_id, result.cost_usd, result.seconds) == ("r-1", 0.42, 61.5)
    assert result.rejected[0]["evidence"] == EVIDENCE_CLAIM


def test_reporter_falls_back_to_the_template_when_the_model_raises(
    tmp_path: Path,
) -> None:
    state, artifacts = engine_state(tmp_path)
    llm = ScriptedLLM(reporter=RuntimeError("model unavailable"))
    result = asyncio.run(report(llm, state, artifacts))

    text = result.methods_md
    facts = report_facts(state, artifacts)
    assert text == render_template(facts)
    assert f"[q1.unstranded] {REJECTED_STATEMENT}" in text
    assert EVIDENCE_CLAIM in text and "[strand_ratio]" in text
    assert "Settled on [q1.reverse]" in text and "(posterior 0.97)" in text
    assert "`same_table`" in text
    assert "[q3.cooks] Use Cook's distance filtering (0.55)" in text
    assert "both are reported" in text
    assert f"`{DE_FILE}` passed the contract checks" in text
    assert stray_numbers(text, facts) == []


def test_reporter_rejects_prose_that_invents_numbers_or_drops_a_rejection(
    tmp_path: Path,
) -> None:
    state, artifacts = engine_state(tmp_path)
    template = render_template(report_facts(state, artifacts))

    def invented(messages: list[dict]) -> dict:
        prose = faithful_prose(messages)["methods_md"]
        return {"methods_md": prose + "\nThe effect replicated in 87.5% of 14 samples."}

    def dropped(messages: list[dict]) -> dict:
        return {"methods_md": "# Methods\n[q1] [q2] [q3] were examined."}

    for writer in (invented, dropped):
        result = asyncio.run(report(ScriptedLLM(reporter=writer), state, artifacts))
        assert result.methods_md == template

    with pytest.raises(BudgetExhausted):
        asyncio.run(
            report(ScriptedLLM(reporter=BudgetExhausted("cap")), state, artifacts)
        )


# --- in the run loop -----------------------------------------------------------------------


class Judges(FakeAgents):
    """Fakes for every role but the referee, critic and reporter, which are the real ones."""

    def __init__(self, lab: FakeLab, scripted: ScriptedLLM, **kw: Any) -> None:
        super().__init__(lab, **kw)
        self.scripted = scripted

    async def referee(self, experiment, observation):
        return await referee(
            self.scripted, experiment, observation, run_dir=self.lab.run_dir
        )

    async def critic(self, artifacts, contract):
        return await critique(self.scripted, artifacts, contract)

    async def report(self, state, artifacts):
        return await report(self.scripted, state, artifacts)


def test_real_judges_drive_the_fake_run_loop(tmp_path: Path) -> None:
    lab = FakeLab()
    deps = fake_deps(lab, clock=FakeClock())
    scripted = ScriptedLLM(referee=VERDICT, reporter=RuntimeError("offline"))
    deps = dataclasses.replace(deps, agents=Judges(lab, scripted, llm=deps.llm))
    cfg = load_config("default", runs_dir=str(tmp_path))
    result = asyncio.run(run(lab, cfg, deps, run_id="r-judges"))

    lines = (tmp_path / "r-judges" / "events.jsonl").read_text("utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    findings = [
        e["payload"]["finding"] for e in events if e["type"] == "critic.finding"
    ]
    assert [(f["id"], f["severity"]) for f in findings] == [
        ("de_table.csv:missing_columns", "block")
    ]
    assert all(a.contract_ok for a in result.artifacts)
    statements = {
        e["payload"]["hypothesis"]["statement"]
        for e in events
        if e["type"] == "hypothesis.proposed"
    }
    referee_calls = [c for c in scripted.calls if c["role"] == "referee"]
    assert referee_calls
    for call in referee_calls:
        assert not any(s in all_text(call) for s in statements)
    for qid in ("q1", "q2"):
        assert f"[{qid}]" in result.methods_md
    assert "## Rejected options" in result.methods_md
