"""DU-15 in the Task 9 run: one leakage policy, the guarded Exa adapter as ``Deps.search``,
pre-registered search diagnostics, and a search experiment that wins the ranking, emits
``search.done`` and moves a posterior. No network: the Exa SDK is a fake."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.proposer import Proposals
from agents.swarm import SwarmRoles
from core.config import load_config
from core.orchestrator import Deps, run
from core.schemas import (
    Experiment,
    Hypothesis,
    Observation,
    OutcomeBin,
    Question,
    Report,
    bin_for,
)
from execution.staging import COUNTS, SAMPLES
from labs.bixbench3.bins import choose_bin
from labs.bixbench3.swarm_lab import build_task9_run
from search.exa import ExaSearch
from search.task9 import (
    SEARCH_DIAGNOSTICS,
    ExaProvider,
    build_task9_search,
    search_diagnostics,
    search_value,
)
from search.task9_policy import DOC_DOMAINS, TASK9_DOI, GuardedSearch, Task9SearchPolicy

VIGNETTE = "https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html"
MANUAL = "https://bioconductor.org/packages/release/bioc/manuals/DESeq2/man/DESeq2.pdf"
BUCKET = "https://storage.googleapis.com/bixbench3-ground-truth/task9/references.csv"
FILTER_EXCERPT = (
    "The results function of the DESeq2 package performs independent filtering by default "
    "using the mean of normalized counts as a filter statistic."
)


def policy() -> Task9SearchPolicy:
    return Task9SearchPolicy()


# --- policy -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        f"DESeq2 documentation {TASK9_DOI}",
        "DESeq2 documentation s12915-024-01879-0",
        "DESeq2 documentation REST-targeting compounds",
        "DESeq2 documentation glioblastoma defaults",
        "DESeq2 defaults for T98G",
        "DESeq2 documentation HEK293 alpha",
        "DESeq2 documentation preprint",
        "DESeq2 documentation biorxiv.org",
        "DESeq2 documentation GEO series",
        "DESeq2 documentation GSE123456",
        "DESeq2 documentation SRA run SRR1234567",
        "DESeq2 documentation bixbench3-ground-truth",
        "DESeq2 documentation storage.googleapis.com/bixbench3-ground-truth",
        "DESeq2 documentation published results",
        "DESeq2 docs site:bioconductor.org",
        "what did the paper find",  # no tool documentation intent
    ],
)
def test_task9_queries_outside_the_policy_are_refused(query):
    assert not policy().allows_query(query)


@pytest.mark.parametrize(
    "url",
    [
        f"https://bmcbiol.biomedcentral.com/articles/{TASK9_DOI}",
        f"https://doi.org/{TASK9_DOI}",
        f"https://link.springer.com/article/{TASK9_DOI}",
        "https://www.biorxiv.org/content/10.1101/2023.01.01.000000v1",
        "https://www.researchsquare.com/article/rs-0000000/v1",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE000000",
        "https://www.ncbi.nlm.nih.gov/geo/",
        "https://www.ncbi.nlm.nih.gov/sra/SRX000000",
        "https://trace.ncbi.nlm.nih.gov/Traces/sra",
        "https://www.ebi.ac.uk/ena/browser/view/PRJNA000000",
        BUCKET,
        "https://storage.googleapis.com/bixbench3-ground-truth/",
        # documentation hosts with study or publication paths
        "https://bioconductor.org/packages/DESeq2/T98G",
        "https://bioconductor.org/packages/DESeq2/supplement",
        f"https://bioconductor.org/packages/{TASK9_DOI}",
        "https://bioconductor.org.example.org/packages/DESeq2",
        "http://bioconductor.org/packages/DESeq2",
    ],
)
def test_task9_urls_outside_the_policy_are_refused(url):
    assert not policy().allows_url(url)


def test_documentation_is_allowed_and_study_excerpts_are_not():
    p = policy()
    assert p.allows_url(VIGNETTE) and p.allows_url(MANUAL)
    assert p.allows_url("https://www.ensembl.org/info/genome/stable_ids/index.html")
    assert p.allows_text(FILTER_EXCERPT)
    for text in (
        "Differential expression in T98G cells treated with",
        f"Cited as doi:{TASK9_DOI}",
        "Counts are in Supplementary Table 2",
        "a REST-targeting compound",
    ):
        assert not p.allows_text(text)


def test_exa_adapter_gets_the_same_exclusions_and_domains():
    p = policy()
    filters = p.rules.sdk_filters()
    assert filters["include_domains"] == list(DOC_DOMAINS)
    for source in (
        "storage.googleapis.com/bixbench3-ground-truth",
        "ncbi.nlm.nih.gov/geo",
        "bmcbiol.biomedcentral.com",
        "biorxiv.org",
    ):
        assert source in filters["exclude_domains"]
    assert len(filters["exclude_domains"]) == len(set(filters["exclude_domains"]))
    assert TASK9_DOI in p.rules.forbidden_terms
    # anything the adapter's own policy refuses, the guard refuses first
    for url in (
        BUCKET,
        f"https://doi.org/{TASK9_DOI}",
        "https://bioconductor.org/x?y=1",
    ):
        assert not p.allows_url(url)


# --- the guarded Exa adapter -------------------------------------------------------------


def hit(url: str, text: str, title: str = "DESeq2") -> SimpleNamespace:
    return SimpleNamespace(url=url, title=title, id=url, highlights=[text])


class FakeExa:
    """Stands in for ``exa_py.AsyncExa`` (no network)."""

    def __init__(self, results=None, *, cost: float = 0.007) -> None:
        self.results = (
            results if results is not None else [hit(VIGNETTE, FILTER_EXCERPT)]
        )
        self.cost = cost
        self.calls: list[dict] = []
        self.failure: Exception | None = None

    async def search(self, **request):
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            results=self.results,
            cost_dollars=SimpleNamespace(total=self.cost),
            request_id="req-test",
        )


def guarded(tmp_path: Path, sdk: FakeExa) -> GuardedSearch:
    cfg = load_config("default", runs_dir=str(tmp_path / "runs"))
    search = build_task9_search(cfg, client=sdk, env={})
    assert isinstance(search, GuardedSearch)
    return search


def test_guarded_exa_reports_what_both_layers_removed(tmp_path):
    sdk = FakeExa(
        [
            hit(VIGNETTE, FILTER_EXCERPT),
            hit(BUCKET, "reference table"),  # removed by the Exa adapter's screen
            hit(MANUAL, "results for T98G"),  # removed by the excerpt screen
        ]
    )
    search = guarded(tmp_path, sdk)

    async def go():
        try:
            found = await search.find(SEARCH_DIAGNOSTICS[0].query)
            assert [e.url for e in found] == [VIGNETTE] and found[0].kind == "search"
            assert search.last_blocked == [BUCKET, MANUAL]
            assert sdk.calls[0]["include_domains"] == list(DOC_DOMAINS)
            again = await search.find(SEARCH_DIAGNOSTICS[0].query)  # cache hit
            assert again == found and search.last_blocked == [BUCKET, MANUAL]
            assert len(sdk.calls) == 1
            with pytest.raises(PermissionError):
                await search.find("DESeq2 documentation for T98G")
            assert len(sdk.calls) == 1  # refused before any request
            exa = search.provider.exa
            kinds = [
                r[0] for r in exa.db.execute("SELECT type FROM audit ORDER BY seq")
            ]
            assert "guard.search.query" in kinds and "search.result_blocked" in kinds
        finally:
            await search.aclose()

    asyncio.run(go())
    assert (tmp_path / "runs" / ".cache" / "search").is_dir()


def test_provider_must_share_the_guard_domains(tmp_path):
    exa = ExaSearch(policy().rules, "r", tmp_path, client=FakeExa())
    provider = ExaProvider(exa)

    async def go():
        try:
            with pytest.raises(ValueError, match="same domains"):
                await provider(
                    "DESeq2 documentation alpha", k=5, include_domains=("x.org",)
                )
        finally:
            await provider.aclose()

    asyncio.run(go())


# --- pre-registered search diagnostics ----------------------------------------------------

# Question texts the proposers raised in the first real Task 9 run (r-20261114-175926-6050).
RUN1_QUESTIONS = {
    "q-design": "What design formula should be used in the DESeq2 analysis for "
    "treatment-versus-control comparisons for each cell line?",
    "q-filter": "Should filtering be applied to genes prior to DESeq2 analysis and/or "
    "downstream reporting?",
    "q-padj": "How should multiple testing correction be handled and what adjusted p-value "
    "threshold should be used for reporting DEGs?",
    "q-effect": "What effect size threshold or criteria should be applied to define "
    "differentially expressed genes beyond statistical significance?",
    "q-sets": "How should the shared and D4-specific gene sets be derived from the DE results?",
    "q-ids": "Which gene identifier format should be used consistently across input counts, "
    "DESeq2 results, and output files?",
    "q-norm": "What normalization method should be applied to the raw count data before "
    "DESeq2 differential expression testing?",
    "q-outliers": "Should outlier detection or filtering be explicitly performed on the count "
    "data or sample replicates before running DESeq2 to avoid aberrant DE results?",
    "q-depth": "Is the coverage and sequencing depth sufficient and balanced across "
    "conditions and replicates to ensure reliable differential expression analysis?",
}
EXPECTED_MATCH = {
    "q-design": "design_formula",
    "q-filter": "independent_filtering",
    "q-padj": "alpha_padj",
    "q-effect": "lfc_shrink",
    "q-norm": "raw_counts",
}


def question(qid: str, text: str, raised_by: str = "statistician") -> Question:
    return Question(id=qid, text=text, kind="categorical", raised_by=raised_by)


def test_documentation_questions_get_one_search_diagnostic_each():
    questions = [question(qid, text) for qid, text in RUN1_QUESTIONS.items()]
    questions.append(
        question("library-sample_quality", "Should filtering of genes", "library")
    )
    found = search_diagnostics(questions)
    assert {
        qid: [e.design["search_library"] for e in exps] for qid, exps in found.items()
    } == {qid: [key] for qid, key in EXPECTED_MATCH.items()}
    for qid, (exp,) in found.items():
        assert exp.kind == "search" and exp.query and exp.code is None
        assert (
            exp.question_ids == [qid] and exp.est_cost_usd > 0 and exp.est_cost_s < 10
        )


def test_every_diagnostic_query_passes_the_policy_and_its_rules_are_safe():
    p = policy()
    for diagnostic in SEARCH_DIAGNOSTICS:
        assert p.allows_query(diagnostic.query), diagnostic.key
        exp = diagnostic.experiment(["q"], est_cost_usd=0.007)
        assert len({b.id for b in exp.bins}) == len(exp.bins) >= 2
        assert all(len(b.rule) <= 512 for b in exp.bins)  # the lab's rule limit
        # No usable documentation: no bin, so no belief can move.
        empty = search_value([])
        assert bin_for(exp.bins, empty) is None
        with pytest.raises(ValueError, match="exactly one"):
            choose_bin(((b.id, b.rule) for b in exp.bins), empty)


def evidence(*texts: str) -> list:
    from core.schemas import Evidence

    return [
        Evidence(
            id=f"e{i}", kind="search", claim=t, author_role="experimenter", url=VIGNETTE
        )
        for i, t in enumerate(texts)
    ]


PREFILTER = "keep <- rowSums(counts(dds) >= 10) >= smallestGroupSize"
ALPHA = "alpha: the significance cutoff used for optimizing the independent filtering (by default 0.1)"
BH = 'results(object, ..., pAdjustMethod = "BH")'
LAST = "you should put the variable of interest at the end of the formula"
GROUP = "dds$group <- factor(paste0(dds$genotype, dds$condition))"
SHRINK = "Shrinkage of effect size (LFC estimates) is useful for visualization and ranking of genes."
THRESHOLD = "res <- results(dds, lfcThreshold=1)"
RAW = "The values in the matrix should be un-normalized counts or estimated counts of reads."
OTHER = "DESeq2 package for differential analysis of count data."


@pytest.mark.parametrize(
    "key,texts,expected",
    [
        ("independent_filtering", [FILTER_EXCERPT], "builtin"),
        ("independent_filtering", [PREFILTER], "prefilter"),
        ("independent_filtering", [FILTER_EXCERPT, PREFILTER], "both"),
        ("independent_filtering", [OTHER], "neither"),
        ("alpha_padj", [ALPHA], "alpha_0_1"),
        ("alpha_padj", [BH], "bh"),
        ("alpha_padj", [ALPHA, BH], "both"),
        ("alpha_padj", [OTHER], "neither"),
        ("design_formula", [LAST], "variable_last"),
        ("design_formula", [GROUP], "group_factor"),
        ("design_formula", [LAST, GROUP], "both"),
        ("design_formula", [OTHER], "neither"),
        ("lfc_shrink", [SHRINK], "shrink_for_ranking"),
        ("lfc_shrink", [THRESHOLD], "threshold_test"),
        ("lfc_shrink", [SHRINK, THRESHOLD], "both"),
        ("lfc_shrink", [OTHER], "neither"),
        ("raw_counts", [RAW], "raw_required"),
        ("raw_counts", [OTHER], "not_stated"),
    ],
)
def test_bins_are_computed_by_code_and_exclusive(key, texts, expected):
    (diagnostic,) = [d for d in SEARCH_DIAGNOSTICS if d.key == key]
    exp = diagnostic.experiment(["q"], est_cost_usd=0.007)
    value = search_value(evidence(*texts))
    # The orchestrator's evaluator and the lab's strict one agree on exactly one bin.
    assert bin_for(exp.bins, value) == expected
    assert choose_bin(((b.id, b.rule) for b in exp.bins), value) == expected


def test_swarm_designer_returns_search_diagnostics_for_matching_questions():
    async def propose(brief, evidence, *, llm, memory, bus, config, library):
        qs = [question(qid, RUN1_QUESTIONS[qid]) for qid in ("q-filter", "q-ids")]
        return Proposals(questions=qs)

    cfg = load_config("default")
    roles = SwarmRoles(None, cfg, propose_fn=propose, library=lambda brief: [])
    questions, _ = asyncio.run(roles.propose({"goal": "g"}, []))
    by_id = {q.id: q for q in questions}
    (exp,) = asyncio.run(roles.design(by_id["q-filter"], [], [], []))
    assert exp.kind == "search" and exp.design == {
        "search_library": "independent_filtering"
    }
    assert asyncio.run(roles.design(by_id["q-ids"], [], [], [])) == []
    # a second propose round does not duplicate the diagnostic
    asyncio.run(roles.propose({"goal": "g"}, []))
    assert len(asyncio.run(roles.design(by_id["q-filter"], [], [], []))) == 1


# --- wiring: build_task9_run --------------------------------------------------------------


def make_case(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / COUNTS).write_text("Geneid\tS1\nENSG00000000001\t5\n", encoding="utf-8")
    (root / SAMPLES).write_text(
        "sample_id\tcell_line\tcondition\nS1\tA\tCtrl\n", encoding="utf-8"
    )
    return root


class NoLLM:
    on_usage = None

    async def structured(self, role, schema, messages, n=1, **_):  # pragma: no cover
        raise AssertionError("no model calls in wiring tests")


@pytest.mark.parametrize(
    "preset,key,expect_search",
    [
        ("default", "test-key", True),
        ("default", None, False),
        ("no_search", "test-key", False),
    ],
)
def test_build_task9_run_wires_search_from_config_and_key(
    tmp_path, monkeypatch, preset, key, expect_search
):
    if key is None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("EXA_API_KEY", key)
    cfg = load_config(preset, runs_dir=str(tmp_path / "runs"))
    lab, deps = build_task9_run(cfg, case_dir=make_case(tmp_path / "case"), llm=NoLLM())
    try:
        if expect_search:
            assert isinstance(deps.search, GuardedSearch)
            assert lab.inner.search is deps.search and lab.owns_search
            assert isinstance(deps.search.provider, ExaProvider)
            assert deps.search.policy == Task9SearchPolicy()
            assert lab.brief()["search_policy"] == "Task 9 documentation policy"
        else:
            assert deps.search is None and lab.inner.search is None
            assert lab.brief()["search_policy"] == "disabled"
    finally:
        asyncio.run(lab.close())
    if expect_search:
        assert not lab.owns_search  # closed once, with the lab


def test_build_task9_run_keeps_a_passed_search_unless_search_is_off(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    mine = object()
    case = make_case(tmp_path / "case")
    cfg = load_config("default", runs_dir=str(tmp_path / "runs"))
    lab, deps = build_task9_run(cfg, case_dir=case, llm=NoLLM(), search=mine)
    assert deps.search is mine and lab.inner.search is mine and not lab.owns_search
    off = load_config("no_search", runs_dir=str(tmp_path / "runs"))
    _, deps = build_task9_run(off, case_dir=case, llm=NoLLM(), search=mine)
    assert deps.search is None


def test_build_task9_search_needs_config_and_key(tmp_path):
    on = load_config("default", runs_dir=str(tmp_path))
    off = load_config("no_search", runs_dir=str(tmp_path))
    assert build_task9_search(on, env={}) is None
    assert build_task9_search(off, env={"EXA_API_KEY": "k"}) is None
    search = build_task9_search(on, env={"EXA_API_KEY": "k"})
    assert isinstance(search, GuardedSearch)
    asyncio.run(search.aclose())


# --- one orchestrator run: the search experiment is ranked, runs and moves a posterior ------

FILTER_OPTIONS = {
    "q-filter-h0": "Apply a minimal count filter, e.g., keep genes with counts > 10 in at "
    "least two samples.",
    "q-filter-h1": "No filtering; use all genes as provided to retain maximal data.",
    "q-filter-h2": "Apply independent filtering as implemented in DESeq2, relying on mean "
    "expression to filter.",
}
# P(bin | option) for the search diagnostic, as a predictor would state them.
SEARCH_ROWS = {
    "q-filter-h0": {"builtin": 0.15, "prefilter": 0.45, "both": 0.3, "neither": 0.1},
    "q-filter-h1": {"builtin": 0.2, "prefilter": 0.1, "both": 0.1, "neither": 0.6},
    "q-filter-h2": {"builtin": 0.7, "prefilter": 0.05, "both": 0.2, "neither": 0.05},
}


def rival_compute() -> Experiment:
    """A slow compute check with weak separation: it should lose the ranking to search."""
    return Experiment(
        id="rival-compute",
        question_ids=[],
        kind="compute",
        goal="Count genes with fewer than 10 reads in every sample",
        language="python",
        code="print('RESULT: {\"n_low\": 0}')",
        bins=[
            OutcomeBin(id="few", description="few low genes", rule="n_low < 100"),
            OutcomeBin(id="many", description="many low genes", rule="n_low >= 100"),
        ],
        est_cost_s=300,
    )


class RowPredictor:
    async def predict(self, exp, hypotheses, context):
        if exp.kind == "search":
            return {h.id: SEARCH_ROWS[h.id] for h in hypotheses}
        return {
            h.id: {"few": 0.5 + 0.05 * i, "many": 0.5 - 0.05 * i}
            for i, h in enumerate(hypotheses)
        }


class QuietLab:
    name = "bixbench3"

    def __init__(self) -> None:
        self.ran: list[str] = []

    def brief(self) -> dict:
        return {"goal": "Task 9 test", "contract": {}}

    async def profile(self):
        return []

    def candidate_designs(self):
        return None

    async def run(self, exp):
        self.ran.append(exp.id)
        return Observation(experiment_id=exp.id, valid=True, value={"n_low": 0})

    async def deliver(self, state):
        return []


async def propose_filter(brief, evidence, *, llm, memory, bus, config, library):
    q = question("q-filter", RUN1_QUESTIONS["q-filter"])
    hyps = [
        Hypothesis(
            id=hid,
            question_id=q.id,
            statement=text,
            author="statistician",
            rationale="-",
            prior=1.0,
            posterior=1.0,
        )
        for hid, text in FILTER_OPTIONS.items()
    ]
    rival = rival_compute().model_copy(update={"question_ids": [q.id]})
    return Proposals(questions=[q], hypotheses=hyps, diagnostics=[rival])


async def referee_ok(llm, exp, obs, *, run_dir=None):
    return obs.valid, "ok"


async def no_findings(llm, artifacts, contract, *, counts_path=None, brief=None):
    return []


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


def run_with_search(tmp_path: Path, sdk: FakeExa) -> tuple[list[dict], QuietLab]:
    cfg = load_config(
        "default", runs_dir=str(tmp_path / "runs"), max_experiments_per_question=1
    )
    search = build_task9_search(cfg, client=sdk, env={})
    roles = SwarmRoles(
        None,
        cfg,
        propose_fn=propose_filter,
        library=lambda brief: [],
        referee_fn=referee_ok,
        critic_fn=no_findings,
        report_fn=report_plain,
    )
    lab = QuietLab()

    async def go():
        try:
            await run(
                lab,
                cfg,
                Deps(agents=roles, predictor=RowPredictor(), search=search),
                run_id="r-search",
            )
        finally:
            await search.aclose()

    asyncio.run(go())
    path = tmp_path / "runs" / "r-search" / "events.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()], lab


def payloads(events: list[dict], kind: str) -> list[dict]:
    return [e["payload"] for e in events if e["type"] == kind]


def test_search_experiment_wins_runs_and_moves_the_posterior(tmp_path):
    sdk = FakeExa([hit(VIGNETTE, FILTER_EXCERPT), hit(BUCKET, "reference table")])
    events, lab = run_with_search(tmp_path, sdk)
    assert events[-1]["type"] == "run.finished"

    (scored,) = payloads(events, "experiment.scored")
    kinds = {r["experiment_id"]: r["kind"] for r in scored["ranking"]}
    assert set(kinds.values()) == {"search", "compute"}  # ranked against compute
    assert (
        kinds[scored["chosen"]] == "search" and scored["ranking"][0]["kind"] == "search"
    )
    assert lab.ran == []  # the compute rival never ran; the orchestrator ran the search

    (done,) = payloads(events, "search.done")
    assert done["query"] == SEARCH_DIAGNOSTICS[0].query
    assert done["urls"] == [VIGNETTE] and done["blocked"] == [BUCKET]
    assert done["bin_id"] == "builtin" and done["experiment_id"] == scored["chosen"]
    assert len(sdk.calls) == 1

    (update,) = payloads(events, "belief.updated")
    assert (
        update["experiment_id"] == scored["chosen"] and update["observed"] == "builtin"
    )
    prior, post = update["prior"], update["posterior"]
    assert post["q-filter-h2"] > prior["q-filter-h2"]  # independent filtering gains
    assert post["q-filter-h1"] < prior["q-filter-h1"]
    (finished,) = payloads(events, "experiment.finished")
    assert finished["observation"]["cost_usd"] == pytest.approx(0.007)
    # the referee sees the excerpts the bin rules were evaluated on
    value = finished["observation"]["value"]
    assert value["n_results"] == 1 and "performs independent filtering" in value["text"]


def test_a_failed_search_is_an_invalid_observation_not_a_crashed_run(tmp_path):
    sdk = FakeExa()
    sdk.failure = TimeoutError("offline")
    events, _ = run_with_search(tmp_path, sdk)
    assert events[-1]["type"] == "run.finished"
    (done,) = payloads(events, "search.done")
    assert done["urls"] == [] and done["bin_id"] is None
    (verdict,) = payloads(events, "referee.verdict")
    assert not verdict["valid"] and "SearchUnavailable" in verdict["note"]
    assert not payloads(events, "belief.updated")
