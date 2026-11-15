"""Exa search inside the Task 9 run: the guarded adapter and pre-registered diagnostics.

``build_task9_search`` returns the ``core.interfaces.Search`` the run uses: ``GuardedSearch``
with the Task 9 policy over ``ExaSearch`` (screening, disk cache and audit configured from
the same policy). It returns ``None`` when ``cfg.search`` is off or ``EXA_API_KEY`` is unset,
and the orchestrator then drops every search experiment.

``search_diagnostics`` is the search half of the DU-42 idea: a few documentation questions
with a fixed query and bins as code rules, matched to the questions the proposers raise
(DESeq2 defaults: independent filtering, alpha and p-value adjustment, design formulas,
``lfcShrink``, raw-count input). The orchestrator runs a search experiment itself
(``_run_search``) and evaluates the rules over ``{"n_results", "text", "urls"}``, where
``text`` is the screened excerpts joined and lower-cased. ``Task9Lab.run`` computes the
same fields when no mapper is injected, so no model ever picks the observed bin. Every bin
requires ``n_results > 0``: a search that returns no usable documentation lands in no bin
and is invalid, so it cannot move a belief.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.schemas import Experiment, OutcomeBin
from search.task9_policy import GuardedSearch, Task9SearchPolicy

if TYPE_CHECKING:
    from core.schemas import Evidence, Question
    from search.exa import ExaSearch

CACHE_SCOPE = "bixbench3-task9"
SEARCH_COST_S = 4.0


class ExaProvider:
    """``ExaSearch`` as a ``GuardedSearch`` provider. Exa gets its domain filters from the
    same policy, so ``include_domains`` must match them."""

    def __init__(self, exa: ExaSearch) -> None:
        self.exa = exa
        self.last_blocked: list[str] = []
        self.last_cost_usd = 0.0

    async def __call__(
        self, query: str, *, k: int, include_domains: tuple[str, ...]
    ) -> list[Evidence]:
        self.last_blocked, self.last_cost_usd = [], 0.0
        if tuple(include_domains) != tuple(self.exa.policy.allowed_sources):
            raise ValueError("The guard and the Exa adapter must use the same domains")
        result = await self.exa.lookup(query, k)
        self.last_blocked, self.last_cost_usd = list(result.blocked), result.cost_usd
        return list(result.evidence)

    async def aclose(self) -> None:
        await self.exa.aclose()


def build_task9_search(
    cfg: Any,
    *,
    root: Path | None = None,
    client: Any = None,
    policy: Task9SearchPolicy | None = None,
    env: Mapping[str, str] | None = None,
) -> GuardedSearch | None:
    """The guarded Exa search for a Task 9 run, or ``None`` when search is off or no key.

    The disk cache and audit live under ``<cfg.runs_dir>/.cache/search`` (next to the LLM
    cache) and are shared by runs, so a repeated query costs nothing. ``client`` replaces
    the Exa SDK (tests); with a client no key is needed.
    """
    from search.exa import ExaSearch

    env = os.environ if env is None else env
    if not getattr(cfg, "search", False):
        return None
    if client is None and not env.get("EXA_API_KEY"):
        return None
    policy = policy or Task9SearchPolicy()
    if root is None:
        root = Path(getattr(cfg, "runs_dir", "runs")) / ".cache" / "search"
    exa = ExaSearch(policy.rules, CACHE_SCOPE, Path(root), config=cfg, client=client)
    # The guard's query, result and denial records go to the adapter's audit table.
    return GuardedSearch(
        ExaProvider(exa),
        policy,
        lambda kind, payload: exa.audit("guard." + kind, payload),
    )


def search_value(evidence: Iterable[Evidence]) -> dict:
    """What a search experiment's bin rules see (the orchestrator builds the same dict)."""
    found = list(evidence)
    return {
        "n_results": len(found),
        "text": " ".join(e.claim for e in found).lower(),
        "urls": [e.url for e in found if e.url],
    }


# --- pre-registered search diagnostics --------------------------------------------------


def _any(*phrases: str) -> str:
    return "(" + " or ".join(f"{p!r} in text" for p in phrases) + ")"


def _two_claims(
    a: tuple[str, str, str], b: tuple[str, str, str], neither: str
) -> tuple[tuple[str, str, str], ...]:
    """Four exclusive bins over two documented claims ``(id, description, rule)``:
    only A, only B, both, neither. Every bin needs at least one screened result."""
    (a_id, a_text, a_rule), (b_id, b_text, b_rule) = a, b
    found = "n_results > 0"
    return (
        (a_id, a_text, f"{found} and {a_rule} and not {b_rule}"),
        (b_id, b_text, f"{found} and {b_rule} and not {a_rule}"),
        ("both", f"both: {a_text}; and {b_text}", f"{found} and {a_rule} and {b_rule}"),
        ("neither", neither, f"{found} and not {a_rule} and not {b_rule}"),
    )


def _one_claim(
    a: tuple[str, str, str], neither: str
) -> tuple[tuple[str, str, str], ...]:
    a_id, a_text, a_rule = a
    return (
        (a_id, a_text, f"n_results > 0 and {a_rule}"),
        ("not_stated", neither, f"n_results > 0 and not {a_rule}"),
    )


@dataclass(frozen=True)
class SearchDiagnostic:
    key: str
    match: re.Pattern[str]  # over the question text
    goal: str
    query: str
    bins: tuple[tuple[str, str, str], ...]
    est_cost_s: float = SEARCH_COST_S

    def experiment(self, question_ids: list[str], *, est_cost_usd: float) -> Experiment:
        return Experiment(
            id=f"search-{self.key}",
            question_ids=list(question_ids),
            kind="search",
            goal=self.goal,
            query=self.query,
            design={"search_library": self.key},
            bins=[OutcomeBin(id=i, description=d, rule=r) for i, d, r in self.bins],
            est_cost_s=self.est_cost_s,
            est_cost_usd=est_cost_usd,
        )


_AUTO_FILTER = _any(
    "performs independent filtering",
    "independent filtering by default",
    "automatically performs independent filtering",
    "independentfiltering = true",
    "independentfiltering=true",
)
_PREFILTER = _any("pre-filter", "prefilter", "rowsums(counts(dds)")
_ALPHA = _any(
    "alpha=0.1", "alpha = 0.1", "by default 0.1", "set to 0.1", "alpha is set to 0.1"
)
_BH = _any('padjustmethod = "bh"', 'padjustmethod="bh"', "benjamini")
_GROUP = _any("paste0(", "~ group", "~group", "combine the factors", "single factor")
_LAST = _any(
    "variable of interest at the end",
    "end of the formula",
    "last variable in the design",
)
_SHRINK_RANK = (
    "('shrink' in text and " + _any("visualization", "ranking", "rank genes") + ")"
)
_THRESHOLD = _any("lfcthreshold")
_RAW = _any(
    "un-normalized",
    "unnormalized",
    "raw counts",
    "raw count",
    "pre-normalized",
    "should not be used as input",
)

SEARCH_DIAGNOSTICS: tuple[SearchDiagnostic, ...] = (
    SearchDiagnostic(
        key="independent_filtering",
        match=re.compile(
            r"(?=.*\bfilter)(?=.*(?:\bgenes?\b|low[\s-]*count|independent))",
            re.IGNORECASE | re.DOTALL,
        ),
        goal=(
            "Check the DESeq2 documentation: does results() filter low-count genes itself "
            "(independent filtering), and does the vignette recommend a manual pre-filter?"
        ),
        query=(
            "DESeq2 documentation: independent filtering default in results() and "
            "pre-filtering of low count genes"
        ),
        bins=_two_claims(
            (
                "builtin",
                "results() filters by default (independent filtering)",
                _AUTO_FILTER,
            ),
            ("prefilter", "a manual pre-filter of low counts is described", _PREFILTER),
            "neither is stated in the excerpts",
        ),
    ),
    SearchDiagnostic(
        key="alpha_padj",
        match=re.compile(
            r"multiple[\s-]*testing|adjusted\s+p|\bp-?adj|\bfdr\b|\balpha\b|benjamini|"
            r"bonferroni|significance\s+(?:threshold|cutoff)",
            re.IGNORECASE,
        ),
        goal=(
            "Check the DESeq2 documentation for the results() defaults: alpha (the FDR "
            "target of independent filtering) and the p-value adjustment method"
        ),
        query=(
            "DESeq2 results function documentation: alpha default and pAdjustMethod "
            "default for adjusted p-values"
        ),
        bins=_two_claims(
            (
                "alpha_0_1",
                "alpha defaults to 0.1 and should match the FDR cutoff",
                _ALPHA,
            ),
            ("bh", "p-values are adjusted with Benjamini-Hochberg by default", _BH),
            "neither default is stated in the excerpts",
        ),
    ),
    SearchDiagnostic(
        key="design_formula",
        match=re.compile(
            r"design\s+formula|\bdesign\b.*\bdeseq2?\b|model\s+formula", re.IGNORECASE
        ),
        goal=(
            "Check the DESeq2 vignette's design conventions: the variable of interest last "
            "in the formula, or a combined group factor for per-combination contrasts"
        ),
        query=(
            "DESeq2 vignette documentation: design formula with the variable of interest "
            "at the end, and combining factors into a group variable for contrasts"
        ),
        bins=_two_claims(
            (
                "variable_last",
                "put the variable of interest at the end of the formula",
                _LAST,
            ),
            (
                "group_factor",
                "combine factors into one group factor for contrasts",
                _GROUP,
            ),
            "neither convention is stated in the excerpts",
        ),
    ),
    SearchDiagnostic(
        key="lfc_shrink",
        match=re.compile(
            r"shrink|effect\s+size|fold[\s-]*change\s+(?:threshold|cutoff)|lfc\s*threshold",
            re.IGNORECASE,
        ),
        goal=(
            "Check the DESeq2 documentation: is lfcShrink for visualization and ranking, "
            "and is an LFC threshold tested in results() with lfcThreshold?"
        ),
        query=(
            "DESeq2 lfcShrink documentation: shrunken log2 fold changes for ranking, and "
            "the lfcThreshold argument of results"
        ),
        bins=_two_claims(
            (
                "shrink_for_ranking",
                "shrinkage is for visualization and ranking",
                _SHRINK_RANK,
            ),
            (
                "threshold_test",
                "a fold-change threshold is tested with lfcThreshold",
                _THRESHOLD,
            ),
            "neither use is stated in the excerpts",
        ),
    ),
    SearchDiagnostic(
        key="raw_counts",
        match=re.compile(
            r"normali[sz]|\bcpm\b|\btmm\b|median[\s-]*of[\s-]*ratios|size\s+factors?",
            re.IGNORECASE,
        ),
        goal=(
            "Check the DESeq2 documentation: must the input be un-normalized counts, so "
            "that no CPM or TMM step comes before DESeq2?"
        ),
        query=(
            "DESeq2 documentation: should the input count matrix be raw un-normalized "
            "counts rather than normalized values?"
        ),
        bins=_one_claim(
            ("raw_required", "DESeq2 takes un-normalized counts as input", _RAW),
            "the excerpts do not state the input requirement",
        ),
    ),
)


def search_diagnostics(
    questions: Iterable[Question],
    *,
    library: tuple[SearchDiagnostic, ...] = SEARCH_DIAGNOSTICS,
    est_cost_usd: float | None = None,
) -> dict[str, list[Experiment]]:
    """Question id -> the search diagnostics whose pattern matches its text.

    One experiment serves every question it matches (``question_ids`` lists them all), so
    a single Exa call updates each of them. Library forks (``raised_by == "library"``)
    keep their compute diagnostics only.
    """
    from search.exa import EXA_ESTIMATED_COST_USD

    cost = EXA_ESTIMATED_COST_USD if est_cost_usd is None else est_cost_usd
    open_questions = [
        q for q in questions if q.kind == "categorical" and q.raised_by != "library"
    ]
    out: dict[str, list[Experiment]] = {}
    for diagnostic in library:
        qids = [q.id for q in open_questions if diagnostic.match.search(q.text)]
        if not qids:
            continue
        exp = diagnostic.experiment(qids, est_cost_usd=cost)
        for qid in qids:
            out.setdefault(qid, []).append(exp)
    return out

