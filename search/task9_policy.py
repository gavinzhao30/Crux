"""Shared Task 9 search boundary for both evaluation arms.

This is the one search policy for Task 9. ``Task9SearchPolicy`` holds the task metadata
(the source paper, its preprint, the study's GEO/SRA records, the ground-truth bucket and
the study's terms) and the documentation domains. Its checks apply two rule sets to that
same data: the lexical screen in this module and the generic documentation policy in
``search.policy`` (``Task9SearchPolicy.rules``). ``rules`` also configures the Exa adapter,
both its server-side domain filters and its own screening, so the provider never allows
anything the guard would block. The exclusions come from public task metadata, never from
the ground truth.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from html import unescape
from typing import TYPE_CHECKING, Protocol
from unicodedata import normalize
from urllib.parse import unquote, urlsplit

from search.policy import PolicyDenied, SearchPolicy

if TYPE_CHECKING:
    from core.schemas import Evidence

DOC_DOMAINS = (
    "bioconductor.org",
    "cran.r-project.org",
    "nf-co.re",
    "docs.seqera.io",
    "subread.sourceforge.net",
    "pydeseq2.readthedocs.io",
    "ensembl.org",
)
# The source paper: BMC Biology (2024), REST-targeting compounds in glioblastoma, studied
# in T98G and HEK293 cells.
TASK9_DOI = "10.1186/s12915-024-01879-0"
_TASK9_TERMS = (
    "glioblastoma",
    "t98g",
    "hek293",
    "rest knockout",
    "rest inhibitors",
    "rest-targeting",
    "rest targeting compound",
    "s12915-024-01879",
    "tagseq_featurecounts_matrix",
    "t98g_slow_shared_degs",
    "hek293_shared_degs",
    "t98g_d4_specific_upregulated_genes",
)
# Host or host/path prefixes. They are screened as text and sent to Exa as exclusions.
_TASK9_EXCLUDED = (
    "storage.googleapis.com/bixbench3-ground-truth",  # the grader's references
    "bmcbiol.biomedcentral.com",  # the source paper's journal
    "biorxiv.org",  # preprint servers
    "medrxiv.org",
    "researchsquare.com",
    "preprints.org",
    "ncbi.nlm.nih.gov/geo",  # the study's GEO and SRA records
    "ncbi.nlm.nih.gov/sra",
    "trace.ncbi.nlm.nih.gov",
    "ebi.ac.uk/ena",
    "ncbi.nlm.nih.gov/pmc",  # full-text mirrors of the paper
    "pubmed.ncbi.nlm.nih.gov",
    "europepmc.org",
)
_RESTRICTED = re.compile(
    r"10\.\d{4,9}[\s_/]|"
    r"ground[\s_-]*truth|supplement\w*|processed\s+(?:counts?|tables?|data)|"
    r"\b(?:GEO|SRA|GSE\d+|SR[ARPX]\d+|PRJNA\d+|BixBench3?)\b|"
    r"\b(?:published|study|paper)\s+(?:results|tables|findings)\b",
    re.IGNORECASE,
)


def _canonical(text: str) -> str:
    for _ in range(3):
        text = unquote(unescape(text))
    return normalize("NFKC", text).casefold()


def _in_domain(host: str, domain: str) -> bool:
    domain = domain.split("/", 1)[0]
    return host == domain or host.endswith("." + domain)


def _passes(check: Callable[[str], object], value: str) -> bool:
    try:
        check(value)
    except PolicyDenied:
        return False
    return True


@dataclass(frozen=True)
class Task9SearchPolicy:
    study_terms: tuple[str, ...] = _TASK9_TERMS
    excluded_domains: tuple[str, ...] = _TASK9_EXCLUDED
    excluded_dois: tuple[str, ...] = (TASK9_DOI,)
    include_domains: tuple[str, ...] = DOC_DOMAINS
    task_id: str = "bixbench3-task9"

    @cached_property
    def rules(self) -> SearchPolicy:
        """The same exclusions as a ``search.policy.SearchPolicy`` (for ``ExaSearch``)."""
        terms = (*self.study_terms, *self.excluded_dois, *self.excluded_domains)
        return SearchPolicy(
            task_id=self.task_id,
            forbidden_terms=tuple(dict.fromkeys(t for t in terms if t.strip())),
            allowed_sources=self.include_domains,
            forbidden_sources=self.excluded_domains,
        )

    def _prohibited(self, text: str) -> bool:
        text = _canonical(text)
        markers = self.study_terms + self.excluded_domains + self.excluded_dois
        return bool(
            _RESTRICTED.search(text)
            or any(marker and _canonical(marker) in text for marker in markers)
        )

    def allows_query(self, query: str) -> bool:
        canonical = _canonical(query)
        return bool(
            query.strip()
            and len(query) <= 1000
            and "://" not in canonical
            and not self._prohibited(query)
            and _passes(self.rules.check_query, query)
        )

    def allows_url(self, url: str) -> bool:
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").lower().rstrip(".")
            port = parsed.port
        except ValueError:
            return False
        return bool(
            parsed.scheme == "https"
            and not parsed.username
            and not parsed.password
            and port in (None, 443)
            and any(_in_domain(host, domain) for domain in self.include_domains)
            and not self._prohibited(url)
            and _passes(self.rules.check_url, url)
        )

    def allows_text(self, text: str) -> bool:
        """May a returned excerpt reach an agent?"""
        return not self._prohibited(text) and _passes(self.rules.check_text, text)


class SearchProvider(Protocol):
    async def __call__(
        self, query: str, *, k: int, include_domains: tuple[str, ...]
    ) -> list["Evidence"]: ...


class GuardedSearch:
    """Apply the policy outside Modal, before returning any evidence to an agent.

    Implements ``core.interfaces.Search``. After each ``find``, ``last_blocked`` lists the
    URLs the policy removed, here or in a screening provider (the orchestrator reports them
    in ``search.done``). A query outside the policy raises ``PermissionError`` before the
    provider is called.
    """

    def __init__(
        self,
        provider: SearchProvider,
        policy: Task9SearchPolicy,
        on_event: Callable[[str, dict], None],
    ):
        self.provider = provider
        self.policy = policy
        self.on_event = on_event
        self.last_blocked: list[str] = []

    async def find(self, query: str, k: int = 5) -> list["Evidence"]:
        self.last_blocked = []
        allowed = self.policy.allows_query(query)
        self.on_event("search.query", {"query": query, "blocked": not allowed})
        if not allowed:
            raise PermissionError("Search query is outside the Task 9 documentation policy")
        results = await self.provider(query, k=k, include_domains=self.policy.include_domains)
        # A provider that screens too (``ExaProvider``) reports what it removed.
        self.last_blocked = [str(u) for u in getattr(self.provider, "last_blocked", None) or []]
        evidence = []
        for result in results:
            allowed = bool(
                result.url
                and self.policy.allows_url(result.url)
                and self.policy.allows_text(result.claim)
            )
            self.on_event(
                "search.result", {"query": query, "url": result.url, "blocked": not allowed}
            )
            if allowed:
                evidence.append(result)
            elif str(result.url or "") not in self.last_blocked:
                self.last_blocked.append(str(result.url or ""))
        self.on_event("search.done", {"query": query, "urls": [item.url for item in evidence]})
        return evidence

    async def aclose(self) -> None:
        """Close the provider if it holds resources (``ExaProvider`` does)."""
        close = getattr(self.provider, "aclose", None)
        if callable(close):
            await close()
