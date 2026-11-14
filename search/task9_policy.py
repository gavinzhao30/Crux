"""Shared Task 9 search boundary for both evaluation arms."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from html import unescape
from typing import TYPE_CHECKING, Protocol
from unicodedata import normalize
from urllib.parse import unquote, urlsplit

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
_TASK9_TERMS = (
    "glioblastoma",
    "t98g",
    "hek293",
    "rest knockout",
    "rest inhibitors",
    "tagseq_featurecounts_matrix",
    "t98g_slow_shared_degs",
    "hek293_shared_degs",
    "t98g_d4_specific_upregulated_genes",
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
    return host == domain or host.endswith("." + domain)


@dataclass(frozen=True)
class Task9SearchPolicy:
    study_terms: tuple[str, ...] = _TASK9_TERMS
    excluded_domains: tuple[str, ...] = ()
    excluded_dois: tuple[str, ...] = ()
    include_domains: tuple[str, ...] = DOC_DOMAINS

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
        )


class SearchProvider(Protocol):
    async def __call__(
        self, query: str, *, k: int, include_domains: tuple[str, ...]
    ) -> list["Evidence"]: ...


class GuardedSearch:
    """Apply the policy outside Modal, before returning any evidence to an agent."""

    def __init__(
        self,
        provider: SearchProvider,
        policy: Task9SearchPolicy,
        on_event: Callable[[str, dict], None],
    ):
        self.provider = provider
        self.policy = policy
        self.on_event = on_event

    async def find(self, query: str, k: int = 5) -> list["Evidence"]:
        allowed = self.policy.allows_query(query)
        self.on_event("search.query", {"query": query, "blocked": not allowed})
        if not allowed:
            raise PermissionError("Search query is outside the Task 9 documentation policy")
        results = await self.provider(query, k=k, include_domains=self.policy.include_domains)
        evidence = []
        for result in results:
            allowed = bool(
                result.url
                and self.policy.allows_url(result.url)
                and not self.policy._prohibited(result.claim)
            )
            self.on_event(
                "search.result", {"query": query, "url": result.url, "blocked": not allowed}
            )
            if allowed:
                evidence.append(result)
        self.on_event("search.done", {"query": query, "urls": [item.url for item in evidence]})
        return evidence
