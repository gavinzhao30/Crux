"""Apply the same task-specific documentation policy before and after search."""

import hashlib
import re
import unicodedata
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_SOURCES = (
    "bioconductor.org/packages",
    "cran.r-project.org/web/packages",
    "nf-co.re/docs",
    "subread.sourceforge.net",
    "www.htslib.org/doc",
    "ensembl.org/info",
    "www.gencodegenes.org/pages",
    "www.biostars.org/p",
    "github.com/bioc/DESeq2",
)
DENIED_DOMAINS = (
    "doi.org",
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "storage.googleapis.com",
    "s3.amazonaws.com",
)
DENIED_TEXT = (
    "bixbench3-ground-truth",
    "ground truth",
    "ground-truth",
    "ground_truth",
    "source paper",
    "preprint",
    "supplement",
    "processed table",
    "answer key",
    "differentially expressed genes",
    "study results",
    "study's results",
)
ACCESSION = re.compile(r"\b(?:GSE|GSM|GDS|SRP|SRR|SRX|ERP|ERR)\d+\b", re.IGNORECASE)
DOI = re.compile(r"\b10\.\d{4,9}/", re.IGNORECASE)
INTENT = re.compile(
    r"\b(?:docs?|documentation|manual|defaults?|error|version|parameter|argument|"
    r"alpha|filtering|reference|annotation|lfcse|lfcshrink|stat)\b",
    re.IGNORECASE,
)


class PolicyDenied(ValueError):
    """A query or result falls outside the run's documentation-only policy."""


def normalized(value: str) -> str:
    # Decode repeatedly so escaped DOI, accession and forbidden URL text is checked.
    for _ in range(4):
        decoded = unquote(value)
        if decoded == value:
            break
        value = decoded
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(c for c in value if unicodedata.category(c) != "Cf").split())


def parsed_url(url: str):
    if any(ord(c) < 32 or ord(c) == 127 for c in url) or "\\" in url:
        raise PolicyDenied("Malformed URL")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise PolicyDenied("Malformed URL port") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise PolicyDenied("Only HTTPS documentation URLs without credentials are allowed")
    if parsed.query:
        raise PolicyDenied("Documentation URLs with query strings need explicit review")
    decoded_path = normalized(parsed.path)
    if (
        "\\" in decoded_path
        or "//" in decoded_path
        or any(ord(c) < 32 for c in decoded_path)
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    ):
        raise PolicyDenied("Noncanonical documentation path")
    return parsed


class SearchPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    # Required per-task source paper, DOI, title, cohort and result identifiers.
    # Supply these from public task metadata, never by reading ground truth.
    forbidden_terms: tuple[str, ...] = Field(min_length=1)
    allowed_sources: tuple[str, ...] = DEFAULT_SOURCES
    forbidden_sources: tuple[str, ...] = ()
    allowed_topics: tuple[str, ...] = (
        "deseq2",
        "edger",
        "featurecounts",
        "lfcshrink",
        "bioconductor",
        "ensembl",
        "gencode",
        "nf-core",
        "samtools",
        "htslib",
        "subread",
    )

    @model_validator(mode="after")
    def validate_policy(self):
        if not self.allowed_sources or not self.allowed_topics:
            raise ValueError("Documentation sources and topics must be explicit")
        if any(len(normalized(term)) < 3 for term in self.forbidden_terms):
            raise ValueError("Study exclusions must be nonempty identifying strings")
        if any(not normalized(topic) for topic in self.allowed_topics):
            raise ValueError("Allowed topics must not be empty")
        for source in (*self.allowed_sources, *self.forbidden_sources):
            parsed = parsed_url("https://" + source)
            if parsed.fragment or parsed.path.endswith("/"):
                raise ValueError("Use host/path prefixes without fragments or trailing slashes")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def check_text(self, text: str, *, query: bool = False) -> None:
        value = normalized(text)
        if re.search(r"%[0-9a-f]{2}", value):
            raise PolicyDenied("Overencoded text cannot be screened reliably")
        if any(normalized(term) in value for term in self.forbidden_terms):
            raise PolicyDenied("Task-specific source or result identifier")
        if any(term in value for term in DENIED_TEXT) or ACCESSION.search(value):
            raise PolicyDenied("Study results, supplements, accessions or ground truth")
        if query and (DOI.search(value) or "doi.org" in value):
            raise PolicyDenied("DOI lookup is outside documentation search")

    def check_query(self, query: str) -> str:
        if not query.strip() or len(query) > 2000:
            raise PolicyDenied("Query must contain 1 to 2000 characters")
        self.check_text(query, query=True)
        value = normalized(query)
        if not INTENT.search(value) or not any(
            re.search(r"(?<!\w)" + re.escape(normalized(t)) + r"(?!\w)", value)
            for t in self.allowed_topics
        ):
            raise PolicyDenied("Query must ask about an approved tool's documentation")
        # Do not allow query text to broaden or override the source filter.
        if re.search(r"\b(?:site|inurl|filetype):|https?://", value):
            raise PolicyDenied("Source operators belong in the server-side policy")
        return " ".join(query.split())

    @staticmethod
    def _matches(url: str, prefix: str) -> bool:
        parsed = parsed_url(url)
        allowed = parsed_url("https://" + prefix)
        host = parsed.hostname.lower().removeprefix("www.")
        expected = allowed.hostname.lower().removeprefix("www.")
        # URL paths are case sensitive even though domain names are not.
        path, required = parsed.path, allowed.path
        for _ in range(4):
            path, required = unquote(path), unquote(required)
        path, required = path.rstrip("/"), required.rstrip("/")
        return host == expected and (
            not required or path == required or path.startswith(required + "/")
        )

    def check_url(self, url: str) -> None:
        parsed = parsed_url(url)
        self.check_text(url)
        host = parsed.hostname.lower().rstrip(".")
        if any(host == domain or host.endswith("." + domain) for domain in DENIED_DOMAINS):
            raise PolicyDenied("Denied source domain")
        path = normalized(parsed.path)
        if any(part in path for part in ("/geo", "/sra", "/supplement", "/preprint")):
            raise PolicyDenied("Result repository or publication path")
        if any(self._matches(url, prefix) for prefix in self.forbidden_sources):
            raise PolicyDenied("Task-specific source URL")
        if not any(self._matches(url, prefix) for prefix in self.allowed_sources):
            raise PolicyDenied("URL is outside approved documentation sources")

    def sdk_filters(self) -> dict:
        return {
            "include_domains": list(self.allowed_sources),
            "exclude_domains": list(dict.fromkeys((*DENIED_DOMAINS, *self.forbidden_sources))),
        }
