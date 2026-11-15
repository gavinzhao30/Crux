"""End critic: check delivered artifacts against the deliverable contract.

``await critique(llm, artifacts, contract)`` backs ``AgentRoles.critic``. Contract rules are
enforced in code (``check_contract``) and come back as ``block`` findings with stable rule names
(``RULES``). Then the model (role ``"critic"``) reviews METHODS.md, if one was delivered, for
departures from the method guidance, analyses stopped early and omissions. Model findings are
``warn`` unless they quote, verbatim, the contract or guidance text they break.

Finding ids are ``<artifact>:<rule>``. A repaired artifact that passes no longer carries the id,
which is how the UI marks a finding resolved, so rule names never change between rounds.

Two contract shapes are accepted: Task 9's ``{key: {path, row_key, kind, required_columns}}``
and the fakes' ``{"artifacts": {name: {"columns": [...]}}}``. Optional per-artifact keys:
``row_key_pattern`` (regex), ``unversioned_ids`` (bool), ``allowed_values`` (column -> labels),
``regulation_labels`` (labels for the ``regulation`` column) and ``required`` (default true).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from core.budget import BudgetExhausted
from core.interfaces import LLM
from core.schemas import Artifact, CriticFinding

PROMPT = Path(__file__).parent / "prompts" / "critic.md"
log = logging.getLogger(__name__)

# Contract rules checked in code. Every one is a ``block`` finding.
MISSING_DELIVERABLE = "missing_deliverable"
SHA256_MISMATCH = "sha256_mismatch"
UNPARSEABLE = "unparseable"
EMPTY_TABLE = "empty_table"
MISSING_COLUMNS = "missing_columns"
ROW_KEY_PATTERN = "row_key_pattern"
DUPLICATE_KEYS = "duplicate_keys"
PADJ_INVALID = "padj_invalid"
VERSIONED_IDS = "versioned_ids"
INVALID_LABELS = "invalid_labels"
BASE_MEAN_MISMATCH = "base_mean_mismatch"
RULES = (
    MISSING_DELIVERABLE,
    SHA256_MISMATCH,
    UNPARSEABLE,
    EMPTY_TABLE,
    MISSING_COLUMNS,
    ROW_KEY_PATTERN,
    DUPLICATE_KEYS,
    PADJ_INVALID,
    VERSIONED_IDS,
    INVALID_LABELS,
    BASE_MEAN_MISMATCH,
)
# METHODS.md review by the model: ``methods_<kind>``, plus one for a review that failed.
METHODS_KINDS = ("guidance", "stopped_early", "omission", "contract_violation")
METHODS_RULES = tuple(f"methods_{k}" for k in METHODS_KINDS) + ("methods_unreviewed",)

_SPEC_KEYS = {"path", "required_columns", "columns", "row_key", "kind", "format"}
_GUIDANCE_KEYS = ("method_guidance", "methods", "guidance")
_KEY_PATTERNS = {
    "gene_id": r"ENS[A-Z]*G\d+(?:\.\d+)?(?:_PAR_Y)?",
    "go_id": r"GO:\d{7}",
    "gene_symbol": r"\S+",
}
_ANY_KEY = r".*\S.*"
_VERSIONED = re.compile(r"[A-Za-z][A-Za-z0-9_]*?\d+\.\d+(?:_PAR_Y)?")
_NA = {"", "na", "nan"}
_COUNT_ANNOTATIONS = {"chr", "start", "end", "strand", "length"}
_COUNT_ID_COLUMNS = ("geneid", "gene_id", "gene", "id")

BASE_MEAN_GENES = 20
BASE_MEAN_MIN_GENES = 5
# A table fails when its median |log2 fold difference| from the recomputed base mean is
# above 1.5 (2.8x), or above 1 (2x) while the gene ranks barely agree (Spearman below 0.3).
# On simulated data with DESeq2 run on one cell line but recomputed over all samples this
# never fired (0 of 300), while shuffled, log-scale and p-value columns failed >= 99%.
BASE_MEAN_MAX_LOG2_RATIO = 1.5
BASE_MEAN_WEAK_LOG2_RATIO = 1.0
BASE_MEAN_WEAK_SPEARMAN = 0.3
MAX_EXAMPLES = 3
MAX_METHODS_CHARS = 20_000


@dataclass(frozen=True)
class DeliverableSpec:
    """One required artifact, normalised from either contract shape."""

    key: str
    path: str
    delimiter: str | None  # None: not a table, only its presence is checked
    kind: str | None = None
    required_columns: tuple[str, ...] = ()
    row_key: str | None = None
    key_pattern: str = _ANY_KEY
    unversioned: bool = False
    labels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    required: bool = True


def deliverable_specs(contract: dict) -> list[DeliverableSpec]:
    """The artifact specs in ``contract``, in contract order."""
    if not isinstance(contract, dict):
        return []
    raw = contract.get("artifacts")
    if isinstance(raw, dict):
        items = [(str(k), v) for k, v in raw.items()]
    elif isinstance(raw, list):
        items = [
            (str(v.get("name") or v.get("path") or i), v)
            for i, v in enumerate(raw)
            if isinstance(v, dict)
        ]
    else:
        items = [(str(k), v) for k, v in contract.items() if k != "recalled_evidence"]
    specs = []
    for key, spec in items:
        if not isinstance(spec, dict) or not _SPEC_KEYS & spec.keys():
            continue
        path = str(spec.get("path") or key)
        fmt = str(spec.get("format") or Path(path).suffix.lstrip(".") or "csv").lower()
        columns = tuple(
            str(c) for c in spec.get("required_columns") or spec.get("columns") or ()
        )
        row_key = spec.get("row_key")
        labels = {
            str(c): tuple(map(str, v))
            for c, v in (spec.get("allowed_values") or {}).items()
        }
        if spec.get("regulation_labels"):
            labels["regulation"] = tuple(map(str, spec["regulation_labels"]))
        specs.append(
            DeliverableSpec(
                key=key,
                path=path,
                delimiter={"csv": ",", "tsv": "\t"}.get(fmt),
                kind=spec.get("kind"),
                required_columns=columns,
                row_key=str(row_key) if row_key else None,
                key_pattern=str(
                    spec.get("row_key_pattern")
                    or _KEY_PATTERNS.get(str(row_key), _ANY_KEY)
                ),
                unversioned=bool(spec.get("unversioned_ids", "gene_id" in columns)),
                labels=labels,
                required=bool(spec.get("required", True)),
            )
        )
    return specs


async def critique(
    llm: LLM,
    artifacts: list[Artifact],
    contract: dict,
    *,
    counts_path: Path | None = None,
    seed: int = 0,
    brief: dict | None = None,
) -> list[CriticFinding]:
    """Contract findings from code, then the model's METHODS.md review.

    ``counts_path`` enables the base_mean spot check (skipped without it). ``brief`` is the
    lab brief; its ``goal`` and any method guidance go to the model with the contract.
    ``contract["recalled_evidence"]`` reaches the model prompt only.
    """
    contract = contract if isinstance(contract, dict) else {}
    findings = check_contract(artifacts, contract, counts_path=counts_path, seed=seed)
    methods = methods_artifact(artifacts)
    if methods is not None:
        findings += await review_methods(
            llm, methods, artifacts, contract, findings, brief
        )
    return findings


def check_contract(
    artifacts: list[Artifact],
    contract: dict,
    *,
    counts_path: Path | None = None,
    seed: int = 0,
) -> list[CriticFinding]:
    """Every contract rule that code can check, as ``block`` findings in contract order."""
    findings: list[CriticFinding] = []
    counts: dict[str, float] | None = None
    for spec in deliverable_specs(contract):
        artifact = _match(spec, artifacts)
        if artifact is None:
            if spec.required:
                findings.append(
                    _block(
                        spec.path, MISSING_DELIVERABLE, f"{spec.path} was not delivered"
                    )
                )
            continue
        if (
            counts is None
            and counts_path is not None
            and "base_mean" in spec.required_columns
        ):
            counts = _counts_base_mean(Path(counts_path))
        findings += _check_artifact(spec, artifact, counts, seed)
    return findings


def methods_artifact(artifacts: list[Artifact]) -> Artifact | None:
    return next(
        (a for a in artifacts if Path(a.name).name.lower() == "methods.md"), None
    )


# --- code checks ----------------------------------------------------------------------------


def _match(spec: DeliverableSpec, artifacts: list[Artifact]) -> Artifact | None:
    want = {spec.path, Path(spec.path).name, spec.key}
    for a in artifacts:
        if a.name in want or Path(a.name).name in want or Path(a.name).stem == spec.key:
            return a
    return None


def _check_artifact(
    spec: DeliverableSpec,
    artifact: Artifact,
    counts: dict[str, float] | None,
    seed: int,
) -> list[CriticFinding]:
    name = artifact.name
    path = Path(artifact.path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [
            _block(
                name,
                MISSING_DELIVERABLE,
                f"{name} cannot be read: {type(exc).__name__}",
            )
        ]
    out = []
    if artifact.sha256 and hashlib.sha256(data).hexdigest() != artifact.sha256:
        out.append(
            _block(name, SHA256_MISMATCH, f"{name} changed after it was exported")
        )
    if spec.delimiter is None:
        return out
    try:
        header, rows = _parse_table(data, spec.delimiter)
    except ValueError as exc:
        return [*out, _block(name, UNPARSEABLE, f"{name} does not parse: {exc}")]
    if not header:
        return [*out, _block(name, EMPTY_TABLE, f"{name} is empty")]
    missing = [c for c in spec.required_columns if c not in header]
    if missing:
        out.append(
            _block(
                name,
                MISSING_COLUMNS,
                f"{name} is missing column(s): {', '.join(missing)}",
            )
        )
    if not rows:
        return [*out, _block(name, EMPTY_TABLE, f"{name} has a header but no rows")]
    col = {c: i for i, c in reversed(list(enumerate(header)))}

    def values(column: str) -> list[str]:
        return [row[col[column]].strip() for row in rows]

    if spec.row_key in col:
        keys = values(spec.row_key)
        pattern = re.compile(spec.key_pattern)
        bad = [k for k in keys if not pattern.fullmatch(k)]
        if bad:
            out.append(
                _block(
                    name,
                    ROW_KEY_PATTERN,
                    f"{len(bad)} of {len(keys)} {spec.row_key} values do not match "
                    f"{spec.key_pattern!r}, e.g. {_examples(bad)}",
                )
            )
        dupes = [k for k, n in Counter(keys).items() if n > 1 and k]
        if dupes:
            out.append(
                _block(
                    name,
                    DUPLICATE_KEYS,
                    f"{len(dupes)} {spec.row_key} value(s) appear more than once, "
                    f"e.g. {_examples(dupes)}",
                )
            )
    if "padj" in col:
        message = _padj_problem(values("padj"))
        if message:
            out.append(_block(name, PADJ_INVALID, f"{name}: {message}"))
    if spec.unversioned and "gene_id" in col:
        versioned = [g for g in values("gene_id") if _VERSIONED.fullmatch(g)]
        if versioned:
            out.append(
                _block(
                    name,
                    VERSIONED_IDS,
                    f"{len(versioned)} gene_id values carry a version suffix where the contract "
                    f"wants unversioned ids, e.g. {_examples(versioned)}",
                )
            )
    for column, allowed in spec.labels.items():
        if column in col:
            bad = sorted({v for v in values(column)} - set(allowed))
            if bad:
                out.append(
                    _block(
                        name,
                        INVALID_LABELS,
                        f"{column} has label(s) {_examples(bad)} outside {list(allowed)}",
                    )
                )
    if counts and "base_mean" in col and "gene_id" in col:
        message = _base_mean_problem(
            values("gene_id"), values("base_mean"), counts, seed, name
        )
        if message:
            out.append(_block(name, BASE_MEAN_MISMATCH, f"{name}: {message}"))
    return out


def _parse_table(data: bytes, delimiter: str) -> tuple[list[str], list[list[str]]]:
    """Header and rows; ``ValueError`` if the bytes are not a well-formed table."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("not UTF-8 text") from exc
    try:
        rows = [
            r
            for r in csv.reader(text.splitlines(), delimiter=delimiter, strict=True)
            if r
        ]
    except csv.Error as exc:
        raise ValueError(str(exc)) from exc
    if not rows:
        return [], []
    header, body = rows[0], rows[1:]
    for line_no, row in enumerate(body, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"row {line_no} has {len(row)} fields, the header has {len(header)}"
            )
    return header, body


def _padj_problem(raw: list[str]) -> str | None:
    non_numeric, out_of_range = [], []
    for v in raw:
        if v.lower() in _NA:
            continue
        try:
            p = float(v)
        except ValueError:
            non_numeric.append(v)
            continue
        if not 0.0 <= p <= 1.0:  # also catches nan and inf
            out_of_range.append(v)
    parts = []
    if non_numeric:
        parts.append(
            f"{len(non_numeric)} padj values are not numeric, e.g. {_examples(non_numeric)}"
        )
    if out_of_range:
        parts.append(
            f"{len(out_of_range)} padj values are outside 0..1, e.g. {_examples(out_of_range)}"
        )
    return "; ".join(parts) or None


def _counts_base_mean(path: Path) -> dict[str, float]:
    """Unversioned gene id -> DESeq2 base mean (mean of median-of-ratios normalised counts).

    Reads a featureCounts-style matrix (``#`` comments, ``Geneid``, annotation columns) or a
    plain gene x sample table. Returns ``{}`` if no size factors can be estimated.
    """
    import pandas as pd

    try:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, comment="#")
    except (OSError, ValueError) as exc:
        log.warning("cannot read counts %s: %s", path, exc)
        return {}
    if df.empty:
        return {}
    lower = {str(c).lower(): c for c in df.columns}
    id_col = next((lower[c] for c in _COUNT_ID_COLUMNS if c in lower), df.columns[0])
    samples = [
        c
        for c in df.columns
        if c != id_col
        and str(c).lower() not in _COUNT_ANNOTATIONS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not samples:
        return {}
    counts = df[samples].to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        logs = np.log(counts)
    log_geo_means = logs.mean(axis=1)
    usable = np.isfinite(log_geo_means)
    if not usable.any():
        return {}
    size_factors = np.exp(np.median(logs[usable] - log_geo_means[usable, None], axis=0))
    base_mean = (counts / size_factors).mean(axis=1)
    out: dict[str, float] = {}
    for gene, value in zip(df[id_col].astype(str), base_mean, strict=True):
        out.setdefault(_unversioned(gene), float(value))
    return out


def _base_mean_problem(
    genes: list[str],
    reported: list[str],
    counts: dict[str, float],
    seed: int,
    name: str,
) -> str | None:
    """Recompute base_mean for ``BASE_MEAN_GENES`` seeded genes and compare.

    The critic does not know which samples each comparison used, so the test is tolerant (see
    ``BASE_MEAN_MAX_LOG2_RATIO``): a DESeq2 base_mean from any subset of the samples passes,
    while constant, shuffled, log-scale or copied-column placeholders fail.
    """
    pairs = {}
    for gene, value in zip(genes, reported, strict=True):
        key = _unversioned(gene)
        if key in counts and key not in pairs:
            pairs[key] = value
    if len(pairs) < BASE_MEAN_MIN_GENES:
        return None
    rng = random.Random(f"{seed}:{name}")
    chosen = rng.sample(sorted(pairs), min(BASE_MEAN_GENES, len(pairs)))
    expected = np.array([counts[g] for g in chosen])
    try:
        got = np.array([float(pairs[g]) for g in chosen])
    except ValueError:
        return "base_mean has non-numeric values for genes in the count matrix"
    if not np.isfinite(got).all() or (got < 0).any():
        return "base_mean has negative or non-finite values"
    rho = _spearman(got, expected)
    ratio = float(np.median(np.abs(np.log2((got + 1.0) / (expected + 1.0)))))
    where = f"{len(chosen)} seeded genes (seed {seed})"
    if rho is None and np.ptp(got) == 0 and np.ptp(expected) > 0:
        return f"base_mean is constant ({got[0]:g}) across {where}, a placeholder"
    weak = rho is None or rho < BASE_MEAN_WEAK_SPEARMAN
    if ratio > BASE_MEAN_MAX_LOG2_RATIO or (weak and ratio > BASE_MEAN_WEAK_LOG2_RATIO):
        spearman = "undefined" if rho is None else f"{rho:.2f}"
        return (
            f"base_mean does not match the normalised counts across {where}: median "
            f"|log2 ratio| {ratio:.2f}, Spearman {spearman}"
        )
    return None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ra, rb = _ranks(a), _ranks(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def _unversioned(gene: str) -> str:
    gene = gene.strip()
    return gene.split(".", 1)[0] if _VERSIONED.fullmatch(gene) else gene


def _examples(items: list[str]) -> str:
    return ", ".join(repr(i) for i in items[:MAX_EXAMPLES])


def _block(artifact: str, rule: str, message: str) -> CriticFinding:
    return _finding(artifact, rule, message, "block")


def _finding(
    artifact: str, rule: str, message: str, severity: Literal["block", "warn"]
) -> CriticFinding:
    return CriticFinding(
        id=f"{artifact}:{rule}",
        artifact=artifact,
        severity=severity,
        rule=rule,
        message=message,
    )


# --- METHODS.md review by the model ---------------------------------------------------------


class MethodsIssue(BaseModel):
    kind: Literal["guidance", "stopped_early", "omission", "contract_violation"]
    message: str
    contract_quote: str


class MethodsReview(BaseModel):
    issues: list[MethodsIssue]


async def review_methods(
    llm: LLM,
    methods: Artifact,
    artifacts: list[Artifact],
    contract: dict,
    code_findings: list[CriticFinding],
    brief: dict | None = None,
) -> list[CriticFinding]:
    """The model's METHODS.md findings, one per kind, ``warn`` unless a quote checks out."""
    try:
        text = Path(methods.path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []  # an unreadable METHODS.md is not a contract deliverable here
    guidance = _guidance(contract, brief)
    shown_contract = {
        s.path: {
            "kind": s.kind,
            "required_columns": list(s.required_columns),
            "row_key": s.row_key,
        }
        for s in deliverable_specs(contract)
    }
    payload = {
        "methods_md": text[:MAX_METHODS_CHARS],
        "contract": shown_contract,
        "method_guidance": guidance,
        "delivered": [a.name for a in artifacts],
        "code_check_failures": [f"{f.artifact}: {f.rule}" for f in code_findings],
        "recalled_evidence": _recalled(contract),
    }
    messages = [
        {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]
    try:
        [review] = await llm.structured("critic", MethodsReview, messages)
    except BudgetExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed call falls back, never crashes
        log.warning("critic model call failed: %s", exc)
        return [
            _finding(
                methods.name,
                "methods_unreviewed",
                f"{methods.name} was not reviewed: the model call failed ({type(exc).__name__})",
                "warn",
            )
        ]
    review = MethodsReview.model_validate(review)
    citable = _norm(json.dumps(shown_contract) + " " + " ".join(guidance))
    grouped: dict[str, list[MethodsIssue]] = {}
    for issue in review.issues:
        if issue.message.strip():
            grouped.setdefault(issue.kind, []).append(issue)
    out = []
    for kind in METHODS_KINDS:
        issues = grouped.get(kind)
        if not issues:
            continue
        cited = any(
            kind == "contract_violation"
            and len(_norm(i.contract_quote)) >= 3
            and _norm(i.contract_quote) in citable
            for i in issues
        )
        message = "; ".join(" ".join(i.message.split()) for i in issues)
        out.append(
            _finding(
                methods.name, f"methods_{kind}", message, "block" if cited else "warn"
            )
        )
    return out


def _guidance(contract: dict, brief: dict | None) -> list[str]:
    texts = []
    for source in (contract, brief or {}):
        for key in (*_GUIDANCE_KEYS, "goal"):
            value = source.get(key)
            if isinstance(value, str) and value.strip() and value not in texts:
                texts.append(value)
    return texts


def _recalled(contract: dict) -> list[dict]:
    out = []
    for item in contract.get("recalled_evidence") or []:
        if isinstance(item, dict) and item.get("claim"):
            out.append({"kind": item.get("kind"), "claim": item["claim"]})
    return out


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[\"'`]", "", str(text)).lower().split()).strip(" .;:,")
