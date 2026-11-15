"""One Letter engine: which HBB variant is on the strip under the Jetson camera?

A standalone Crux loop over one categorical question, ``q-variant``, written against the
reader API and the events in tools/ONE_LETTER_CONTRACT.md. It reuses ``core.scoring`` (rank,
update, is_surprise, predictive), the ``core.schemas`` models and ``core.events.EventBus``;
``core`` itself is unchanged.

Per step:

1. Score the designs: each window not read yet on this strip (``window-1..3``) and a
   replicate re-read of the last window read. Outcome bins are the distinct 9-letter strings
   the real hypotheses predict for that window plus ``other``; a read matches a predicted
   string with probability ``prod((1 - e) or e / 3)`` per base (read error ``e`` = 0.02) and
   ``other`` takes the rest. The null ``h0`` gets a uniform row, as in ``core.eig``, and is
   left out of the EIG (Crux's convention), so windows 1 and 2, where every variant agrees,
   are worth exactly 0 bits and window 3 is chosen first.
2. Emit ``dna.request``, ``POST /show`` so the strip page renders the window, and poll
   ``GET /read`` until that window is ``stable`` (with a timeout and ``experiment.log``
   progress lines). Then emit ``experiment.started``, ``experiment.finished``, ``dna.read``
   and ``belief.updated``.
3. A read in ``other`` (or a Crux surprise: low predictive probability or ``h0`` above 0.4)
   raises ``surprise.raised`` and a red overlay on the mismatching cells, then a replicate.
   If the replicate reads the same letters, the engine regenerates: it looks the read up in
   the known-variant library (``variants.LIBRARY``) and adds the match, or names an unlisted
   change from the standard genetic code, re-scores every read of this strip under the new
   hypothesis set and continues until a hypothesis passes ``settle``.
4. Settle: ``question.settled``, a green overlay, ``dna.verdict`` (text from the reader's
   ``POST /verdict``, i.e. the Jetson's local model, or a template if that fails) and, for a
   regenerated variant, a lesson in ``runs/oneletter_lessons.jsonl`` and ``dna.lesson``.
5. Watch mode (or ``strips > 1``): keep reading the settled window; when its letters change,
   the judge loaded another sample, and the question is reopened.

Regeneration re-emits ``hypothesis.merged`` (the new priors) and ``experiment.proposed`` (the
new tables) before replaying the strip's reads, so ``core.replay`` rebuilds every belief.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from core import events as E
from core import scoring
from core.events import EventBus, jsonable, new_run_id
from core.schemas import (
    Experiment,
    Hypothesis,
    Observation,
    OutcomeBin,
    Question,
    Report,
    RunEvent,
    bin_for,
)
from labs.oneletter import variants as V
from labs.oneletter.verdict_check import check_verdict

logger = logging.getLogger(__name__)

LAB = "oneletter"
QUESTION_ID = "q-variant"
QUESTION_TEXT = "Which HBB variant is on this strip?"
NULL_ID = "h0"
OTHER = "other"
READER_ENV = "CRUX_ONELETTER_READER"
# A file, not a folder: the API lists every folder under runs/ as a run.
DEFAULT_LESSONS = Path("runs") / "oneletter_lessons.jsonl"

DNA_REQUEST = "dna.request"
DNA_READ = "dna.read"
DNA_VARIANT = "dna.variant"
DNA_VERDICT = "dna.verdict"
DNA_LESSON = "dna.lesson"
DNA_EVENTS = (DNA_REQUEST, DNA_READ, DNA_VARIANT, DNA_VERDICT, DNA_LESSON)
EVENT_TYPES = (*E.EVENT_TYPES, *DNA_EVENTS)

RESIDUE = 6
EIG_ZERO = 1e-9


@dataclass
class OneLetterConfig:
    """Run settings. ``to_event()`` is what ``run.started.config`` records."""

    reader_url: str
    timeout_s: float = 120.0  # wait for one requested window
    poll_s: float = 0.25
    read_error: float = 0.02  # per-base probability that the camera reads a wrong letter
    settle: float = 0.95
    null_prior: float = 0.1
    cost_lambda: float = 0.2
    est_cost_s: float = 10.0  # the page changes and the camera settles: same for every design
    max_reads: int = 8  # per strip
    max_rereads: int = 3  # surprising reads in a row before the question is unresolved
    replicate_gap_s: float = 1.0  # a replicate uses frames this much newer (reader clock)
    strips: int = 1  # samples to settle before finishing
    watch: bool = False  # keep watching for new samples until stopped
    watch_timeout_s: float | None = None
    swap_confirm: int = 2  # stable reads of the new letters that confirm a swap
    lessons_path: str = DEFAULT_LESSONS.as_posix()
    fake_reader: bool = False
    verdict_max_tokens: int = 100
    verdict_timeout_s: float = 60.0
    verdict_attempts: int = 2  # model sentences tried before the template; each is fact-checked
    progress_every_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.reader_url:
            raise ValueError(f"reader_url is required (or set {READER_ENV})")
        if not 0.0 < self.read_error < 0.75:
            raise ValueError("read_error must be in (0, 0.75)")
        if not 0.5 < self.settle <= 1.0:
            raise ValueError("settle must be in (0.5, 1]")
        if not 0.0 < self.null_prior < 1.0:
            raise ValueError("null_prior must be in (0, 1)")
        if self.timeout_s <= 0 or self.poll_s < 0 or self.max_reads < 1 or self.strips < 1:
            raise ValueError("timeout_s > 0, poll_s >= 0, max_reads >= 1 and strips >= 1")

    @property
    def annotated_url(self) -> str:
        return self.reader_url.rstrip("/") + "/annotated_stream"

    def to_event(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["annotated_url"] = self.annotated_url
        return out


# --- the reader --------------------------------------------------------------------------


class ReaderError(RuntimeError):
    """The reader could not be reached or answered with an error."""


class Reader(Protocol):
    async def read(self) -> dict: ...
    async def show(self, window: int) -> None: ...
    async def overlay(self, body: dict) -> None: ...
    async def verdict(self, prompt: str, max_tokens: int, timeout_s: float) -> dict: ...
    async def aclose(self) -> None: ...


class HttpReader:
    """The reader service over HTTP (``transport`` lets tests serve a fake in process)."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, transport=transport, timeout=timeout_s
        )

    async def _call(self, method: str, path: str, **kw: Any) -> dict:
        try:
            r = await self._client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise ReaderError(f"{method} {path}: {type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise ReaderError(f"{method} {path}: HTTP {r.status_code}")
        try:
            body = r.json()
        except ValueError as exc:
            raise ReaderError(f"{method} {path}: not JSON") from exc
        if not isinstance(body, dict):
            raise ReaderError(f"{method} {path}: not a JSON object")
        return body

    async def read(self) -> dict:
        return await self._call("GET", "/read")

    async def show(self, window: int) -> None:
        await self._call("POST", "/show", json={"window": window})

    async def overlay(self, body: dict) -> None:
        await self._call("POST", "/overlay", json=body)

    async def verdict(self, prompt: str, max_tokens: int, timeout_s: float) -> dict:
        return await self._call(
            "POST",
            "/verdict",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=timeout_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --- likelihoods ---------------------------------------------------------------------------


def verdict_prompt(v: V.Variant, p: float, w: int, mine: list[str]) -> str:
    """The facts the Jetson model gets for its one-sentence verdict (and nothing else)."""
    agree = sum(b == v.window(w) for b in mine)
    start, end = V.WINDOWS[w]
    facts = [
        (
            f"A camera read window {w} (nucleotides {start}-{end} of the human HBB "
            f"gene, RefSeq {V.REFSEQ}) on a demo strip shown on a screen "
            f"{len(mine)} times: {', '.join(mine)}. {agree} of them match the "
            "answer below."
        ),
        f"Codon 6 reads {v.codon} ({v.amino_acid}); the normal codon 6 is GAG (Glu).",
    ]
    if v.key == "hba":
        facts.append("Answer: HbA, normal haemoglobin.")
    elif v.known:
        facts.append(f"Answer: {v.name}, {v.hgvs}, dbSNP {v.rsid}, legacy name {v.legacy}.")
    else:
        facts.append(
            f"Answer: an unknown variant ({v.legacy}); no variant in the demo's library "
            "(HbS, HbC) matches it."
        )
    facts.append(f"Posterior probability {p:.2f}. This is a demo strip, not a patient sample.")
    return (
        "Write one plain sentence of at most 30 words for a live science demo screen. "
        "Use only these facts: copy every sequence, codon and code exactly as written, "
        "and add no other facts and no medical claims or advice.\nFacts:\n"
        + "\n".join(f"- {f}" for f in facts)
    )


def read_probability(observed: str, true: str, error: float) -> float:
    """P(camera reads ``observed`` | the window carries ``true``), independent per base."""
    p = 1.0
    for a, b in zip(observed, true, strict=True):
        p *= (1.0 - error) if a == b else error / 3.0
    return p


def window_table(
    window: int, real: list[V.Variant], error: float
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Bins (distinct predicted strings, then ``other``) and P(bin | hypothesis)."""
    bins: list[str] = []
    for v in real:
        s = v.window(window)
        if s not in bins:
            bins.append(s)
    lik: dict[str, dict[str, float]] = {}
    for v in real:
        row = {b: read_probability(b, v.window(window), error) for b in bins}
        row[OTHER] = max(0.0, 1.0 - sum(row.values()))
        lik[v.hypothesis_id] = row
    n = len(bins) + 1
    lik[NULL_ID] = {b: 1.0 / n for b in [*bins, OTHER]}  # uniform null, as in core.eig
    return [*bins, OTHER], lik


@dataclass
class ReadRecord:
    experiment_id: str
    window: int
    bases: str
    confidence: list[float]
    reader_ts: float | None
    bin_id: str = OTHER


@dataclass
class StripResult:
    strip: int
    outcome: str  # settled | unresolved | timeout
    hypothesis_id: str | None = None
    posterior: float | None = None
    reads: list[ReadRecord] = field(default_factory=list)
    reason: str = ""


EventHook = Callable[[RunEvent], Awaitable[None] | None]


# --- the engine ----------------------------------------------------------------------------


class OneLetterEngine:
    def __init__(
        self,
        cfg: OneLetterConfig,
        reader: Reader,
        bus: EventBus,
        *,
        on_event: EventHook | None = None,
    ) -> None:
        self.cfg = cfg
        self.reader = reader
        self.bus = bus
        self.run_id = bus.run_id
        self.on_event = on_event
        self.lessons_path = Path(cfg.lessons_path)
        self.question = Question(
            id=QUESTION_ID, text=QUESTION_TEXT, kind="categorical", raised_by=LAB
        )
        self.real: list[V.Variant] = []
        self.hyp: dict[str, Hypothesis] = {}
        self.belief: dict[str, float] = {}
        self.from_lessons: set[str] = set()
        self.regenerated: set[str] = set()
        self.experiments: dict[str, Experiment] = {}
        self._published: dict[str, str] = {}
        self.strip = 1
        self.reads: list[ReadRecord] = []
        self.results: list[StripResult] = []
        self.shown: int | None = None  # the last window requested with /show
        self.last_ts: float | None = None  # reader clock of the last accepted read
        self._started = False
        self._terminal = False

    # --- events

    async def emit(self, type_: str, payload: dict) -> RunEvent:
        event = await self.bus.emit(type_, jsonable(payload))
        if self.on_event is not None:
            try:
                out = self.on_event(event)
                if asyncio.iscoroutine(out):
                    await out
            except Exception:  # a display hook must not stop the run
                logger.exception("on_event hook failed")
        return event

    async def log(self, experiment_id: str, line: str) -> None:
        await self.emit(
            E.EXPERIMENT_LOG,
            {"experiment_id": experiment_id, "stream": "stdout", "line": line[:500]},
        )

    async def _overlay(self, body: dict, eid: str) -> None:
        try:
            await self.reader.overlay(body)
        except ReaderError as exc:
            await self.log(eid, f"overlay not shown ({exc})")

    # --- hypotheses

    def _variant(self, hid: str) -> V.Variant:
        return next(v for v in self.real if v.hypothesis_id == hid)

    def _hypothesis(self, v: V.Variant, prior: float) -> Hypothesis:
        if v.hypothesis_id in self.regenerated:
            author, why = "regenerator", f"proposed after a surprise: {v.source}"
        elif v.hypothesis_id in self.from_lessons:
            author, why = "lesson", f"saved as a lesson by an earlier run: {v.source}"
        else:
            author, why = "variant-library", v.source
        return Hypothesis(
            id=v.hypothesis_id,
            question_id=QUESTION_ID,
            statement=v.statement(),
            author=author,
            rationale=why,
            prior=prior,
            posterior=prior,
        )

    def _null(self, prior: float) -> Hypothesis:
        return Hypothesis(
            id=NULL_ID,
            question_id=QUESTION_ID,
            statement="None of these: the strip carries letters no listed variant predicts",
            author=LAB,
            rationale="built-in null option (an unlisted variant, or a camera misread)",
            is_null=True,
            prior=prior,
            posterior=prior,
        )

    @staticmethod
    def _null_variant() -> dict:
        return {
            "hypothesis_id": NULL_ID,
            "name": "None of these",
            "hgvs": "",
            "legacy": "",
            "rsid": "",
            "codon": "",
            "amino_acid": "",
            "source": "built-in null option",
        }

    def _priors(self) -> dict[str, float]:
        n = len(self.real)
        out = {v.hypothesis_id: (1.0 - self.cfg.null_prior) / n for v in self.real}
        out[NULL_ID] = self.cfg.null_prior
        return out

    async def _reset_beliefs(self, *, announce: list[str]) -> None:
        """Equal priors over the real options, ``null_prior`` on h0; ``hypothesis.proposed``
        and ``dna.variant`` for each id in ``announce``; then ``hypothesis.merged``."""
        priors = self._priors()
        self.hyp = {
            v.hypothesis_id: self._hypothesis(v, priors[v.hypothesis_id]) for v in self.real
        }
        self.hyp[NULL_ID] = self._null(priors[NULL_ID])
        self.belief = dict(priors)
        for hid in announce:
            await self.emit(E.HYPOTHESIS_PROPOSED, {"hypothesis": self.hyp[hid]})
            variant = self._null_variant() if hid == NULL_ID else self._variant(hid).dna_variant()
            await self.emit(DNA_VARIANT, variant)
        await self.emit(
            E.HYPOTHESIS_MERGED,
            {"question_id": QUESTION_ID, "kept": "", "merged": [], "priors": priors},
        )

    def _sync(self) -> None:
        for hid, p in self.belief.items():
            self.hyp[hid].posterior = p

    # --- designs

    def _eid(self, window: int, n: int) -> str:
        eid = f"window-{window}" + (f"-r{n}" if n > 1 else "")
        return eid + (f"-s{self.strip}" if self.strip > 1 else "")

    def _designs(self) -> list[tuple[str, int, bool]]:
        read = [r.window for r in self.reads]
        out = [(self._eid(w, 1), w, False) for w in V.WINDOWS if w not in read]
        if self.reads:
            last = self.reads[-1].window
            out.append((self._eid(last, read.count(last) + 1), last, True))
        return out

    def _experiment(self, eid: str, window: int, replicate: bool) -> Experiment:
        bins, lik = window_table(window, self.real, self.cfg.read_error)
        start, end = V.WINDOWS[window]
        owners = {
            b: [v.hypothesis_id for v in self.real if v.window(window) == b] for b in bins[:-1]
        }
        outcome = [
            OutcomeBin(
                id=b,
                description=f"reads {b} (predicted by {', '.join(owners[b])})",
                rule=f"bases == '{b}'",
            )
            for b in bins[:-1]
        ]
        outcome.append(
            OutcomeBin(
                id=OTHER,
                description="any other 9 letters (no hypothesis predicts it)",
                rule=f"bases not in {bins[:-1]!r}",
            )
        )
        verb = "Re-read" if replicate else "Read"
        return Experiment(
            id=eid,
            question_ids=[QUESTION_ID],
            kind="measure",
            goal=f"{verb} window {window} (nt {start}-{end})"
            + (" (replicate)" if replicate else ""),
            design={
                "window": window,
                "replicate": replicate,
                "nt": [start, end],
                "strip": self.strip,
            },
            bins=outcome,
            likelihood=lik,
            likelihood_source="exact",
            est_cost_s=self.cfg.est_cost_s,
        )

    async def _publish(self, exp: Experiment) -> None:
        """Store ``exp`` and emit ``experiment.proposed`` when it is new or its table changed."""
        sig = exp.model_dump_json(include={"bins", "likelihood"})
        self.experiments[exp.id] = exp
        if self._published.get(exp.id) != sig:
            self._published[exp.id] = sig
            await self.emit(E.EXPERIMENT_PROPOSED, {"experiment": exp})

    def rank(self) -> list[Experiment]:
        """Candidate designs with ``eig_bits`` and ``score`` set, best first (core.scoring)."""
        self._sync()
        exps = [self._experiment(*d) for d in self._designs()]
        return scoring.rank(self.question, list(self.hyp.values()), exps, lam=self.cfg.cost_lambda)

    # --- reading

    async def _wait_for(self, window: int, eid: str, min_ts: float | None) -> dict | None:
        """Poll ``/read`` until ``window`` is stable with 9 valid letters (newer than
        ``min_ts`` on the reader clock), or return ``None`` after ``timeout_s``."""
        t0 = time.monotonic()
        last_state, last_note = "", t0
        while True:
            try:
                r = await self.reader.read()
            except ReaderError as exc:
                state = f"reader error: {exc}"
            else:
                w, bases, ts = r.get("window"), r.get("bases"), r.get("ts")
                fresh = min_ts is None or not isinstance(ts, (int, float)) or ts > min_ts
                if w == window and r.get("stable") is True and V.is_bases(bases) and fresh:
                    return r
                if w is None:
                    state = "the camera sees no window card"
                elif w != window:
                    state = f"the camera sees window {w}, waiting for window {window}"
                elif not V.is_bases(bases):
                    state = f"window {window} seen, letters not readable yet"
                elif not fresh:
                    state = f"window {window} seen, waiting for a fresh frame"
                else:
                    state = f"window {window} seen ({bases}), waiting for a stable read"
            now = time.monotonic()
            if state != last_state:
                await self.log(eid, state)
                last_state, last_note = state, now
            elif now - last_note >= self.cfg.progress_every_s:
                await self.log(eid, f"still waiting for window {window} ({now - t0:.0f} s)")
                last_note = now
            if now - t0 >= self.cfg.timeout_s:
                await self.log(eid, f"timed out after {self.cfg.timeout_s:.0f} s")
                return None
            await asyncio.sleep(self.cfg.poll_s)

    async def _show(self, window: int, eid: str) -> None:
        try:
            await self.reader.show(window)
            self.shown = window
        except ReaderError as exc:  # an older reader without /show: a person shows it
            await self.log(eid, f"POST /show failed ({exc}); show window {window} by hand")

    async def _request(self, exp: Experiment, ranked: list[Experiment], why: str) -> dict | None:
        window, replicate = exp.design["window"], exp.design["replicate"]
        await self.emit(
            DNA_REQUEST,
            {
                "window": window,
                "reason": why,
                "eig_bits": exp.eig_bits or 0.0,
                "replicate": replicate,
                "experiment_id": exp.id,
                "strip": self.strip,
            },
        )
        await self._show(window, exp.id)
        min_ts = self.last_ts
        if replicate:
            prev = [r.reader_ts for r in self.reads if r.window == window and r.reader_ts]
            if prev:
                min_ts = max(min_ts or 0.0, prev[-1] + self.cfg.replicate_gap_s)
        return await self._wait_for(window, exp.id, min_ts)

    def _reason(self, chosen: Experiment, ranked: list[Experiment]) -> str:
        w = chosen.design["window"]
        others = ", ".join(f"{e.id} {e.eig_bits or 0.0:.2f}" for e in ranked if e.id != chosen.id)
        head = (
            f"re-read window {w} to confirm"
            if chosen.design["replicate"]
            else f"window {w} is the most informative read"
        )
        return f"{head}: {chosen.eig_bits or 0.0:.2f} bits (others: {others or 'none'})"

    def _expected(self, window: int) -> dict[str, str]:
        return {v.hypothesis_id: v.window(window) for v in self.real}

    def _closest(self, window: int, bases: str) -> tuple[str, list[int]]:
        expected = self._expected(window)
        hid = min(expected, key=lambda h: V.hamming(bases, expected[h]))
        return hid, V.mismatches(bases, expected[hid])

    async def _observe(
        self, exp: Experiment, r: dict, seconds: float
    ) -> tuple[ReadRecord, bool, float]:
        """Emit the observation and update the belief. Returns the read, whether it is a
        surprise, and its predictive probability."""
        window, bases = exp.design["window"], r["bases"]
        conf = [float(c) for c in (r.get("confidence") or [])]
        ts = r.get("ts") if isinstance(r.get("ts"), (int, float)) else None
        bin_id = bin_for(exp.bins, {"bases": bases}) or OTHER
        prior = dict(self.belief)
        pbar = scoring.predictive(prior, exp.likelihood, bin_id)
        posterior = scoring.update(prior, exp.likelihood, bin_id)
        closest, mism = self._closest(window, bases)
        await self.emit(
            E.EXPERIMENT_STARTED,
            {"experiment_id": exp.id, "question_ids": [QUESTION_ID], "kind": exp.kind},
        )
        obs = Observation(
            experiment_id=exp.id,
            valid=True,
            value={"window": window, "bases": bases, "confidence": conf},
            bin_id=bin_id,
            surprise=-math.log(pbar) if pbar > 0 else 0.0,
            seconds=seconds,
        )
        await self.emit(E.EXPERIMENT_FINISHED, {"observation": obs})
        await self.emit(
            DNA_READ,
            {
                "experiment_id": exp.id,
                "window": window,
                "bases": bases,
                "expected": self._expected(window),
                "mismatch_positions": mism,
                "confidence": conf,
                "closest": closest,
                "bin_id": bin_id,
                "replicate": exp.design["replicate"],
                "strip": self.strip,
            },
        )
        await self.emit(
            E.BELIEF_UPDATED,
            {
                "question_id": QUESTION_ID,
                "experiment_id": exp.id,
                "observed": bin_id,
                "prior": prior,
                "posterior": posterior,
            },
        )
        surprise = bin_id == OTHER or scoring.is_surprise(
            prior, exp.likelihood, bin_id, null_ids={NULL_ID}
        )
        self.belief = dict(posterior)
        self._sync()
        rec = ReadRecord(exp.id, window, bases, conf, ts, bin_id)
        self.reads.append(rec)
        if ts is not None:
            self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)
        return rec, surprise, pbar

    async def _surprise(self, rec: ReadRecord, pbar: float, reason: str) -> None:
        closest, mism = self._closest(rec.window, rec.bases)
        await self.emit(
            E.SURPRISE_RAISED,
            {
                "question_id": QUESTION_ID,
                "experiment_id": rec.experiment_id,
                "reason": reason,
                "z_score": None,
                "null_mass": self.belief.get(NULL_ID),
                "window": rec.window,
                "bases": rec.bases,
                "mismatch_positions": mism,
                "closest": closest,
                "predictive": pbar,
            },
        )
        await self._overlay(
            {
                "highlight": mism,
                "color": "red",
                "message": f"Surprise: window {rec.window} reads {rec.bases}; "
                "no hypothesis predicts it. Re-reading.",
            },
            rec.experiment_id,
        )

    # --- regeneration

    async def _regenerate(self, rec: ReadRecord) -> str | None:
        """Add the variant that explains ``rec`` and replay the strip. Returns an
        unresolved reason when nothing in HBB explains it."""
        have = {v.key for v in self.real}
        v = V.match_library(rec.window, rec.bases, exclude=have)
        if v is None:
            v = V.novel_variant(rec.window, rec.bases)
            if v is None or v.key in have:
                ref = V.reference_window(rec.window)
                n = V.hamming(rec.bases, ref)
                return (
                    f"window {rec.window} reads {rec.bases}, {n} of 9 letters away from the HBB "
                    f"reference {ref}: not an HBB variant this demo can name"
                )
        self.real.append(v)
        self.regenerated.add(v.hypothesis_id)
        priors = self._priors()
        old = self.hyp
        self.hyp = {}
        for x in self.real:
            h = old.get(x.hypothesis_id) or self._hypothesis(x, priors[x.hypothesis_id])
            h.prior = h.posterior = priors[x.hypothesis_id]
            self.hyp[x.hypothesis_id] = h
        self.hyp[NULL_ID] = old[NULL_ID]
        self.hyp[NULL_ID].prior = self.hyp[NULL_ID].posterior = priors[NULL_ID]
        self.belief = dict(priors)
        await self.emit(E.HYPOTHESIS_PROPOSED, {"hypothesis": self.hyp[v.hypothesis_id]})
        await self.emit(DNA_VARIANT, v.dna_variant())
        await self.emit(
            E.HYPOTHESIS_MERGED,
            {"question_id": QUESTION_ID, "kept": "", "merged": [], "priors": priors},
        )
        seen: dict[str, Experiment] = {}
        for r in self.reads:
            if r.experiment_id not in seen:
                old_exp = self.experiments[r.experiment_id]
                exp = self._experiment(r.experiment_id, r.window, old_exp.design["replicate"])
                await self._publish(exp)
                seen[r.experiment_id] = exp
        for r in self.reads:
            exp = seen[r.experiment_id]
            r.bin_id = bin_for(exp.bins, {"bases": r.bases}) or OTHER
            prior = dict(self.belief)
            posterior = scoring.update(prior, exp.likelihood, r.bin_id)
            await self.emit(
                E.BELIEF_UPDATED,
                {
                    "question_id": QUESTION_ID,
                    "experiment_id": r.experiment_id,
                    "observed": r.bin_id,
                    "prior": prior,
                    "posterior": posterior,
                    "replay": True,
                },
            )
            self.belief = dict(posterior)
        self._sync()
        return None

    # --- settling

    def _best(self) -> tuple[str, float]:
        hid = max((v.hypothesis_id for v in self.real), key=lambda h: self.belief.get(h, 0.0))
        return hid, self.belief.get(hid, 0.0)

    def _facts(self, v: V.Variant, p: float) -> tuple[list[str], int, int]:
        w = 3 if any(r.window == 3 for r in self.reads) else self.reads[-1].window
        mine = [r.bases for r in self.reads if r.window == w]
        agree = sum(b == v.window(w) for b in mine)
        return mine, agree, w

    def _template(self, v: V.Variant, p: float) -> str:
        mine, agree, w = self._facts(v, p)
        reads = f"{agree} of {len(mine)} reads of window {w}"
        tail = f"posterior {p:.2f}. Demo strip on one gene, not a patient sample."
        if v.key == "hba":
            return f"Codon 6 reads GAG (Glu), the normal codon, in {reads}: HbA (normal); {tail}"
        if not v.known:
            return (
                f"Window {w} reads {v.window(w)} in {reads}, which no variant in our library "
                f"(HbS, HbC) predicts: an unknown variant, {v.legacy}; {tail}"
            )
        ids = ", ".join(x for x in (v.hgvs.split(":", 1)[-1].split(" ")[0], v.rsid) if x)
        return (
            f"Codon 6 reads {v.codon} ({v.amino_acid}) instead of GAG (Glu) in {reads}: "
            f"{v.name}, {ids}, {v.legacy}; {tail}"
        )

    def _prompt(self, v: V.Variant, p: float) -> str:
        mine, _, w = self._facts(v, p)
        return verdict_prompt(v, p, w, mine)

    async def _verdict(self, v: V.Variant, p: float) -> tuple[str, str, float | None]:
        """The Jetson model's sentence if it passes the fact check, else the template.

        The small model gets facts wrong about one time in four even with them in its prompt,
        so each sentence goes through ``check_verdict``; a rejected one gets one more try.
        """
        prompt = self._prompt(v, p)
        mine, _, w = self._facts(v, p)
        for attempt in range(1, max(1, self.cfg.verdict_attempts) + 1):
            try:
                out = await self.reader.verdict(
                    prompt, self.cfg.verdict_max_tokens, self.cfg.verdict_timeout_s
                )
                text = " ".join(str(out.get("text") or "").split()).strip("\"'")
                if out.get("ok") is False or not text:
                    raise ReaderError("the model returned no text")
                model = str(out.get("model") or "unknown")
                latency = out.get("latency_ms")
                latency = float(latency) if latency is not None else None
            except (ReaderError, TypeError, ValueError) as exc:
                await self.log("verdict", f"verdict from the template ({exc})")
                return self._template(v, p), "template", None
            problems = check_verdict(
                text, variant=v, posterior=p, window=w, reads=mine, prompt=prompt
            )
            if not problems:
                return text, f"jetson/{model}", latency
            await self.log(
                "verdict",
                f"model sentence {attempt} rejected by the fact check: {'; '.join(problems[:3])}",
            )
        await self.log("verdict", "verdict from the template (no model sentence passed the check)")
        return self._template(v, p), "template", None

    async def _settle(self, hid: str, p: float) -> StripResult:
        v = self._variant(hid)
        self.hyp[hid].status = "accepted"
        await self.emit(
            E.QUESTION_SETTLED,
            {"question_id": QUESTION_ID, "hypothesis_id": hid, "posterior": p},
        )
        await self._overlay(
            {
                "highlight": [0, 1, 2],
                "color": "green",
                "message": f"{v.short}: posterior {p:.2f}",
            },
            "verdict",
        )
        text, author, latency = await self._verdict(v, p)
        await self.emit(
            DNA_VERDICT,
            {
                "hypothesis_id": hid,
                "name": v.name,
                "posterior": p,
                "reads": len(self.reads),
                "text": text,
                "author": author,
                "latency_ms": latency,
                "pdb_id": v.pdb_id,
                "residue": RESIDUE,
                "legacy": v.legacy,
                "strip": self.strip,
            },
        )
        if hid in self.regenerated:
            saved = {x.key for x in V.load_lessons(self.lessons_path)}
            if v.key not in saved:
                total = V.append_lesson(
                    self.lessons_path,
                    v,
                    run_id=self.run_id,
                    strip=self.strip,
                    posterior=p,
                    reads=[r.bases for r in self.reads],
                    added_at=datetime.now(UTC).isoformat(),
                )
                await self.emit(DNA_LESSON, {"added": [hid], "lessons_total": total})
        return StripResult(self.strip, "settled", hid, p, list(self.reads))

    async def _unresolve(self, reason: str, outcome: str = "unresolved") -> StripResult:
        top = sorted(self.belief, key=lambda h: -self.belief[h])
        await self.emit(
            E.QUESTION_UNRESOLVED,
            {"question_id": QUESTION_ID, "reason": reason, "top": top},
        )
        await self._overlay({"highlight": [], "color": "yellow", "message": reason[:200]}, "q")
        return StripResult(self.strip, outcome, None, None, list(self.reads), reason)

    # --- one strip

    async def strip_loop(self) -> StripResult:
        self.reads = []
        pending: ReadRecord | None = None  # the surprising read a replicate must check
        surprises = 0
        while True:
            if len(self.reads) >= self.cfg.max_reads:
                hid, p = self._best()
                return await self._unresolve(
                    f"no hypothesis reached {self.cfg.settle} in {len(self.reads)} reads "
                    f"(best {hid} at {p:.2f})"
                )
            ranked = self.rank()
            for exp in ranked:
                await self._publish(exp)
            chosen = ranked[0]
            why = self._reason(chosen, ranked)
            rep = next((e for e in ranked if e.design["replicate"]), None)
            if pending is not None and rep is not None:
                chosen = rep
                why = (
                    f"re-read window {pending.window} to rule out a camera misread of "
                    f"{pending.bases} ({rep.eig_bits or 0.0:.2f} bits)"
                )
            elif (chosen.eig_bits or 0.0) <= EIG_ZERO and rep is not None:
                chosen = rep
                why = (
                    "every design is worth 0 bits between the listed variants; "
                    f"re-read window {rep.design['window']} to test 'none of these'"
                )
            self.experiments[chosen.id] = chosen
            await self.emit(
                E.EXPERIMENT_SCORED,
                {
                    "question_id": QUESTION_ID,
                    "chosen": chosen.id,
                    "ranking": [
                        {
                            "experiment_id": e.id,
                            "kind": e.kind,
                            "goal": e.goal,
                            "eig_bits": e.eig_bits,
                            "est_cost_s": e.est_cost_s,
                            "est_cost_usd": e.est_cost_usd,
                            "score": e.score,
                            "window": e.design["window"],
                            "replicate": e.design["replicate"],
                        }
                        for e in ranked
                    ],
                },
            )
            t0 = time.monotonic()
            r = await self._request(chosen, ranked, why)
            if r is None:
                w = chosen.design["window"]
                return await self._unresolve(
                    f"no stable read of window {w} within {self.cfg.timeout_s:.0f} s",
                    "timeout",
                )
            rec, surprise, pbar = await self._observe(chosen, r, time.monotonic() - t0)

            if pending is not None and rec.bin_id == OTHER and rec.bases == pending.bases:
                await self._surprise(
                    rec,
                    pbar,
                    f"the replicate read {rec.bases} again, so it is not a one-off camera "
                    "misread; looking for an explanation in the known-variant library",
                )
                why_not = await self._regenerate(rec)
                if why_not is not None:
                    return await self._unresolve(why_not)
                pending, surprises = None, 0
            elif surprise:
                surprises += 1
                if surprises > self.cfg.max_rereads:
                    return await self._unresolve(
                        f"{surprises} surprising reads of window {rec.window} in a row that "
                        "do not agree with each other"
                    )
                closest, mism = self._closest(rec.window, rec.bases)
                if rec.bin_id == OTHER:
                    reason = (
                        f"window {rec.window} read {rec.bases}, which no hypothesis predicts "
                        f"(closest: {closest}, {len(mism)} mismatch"
                        f"{'es' if len(mism) != 1 else ''} at position "
                        f"{', '.join(map(str, mism))}); predictive probability {pbar:.3f}"
                    )
                else:
                    reason = (
                        f"window {rec.window} read {rec.bases} with predictive probability "
                        f"{pbar:.3f}; 'none of these' now holds {self.belief[NULL_ID]:.2f}"
                    )
                await self._surprise(rec, pbar, reason)
                pending = rec
                continue
            elif pending is not None:
                mism = V.mismatches(pending.bases, rec.bases)
                await self.log(
                    rec.experiment_id,
                    f"the replicate read {rec.bases} ({rec.bin_id}); the earlier read "
                    f"{pending.bases} was a camera misread",
                )
                await self._overlay(
                    {
                        "highlight": mism,
                        "color": "yellow",
                        "message": f"Re-read agrees with {rec.bin_id}: earlier read was a camera "
                        "misread",
                    },
                    rec.experiment_id,
                )
                pending, surprises = None, 0
            hid, p = self._best()
            if p >= self.cfg.settle:
                return await self._settle(hid, p)

    # --- watching for a new sample

    async def watch(self, window: int, settled: str) -> str | None:
        """Read ``window`` until its letters change (confirmed ``swap_confirm`` times);
        returns the new letters, or ``None`` after ``watch_timeout_s``."""
        eid = f"watch-s{self.strip}"
        if self.shown != window:
            await self._show(window, eid)
        await self.log(
            eid,
            f"watching window {window} (now {settled}) for a new sample; swap it on the strip page",
        )
        t0 = time.monotonic()
        candidate, count, last_ts = None, 0, None
        while True:
            try:
                r = await self.reader.read()
            except ReaderError:
                r = {}
            bases, ts = r.get("bases"), r.get("ts")
            new_frame = ts is None or ts != last_ts
            if (
                r.get("window") == window
                and r.get("stable") is True
                and V.is_bases(bases)
                and new_frame
            ):
                last_ts = ts
                if bases == settled:
                    candidate, count = None, 0
                elif bases == candidate:
                    count += 1
                else:
                    candidate, count = bases, 1
                if candidate is not None and count >= self.cfg.swap_confirm:
                    await self.log(eid, f"window {window} changed: {settled} -> {candidate}")
                    return candidate
            if (
                self.cfg.watch_timeout_s is not None
                and time.monotonic() - t0 >= self.cfg.watch_timeout_s
            ):
                await self.log(eid, "stopped watching (watch timeout)")
                return None
            await asyncio.sleep(self.cfg.poll_s)

    async def reopen(self, window: int, old: str, new: str) -> None:
        self.strip += 1
        self.reads = []
        self.question = self.question.model_copy(update={"status": "open"})
        await self._overlay({"clear": True}, "reopen")
        await self.emit(
            E.QUESTION_OPENED,
            {
                "question": self.question,
                "reopened": True,
                "strip": self.strip,
                "reason": f"window {window} letters changed from {old} to {new}: a new sample",
            },
        )
        await self._reset_beliefs(announce=[*self.hyp])

    # --- the run

    async def start(self) -> None:
        self.real = [V.VARIANTS[k] for k in V.START_KEYS]
        for v in V.load_lessons(self.lessons_path):
            if v.key not in {x.key for x in self.real}:
                self.real.append(v)
                self.from_lessons.add(v.hypothesis_id)
        config = self.cfg.to_event()
        config["lessons_loaded"] = sorted(self.from_lessons)
        await self.emit(
            E.RUN_STARTED,
            {
                "lab": LAB,
                "goal": QUESTION_TEXT,
                "config": config,
                "started_at": datetime.now(UTC).isoformat(),
            },
        )
        self._started = True
        await self._overlay({"clear": True}, "start")
        await self.emit(E.QUESTION_OPENED, {"question": self.question, "strip": self.strip})
        ids = [v.hypothesis_id for v in self.real] + [NULL_ID]
        await self._reset_beliefs(announce=ids)

    async def _finish(self, status: str, t0: float) -> None:
        self._terminal = True
        lines = []
        for res in self.results:
            if res.outcome == "settled":
                lines.append(
                    f"- strip {res.strip}: settled on {res.hypothesis_id} "
                    f"(posterior {res.posterior:.3f}) after {len(res.reads)} reads"
                )
            else:
                lines.append(f"- strip {res.strip}: {res.outcome} ({res.reason})")
        methods = (
            "One Letter: categorical EIG over the strip windows (core.scoring), read error "
            f"{self.cfg.read_error} per base, settle at {self.cfg.settle}."
            + ("\n" + "\n".join(lines) if lines else "")
        )
        report = Report(
            run_id=self.run_id,
            artifacts=[],
            methods_md=methods,
            rejected=[],
            unresolved=[QUESTION_ID] if any(r.outcome != "settled" for r in self.results) else [],
            cost_usd=0.0,
            seconds=time.monotonic() - t0,
        )
        await self.emit(E.RUN_FINISHED, {"report": report, "status": status})

    async def run(self) -> list[StripResult]:
        t0 = time.monotonic()
        where = "start"
        try:
            await self.start()
            status = "ok"
            while True:
                where = f"strip {self.strip}"
                res = await self.strip_loop()
                self.results.append(res)
                if res.outcome == "timeout":
                    status = "stopped"
                    break
                if not self.cfg.watch and len(self.results) >= self.cfg.strips:
                    break
                if not self.reads:
                    break
                window, old = self.reads[-1].window, self.reads[-1].bases
                where = f"watch {self.strip}"
                new = await self.watch(window, old)
                if new is None:
                    break
                await self.reopen(window, old, new)
            await self._finish(status, t0)
            return self.results
        except asyncio.CancelledError:
            if self._started and not self._terminal:
                await self._finish("stopped", t0)
            raise
        except Exception as exc:
            if self._started and not self._terminal:
                self._terminal = True
                await self.emit(
                    E.RUN_ERROR,
                    {"message": f"{type(exc).__name__}: {exc}"[:1000], "where": where},
                )
            raise


async def run_oneletter(
    cfg: OneLetterConfig,
    *,
    run_id: str | None = None,
    runs_dir: str | Path = "runs",
    events_path: str | Path | None = None,
    reader: Reader | None = None,
    on_event: EventHook | None = None,
) -> list[StripResult]:
    """Run the engine; events go to ``<runs_dir>/<run_id>/events.jsonl`` or ``events_path``."""
    bus = EventBus(run_id or new_run_id(), Path(runs_dir))
    if events_path is not None:
        bus.path = Path(events_path)
    reader = reader or HttpReader(cfg.reader_url)
    try:
        return await OneLetterEngine(cfg, reader, bus, on_event=on_event).run()
    finally:
        await reader.aclose()


# --- the API launcher ----------------------------------------------------------------------

_OVERRIDES: dict[str, type] = {
    "reader_url": str,
    "timeout_s": float,
    "poll_s": float,
    "read_error": float,
    "settle": float,
    "null_prior": float,
    "max_reads": int,
    "strips": int,
    "watch": bool,
    "watch_timeout_s": float,
    "replicate_gap_s": float,
}
_FAKE_OVERRIDES = ("fake", "fake_show_delay_s", "fake_swap_delay_s")


def launch(overrides: dict, *, run_id: str, runs_dir: str | Path) -> Awaitable[Any]:
    """Validate a ``POST /runs`` request for ``oneletter`` and return the run's coroutine.

    ``overrides.reader_url`` defaults to ``$CRUX_ONELETTER_READER``. ``overrides.fake``
    (a list of samples, e.g. ``["hbs", "hbc"]``) runs against the in-process fake reader
    instead: the first sample is loaded, the others are the judge's swaps after each settle.
    Raises ``ValueError`` for a bad request.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(_OVERRIDES) - set(_FAKE_OVERRIDES)
    if unknown:
        raise ValueError(f"unknown oneletter overrides: {', '.join(sorted(unknown))}")
    kw: dict[str, Any] = {}
    for key, typ in _OVERRIDES.items():
        if key in overrides and overrides[key] is not None:
            value = overrides[key]
            if typ is bool and not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
            try:
                kw[key] = typ(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key}: {exc}") from exc
    kw["lessons_path"] = (Path(runs_dir) / "oneletter" / "lessons.jsonl").as_posix()
    reader: Reader | None = None
    fake = overrides.get("fake")
    if not fake:
        kw.setdefault("reader_url", os.environ.get(READER_ENV, ""))
        cfg = OneLetterConfig(**kw)
        return run_oneletter(cfg, run_id=run_id, runs_dir=runs_dir)

    from labs.oneletter.fake_reader import FakeReader, create_app

    if isinstance(fake, str):
        samples = [fake]
    elif isinstance(fake, list) and fake and all(isinstance(s, str) for s in fake):
        samples = list(fake)
    else:
        samples = ["hbs"]
    try:
        fr = FakeReader(
            samples[0],
            swaps=samples[1:],
            show_delay_s=float(overrides.get("fake_show_delay_s", 1.0)),
            swap_delay_s=float(overrides.get("fake_swap_delay_s", 3.0)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fake: {exc}") from exc
    kw.update(reader_url="fake://in-process", fake_reader=True)
    kw.setdefault("strips", len(samples))
    cfg = OneLetterConfig(**kw)
    reader = HttpReader("http://fake-reader", transport=httpx.ASGITransport(app=create_app(fr)))
    return run_oneletter(cfg, run_id=run_id, runs_dir=runs_dir, reader=reader)
