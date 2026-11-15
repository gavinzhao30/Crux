"""Reviewed development lessons, frozen before reuse on declared held-out cases.

The corpus is separate from task-run memory. It contains procedural advice and source
IDs, never evaluation grades. A review is supplied by the caller, not invented here.
"""

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from core.config import RunConfig
from core.schemas import Evidence
from memory.base import MemoryRecord, RunContext
from memory.instrumentation import format_memory
from memory.pastdev import PastMemory

LESSON_ROLES = ("reporter", "proposer", "statistician", "bioinformatician", "sceptic", "designer")
READERS = set(LESSON_ROLES) - {"reporter"}
BLOCKED_CONTENT = re.compile(
    r"bixbench3-ground-truth|ground[-_ ]truth|answer[-_ ]key|reference artifact|"
    r"grade\.json|official score",
    re.IGNORECASE,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Applicability(FrozenModel):
    key: str = Field(min_length=1)
    value: StrictStr | StrictBool | StrictInt | StrictFloat

    def matches(self, context: dict) -> bool:
        actual = context.get(self.key)
        return type(actual) is type(self.value) and actual == self.value


class DevelopmentEvidence(FrozenModel):
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    split: Literal["dev"]
    evidence_id: str = Field(min_length=1)
    kind: Literal["observation", "objection", "diagnostic", "execution_failure"]
    input_hash: str = Field(min_length=1)
    code_hash: str | None = None


class ReviewedLesson(FrozenModel):
    id: str = Field(min_length=1)
    when: str = Field(min_length=1)
    mistake: str = Field(min_length=1)
    diagnostic: str = Field(min_length=1)
    repair: str = Field(min_length=1)
    exceptions: tuple[str, ...]
    sources: tuple[DevelopmentEvidence, ...] = Field(min_length=1)
    requires: tuple[Applicability, ...] = ()
    excludes: tuple[Applicability, ...] = ()
    approved: Literal[True]
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime

    @field_validator("reviewed_at")
    @classmethod
    def aware_time(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Review times need an explicit timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def procedural_only(self):
        if any(
            not value.strip() for value in (self.when, self.mistake, self.diagnostic, self.repair)
        ):
            raise ValueError("Procedural lesson fields must not be blank")
        text = "\n".join([self.when, self.mistake, self.diagnostic, self.repair, *self.exceptions])
        if BLOCKED_CONTENT.search(text):
            raise ValueError("Lessons cannot contain grading or reference-answer material")
        if not self.reviewed_by.strip() or not self.id.strip():
            raise ValueError("Lesson and reviewer identifiers must not be blank")
        if len(text) > 8000:
            raise ValueError("Store compact procedures, not analysis data")
        return self

    def applies(self, context: dict) -> bool:
        return all(c.matches(context) for c in self.requires) and not any(
            c.matches(context) for c in self.excludes
        )


class FrozenLessons(FrozenModel):
    version: Literal[1] = 1
    synthetic: bool = False
    frozen_at: datetime
    held_out_case_ids: tuple[str, ...] = Field(min_length=1)
    lessons: tuple[ReviewedLesson, ...]

    @field_validator("frozen_at")
    @classmethod
    def aware_time(cls, value):
        return ReviewedLesson.aware_time(value)

    @model_validator(mode="after")
    def validate_freeze(self):
        if len(set(self.held_out_case_ids)) != len(self.held_out_case_ids):
            raise ValueError("Held-out case IDs must be unique")
        if any(not case.strip() for case in self.held_out_case_ids):
            raise ValueError("Held-out case IDs must not be blank")
        ids = [lesson.id for lesson in self.lessons]
        if len(set(ids)) != len(ids):
            raise ValueError("Lesson IDs must be unique")
        for lesson in self.lessons:
            if lesson.reviewed_at > self.frozen_at:
                raise ValueError("Lessons must be reviewed before the freeze")
            if any(s.case_id in self.held_out_case_ids for s in lesson.sources):
                raise ValueError("Held-out cases cannot contribute lessons")
        return self

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def save(self, directory: Path) -> Path:
        """Content-addressed, create-only manifest; an existing file is never overwritten."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"lessons-{self.digest}.json"
        body = (
            json.dumps(
                {"sha256": self.digest, "manifest": self.model_dump(mode="json")},
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(body)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != body:
                raise ValueError("An existing frozen manifest has changed") from None
        return path

    @classmethod
    def load(cls, path: Path):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        if set(wrapper) != {"sha256", "manifest"}:
            raise ValueError("Expected a frozen lesson manifest and its digest")
        corpus = cls.model_validate(wrapper["manifest"])
        if corpus.digest != wrapper["sha256"]:
            raise ValueError("Frozen lesson digest mismatch")
        return corpus

    def evidence(self, lesson: ReviewedLesson) -> Evidence:
        return Evidence(
            id="lesson-" + self.digest[:16] + ":" + lesson.id,
            kind="lesson",
            author_role="reporter",
            claim="\n".join(
                [
                    f"When: {lesson.when}",
                    f"Mistake: {lesson.mistake}",
                    f"Diagnostic: {lesson.diagnostic}",
                    f"Repair: {lesson.repair}",
                    "Exceptions: " + ("; ".join(lesson.exceptions) or "None declared by reviewer"),
                ]
            ),
            source_ref=lesson.sources[0].evidence_id,
            conditions={
                "corpus_sha256": self.digest,
                "lesson_id": lesson.id,
                "reviewed_by": lesson.reviewed_by,
                "reviewed_at": lesson.reviewed_at.isoformat(),
                "development_sources": [s.model_dump() for s in lesson.sources],
            },
        )


class LessonMemory:
    """A dedicated frozen corpus, read independently of current task-run evidence.

    Use an async context manager. prepare() is an explicit indexing barrier. Off means
    no provider is constructed, no records are uploaded and no recall occurs.
    """

    def __init__(
        self,
        corpus: FrozenLessons,
        config: RunConfig,
        root: Path = Path("runs/lessons"),
        *,
        allow_synthetic: bool = False,
        memory_factory=PastMemory,
        **provider_options,
    ):
        if config.lessons and corpus.synthetic and not allow_synthetic:
            raise ValueError("Synthetic corpora require an explicit allow_synthetic opt-in")
        self.corpus = corpus
        self.enabled = config.lessons
        self.context = RunContext(
            run_id="lessons-" + corpus.digest,
            task_id="reviewed-procedural-lessons",
            config_id="frozen-corpus-v1",
            roles=LESSON_ROLES,
        )
        self.root = root
        self.memory_factory = memory_factory
        self.backend_label = (
            "pastdev_lessons" if memory_factory is PastMemory else "fixture_lessons"
        )
        self.provider_options = provider_options
        self.memory = None
        self.ready = False
        self.closed = False
        self.events: list[dict] = []
        self._by_id = {corpus.evidence(lesson).id: lesson for lesson in corpus.lessons}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True
        if self.memory:
            await self.memory.aclose()

    async def prepare(self) -> None:
        if self.closed:
            raise RuntimeError("Lesson memory is closed")
        if not self.enabled or self.ready:
            return
        if self.memory is None:
            self.memory = self.memory_factory(self.context, self.root, **self.provider_options)
        for lesson in self.corpus.lessons:
            evidence = self.corpus.evidence(lesson)
            await self.memory.write_record(
                MemoryRecord(
                    event_id=evidence.id,
                    run_id=self.context.run_id,
                    task_id=self.context.task_id,
                    evidence=evidence,
                    event_type="lesson",
                    occurred_at=lesson.reviewed_at,
                    evidence_ids=[source.evidence_id for source in lesson.sources],
                    current_status="supported",
                )
            )
            if hasattr(self.memory, "store"):
                self.memory.store.audit(
                    "lesson.written",
                    {
                        "lesson": evidence.model_dump(),
                        "corpus_sha256": self.corpus.digest,
                    },
                )
        await self.memory.flush()
        self.ready = True

    async def recall(
        self,
        query: str,
        *,
        case_id: str,
        role: str,
        context: dict,
        k: int = 3,
    ) -> list[Evidence]:
        if self.closed:
            raise RuntimeError("Lesson memory is closed")
        if case_id not in self.corpus.held_out_case_ids:
            raise ValueError("Case is outside the frozen comparison cohort")
        if role not in READERS or not 1 <= k <= 200:
            raise ValueError("Lesson recall is for proposers/designers with k between 1 and 200")
        if not self.enabled:
            self.events.append(
                {
                    "case_id": case_id,
                    "role": role,
                    "enabled": False,
                    "evidence_ids": [],
                    "corpus_sha256": self.corpus.digest,
                }
            )
            return []
        if not self.ready:
            raise RuntimeError("Call prepare() and wait for indexing before a comparison")
        started = time.perf_counter()
        recalled = await self.memory.recall(query, self.context.run_id, role, k=200)
        selected = {}
        for evidence in recalled:
            lesson = self._by_id.get(evidence.id)
            if lesson is not None and lesson.applies(context):
                selected.setdefault(evidence.id, self.corpus.evidence(lesson))
                if len(selected) == k:
                    break
        prompt = format_memory(list(selected.values()))
        payload = {
            "case_id": case_id,
            "role": role,
            "reader_role": role,
            "enabled": True,
            "evidence_ids": list(selected),
            "authors": [ev.author_role for ev in selected.values()],
            "query": query,
            "backend": self.backend_label,
            "corpus_sha256": self.corpus.digest,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "prompt_tokens_estimate": (len(prompt.encode("utf-8")) + 3) // 4,
            "token_method": "estimate_utf8_bytes_div4",
        }
        self.events.append(payload)
        if hasattr(self.memory, "store"):
            self.memory.store.audit("memory.recalled", payload)
        return list(selected.values())
