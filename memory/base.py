"""Application-owned provenance envelope; memory never updates scientific beliefs."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.schemas import Evidence

DEFAULT_ROLES = (
    "experimenter",
    "critic",
    "designer",
    "reporter",
    "referee",
    "predictor",
    "statistician",
    "bioinformatician",
    "sceptic",
)


class RunContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    config_id: str = "default"
    repeat: int = Field(default=0, ge=0)
    roles: tuple[str, ...] = DEFAULT_ROLES

    @field_validator("roles")
    @classmethod
    def valid_roles(cls, roles: tuple[str, ...]) -> tuple[str, ...]:
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("roles must be nonempty and unique")
        if any(not r or len(r) > 60 or not r.isascii() or not r.isprintable() for r in roles):
            raise ValueError("roles must be short printable ASCII identifiers")
        return roles

    @property
    def audience(self) -> str:
        scope = json.dumps([self.run_id, self.task_id, self.config_id, self.repeat])
        return "crux-" + hashlib.sha256(scope.encode()).hexdigest()[:32]

    def identity(self, role: str) -> str:
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        return f"{self.audience}:{role}"


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    run_id: str
    task_id: str
    evidence: Evidence
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step_id: str | None = None
    event_type: str = "observation"
    claim_id: str | None = None
    parent_event_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    input_hash: str | None = None
    code_hash: str | None = None
    environment_id: str | None = None
    parameters: dict = Field(default_factory=dict)
    outcome: dict = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    current_status: Literal[
        "proposed", "supported", "contradicted", "inconclusive", "invalid_analysis"
    ] = "proposed"
    supersedes: list[str] = Field(default_factory=list)
    mandatory: bool = False

    @field_validator("occurred_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must have a timezone")
        return value.astimezone(UTC)


class RecallUnavailable(RuntimeError):
    """Remote recall failed; distinct from a successful recall with no matches."""


class IngestionFailed(RuntimeError):
    """The provider rejected or failed an ingestion; durable work remains queued."""


class ContentRejected(IngestionFailed):
    """A source needs a corrected new event; retrying the same body cannot repair it."""
