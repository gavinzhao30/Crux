"""Evidence contract from docs/ERD.md; remaining shared schemas belong to DU-3."""

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    id: str
    kind: Literal["profile", "observation", "search", "objection", "lesson"]
    claim: str
    author_role: str
    source_ref: str | None = None
    url: str | None = None
    conditions: dict = Field(default_factory=dict)
