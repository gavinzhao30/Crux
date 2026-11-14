"""Crux v2 run settings shared by the experiment arms."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.llm import LLMConfig


class RunConfig(BaseModel):
    """Configure experiment selection, agent roles, and evidence sharing."""

    model_config = ConfigDict(extra="forbid")

    selection: Literal["eig", "random", "model_pick"] = "eig"
    proposers: Literal[3, 1] = 3
    critic: bool = True
    predictor: Literal["sampled", "exact_only", "jev", "stated"] = "sampled"
    memory: Literal["pastdev", "local_role_only", "vector"] = "pastdev"
    search: bool = True
    lessons: bool = False
    null_hypothesis: bool = True
    llm: LLMConfig = Field(default_factory=LLMConfig)
