"""Search and memory ablation flags; remaining run configuration is owned by DU-2."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    search: Literal["on", "off"] = "on"
    memory: Literal["pastdev", "local_role_only"] = "pastdev"
    lessons: bool = False
    # Auto search, <=10 results, highlights, no summaries. Override as pricing changes.
    exa_estimated_cost_usd: float = Field(default=0.007, ge=0, allow_inf_nan=False)
