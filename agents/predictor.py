"""Neutral predictor: P(outcome bin | option) for each experiment.

Stand-in until DU-10's predictor lands; it keeps the ``Predictor.predict`` interface, so the
swap is one line in the wiring.

* Exact rows. A library diagnostic whose outcome an option fully determines carries
  ``design["exact"] = {answer_key: {bin_id: p}}``. Options with a matching ``answer_key``
  get that row as-is: no model call.
* Sampled rows. For every other option the predictor model is asked ``n`` times, blind to
  every other option, which bin the experiment will land in if that option is true. The
  counts become probabilities with ``core.scoring.likelihood_from_samples`` (add-half
  smoothing plus an epsilon floor, so no bin is ever impossible).
"""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from core import scoring
from core.schemas import Experiment, Hypothesis

_SYSTEM = """You predict the outcome of one pre-registered diagnostic, assuming one stated
option is true. You see the diagnostic's goal, its code and its outcome bins. Answer with
the id of the single bin the result will most likely land in, and one short reason.
Reason from what the code computes; do not assume the option is likely or unlikely."""


class BinChoice(BaseModel):
    bin_id: str = Field(min_length=1)
    reason: str = ""


def exact_row(exp: Experiment, hypothesis: Hypothesis) -> dict[str, float] | None:
    """The library's deterministic row for this option, renormalised to the exp's bins."""
    table = (exp.design or {}).get("exact") or {}
    row = table.get(hypothesis.answer_key or "")
    if not isinstance(row, dict):
        return None
    bins = [b.id for b in exp.bins]
    values = {b: float(row.get(b, 0.0)) for b in bins}
    total = sum(values.values())
    if total <= 0 or any(v < 0 for v in values.values()):
        return None
    return {b: v / total for b, v in values.items()}


class SampledPredictor:
    def __init__(self, llm, *, n: int = 8, eps: float = 0.1) -> None:
        self.llm = llm
        self.n = n
        self.eps = eps

    async def predict(
        self, exp: Experiment, hypotheses: list[Hypothesis], context: str
    ) -> dict[str, dict[str, float]]:
        bins = [b.id for b in exp.bins]
        table: dict[str, dict[str, float]] = {}
        sampled: list[Hypothesis] = []
        for h in hypotheses:
            row = exact_row(exp, h)
            if row is not None:
                table[h.id] = row
            else:
                sampled.append(h)
        rows = await asyncio.gather(*(self._sample(exp, h, context) for h in sampled))
        for h, counts in zip(sampled, rows, strict=True):
            table[h.id] = scoring.likelihood_from_samples(
                {b: counts.get(b, 0) for b in bins}, len(bins), self.eps
            )
        return table

    async def _sample(
        self, exp: Experiment, h: Hypothesis, context: str
    ) -> dict[str, int]:
        spec = {
            "goal": exp.goal,
            "language": exp.language,
            "code": exp.code,
            "query": exp.query,
            "bins": [
                {"id": b.id, "description": b.description, "rule": b.rule}
                for b in exp.bins
            ],
        }
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "assume_true": h.statement,
                        "diagnostic": spec,
                        "context": context[:4000],
                    }
                ),
            },
        ]
        choices = await self.llm.structured("predictor", BinChoice, messages, n=self.n)
        valid = {b.id for b in exp.bins}
        counts: dict[str, int] = {}
        for choice in choices:
            bin_id = BinChoice.model_validate(choice).bin_id
            if bin_id in valid:
                counts[bin_id] = counts.get(bin_id, 0) + 1
        return counts
