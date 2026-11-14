"""Run budget: money, model calls and wall time.

``Budget.on_usage`` accepts anything shaped like ``agents.llm.Usage`` (attributes or a dict
with ``input_tokens``, ``output_tokens`` and ``cost_usd``). A call whose ``cost_usd`` is
``None`` (a model with no price) is counted as a call but adds no cost. ``check`` raises
``BudgetExhausted`` once any cap is reached; the orchestrator then ends the run cleanly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

__all__ = ["Budget", "BudgetExhausted"]


class BudgetExhausted(RuntimeError):
    """A budget cap was reached. ``reason`` names the cap."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class Budget:
    """Tracks spend against ``max_usd``, ``max_llm_calls`` and ``max_seconds``.

    ``clock`` returns seconds (any origin); elapsed time is measured from construction.
    """

    def __init__(
        self,
        max_usd: float,
        max_llm_calls: int,
        max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_usd = float(max_usd)
        self.max_llm_calls = int(max_llm_calls)
        self.max_seconds = float(max_seconds)
        self._clock = clock
        self._start = clock()
        self.cost_usd = 0.0
        self.llm_calls = 0
        self.unpriced_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.experiments = 0
        self.experiment_seconds = 0.0

    @property
    def seconds(self) -> float:
        """Wall time since the budget started."""
        return max(self._clock() - self._start, 0.0)

    def on_usage(self, usage: Any) -> None:
        """Count one model call (an ``agents.llm.Usage`` or a dict of the same fields)."""
        self.llm_calls += 1
        self.tokens_in += int(_field(usage, "input_tokens") or 0)
        self.tokens_out += int(_field(usage, "output_tokens") or 0)
        cost = _field(usage, "cost_usd")
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += float(cost)

    def on_experiment(self, seconds: float, cost_usd: float) -> None:
        """Count one finished experiment and its compute cost."""
        self.experiments += 1
        self.experiment_seconds += max(float(seconds or 0.0), 0.0)
        self.cost_usd += max(float(cost_usd or 0.0), 0.0)

    def exhausted_reason(self) -> str | None:
        """Which cap is reached, or ``None``."""
        if self.cost_usd >= self.max_usd:
            return f"cost ${self.cost_usd:.4f} reached the ${self.max_usd:.2f} cap"
        if self.llm_calls >= self.max_llm_calls:
            return f"{self.llm_calls} model calls reached the cap of {self.max_llm_calls}"
        elapsed = self.seconds
        if elapsed >= self.max_seconds:
            return f"{elapsed:.0f} s reached the {self.max_seconds:.0f} s cap"
        return None

    @property
    def exhausted(self) -> bool:
        return self.exhausted_reason() is not None

    def check(self) -> None:
        """Raise ``BudgetExhausted`` if any cap is reached."""
        reason = self.exhausted_reason()
        if reason is not None:
            raise BudgetExhausted(reason)

    def snapshot(self) -> dict[str, float | int]:
        """The ``budget.tick`` payload."""
        return {
            "cost_usd": round(self.cost_usd, 10),
            "llm_calls": self.llm_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "seconds": round(self.seconds, 6),
        }
