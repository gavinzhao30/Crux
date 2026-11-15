"""Learned predictor: P(outcome bin | option) from a LoRA fine-tune served by Tinker (DU-29).

Same ``Predictor.predict`` interface as ``agents.predictor.SampledPredictor``:

* Library rows win, as in ``SampledPredictor``. Options whose ``answer_key`` has a row in
  ``design["exact"]`` or ``design["stated"]`` get it floored by ``LIBRARY_FLOOR``
  (``agents.predictor.library_row``), with no model call. ``use_stated_rows=False`` keeps
  only the exact rows and lets the model replace the measured stated rows (the setting
  ``eval/eval_predictor.py`` measures; see its report before turning it on).
* Scored rows. For every other option, each bin is scored by the model's log-probability of
  the bin's completion (``bin_completion``) after the prompt (``build_prompt``): one
  ``compute_logprobs`` prefill per bin, all of them concurrent. The bin log-probabilities are
  softmaxed over the experiment's bins and floored like ``core.scoring.likelihood_from_samples``:
  ``(1 - eps) * p + eps / n_bins``.
* Fallback. If Tinker errors, the injected ``fallback`` predictor fills the remaining rows
  (and the probe experiment's ``likelihood_source`` is set to ``"sampled"``); with no
  fallback the error propagates.

``build_prompt`` and ``bin_completion`` are the only prompt code: the training script
(``eval/train_tinker_predictor.py``) and the evaluation use them too, so training and
inference see identical text. The prompt holds the diagnostic's goal, its bins (id and
description, not code), the option assumed true and a compact summary of the lab's data
profile (``profile_summary``: the parsed result of ``labs.bixbench3.task9._PROFILE``).
The orchestrator's ``context`` (goal and question text) is not used: the option statement
carries the question, and the profile reaches the predictor through the injected
``profile`` callable, since ``Predictor.predict`` has no channel for data.

Training labels exist only for the option that was true in each synthetic case, so a row for
an option the data contradict is an extrapolation; see ``eval/results/predictor_calibration.md``.

The text uses the Qwen3 chat format with thinking disabled, so the base model can be scored
zero-shot with the same prompt.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import statistics
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agents.predictor import LIBRARY_FLOOR, exact_row, library_row
from core.schemas import Experiment, Hypothesis

logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-8B"
END = "<|im_end|>"
_SYSTEM = (
    "You predict which outcome bin a pre-registered diagnostic will land in, assuming one "
    "stated option is true. Use the data profile. Answer with the bin id only."
)


# --- prompt ---------------------------------------------------------------------------


def _short_count(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.2g}M"
    if value >= 1e3:
        return f"{value / 1e3:.2g}k"
    return f"{value:.0f}"


def _latest_profile(profile: Sequence[Mapping] | Mapping | None) -> Mapping | None:
    if isinstance(profile, Mapping):
        return profile
    for facts in reversed(list(profile or [])):
        if isinstance(facts, Mapping) and "shape" in facts:
            return facts
    return None


def profile_summary(profile: Sequence[Mapping] | Mapping | None) -> str:
    """One compact line from the lab's parsed profile (``SwarmTask9Lab.profile_facts``).

    Samples per group, library-size spread, the versioned gene ID fraction and the count
    column layout. Missing or malformed fields are left out, never guessed.
    """
    facts = _latest_profile(profile)
    if facts is None:
        return "unavailable"
    parts: list[str] = []
    shape = facts.get("shape") or []
    n_genes = int(shape[0]) if len(shape) >= 1 else 0
    if len(shape) >= 2:
        parts.append(f"{n_genes} genes x {int(shape[1])} libraries")
    samples = [s for s in facts.get("samples") or [] if isinstance(s, Mapping)]
    if samples:
        groups = Counter(
            (str(s.get("cell_line", "")), str(s.get("condition", ""))) for s in samples
        )
        lines: dict[str, int] = {}
        for line, _ in groups:
            lines[line] = lines.get(line, 0) + 1
        parts.append(
            "groups per cell line: "
            + ", ".join(f"{k} {v}" for k, v in sorted(lines.items()))
        )
        sizes = sorted(groups.values())
        reps = f"{sizes[0]}" if sizes[0] == sizes[-1] else f"{sizes[0]}-{sizes[-1]}"
        parts.append(f"libraries per group: {reps}")
        has_ctrl = all(
            any(c == "Ctrl" for (line2, c) in groups if line2 == line) for line in lines
        )
        if not has_ctrl:
            parts.append("a cell line has no Ctrl group")
        layout = "count_column" if "count_column" in samples[0] else "sample_id"
        parts.append(f"count columns named by {layout}")
    sizes_map = facts.get("library_sizes") or {}
    values = [float(v) for v in sizes_map.values() if isinstance(v, (int, float))]
    if values and max(values) > 0:
        median = statistics.median(values)
        if median > 0:
            ordered = sorted(values)
            parts.append(
                f"library size median {_short_count(median)}, "
                f"min/median {ordered[0] / median:.2f}, "
                f"second-smallest/median {ordered[min(1, len(ordered) - 1)] / median:.2f}, "
                f"max/median {ordered[-1] / median:.2f}"
            )
    versioned = facts.get("versioned_gene_ids")
    if n_genes > 0 and isinstance(versioned, (int, float)):
        parts.append(f"versioned gene IDs {100.0 * versioned / n_genes:.1f}%")
    return "; ".join(parts) if parts else "unavailable"


def build_prompt(
    exp: Experiment, hypothesis: Hypothesis, profile: Sequence[Mapping] | Mapping | None
) -> str:
    """The full prompt text (chat-formatted) for one experiment, option and data profile."""
    bins = "\n".join(f"- {b.id}: {b.description}" for b in exp.bins)
    user = (
        f"Diagnostic: {exp.goal}\n"
        f"Outcome bins:\n{bins}\n"
        f"Assume true: {hypothesis.statement}\n"
        f"Data profile: {profile_summary(profile)}\n"
        "Which bin will the result land in?"
    )
    return (
        f"<|im_start|>system\n{_SYSTEM}{END}\n"
        f"<|im_start|>user\n{user}{END}\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def bin_completion(bin_id: str) -> str:
    """The completion scored (and trained) for one bin; the end token makes ids prefix-safe."""
    return f"{bin_id}{END}"


# --- scoring math ---------------------------------------------------------------------


def logprobs_to_row(
    logprobs: Mapping[str, float], eps: float = 0.1
) -> dict[str, float]:
    """Softmax of per-bin sequence log-probabilities, then the predictor's epsilon floor."""
    if not logprobs:
        raise ValueError("no bins to score")
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    finite = {b: float(v) for b, v in logprobs.items() if math.isfinite(float(v))}
    n = len(logprobs)
    if not finite:
        return {b: 1.0 / n for b in logprobs}
    top = max(finite.values())
    weights = {b: math.exp(finite[b] - top) if b in finite else 0.0 for b in logprobs}
    total = sum(weights.values())
    return {b: (1.0 - eps) * w / total + eps / n for b, w in weights.items()}


def completion_logprob(logprobs: Sequence[float | None], n_completion: int) -> float:
    """Sum of the last ``n_completion`` token log-probabilities (the completion's)."""
    tail = list(logprobs)[-n_completion:]
    if len(tail) != n_completion or any(v is None for v in tail):
        raise ValueError("compute_logprobs returned no score for a completion token")
    return float(sum(tail))  # type: ignore[arg-type]


class _Tokens:
    """Stand-in for ``tinker.ModelInput`` when the SDK is not installed (tests)."""

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = list(tokens)

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


def model_input(tokens: list[int]) -> Any:
    try:
        from tinker import types
    except ImportError:  # pragma: no cover - exercised only without the SDK
        return _Tokens(tokens)
    return types.ModelInput.from_ints(list(tokens))


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    if hasattr(value, "result") and callable(value.result):
        return await asyncio.to_thread(value.result)
    return value


# --- predictor ------------------------------------------------------------------------


class TinkerPredictor:
    """``Predictor`` backed by a Tinker sampling client (a fine-tuned or base model).

    ``sampling_client_or_path`` is a ``tinker.SamplingClient`` (or anything with
    ``compute_logprobs`` and ``get_tokenizer``) or a ``tinker://`` path, opened lazily on the
    first prediction. ``profile`` returns the lab's parsed profile facts at prediction time.
    ``use_stated_rows`` keeps the library's stated rows (default, like ``SampledPredictor``);
    exact rows always win. ``tokenizer`` overrides the client's (tests).
    """

    def __init__(
        self,
        sampling_client_or_path: Any,
        *,
        profile: Callable[[], Sequence[Mapping]] | None = None,
        eps: float = 0.1,
        fallback: Any = None,
        tokenizer: Any = None,
        base_model: str | None = None,
        max_concurrency: int = 32,
        use_stated_rows: bool = True,
    ) -> None:
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must be in [0, 1]")
        self._target = sampling_client_or_path
        self._client: Any = (
            None
            if isinstance(sampling_client_or_path, str)
            else sampling_client_or_path
        )
        self._lock = threading.Lock()
        self.profile = profile
        self.eps = eps
        self.fallback = fallback
        self._tokenizer = tokenizer
        self.base_model = base_model
        self.max_concurrency = max_concurrency
        self.use_stated_rows = use_stated_rows
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self.calls = 0
        self.prompt_tokens = 0

    # -- clients -----------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        with self._lock:
            return self._open_client()

    def _open_client(self) -> Any:
        if self._client is None:
            import tinker

            service = tinker.ServiceClient()
            path = str(self._target)
            if path.startswith("tinker://"):
                self._client = service.create_sampling_client(model_path=path).result()
            else:
                self._client = service.create_sampling_client(base_model=path).result()
        return self._client

    def _ensure_tokenizer(self) -> Any:
        with self._lock:
            if self._tokenizer is None:
                self._tokenizer = self._open_client().get_tokenizer()
            return self._tokenizer

    async def ready(self) -> None:
        """Open the client and tokenizer (off the event loop) if they are not open yet."""
        if self._client is None or self._tokenizer is None:
            await asyncio.to_thread(self._ensure_tokenizer)

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrency)
            self._sem_loop = loop
        return self._sem

    def encode(self, text: str) -> list[int]:
        return list(self._ensure_tokenizer().encode(text, add_special_tokens=False))

    # -- Predictor ---------------------------------------------------------------------

    def library_row(self, exp: Experiment, h: Hypothesis) -> dict[str, float] | None:
        """The floored library row this predictor uses for ``h``, or None to score it."""
        if self.use_stated_rows:
            return library_row(exp, h)
        row = exact_row(exp, h)
        if row is None:
            return None
        return {b: (1 - LIBRARY_FLOOR) * p + LIBRARY_FLOOR / len(row) for b, p in row.items()}

    async def predict(
        self, exp: Experiment, hypotheses: list[Hypothesis], context: str
    ) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        pending: list[Hypothesis] = []
        for h in hypotheses:
            row = self.library_row(exp, h)
            if row is not None:
                table[h.id] = row
            else:
                pending.append(h)
        if not pending:
            return table
        try:
            facts = list(self.profile() or []) if self.profile is not None else []
            scored = await asyncio.gather(*(self.score(exp, h, facts) for h in pending))
        except Exception as exc:
            if self.fallback is None:
                raise
            logger.warning(
                "Tinker predictor failed (%s); using the fallback predictor", exc
            )
            rows = await self.fallback.predict(exp, pending, context)
            exp.likelihood_source = "sampled"
            table.update(rows or {})
            return table
        for h, logprobs in zip(pending, scored, strict=True):
            table[h.id] = logprobs_to_row(logprobs, self.eps)
        return table

    async def score(
        self, exp: Experiment, hypothesis: Hypothesis, profile: Sequence[Mapping] | None
    ) -> dict[str, float]:
        """Per-bin sequence log-probabilities (unnormalised) for one option."""
        await self.ready()
        client = self._client
        prompt = self.encode(build_prompt(exp, hypothesis, profile))

        async def one(bin_id: str) -> float:
            completion = self.encode(bin_completion(bin_id))
            async with self._semaphore():
                result = await _resolve(
                    client.compute_logprobs(
                        model_input(prompt + completion),
                        prompt_logprobs_last_n=len(completion),
                    )
                )
            self.calls += 1
            self.prompt_tokens += len(prompt) + len(completion)
            return completion_logprob(result, len(completion))

        values = await asyncio.gather(*(one(b.id) for b in exp.bins))
        return {b.id: v for b, v in zip(exp.bins, values, strict=True)}


def load_model_path(path: str | None = None) -> str | None:
    """The ``tinker://`` sampler path: ``path``, else env ``CRUX_TINKER_PREDICTOR``."""
    import os

    value = path if path is not None else os.environ.get("CRUX_TINKER_PREDICTOR")
    return value.strip() if value and value.strip() else None
