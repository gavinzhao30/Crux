"""Live capability check for the configured OpenAI key.

Run from the repository root:

    uv run --python 3.12 --no-project --with "openai>=1.60" --with "pydantic>=2" \
        python scripts/check_models.py [--model MODEL] [--structured-model MODEL]

It does three things and prints a short report (never the key itself):

1. Lists the model IDs visible to the key (sorted, IDs only).
2. Sends ONE Chat Completions request for a single-letter answer (A/B/C) with
   ``logprobs=True, top_logprobs=5, max_tokens=1`` and no JSON schema, and reports whether
   logprobs came back non-empty.
3. Sends ONE structured-output request with a tiny pydantic schema through whichever parse
   helper the installed SDK exposes (``responses.parse`` first, then
   ``chat.completions.parse``, then ``beta.chat.completions.parse``) and reports success and
   the token usage fields.

Rejected parameters (some reasoning models refuse ``temperature``, ``logprobs`` or
``max_tokens``) are reported, not raised. Total cost is a fraction of a cent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Literal

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

# Preference order for the cheap probe model; the first one visible to the key is used.
CHEAP_CANDIDATES = (
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-5-nano",
    "gpt-5-mini",
)

QUESTION = (
    "Which planet is known as the Red Planet? A) Venus B) Mars C) Jupiter. "
    "Answer with a single letter only."
)


class Answer(BaseModel):
    letter: Literal["A", "B", "C"]
    confidence: float


def pick_model(ids: list[str], requested: str | None) -> str | None:
    if requested:
        return requested
    for name in CHEAP_CANDIDATES:
        if name in ids:
            return name
    return None


async def list_models(client: AsyncOpenAI) -> list[str]:
    ids = [m.id async for m in client.models.list()]
    return sorted(ids)


async def check_logprobs(client: AsyncOpenAI, model: str) -> None:
    print(f"\n[logprobs] model={model}")
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": QUESTION}],
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": 1,
        "temperature": 0,
    }
    for _ in range(3):
        try:
            resp = await client.chat.completions.create(**kwargs)
            break
        except openai.BadRequestError as exc:
            param = getattr(exc, "param", None) or ""
            msg = str(exc)
            print(f"  rejected: param={param!r} message={msg[:200]!r}")
            if "temperature" in (param + msg) and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            if "max_tokens" in (param + msg) and "max_tokens" in kwargs:
                kwargs.pop("max_tokens")
                kwargs["max_completion_tokens"] = 1
                continue
            print("  logprobs: NOT supported (request rejected)")
            return
    else:
        print("  logprobs: could not complete request")
        return

    choice = resp.choices[0]
    content = choice.logprobs.content if choice.logprobs else None
    print(f"  answer text: {choice.message.content!r}")
    if content:
        top = [(t.token, round(t.logprob, 3)) for t in content[0].top_logprobs]
        print(f"  logprobs: NON-EMPTY, top_logprobs={top}")
    else:
        print("  logprobs: EMPTY")
    if resp.usage:
        print(
            f"  usage: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}"
        )


async def check_structured(client: AsyncOpenAI, model: str) -> None:
    print(f"\n[structured] model={model}")
    messages = [{"role": "user", "content": QUESTION + " Also give your confidence in [0, 1]."}]

    responses = getattr(client, "responses", None)
    if responses is not None and hasattr(responses, "parse"):
        path = "client.responses.parse(text_format=...)"
        kwargs: dict = {"model": model, "input": messages, "text_format": Answer, "temperature": 0}
        try:
            resp = await responses.parse(**kwargs)
        except openai.BadRequestError as exc:
            print(f"  rejected with temperature: {str(exc)[:200]!r}; retrying without it")
            kwargs.pop("temperature")
            resp = await responses.parse(**kwargs)
        print(f"  path: {path}")
        print(f"  parsed: {resp.output_parsed!r}")
        u = resp.usage
        if u is not None:
            print(
                f"  usage: input_tokens={u.input_tokens} output_tokens={u.output_tokens} "
                f"total_tokens={u.total_tokens} "
                f"cached_input={getattr(u.input_tokens_details, 'cached_tokens', None)} "
                f"reasoning={getattr(u.output_tokens_details, 'reasoning_tokens', None)}"
            )
        return

    completions = client.chat.completions
    if not hasattr(completions, "parse"):
        completions = client.beta.chat.completions
    path = f"{type(completions).__module__}.parse(response_format=...)"
    resp = await completions.parse(model=model, messages=messages, response_format=Answer)
    print(f"  path: {path}")
    print(f"  parsed: {resp.choices[0].message.parsed!r}")
    u = resp.usage
    if u is not None:
        print(
            f"  usage: prompt_tokens={u.prompt_tokens} completion_tokens={u.completion_tokens} "
            f"total_tokens={u.total_tokens}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="model for the logprobs probe")
    parser.add_argument("--structured-model", help="model for the structured-output probe")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    client = AsyncOpenAI(max_retries=1, timeout=60)
    print(f"openai SDK {openai.__version__}")

    ids = await list_models(client)
    print(f"\n[models] {len(ids)} visible:")
    for model_id in ids:
        print(f"  {model_id}")
    if args.list_only:
        return 0

    model = pick_model(ids, args.model)
    if model is None:
        print("\nno cheap candidate model visible; pass --model")
        return 1
    try:
        await check_logprobs(client, model)
    except openai.OpenAIError as exc:
        print(f"  logprobs probe failed: {type(exc).__name__}: {str(exc)[:200]}")
    try:
        await check_structured(client, args.structured_model or model)
    except openai.OpenAIError as exc:
        print(f"  structured probe failed: {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
