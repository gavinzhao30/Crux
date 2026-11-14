"""Tests for agents/llm.py. No network and no API key: every test uses a fake client."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai
import pytest
from pydantic import BaseModel

import agents.llm as llm_mod
from agents.llm import LLM, LLMConfig, LLMOutputError, Usage, cache_key

try:  # the SDK's HTTP library; newer SDK releases ship it as httpx2
    import httpx2 as httpx
except ImportError:  # pragma: no cover - depends on the installed SDK
    import httpx

REQUEST = httpx.Request("POST", "https://api.example.invalid/v1/responses")


# --- schemas ------------------------------------------------------------------------------


class Item(BaseModel):
    value: int
    label: str = "x"


class OtherItem(BaseModel):
    value: int
    note: str = ""


# --- fakes --------------------------------------------------------------------------------


def status_error(cls: type, status: int, message: str, *, param=None, code=None, headers=None):
    response = httpx.Response(status, request=REQUEST, headers=headers or {})
    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return cls(message, response=response, body=body)


def rate_limit(code: str | None = None) -> openai.RateLimitError:
    return status_error(openai.RateLimitError, 429, "Rate limit reached", code=code)


def temperature_rejection() -> openai.BadRequestError:
    return status_error(
        openai.BadRequestError,
        400,
        "Unsupported parameter: 'temperature' is not supported with this model.",
        param="temperature",
        code="unsupported_parameter",
    )


class FakeClient:
    """Mimics ``client.responses.parse`` (or the chat ``parse`` helpers with ``api=...``).

    ``script`` is consumed one entry per call: an exception is raised, a callable is called
    with the call's kwargs and returns the response, ``None`` produces the default response.
    The default response's parsed output is ``schema(value=<call number>)``.
    """

    def __init__(
        self,
        *,
        script: list | None = None,
        usage: tuple[int, int] = (100, 20),
        delay: float | Callable[[int], float] = 0.0,
        api: str = "responses",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script = list(script or [])
        self.usage = usage
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.api = api
        endpoint = SimpleNamespace(parse=self._parse)
        if api == "responses":
            self.responses = endpoint
        elif api == "chat":
            self.chat = SimpleNamespace(completions=endpoint)
        elif api == "beta_chat":
            self.beta = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
        else:  # pragma: no cover - test misuse
            raise ValueError(api)

    async def _parse(self, **kwargs: Any) -> Any:
        number = len(self.calls)
        self.calls.append(kwargs)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            delay = self.delay(number) if callable(self.delay) else self.delay
            await asyncio.sleep(delay)
            entry = self.script.pop(0) if self.script else None
            if isinstance(entry, BaseException):
                raise entry
            if callable(entry):
                return entry(kwargs)
            schema = kwargs.get("text_format") or kwargs.get("response_format")
            return self.make_response(schema(value=number))
        finally:
            self.in_flight -= 1

    def make_response(self, parsed: Any, *, refusal: str | None = None) -> Any:
        usage_in, usage_out = self.usage
        if self.api == "responses":
            output = []
            if refusal is not None:
                output = [
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="refusal", refusal=refusal)],
                    )
                ]
            return SimpleNamespace(
                output_parsed=parsed,
                output=output,
                status="completed",
                incomplete_details=None,
                usage=SimpleNamespace(input_tokens=usage_in, output_tokens=usage_out),
            )
        message = SimpleNamespace(parsed=parsed, refusal=refusal)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=usage_in, completion_tokens=usage_out),
        )


MESSAGES = [
    {"role": "system", "content": "Reply with a JSON object."},
    {"role": "user", "content": "Give me a number."},
]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(llm_mod, "_sleep", fake_sleep)
    return delays


def make_llm(tmp_path: Path, client: FakeClient, **overrides: Any) -> tuple[LLM, list[Usage]]:
    seen: list[Usage] = []
    settings: dict[str, Any] = {
        "role_models": {"proposer": "cheap-m", "designer": "strong-m", "default": "default-m"},
        "role_temperature": {"proposer": 0.7, "predictor": 0.8, "default": 0.2},
        "cache_dir": tmp_path / "cache",
    }
    settings.update(overrides)
    return LLM(LLMConfig(**settings), client=client, on_usage=seen.append), seen


def cache_files(tmp_path: Path) -> list[Path]:
    root = tmp_path / "cache"
    return sorted(root.rglob("*.json")) if root.exists() else []


# --- 1. parallel samples ------------------------------------------------------------------


def test_n_samples_in_order_and_bounded_concurrency(tmp_path: Path) -> None:
    # Earlier calls take longer, so completion order is the reverse of start order.
    client = FakeClient(delay=lambda number: 0.04 - 0.004 * number)
    llm, seen = make_llm(tmp_path, client, max_concurrency=3, cache=False)

    results = asyncio.run(llm.structured("proposer", Item, MESSAGES, n=8))

    assert [r.value for r in results] == list(range(8))
    assert all(isinstance(r, Item) for r in results)
    assert len(client.calls) == 8
    assert client.max_in_flight == 3
    assert len(seen) == 8


def test_semaphore_is_shared_across_concurrent_calls(tmp_path: Path) -> None:
    client = FakeClient(delay=0.01)
    llm, _ = make_llm(tmp_path, client, max_concurrency=2, cache=False)

    async def run() -> list[list[Item]]:
        return await asyncio.gather(
            llm.structured("proposer", Item, MESSAGES, n=4),
            llm.structured("designer", Item, MESSAGES, n=4),
        )

    first, second = asyncio.run(run())
    assert len(first) == len(second) == 4
    assert client.max_in_flight == 2


def test_n_must_be_positive(tmp_path: Path) -> None:
    llm, _ = make_llm(tmp_path, FakeClient())
    with pytest.raises(ValueError):
        asyncio.run(llm.structured("proposer", Item, MESSAGES, n=0))


# --- 2. disk cache ------------------------------------------------------------------------


def test_cache_hit_makes_no_calls(tmp_path: Path) -> None:
    client = FakeClient()
    llm, seen = make_llm(tmp_path, client)

    first = asyncio.run(llm.structured("proposer", Item, MESSAGES, n=2))
    assert len(client.calls) == 2
    second = asyncio.run(llm.structured("proposer", Item, MESSAGES, n=2))

    assert len(client.calls) == 2
    assert second == first
    assert [u.cached for u in seen] == [False, False, True, True]
    assert all(u.cost_usd == 0.0 and u.input_tokens == 0 for u in seen[2:])


def test_cache_file_layout_and_contents(tmp_path: Path) -> None:
    llm, _ = make_llm(tmp_path, FakeClient())
    asyncio.run(llm.structured("proposer", Item, MESSAGES))

    key = cache_key(
        model="cheap-m",
        role="proposer",
        messages=MESSAGES,
        schema=Item,
        sample_index=0,
        temperature=0.7,
    )
    path = tmp_path / "cache" / key[:2] / f"{key}.json"
    assert cache_files(tmp_path) == [path]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["output"] == {"value": 0, "label": "x"}
    assert data["usage"]["input_tokens"] == 100
    assert data["usage"]["model"] == "cheap-m"
    assert not list((tmp_path / "cache").rglob("*.tmp*"))


def test_cache_misses_on_temperature_schema_and_sample_index(tmp_path: Path) -> None:
    client = FakeClient()
    llm, _ = make_llm(tmp_path, client)

    asyncio.run(llm.structured("proposer", Item, MESSAGES, n=2))
    assert len(client.calls) == 2

    asyncio.run(llm.structured("proposer", Item, MESSAGES, n=2, temperature=1.1))
    assert len(client.calls) == 4  # new temperature: both samples miss

    asyncio.run(llm.structured("proposer", OtherItem, MESSAGES, n=2))
    assert len(client.calls) == 6  # new schema: both samples miss

    asyncio.run(llm.structured("proposer", Item, MESSAGES, n=3))
    assert len(client.calls) == 7  # samples 0 and 1 hit, sample 2 misses

    asyncio.run(llm.structured("proposer", Item, MESSAGES + [{"role": "user", "content": "!"}]))
    assert len(client.calls) == 8  # new messages miss


def test_cache_disabled_never_writes(tmp_path: Path) -> None:
    client = FakeClient()
    llm, seen = make_llm(tmp_path, client, cache=False)

    asyncio.run(llm.structured("proposer", Item, MESSAGES, n=2))
    asyncio.run(llm.structured("proposer", Item, MESSAGES, n=2))

    assert len(client.calls) == 4
    assert cache_files(tmp_path) == []
    assert not any(u.cached for u in seen)


def test_corrupt_cache_entry_is_a_miss(tmp_path: Path) -> None:
    client = FakeClient()
    llm, _ = make_llm(tmp_path, client)
    asyncio.run(llm.structured("proposer", Item, MESSAGES))
    [path] = cache_files(tmp_path)
    path.write_text("{not json", encoding="utf-8")

    [result] = asyncio.run(llm.structured("proposer", Item, MESSAGES))

    assert len(client.calls) == 2
    assert result.value == 1
    assert json.loads(path.read_text(encoding="utf-8"))["output"]["value"] == 1  # rewritten


def test_fully_cached_run_needs_no_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_client(timeout_s: float) -> Any:
        raise AssertionError("a cache hit must not create a client")

    monkeypatch.setattr(llm_mod, "_make_default_client", no_client)
    warm, _ = make_llm(tmp_path, FakeClient())
    expected = asyncio.run(warm.structured("proposer", Item, MESSAGES, n=3))

    cold = LLM(warm.config)  # no client injected, none created on cache hits
    assert asyncio.run(cold.structured("proposer", Item, MESSAGES, n=3)) == expected
    assert cold._client is None
    assert cold.usage_totals()["cached_calls"] == 3


# --- 3. role mapping ----------------------------------------------------------------------


def test_role_model_and_temperature_mapping(tmp_path: Path) -> None:
    client = FakeClient()
    llm, seen = make_llm(tmp_path, client, cache=False)

    async def run() -> None:
        await llm.structured("proposer", Item, MESSAGES)
        await llm.structured("designer", Item, MESSAGES)
        await llm.structured("mystery-role", Item, MESSAGES)
        await llm.structured("predictor", Item, MESSAGES, temperature=1.3)

    asyncio.run(run())
    sent = [(c["model"], c["temperature"]) for c in client.calls]
    assert sent == [
        ("cheap-m", 0.7),
        ("strong-m", 0.2),
        ("default-m", 0.2),
        ("default-m", 1.3),
    ]
    assert [u.model for u in seen] == ["cheap-m", "strong-m", "default-m", "default-m"]
    first = client.calls[0]
    assert first["text_format"] is Item
    assert first["input"] == MESSAGES


def test_default_config_maps_every_role() -> None:
    config = LLMConfig()
    for role in ("proposer", "predictor", "designer", "referee", "critic", "reporter"):
        assert config.model_for(role)
    assert config.model_for("proposer") == llm_mod.CHEAP_MODEL
    assert config.model_for("referee") == llm_mod.STRONG_MODEL
    assert config.temperature_for("predictor") == 0.8
    assert config.temperature_for("proposer") == 0.7
    assert config.temperature_for("critic") == 0.2
    with pytest.raises(ValueError):
        LLMConfig(role_models={"proposer": "m"})


# --- 4. retries ---------------------------------------------------------------------------


def test_rate_limit_is_retried(tmp_path: Path, no_backoff: list[float]) -> None:
    client = FakeClient(script=[rate_limit(), rate_limit(), None])
    llm, seen = make_llm(tmp_path, client, cache=False)

    results = asyncio.run(llm.structured("proposer", Item, MESSAGES))

    assert len(results) == 1
    assert len(client.calls) == 3
    assert len(no_backoff) == 2
    assert no_backoff[0] <= 1.0 <= no_backoff[1]  # exponential growth, with jitter
    assert len(seen) == 1


@pytest.mark.parametrize(
    "error",
    [
        lambda: openai.APITimeoutError(request=REQUEST),
        lambda: openai.APIConnectionError(request=REQUEST),
        lambda: status_error(openai.InternalServerError, 503, "Service unavailable"),
    ],
)
def test_transient_errors_are_retried(tmp_path: Path, error: Any) -> None:
    client = FakeClient(script=[error(), None])
    llm, _ = make_llm(tmp_path, client, cache=False)
    assert len(asyncio.run(llm.structured("proposer", Item, MESSAGES))) == 1
    assert len(client.calls) == 2


def test_retries_are_bounded(tmp_path: Path) -> None:
    client = FakeClient(script=[rate_limit() for _ in range(10)])
    llm, _ = make_llm(tmp_path, client, cache=False, max_retries=2)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(llm.structured("proposer", Item, MESSAGES))
    assert len(client.calls) == 3


def test_insufficient_quota_is_not_retried(tmp_path: Path) -> None:
    client = FakeClient(script=[rate_limit(code="insufficient_quota")])
    llm, _ = make_llm(tmp_path, client, cache=False)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(llm.structured("proposer", Item, MESSAGES))
    assert len(client.calls) == 1


def test_retry_after_header_is_honoured() -> None:
    error = status_error(
        openai.RateLimitError, 429, "slow down", headers={"retry-after-ms": "7500"}
    )
    assert llm_mod._retry_after_s(error) == pytest.approx(7.5)
    assert llm_mod._backoff_delay(0, 7.5) >= 7.5
    assert 0.5 <= llm_mod._backoff_delay(0) <= 1.0
    assert llm_mod._backoff_delay(20) <= llm_mod.BACKOFF_MAX_S


@pytest.mark.parametrize(
    "error",
    [
        lambda: status_error(openai.BadRequestError, 400, "Invalid schema", param="text"),
        lambda: status_error(openai.AuthenticationError, 401, "Incorrect API key provided"),
        lambda: status_error(openai.NotFoundError, 404, "The model does not exist"),
    ],
)
def test_client_errors_are_not_retried(tmp_path: Path, error: Any) -> None:
    raised = error()
    client = FakeClient(script=[raised, None])
    llm, _ = make_llm(tmp_path, client, cache=False)
    with pytest.raises(type(raised)):
        asyncio.run(llm.structured("proposer", Item, MESSAGES))
    assert len(client.calls) == 1


# --- 5. temperature rejection -------------------------------------------------------------


def test_temperature_rejection_is_remembered_per_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(script=[temperature_rejection()])
    llm, _ = make_llm(tmp_path, client, cache=False)

    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        asyncio.run(llm.structured("designer", Item, MESSAGES))
        asyncio.run(llm.structured("designer", Item, MESSAGES))
        asyncio.run(llm.structured("proposer", Item, MESSAGES))

    assert len(client.calls) == 4
    assert client.calls[0]["temperature"] == 0.2
    assert "temperature" not in client.calls[1]  # retried without it
    assert "temperature" not in client.calls[2]  # omitted from the start next time
    assert client.calls[3]["model"] == "cheap-m"
    assert client.calls[3]["temperature"] == 0.7  # other models unaffected
    warnings = [r for r in caplog.records if "temperature" in r.getMessage()]
    assert len(warnings) == 1


def test_temperature_rejection_with_parallel_samples(tmp_path: Path) -> None:
    client = FakeClient(script=[temperature_rejection() for _ in range(4)], delay=0.005)
    llm, _ = make_llm(tmp_path, client, cache=False)
    results = asyncio.run(llm.structured("designer", Item, MESSAGES, n=4))
    assert len(results) == 4
    assert len(client.calls) == 8
    assert all("temperature" not in c for c in client.calls[4:])


# --- 6. output errors ---------------------------------------------------------------------


def test_refusal_raises_output_error(tmp_path: Path) -> None:
    client = FakeClient()
    client.script = [lambda kw: client.make_response(None, refusal="I can't help with that.")]
    llm, seen = make_llm(tmp_path, client)

    with pytest.raises(LLMOutputError) as info:
        asyncio.run(llm.structured("designer", Item, MESSAGES))

    err = info.value
    assert (err.role, err.model, err.sample_index) == ("designer", "strong-m", 0)
    assert "refused" in str(err)
    assert len(seen) == 1 and seen[0].input_tokens == 100  # billed tokens still counted
    assert cache_files(tmp_path) == []


def test_unparseable_output_raises_output_error(tmp_path: Path) -> None:
    def sdk_validation_failure(kwargs: dict) -> Any:
        return Item.model_validate_json('{"value": "not a number"')  # raises ValidationError

    client = FakeClient(script=[None, sdk_validation_failure])
    llm, _ = make_llm(tmp_path, client, cache=False)

    with pytest.raises(LLMOutputError) as info:
        asyncio.run(llm.structured("proposer", Item, MESSAGES, n=2))
    assert info.value.sample_index in (0, 1)
    assert "ValidationError" in str(info.value)


def test_wrong_shape_output_raises_output_error(tmp_path: Path) -> None:
    client = FakeClient()
    client.script = [lambda kw: client.make_response({"label": "missing value"})]
    llm, _ = make_llm(tmp_path, client, cache=False)
    with pytest.raises(LLMOutputError, match="does not match Item"):
        asyncio.run(llm.structured("proposer", Item, MESSAGES))


def test_failure_cancels_sibling_samples(tmp_path: Path) -> None:
    client = FakeClient(delay=lambda number: 0.0 if number == 0 else 5.0)
    client.script = [lambda kw: client.make_response(None, refusal="no")]
    llm, _ = make_llm(tmp_path, client, cache=False)

    async def run() -> None:
        with pytest.raises(LLMOutputError):
            await llm.structured("proposer", Item, MESSAGES, n=3)
        assert client.in_flight == 0

    asyncio.run(asyncio.wait_for(run(), timeout=2))


# --- 7. usage accounting ------------------------------------------------------------------


def test_usage_hook_and_totals(tmp_path: Path) -> None:
    client = FakeClient(usage=(1000, 200))
    llm, seen = make_llm(
        tmp_path, client, prices_per_mtok={"cheap-m": (0.4, 1.6)}, cache_dir=tmp_path / "cache"
    )

    asyncio.run(llm.structured("proposer", Item, MESSAGES, n=3))
    asyncio.run(llm.structured("designer", Item, MESSAGES))  # no price for strong-m
    asyncio.run(llm.structured("proposer", Item, MESSAGES, n=3))  # all cached

    assert len(seen) == 7
    expected_cost = (1000 * 0.4 + 200 * 1.6) / 1_000_000
    for usage in seen[:3]:
        assert (usage.role, usage.model) == ("proposer", "cheap-m")
        assert (usage.input_tokens, usage.output_tokens) == (1000, 200)
        assert usage.cost_usd == pytest.approx(expected_cost)
        assert usage.cached is False and usage.seconds >= 0
    assert seen[3].cost_usd is None
    assert all(u.cached and u.cost_usd == 0.0 for u in seen[4:])

    totals = llm.usage_totals()
    assert totals["calls"] == 7
    assert totals["cached_calls"] == 3
    assert totals["input_tokens"] == 4000
    assert totals["output_tokens"] == 800
    assert totals["cost_usd"] == pytest.approx(3 * expected_cost)
    assert totals["unpriced_calls"] == 1


# --- API path detection and tracing fallback ----------------------------------------------


@pytest.mark.parametrize("api", ["chat", "beta_chat"])
def test_chat_parse_fallback_paths(tmp_path: Path, api: str) -> None:
    client = FakeClient(api=api, usage=(10, 5))
    llm, seen = make_llm(tmp_path, client, cache=False)

    [result] = asyncio.run(llm.structured("proposer", Item, MESSAGES))

    assert result == Item(value=0)
    call = client.calls[0]
    assert call["response_format"] is Item
    assert call["messages"] == MESSAGES
    assert (seen[0].input_tokens, seen[0].output_tokens) == (10, 5)


def test_chat_refusal_raises_output_error(tmp_path: Path) -> None:
    client = FakeClient(api="chat")
    client.script = [lambda kw: client.make_response(None, refusal="nope")]
    llm, _ = make_llm(tmp_path, client, cache=False)
    with pytest.raises(LLMOutputError, match="refused"):
        asyncio.run(llm.structured("proposer", Item, MESSAGES))


def test_client_without_parse_helper_is_rejected() -> None:
    with pytest.raises(TypeError):
        LLM(client=SimpleNamespace())


def test_missing_langfuse_falls_back_to_plain_client(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setitem(sys.modules, "langfuse", None)  # makes the import fail
    monkeypatch.setattr(llm_mod, "_tracing_warned", False)

    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        first, tracing = llm_mod._make_default_client(5.0)
        llm_mod._make_default_client(5.0)

    assert isinstance(first, openai.AsyncOpenAI)
    assert tracing is False
    assert first.max_retries == 0
    assert len([r for r in caplog.records if "Langfuse" in r.getMessage()]) == 1


def test_trace_metadata_rejection_is_dropped(tmp_path: Path) -> None:
    rejection = status_error(
        openai.BadRequestError, 400, "The 'metadata' parameter is only allowed when 'store' is on"
    )
    client = FakeClient(script=[rejection])
    llm, _ = make_llm(tmp_path, client, cache=False)
    llm.trace_tags = {"run_id": "r-1"}
    llm._tracing = True

    asyncio.run(llm.structured("proposer", Item, MESSAGES, step="propose"))
    asyncio.run(llm.structured("proposer", Item, MESSAGES))

    assert client.calls[0]["metadata"] == {
        "run_id": "r-1",
        "role": "proposer",
        "step": "propose",
        "langfuse_session_id": "r-1",
    }
    assert "metadata" not in client.calls[1]
    assert "metadata" not in client.calls[2]
