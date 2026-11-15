"""Run-scoped telemetry. Export is isolated from inference, bounded and best effort.

Generation records go to Langfuse and, when configured, a local Phoenix collector.
No SDK instrumentation wraps inference: an exporter failure cannot replay a paid call.
"""

# Telemetry is a deliberate exception boundary: SDK failures must not reach inference.
# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from uuid import uuid4

logger = logging.getLogger(__name__)
current_run: contextvars.ContextVar[TraceSession | None] = contextvars.ContextVar(
    "crux_trace_run", default=None
)


@dataclass
class TraceRef:
    trace_id: str
    run_id: str
    role: str
    step: str
    prompt_sha256: str
    model: str
    cached: bool
    url: str | None = None


class TraceSession:
    def __init__(self, run_id: str, step, *, exporter=None):
        self.run_id = run_id
        self.step = step
        self.refs: list[dict] = []
        self.errors: list[str] = []
        self.dropped = 0
        self._queue: queue.Queue = queue.Queue(maxsize=256)
        self._exporter = exporter
        self._thread: threading.Thread | None = None
        self._closed = False

    def record(
        self,
        *,
        role,
        model,
        prompt_sha256,
        messages,
        output,
        usage,
        error=None,
        step=None,
    ):
        project = os.getenv("LANGFUSE_PROJECT_ID")
        enabled = bool(
            os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
        )
        trace_id = uuid4().hex
        base = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
        ref = TraceRef(
            trace_id,
            self.run_id,
            role,
            str(step or self.step()),
            prompt_sha256,
            model,
            bool(usage.get("cached")),
            f"{base}/project/{project}/traces/{trace_id}"
            if enabled and project
            else None,
        )
        data = {
            **asdict(ref),
            "input": messages,
            "output": output,
            "usage": usage,
            "error": error,
            "timestamp": time.time(),
        }
        self.refs.append(asdict(ref))
        if self._closed or not (
            enabled or os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or self._exporter
        ):
            return
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            self.dropped += 1
            return
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._work, daemon=True, name="crux-telemetry"
            )
            self._thread.start()

    def _work(self):
        exporter = None
        try:
            exporter = self._exporter or _Exporters()
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closed:
                        break
                    continue
                try:
                    if item is None:
                        break
                    exporter.export(item)
                except Exception as exc:
                    # Do not log exception messages: providers can include credentials.
                    self.errors.append(type(exc).__name__)
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self.errors.append(type(exc).__name__)
        finally:
            if exporter is not None:
                try:
                    exporter.flush()
                except Exception as exc:
                    self.errors.append(type(exc).__name__)

    async def flush(self, timeout: float = 2.0):
        self._closed = True
        if self._thread is None:
            return
        deadline = time.monotonic() + timeout
        while self._thread.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if self._thread.is_alive():
            self.errors.append("FlushTimeout")

    def status(self):
        return {
            "errors": list(self.errors),
            "dropped": self.dropped,
            "pending": self._queue.unfinished_tasks,
        }


class _Exporters:
    """SDK work runs only on the daemon worker, never in the inference task.

    Phoenix receives the same records alongside Langfuse, so asynchronous export
    failures at Langfuse cannot prevent the local copy being exported.
    """

    def __init__(self):
        self.langfuse = None
        self.phoenix = None
        self.provider = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse

                self.langfuse = Langfuse(timeout=2)
            except Exception:
                logger.warning("Langfuse unavailable; local tracing remains available")
        if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
            try:
                from phoenix.otel import register

                self.provider = register(
                    project_name="crux",
                    endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
                    auto_instrument=False,
                    set_global_tracer_provider=False,
                    batch=True,
                )
                self.phoenix = self.provider.get_tracer("crux")
            except Exception:
                logger.warning("Phoenix unavailable; inference continues")

    def export(self, item):
        metadata = {
            k: item[k] for k in ("run_id", "role", "step", "prompt_sha256", "cached")
        }
        metadata["inference_seconds"] = item["usage"].get("seconds", 0)
        failure = None
        if self.langfuse is not None:
            try:
                from langfuse import propagate_attributes

                with propagate_attributes(session_id=item["run_id"], tags=[item["role"], item["step"]]):
                    span = self.langfuse.start_observation(
                        as_type="generation",
                        name=f"{item['role']}:{item['step']}",
                        trace_context={"trace_id": item["trace_id"]},
                        model=item["model"],
                        metadata=metadata,
                        input=item["input"],
                    )
                usage = item["usage"]
                span.update(
                    output=item["output"],
                    usage_details={
                        "input": usage.get("input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                    },
                    cost_details={"total": usage["cost_usd"]}
                    if usage.get("cost_usd") is not None
                    else None,
                    level="ERROR" if item["error"] else "DEFAULT",
                    status_message=item["error"],
                )
                span.end()
            except Exception as exc:
                failure = exc
        if self.phoenix is not None:
            import json

            from opentelemetry.trace import (
                NonRecordingSpan,
                SpanContext,
                TraceFlags,
                set_span_in_context,
            )

            parent = NonRecordingSpan(
                SpanContext(
                    trace_id=int(item["trace_id"], 16),
                    span_id=int(uuid4().hex[:16], 16),
                    is_remote=True,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            )
            with self.phoenix.start_as_current_span(
                item["role"],
                context=set_span_in_context(parent),
                start_time=int((item["timestamp"] - item["usage"].get("seconds", 0)) * 1e9),
            ) as span:
                span.set_attributes(
                    {
                        "openinference.span.kind": "LLM",
                        "llm.model_name": item["model"],
                        "metadata": json.dumps(metadata),
                        "input.value": json.dumps(item["input"]),
                        "output.value": json.dumps(item["output"]),
                        "llm.token_count.prompt": item["usage"].get("input_tokens", 0),
                        "llm.token_count.completion": item["usage"].get("output_tokens", 0),
                    }
                )
        if failure is not None:
            raise failure

    def flush(self):
        try:
            if self.langfuse is not None:
                self.langfuse.flush()
        finally:
            if self.provider is not None:
                self.provider.force_flush(timeout_millis=1000)
