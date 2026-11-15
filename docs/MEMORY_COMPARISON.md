# Role-local memory and decision replay (DU-17)

DU-17 compares the required memory arms: `local_role_only` and `pastdev`. It reuses
DU-16's evidence contract, canonical status checks and durable store. The local arm
needs no account or API key. The optional vector and full-context arms are not implemented.

## Role-local baseline

`LocalMemory(..., mode="role_only")` implements the ERD Memory protocol. Every role writes
its records to a durable ledger, but a reader receives **only records it authored**.
The author filter runs before selection and before the mandatory-review buffer. Another
role's objection is never smuggled into this arm through local fallback context.

Recall returns the reader's current mandatory records first and then its most recent own
history up to `k`. Mandatory records can exceed `k`, matching the shared backend's behavior.
There is no semantic local retriever or hidden broadcast of other agents' records.
Superseded evidence is excluded using the same canonical status gate as the shared arm.
Canonical invalidation can remove an own record without exposing another role's review.
The full orchestrator's mandatory validity controls must remain the same across arms.

The local ledger lives under `runs/local_role_only/<scope>/memory.sqlite3`; the provider
ledger and outbox use DU-16's separate scope directory. Local writes create no upload
outbox entries and never construct a remote client.

```python
from core.config import RunConfig
from memory.base import RunContext
from memory.factory import create_memory

# Inside an async function; same write()/recall() interface in both arms.
config = RunConfig(memory="local_role_only")  # or memory="pastdev"
context = RunContext(run_id="r1", task_id="task9", config_id=config.memory)
async with create_memory(context, config) as memory:
    own_history = await memory.recall("What did I check?", "r1", "designer")
```

Reopen the same context and root to resume local history. Evidence IDs are immutable;
corrections use a new ID with `supersedes`. The factory rejects provider-only options
for the role-local arm. The `pastdev` arm requires `PAST_API_KEY` in the server environment.

## Shared instrumentation

Both backends use `MemoryTelemetry` and `format_memory(evidence)`. Audit records retain:

- writer and reader roles, evidence IDs, ordered authors and source references;
- per-call latency in milliseconds, using a high-resolution elapsed-time clock;
- memory-fragment prompt size, counting method and SHA-256;
- backend identity, remote failure/fallback status and the original recall query.

`memory.written` and `memory.recalled` payloads include the required fields in
[EVENTS.md](EVENTS.md). Additional measurement fields are compatible with its rule to
ignore unknown payload fields. A sync or async `on_event(event_type, payload)` callback
can forward these to the event bus. Callback failures are audited without losing evidence.
In a run, the orchestrator emits its own `memory.written` and `memory.recalled` events for
every write and recall it makes; this telemetry adds the measurement fields to the run's
memory journal (`runs/<run_id>/memory/`).

The default prompt count is **an estimate: UTF-8 bytes divided by four, rounded up**.
It is labeled `estimate_utf8_bytes_div4`, not reported as actual model usage. To measure
actual memory-fragment tokens, inject a tokenizer for the chosen model and name it:

```python
from memory.instrumentation import MemoryTelemetry

# tokenizer is supplied by the caller; no tokenizer is downloaded implicitly.
telemetry = MemoryTelemetry(
    "local_role_only",
    token_counter=lambda text: len(tokenizer.encode(text)),
    token_method="your-model-tokenizer-version",
)
```

Insert the exact output of `format_memory()` in the prompt to use this count. Outer
instructions, ordinary messages, model outputs and tokenizer overhead are not counted
here. The LLM wrapper remains responsible for billed usage. Telemetry summaries cover
the current adapter instance; durable audit records support aggregation across restarts.

## Frozen decision replay

```sh
uv sync --locked
uv run python -m eval.memory_replay \
  --fixture eval/fixtures/memory_replay.json \
  --output-dir eval/results/memory-replay
```

This writes `comparison.csv`, `comparison.md` and `report.json`. Without `PAST_API_KEY`,
the local arm is measured and past.dev gets a `skipped_missing_key` row with no metrics.
Set the key and rerun to measure both required arms. A failed provider call is reported
as a failure or degraded fallback, never as a successful empty recall. A live run sends
the supplied fixture records to past.dev and uses provider credits.

Each invocation uses fresh, disjoint run/config scopes, and each probe exposes only
the declared prefix of the frozen history. Queries, roles, source records, canonical
status gates, decision rules and prompt formatting are held fixed across arms. The
harness waits for ingestion to complete before static retrieval measurements, and
records ingestion waiting separately from recall latency and total wall time.

The checked-in fixture is **synthetic plumbing data**, not BixBench3 ground truth or a
representative scientific evaluation. Its action hints are part of the permitted source
evidence. The fixed decision proxy reads the current role's `next_action_by_role` hints;
absent hints keep the predeclared plan, and conflicting hints request review. Labels
are never executed. Hidden `decisive_evidence_ids` annotations are used only for scoring
retrieval and are never passed to memory or the decision policy.

Reported metrics include:

- decisive-evidence recall hits and changes from the predeclared action;
- changes attributable to another role's evidence, checked by removing foreign records
  from the same retrieved context and reevaluating the deterministic policy;
- repeated diagnostic **decision labels** under identical `conditions_hash` values;
- choices repeating a rejection recorded in current, nonsuperseded source evidence;
- write/recall latency p50 and p95, prompt size, ingestion waiting and wall time.

The diagnostic and rejection counts are decision-replay proxies. No pipeline, actual
diagnostic or LLM agent is run. A fixture deliberately requiring cross-role evidence
cannot establish average uplift, scientific correctness or superiority to other shared
retrievers. A baseline might have lower prompt volume and the provider might show no
gain; report the observed values without manufacturing a success claim.

Tests exercise the actual past.dev adapter against mock HTTP, including both arms,
future-evidence cutoffs and the required event payloads. Injected provider factories
are accepted only with the explicit `mock_http` label. Mock results cannot masquerade
as live past.dev measurements.

## Integration status

Both arms run inside the engine. `memory.run.RunMemory` binds either backend to one run on
its first call, and `build_task9_run` builds it from `cfg.memory`: `pastdev` when
`PAST_API_KEY` is set, `local_role_only` always (it needs no key), otherwise no memory.
The orchestrator writes and recalls at the same points in both arms and drops other
roles' records in the role-only arm, so the call sites, prompts and instrumentation match.
The shared `RunConfig` is `main`'s; this work adds no config fields.

Still open: a reviewed frozen development history (the replay fixture is synthetic). The
vector and full-context extensions remain optional; `memory="vector"` runs without memory.

Existing dependency credits are in the README. This work additionally uses only Python's
standard library for the local backend, metrics, fixture hashing and report generation.
