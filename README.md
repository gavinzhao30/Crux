# Crux

Crux is a team of scientific agents that resolves disagreements by choosing informative
experiments, running them in a sandbox, and recording the evidence behind its decisions.
The engineering contract is in [docs/ERD.md](docs/ERD.md).

## Development

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --locked
uv run pytest -q
uv run ruff check .
```

The current implementation is the DU-16 memory adapter. It includes only the shared
`Evidence` model needed from DU-3; the remaining engine, orchestration and event-bus
modules are separate workstreams. Tests use an HTTP-contract fake and need no API keys.

## Memory integration

DU-17 adds a role-local baseline and a frozen-history decision replay. See
[docs/MEMORY_COMPARISON.md](docs/MEMORY_COMPARISON.md) for the backend switch,
common instrumentation and comparison command. The local arm works without API keys.

Set `PAST_API_KEY` in the server environment. `.env.example` documents the variable;
the application does not automatically load `.env` files. A live synthetic integration
check (uses provider credits) is available with:

```sh
uv run python -m memory.demo
```

It exercises experimenter observation → critic recall and objection → designer recall
and changed action, then checks that a second run cannot recall a private sentinel.
This is a synthetic plumbing check, not a scientific result or a measured memory gain.

```python
from pathlib import Path
from core.schemas import Evidence
from memory.base import RunContext
from memory.pastdev import PastMemory

# Inside an async function:
async with PastMemory(RunContext(run_id="r1", task_id="task9"), Path("runs")) as memory:
    await memory.write(
        Evidence(id="obs1", kind="observation", claim="Two samples share a subject",
                 author_role="experimenter"),
        run_id="r1", author_role="experimenter",
    )
    await memory.flush()  # optional barrier: wait for indexing, e.g. before a benchmark
    evidence = await memory.recall("Which samples share subjects?", "r1", "critic")
```

`write()` commits locally without making a network call. Background upload retries
transient errors; closing the adapter preserves unfinished work for a later restart.
`flush()` has a bounded timeout and surfaces failures. It resumes polling the saved
ingestion ID, which is distinct from the evidence/source ID. Rejected content requires
a corrected new event rather than automatic retries of the same body.

Use `write_record(MemoryRecord(...))` for the full provenance envelope, including
conditions, parent events, artifact references, review status and `supersedes` evidence
IDs. IDs are immutable: retries preserve the exact body and idempotency key; corrections
get a new ID. Objections, invalid analyses and corrections are mandatory context until
superseded. They can exceed recall's `k` so a veto is never silently dropped. Callers
must budget this mandatory context in their prompts.

A failed remote recall is recorded separately from an empty successful response. Pending
and mandatory local records provide an explicitly logged fallback; otherwise the adapter
raises `RecallUnavailable`. Completed ordinary history is selected only by remote recall.
Returned source IDs must resolve to current canonical local records. Retrieval preserves
evidence IDs and never emits a new observation or updates a posterior: the orchestrator
must deduplicate evidence IDs before applying Bayesian updates.

## Which memory holds what

| Store | Contents and authority |
| --- | --- |
| `runs/<scope>/memory.sqlite3` | Durable structured memory journal, canonical record status, outbox receipts and retries, exact retrieval requests/responses, selected evidence IDs and latency. Authority for this adapter's evidence. |
| past.dev | Compact evidence envelopes and artifact pointers; supplies scoped relevant context, not scientific truth. |
| Modal/artifact storage (planned) | Matrices, raw data, generated code, artifacts and environment manifests; large files stay out of memory. |
| Run event log (DU-3 integration pending) | Will determine overall orchestration state and belief replay; memory's audit table does not replace it. |

`RunContext` binds run, task, config and repeat to a deterministic audience and role-specific
reader identities. Reopen the same context/root to resume its outbox. Roles must be declared
in `RunContext.roles` before use; the defaults cover the ERD's scientific roles.

## Leakage controls

Every upload names a fixed run/config/repeat audience, including corrections. Readers
are selected by server-side context, never by query text. Run-ID prefixes and metadata
are not recall filters. Do not put task evidence into project-visible memory. A project
key can access every audience in its project; use separate projects for stronger isolation.
The tests check audience membership, negative cross-run recall and foreign-source rejection.
Grader outputs and ground truth must stay outside agent-visible stores. No grader is
imported by the memory implementation.

## Credits

- [past.dev](https://past.dev/) — memory service;
  [API contract](https://past.dev/docs/memory-api/api-reference) and
  [audience access controls](https://past.dev/docs/memory-api/access-control).
- [HTTPX](https://www.python-httpx.org/) — async HTTP transport.
- [Pydantic](https://docs.pydantic.dev/) — ERD Evidence contract and provenance validation.
- [uv](https://docs.astral.sh/uv/), [pytest](https://docs.pytest.org/) and
  [Ruff](https://docs.astral.sh/ruff/) — environment management and development checks.

## Contributing

Use one branch per Linear issue. Review changes and commit under your own GitHub identity.
Commits and pull requests must have no tool or bot authors, co-author or generated-by
trailers, and no tool listed as an author or contributor in documentation.
