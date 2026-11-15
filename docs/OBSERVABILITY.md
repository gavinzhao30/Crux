# Run provenance and tracing (DU-45)

Every orchestrated run writes `runs/<run_id>/manifest.json`. The manifest starts
before profiling and finalises on success, budget exhaustion, cancellation or
error. Atomic replacement prevents readers from seeing partially written JSON.
Existing run IDs are rejected without overwriting their manifest.

The manifest records configured and observed model IDs, hashes of prompt files
and actual request messages, Git revision and dirty status, grader revision,
verified staged input digests and bundle ID, sandbox/image IDs and image recipe
hash, seeds, full non-secret configuration, timestamps, tokens, cost, diagnostics,
artifact versions and trace references. Local synthetic labs have no staged input
bundle or Modal image, so those fields are empty/null. For Modal runs the executor
records the resolved image ID after image hydration. Prompt/output text is exported
only to the configured tracing services, not embedded in trace references.

Each diagnostic's canonical code/query/design/bins and SHA-256 are saved before
`experiment.started` and `lab.run()`. Artifact digests are calculated from actual
file bytes rather than trusting an adapter's claimed digest. Unknown model prices
produce `cost_usd: null` and an explicit `unpriced_calls` count. Set
`llm.prices_per_mtok` for the models in use to calculate a complete total; the
manifest also keeps the known subtotal and the budget's accounting snapshot.

## Langfuse Cloud

Create or select a project, then create project API keys under Settings → API Keys.
Save credentials only in the ignored `.env` file:

```dotenv
LANGFUSE_PUBLIC_KEY=your-project-public-key
LANGFUSE_SECRET_KEY=your-project-secret-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PROJECT_ID=the-id-after-project-in-the-project-url
```

Use `https://us.cloud.langfuse.com` for the US region, or the regional URL shown in
your project's setup instructions. Load the file when launching the application,
for example `uv run --env-file .env uvicorn api.server:app`.

All structured calls, including cache hits and failed calls, have run-scoped
`run_id`, `role`, `step` and prompt digest references. Model trace links on question
and experiment cards open the recorded generation directly. Different concurrent
runs use context-local trace sessions and usage hooks.

The SDK's drop-in `langfuse.openai` integration can throw after an inference has
already succeeded. Retrying that exception can duplicate a paid request. Crux
therefore uses explicit generation records through the Langfuse SDK on a separate
daemon worker, with plain OpenAI inference. This intentionally replaces the
original drop-in integration to satisfy the fail-open requirement. SDK setup,
export and flush errors cannot retry or fail inference. The queue is bounded;
run exit waits at most two seconds and records export errors, pending records and
drops. A trace URL identifies its destination; it does not prove delivery. SDK
network export may fail asynchronously and cannot always be reported by the SDK.
Generation metadata includes actual inference duration; the Langfuse generation's
span timestamps represent background export processing.

## Local Phoenix fallback

Start a local Phoenix server in a separate terminal:

```shell
uvx --from arize-phoenix phoenix serve
```

Add `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces` to `.env` and open
`http://localhost:6006`. Crux exports the same calls with matching trace IDs using
OpenInference attributes. Phoenix receives a parallel local copy even if Langfuse
is down or rate-limited, including failures reported asynchronously by Langfuse.
No Phoenix Cloud account is needed for the local server. If neither backend is
available, the run continues with local manifest/event references. No tracing
service is silently installed or launched by an ordinary run.

## Replay and acceptance

```shell
uv run python -m core.replay RUN_ID
uv run python -m core.replay runs/RUN_ID/events.jsonl
```

Both categorical Bayes updates and numeric family updates are rederived from the
event log alone and compared with recorded priors/posteriors. Code is not executed
by replay. Numeric sufficient statistics validate probability updates rather than
reproducing the complete particle sampler. The checker also verifies proposed spec
hashes and rejects corrupted trajectories. Legacy numeric logs cannot establish
full replay because they lack the likelihood statistics.

Offline tests cover real Langfuse SDK serialization, real Phoenix OTLP export
while Langfuse fails, bounded flush, no duplicate inference, concurrent runs,
artifact hashing, pre-execution persistence and success/error/cancellation cleanup.
Frontend tests verify card links and reject unsafe URLs and other runs' references.

Final live acceptance still requires an authenticated Langfuse Cloud project, a
real model call whose recorded link opens the matching trace, and replay on the
chosen scientific hero run. An offline synthetic run is a reproducible integration
fixture, not evidence that a scientific hero run is complete.
