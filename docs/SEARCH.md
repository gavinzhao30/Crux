# Documentation search (DU-15)

Search is an experiment: a candidate query competes against compute diagnostics in the
orchestrator. Exa returns cited tool documentation; fixed code rules determine the observed
bin. The orchestrator owns ranking and Bayesian updates. The adapter never chooses an
experiment or changes a posterior itself.

## In the Task 9 run

`build_task9_run` gives the run a search when `cfg.search` is on and `EXA_API_KEY` is set
(`search.task9.build_task9_search`). It becomes `Deps.search` and the lab's search, and the
lab closes it at the end of the run. Without the key, or with the `no_search` preset, there
is no search and the orchestrator drops every search experiment.

The search is `GuardedSearch` (Task 9 policy) over `ExaSearch`:

1. `SwarmRoles.design` returns pre-registered search diagnostics
   (`search.task9.SEARCH_DIAGNOSTICS`) for questions whose text documentation can decide:
   independent filtering and pre-filtering, `alpha` and p-value adjustment, design formula
   conventions, `lfcShrink` versus `lfcThreshold`, and raw-count input. Each has a fixed
   query and bins that are code rules. One experiment serves every question it matches.
2. The predictor fills `P(bin | option)` and the orchestrator ranks the search against the
   question's compute diagnostics by EIG per cost (about 4 s and $0.007 per search).
3. When a search wins, the orchestrator calls `find`, evaluates the bin rules over
   `{"n_results", "text", "urls"}` (`text` is the screened excerpts joined and lower-cased),
   keeps that dict as the observation's value so the referee sees the rules' inputs, emits
   `search.done` (query, URLs, blocked URLs, bin) and updates the posterior after the
   referee. Every bin requires `n_results > 0`, so a search without usable documentation is
   invalid and moves nothing. A refused query or a provider error is an invalid
   observation, not a crashed run.

`Task9Lab.run` computes the same fields when no `summarize_search` mapper is injected, so
no model chooses the observed bin on either path. `ExaSearch`'s disk cache and audit live
in `<runs_dir>/.cache/search`, shared by runs, so a repeated query costs nothing.

## Setup

```sh
uv sync --locked
uv run python -m pytest -q
uv run ruff check .
```

Tests need no API keys and exercise the installed Exa SDK with a mocked HTTP transport.
A live check requires `EXA_API_KEY` in the server environment. Environment files are not
loaded automatically; never expose keys to the browser or agent-generated analysis code.

## Policy

`search/task9_policy.py` is the one Task 9 policy. `Task9SearchPolicy` holds the public
task metadata: the source paper's DOI (`10.1186/s12915-024-01879-0`) and article id, its
journal, preprint servers, the GEO/SRA/ENA record pages, PubMed and PMC mirrors, the
`storage.googleapis.com/bixbench3-ground-truth` bucket, and study terms (glioblastoma,
T98G, HEK293, REST-targeting compounds and the output file names). Its checks combine a
lexical screen with the generic `search.policy.SearchPolicy` built from the same data
(`Task9SearchPolicy.rules`). `rules` also configures `ExaSearch`, both Exa's include and
exclude domain filters and its own screening, so anything the adapter refuses the guard
refuses first. Do not obtain exclusions by inspecting ground truth.

For another task, construct a `SearchPolicy` from that task's public metadata: include the
source paper's title, DOI, study/cohort identifiers and distinctive result terms in
`forbidden_terms`, and source paper/preprint/supplement URLs in `forbidden_sources`. The
constructor rejects an empty exclusion list.

```python
from search.policy import SearchPolicy

# Synthetic example only; replace these exclusions with the actual public task metadata.
policy = SearchPolicy(
    task_id="synthetic-demo",
    forbidden_terms=("private-study-zeta", "10.1234/example-source-study"),
    forbidden_sources=("github.com/example/source-study",),
)
```

For a live smoke check (one paid query, about $0.007), run the demo with the Task 9 policy,
or pass a saved policy with `--policy`:

```sh
uv run python -m search.demo --query "DESeq2 documentation alpha default"
uv run python -m search.demo --policy work/task-policy.json --query "DESeq2 alpha default"
```

## API and outcomes

`ExaSearch.find(query, k=5) -> list[Evidence]` implements the `core.interfaces.Search`
protocol. `lookup()` also returns incremental cost, price provenance, cache status, the
URLs the policy removed and source metadata. Use an async context manager, or call
`aclose()`. Injected SDK clients remain caller-owned. By default the evidence author is
`experimenter`, so it can be written to memory with that scientific role. A caller can
supply another declared `author_role`.

```python
from core.config import RunConfig
from search.exa import ExaSearch

# Inside an async function, using the task policy above:
async with ExaSearch(policy, run_id="r1", config=RunConfig(search=True)) as search:
    evidence = await search.find("DESeq2 documentation alpha default")
```

`SearchExperiments.run(Experiment(..., kind="search")) -> Observation` is a standalone
runner (for example for the single-agent baseline). It durably saves the query, question
IDs, policy fingerprint, outcome rules and their hash **before** the request. The same ID
or spec cannot run twice, including after a process restart. A process interrupted during
a search leaves its spec spent; inspect its audit rather than rerunning it silently. The
swarm does not use it: the orchestrator keeps its own spent-spec set and bin evaluation.

Its result JSON contains `text` (joined source highlights), `n_sources`, cited evidence,
the spec hash and cost provenance. Each `OutcomeBin.rule` is a restricted expression over
`text` and `n_sources`, for example `n_sources > 0 and 'alpha = 0.1' in text`. Supported
operators are comparisons, membership, `and`, `or` and `not`. Calls, attribute access,
indexing, comprehensions and arithmetic are rejected. Agent-written Python is never executed.
Rules must return booleans, and exactly one bin must match. Missing usable sources or
ambiguous bins produce an invalid observation, never an arbitrary winner.

These literal rules are intentionally limited. A task author must account for versions,
negation and conflicting examples; the engine does not infer that any mention of a number
establishes a tool default. The separate referee must validate a result before a belief update.
The older DU-15 text proposes model-selected bins; this implementation follows the stricter
ERD rule that code computes them.

After the orchestrator computes a posterior change, call
`record_belief_effect(experiment_id, question_id, bits)` to log which question moved and
the realized information change. It rejects invalid or unrelated observations and does not
perform posterior arithmetic.

## Cache, audit and cost

`<root>/<run-hash>/search.sqlite3` stores accepted evidence, experiment specs/observations,
and timestamped audit records. Cache keys include query, result limit, request options,
policy and author role. Concurrent identical requests through one adapter share one call;
the cache survives restarts. Changed policies cannot reuse entries from a different policy.
In the Task 9 run, `GuardedSearch` writes its own query, result and denial records to the
same audit table (`guard.*`).

Every outgoing query and every returned URL is audited, including blocked attempts. Rejected
excerpts do not enter the cache or evidence. Errors are distinct from a successful empty
result; failed calls are not cached and their cost is marked unknown. Keys and request
headers are never logged.

If a remote request fails after starting, the invalid experiment carries a configurable
cost reservation rather than assuming it was free; `actual_cost_known` is false. A missing
key fails before a request starts and reserves zero. Reconcile reservations with provider
usage before claiming exact spending totals.

The adapter records Exa's `cost_dollars.total` when available, labeled `provider_estimate`.
Otherwise it uses `ExaSearch(estimated_cost_usd=...)` (default $0.007) and labels the value
`configured_estimate`. The default follows Exa's published Auto search rate for up to 10
results with highlights and no summaries, checked November 14, 2026. It is not an invoice or
an independent spending cap. Cached requests have zero incremental provider cost. The run's
budget counts each search experiment at its `est_cost_usd`.

`RunConfig(search=False)` disables search without constructing a client or requiring a key.
This flag applies to both direct lookups and search experiments.

## Leakage controls

The query gate requires an approved tool and documentation intent. It screens task-specific
terms, DOI lookups, accessions, result queries and source operators. Exa receives an include
list of documentation hosts and an exclude list. Returned URLs are independently checked at
host and path boundaries; HTTP, credentials, query strings, dot traversal and unapproved
hosts (including subdomains such as `support.bioconductor.org`) are rejected. Snippets and
titles are also screened for excluded terms. Adjust the reviewed sources when legitimate
docs are rejected.

This is a conservative lexical/documentation policy, not a proof of semantic isolation.
Public forums and tool repositories can contain study-specific text; the task's exclusion
list must be reviewed. It complements DU-39's sandbox/network and grader isolation; it
does not implement those controls or inspect ground truth.

## Dependencies and references

- [Exa Python SDK](https://github.com/exa-labs/exa-py),
  [Search API](https://exa.ai/docs/reference/search),
  [pricing](https://exa.ai/docs/admin/pricing).
- [Pydantic](https://docs.pydantic.dev/) for ERD contracts and configuration.
- [HTTPX](https://www.python-httpx.org/) for SDK transport tests.
- [uv](https://docs.astral.sh/uv/), [pytest](https://docs.pytest.org/) and
  [Ruff](https://docs.astral.sh/ruff/) for development.
