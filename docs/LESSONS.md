# Reviewed procedural lessons (DU-30)

Lessons carry a procedure learned from development cases into later runs. They change
the agents' context, not their weights. A reusable lesson records when it applies, the
mistake, diagnostic, repair, supporting event IDs, limitations, reviewer and review time.
The framework does not generate or approve scientific lessons on a person's behalf.

## Review and freeze

`ReviewedLesson` requires an explicit approval and reviewer, plus source evidence whose
split is literally `dev`. Source kinds are observation, objection, diagnostic or execution
failure; grading sources and extra score/reference fields are rejected. Known answer and
grading markers are also screened from procedure text. This is an additional guard, not
a substitute for a person's review of provenance and scientific content.

`FrozenLessons` declares the entire held-out case cohort before comparison. It rejects
lessons sourced from those cases, reviews after the freeze, duplicate IDs and naive
timestamps. Applicability has two forms: readable `when`/`exceptions`, and machine gates
in `requires`/`excludes`. Every requirement must match the public context; any matching
exclusion removes the lesson. Values are compared with exact types.

```python
from pathlib import Path
from memory.lessons import FrozenLessons

# This file is prepared and reviewed outside the evaluation loop.
corpus = FrozenLessons.model_validate_json(Path("work/approved-lessons.json").read_text())
path = corpus.save(Path("runs/frozen-lessons"))
corpus = FrozenLessons.load(path)
```

The saved file is content-addressed, create-only and verified against its SHA-256 on
load. Editing a frozen lesson creates a different corpus; it cannot silently alter the
existing manifest. The checked-in example is explicitly synthetic and uses a fictional
fixture reviewer. It is not a review of actual study results.

## Start-of-run recall

The lesson corpus has its own deterministic audience and local ledger, separate from
current task-run evidence. A reporter writes only approved procedures and development
source references. It does not upload the held-out case list, grades, matrices or source
analysis results. This separation preserves DU-16's isolation between task runs.

```python
from core.config import RunConfig
from memory.lessons import LessonMemory

# Inside an async function; PAST_API_KEY is read from the environment.
async with LessonMemory(corpus, RunConfig(lessons=True)) as lessons:
    await lessons.prepare()  # wait for ingestion/indexing before static comparisons
    context = await lessons.recall(
        "What preflight checks apply?",
        case_id="a-case-declared-before-freeze",
        role="designer",
        context={"analysis": "de_from_counts"},
    )
```

Allowed readers are the generic proposer, the three proposer personas, and the designer.
Recall validates the cohort and role, filters applicability, deduplicates source IDs and
resolves results to the frozen canonical lesson text. Unknown remote sources cannot enter
the lesson prompt. Review metadata and development-source pointers accompany each result.
`lessons=False` constructs no provider, uploads nothing and performs no recall.

`prepare()` writes `lesson.written` audit payloads using the existing event contract. The
final filtered recall is audited with writer/reader roles, corpus hash, case ID, latency
and labeled prompt-size estimates. Lessons must be loaded before independent proposals,
without sharing one proposer's current proposals with another.

The Task 9 run does not load a reviewed corpus yet: that waits for real lessons reviewed
from development runs. In a run, `lessons=True` (off by default) only makes the
orchestrator write each settled or robust question's lesson, authored by the reporter, to
the run's own memory and emit `lesson.written`. Those are candidates for review. They are
never reused unreviewed, and the run-scoped audience keeps them out of later runs.

## Paired comparison

`eval.lesson_comparison.compare()` accepts a frozen corpus, the complete declared cohort,
fixed run conditions and an async case-runner callback. The callback receives:

1. the held-out case's public inputs and input hash;
2. `RunConfig(lessons=False)` or `RunConfig(lessons=True)`;
3. identical role-to-model IDs, sandbox image, seed and budget;
4. the selected lesson evidence for proposer and designer.

The settings and cases are cloned per arm, and mutation invalidates a result. Arm order
alternates across cases. Failed arms remain in the report; a corpus-preparation failure
fails the on arm rather than pretending recall succeeded with no lessons. The off arm
can still run. Evaluating only a selected subset of the frozen cohort is rejected.

The runner returns diagnostics before finding the fault, runner cost and official score,
or `None` for anything not measured. Official grades remain in `eval/` report files and
are never sent back to memory. CSV/Markdown/JSON reports include every arm and paired
on-minus-off deltas. Preparation/retrieval billing is separate from runner cost and must
be accounted for before making matched-budget claims. Prompt counts are labeled UTF-8
byte/4 estimates for memory fragments; model billing comes from the LLM wrapper.

For a synthetic context-only check:

```sh
uv run python -m eval.lesson_comparison --output-dir eval/results/lessons-demo
```

For the same synthetic check through the real memory service:

```sh
uv run --env-file .env python -m eval.lesson_comparison --live \
  --output-dir eval/results/lessons-live
```

The CLI explicitly labels its fixture as synthetic, does not invoke a model or pipeline,
and leaves diagnostics, cost and official score unmeasured. Without `--live`, the service
is labeled `fixture_memory`; with it, the real past.dev adapter is used. Synthetic corpora
require explicit opt-in and cannot report measured scientific results. The CLI fails if
the on-arm context is empty or the off-arm context contains lessons.

## Remaining validation

DU-30's real before/after requires reviewed lessons from actual development cases and a
held-out case runner with external grading. The planned scientific comparison follows the
development/overnight results; this branch prepares that path now. The optional predictor
self-calibration/Brier-score experiment is not implemented. Report real gains, no gains
and failures alike; synthetic retrieval success alone is not evidence of self-improvement.

The implementation uses existing Pydantic and past.dev dependencies plus Python's
standard library. Dependency credits are in the README and existing memory documentation.
