# Task 9 comparison protocol: swarm vs single-agent baseline

Written and committed **before** batch-01 ran. Any later change gets a new batch name and is reported as a change.

## What is compared

| | Swarm (`core.cli --lab bixbench3`) | Baseline (`eval.baseline_agent`, DU-22) |
| --- | --- | --- |
| Agents | 3 proposers, the DU-42 fork library, sampled predictor, blind referee, deliverer, critic (2 repair rounds max), reporter | One ReAct agent that plans, runs code, checks and revises |
| Models | gpt-4.1-mini for proposers and predictor; gpt-5.5 for referee, deliverer, critic and reporter | gpt-5.5 throughout |
| Budget | $2 and 40 model calls (`core.config`), 30 min | $2 and 40 model calls, 60 min of execution |
| Sandbox | The same Modal image (R + DESeq2), network blocked, one sandbox per run | The same |
| Inputs | Staged bundle `160253fe1e2f…`: the published count matrix plus `samples.tsv` | The same |
| Prompt | `load_prompt()` fallback (the official prompt is in the gated HF dataset) | The same |
| Web search and shared memory | Off (Exa and past.dev integration aren't merged yet) | Off |

## Measurement

- **Grader:** Edison's deterministic grader at commit `d0e0bbb`, on the 9 required artifacts. Its self-test scores the references 1.0.
- **Score:** a run's score is the mean over the 9 artifacts. We also report the fraction scoring at least 0.80.
- **Runs:** 3 per arm, started in parallel. A failed or ungraded run counts as **0**.
- **Reported:** the mean, SD, min and max per arm, plus cost, model calls and wall time per run. Every run is listed, failures included.

## Known asymmetries, stated up front

- The baseline caps each answer at 4,096 tokens (Jason's design). The swarm's gpt-5.5 roles have no per-answer cap. Both have the same $2 total cap.
- The baseline uses gpt-5.5 for everything, while the swarm uses gpt-4.1-mini for its cheap roles.
- One task, starting from counts: neither arm processes raw reads, so neither score is comparable to the 0.523 full-suite BixBench3 record.
- An earlier swarm run on the same task (0.586, `task9_runs.md`) is already known to us. Nothing was tuned on its grades before this batch.
