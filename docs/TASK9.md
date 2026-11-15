# Crux Task 9


## Task 9 infrastructure

This is the downstream BixBench3 pilot using published counts as input. The sandbox smoke test checks one contrast. Task 9 prompting, full artifact contracts, official grading, and study-level reporting live in the benchmark and evaluation workstreams.

Use Python 3.12 with `modal>=1.6.4,<2` and the project's dependencies. Authenticate Modal in your shell before staging inputs. Prepare three local inputs from the Task 9 data handoff:

- The published matrix named `tagseq_featurecounts_matrix.tsv`.
- A tab-separated sample sheet with `sample_id`, `cell_line`, and `condition`. Add `count_column` if matrix headers differ from sample IDs.
- Optionally, the supplied `inputs_metadata.json` with agent-visible metadata.

```sh
python -m execution.staging \
  --counts /path/tagseq_featurecounts_matrix.tsv --samples /path/samples.tsv \
  --manifest /path/task9-input-manifest.json

python -m execution.smoke_task9 \
  --manifest /path/task9-input-manifest.json --run-id task9-smoke-01 \
  --cell-line T98G --treatment C10 --reference Ctrl
```

Pass `--minimal` on the smoke command if the full conda image cannot resolve. The smoke run checks the mount and egress, runs sequential commands and a timeout, and exports a DE CSV and `METHODS.md` under `runs/<run-id>/artifacts/`. The execution log records command logs and sandbox startup time.

For orchestration, open the sandbox with `ModalExecutor.open("bixbench3", run_id)`, then construct `Task9Lab` with the resulting sandbox ID, task prompt, artifact contract, and an async delivery planner that maps settled state to ordered `(language, code)` steps. Set `de_tables` to the contract paths for DE CSVs. The lab exports artifacts after each step. The caller closes the executor in a `finally` block. Pass the run event sink to the executor to forward command logs to the live UI.

### Leakage controls

Only the staged, hashed input bundle is mounted, read-only. The sandbox starts with all outbound networking blocked and no injected credentials. The published count matrix is the input exception for this downstream pilot. Keep reference artifacts, fault labels, and grades in grader-only storage. Give the isolated grader a copy of the exported artifacts.

Give both evaluation arms the same `GuardedSearch` with the Task 9 policy. The provider must use the supplied documentation domain filter. `Task9SearchPolicy` carries the source paper's DOI, its preprint servers, the study's GEO/SRA pages, the ground-truth bucket and study-specific keywords from the task metadata; `build_task9_run` wires it over Exa (see [SEARCH.md](SEARCH.md)). The wrapper checks queries before search and filters URLs and snippets before returning evidence. Connect its audit callback to the run log to capture every query, result URL, and denial. This filter covers configured terms and approved documentation domains. Search experiments can use an injected evidence-to-result mapper with the same predeclared outcome predicates as compute experiments.

References: [BixBench3](https://github.com/EdisonScientific/BixBench3), [Modal Sandboxes](https://modal.com/docs/guide/sandboxes), [DESeq2](https://bioconductor.org/packages/DESeq2).

## Task 9 evaluation

The pilot uses the published count matrix as input and grades nine downstream artifacts
with Edison's deterministic grader. It reports the continuous mean and the fraction
scoring at least 0.80. The four GO outputs are optional and can be selected explicitly
with `--only`. Task 9 pilot scores use a different protocol from full-suite leaderboard
runs.

### Generate and run cases

Use Python 3.12 with `modal`, `openai`, `pydantic`, `numpy`, and `pandas`.
Generate the suite once and keep the evaluator truth key out of Modal, search, and memory.

```sh
python -m eval.suites.task9.make_cases \
  --counts /path/tagseq_featurecounts_matrix.tsv --samples /path/samples.tsv \
  --output /path/task9-suite
```

The suite contains one clean case and two variants of each of six planted faults.
The count headers and sample IDs are blinded across cases. Only a case's counts and
sample sheet go to `execution.staging`. The evaluator folder holds the case split and
truth key. Use the clean case and the three designated dev variants for tuning.

Accept the [BixBench3 dataset gate](https://huggingface.co/datasets/EdisonScientific/BixBench3)
and supply Task 9's `agent/prompt.md` with `--prompt`. A local fallback derived from the
public output specification is available and reported in `baseline.json`.

```sh
python -m eval.baseline_agent --manifest /path/case-manifest.json --run-id clean-baseline \
  --prompt /path/task9-prompt.md --model "$EVAL_MODEL" \
  --input-rate "$EVAL_INPUT_USD_PER_MILLION" --output-rate "$EVAL_OUTPUT_USD_PER_MILLION"
```

The CLI runs with search disabled. For search-enabled paired runs, pass the same
`GuardedSearch` policy and provider into `run_baseline` and the swarm. Match the model,
rates, budgets, prompt, inputs, image, and enabled tools across the two arms. Set the
same `ModalExecutor.diagnostic_timeout_s` for both arms (default and maximum 120 seconds).

### Grade outside the agent run directory

Check out EdisonScientific/BixBench3 at
`d0e0bbb41222335b1b8878f533a58466a3a782dc` and install NumPy, pandas, and SciPy (use the versions in its lockfile) in its
own Python 3.12 environment. Put the official Task 9 `artifacts_grading.json` and
reference `artifacts/` in evaluator-only storage.

```sh
python -m eval.grade_bixbench3 --task 9 --artifacts-dir runs/clean-baseline/artifacts \
  --references /path/evaluator/references --output-dir /path/evaluator/clean-baseline \
  --grader-repo /path/BixBench3 --python /path/BixBench3/.venv/bin/python
```

Use `--self-test` to compare reference artifacts against themselves in evaluator storage.
The wrapper copies the selected output files into a temporary evaluator directory.
Missing deliverables stay in the score denominator.

For batch grading use `python -m eval.suites.task9.grade_suite --help`. Supply a run index
keyed by opaque case ID, then arm name, with an `artifacts` path. Specify all requested
arms with `--arms`, including any with zero submissions. Set a failed run's `status` to
`failed` or `error`. Missing and failed runs count as zero in the suite means. An optional evaluator
`review` records detection, repair, diagnostic, and rejected explanations. Missing runs
stay visible in coverage. The report displays the clean case separately.
