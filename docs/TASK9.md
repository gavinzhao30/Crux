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

Give both evaluation arms the same `GuardedSearch` with the Task 9 policy. The provider must use the supplied documentation domain filter. Configure the source paper's DOI, domains, and study-specific keywords from the task metadata. The wrapper checks queries before search and filters URLs and snippets before returning evidence. Connect its audit callback to the run log to capture every query, result URL, and denial. This filter covers configured terms and approved documentation domains. Search experiments can use an injected evidence-to-result mapper with the same predeclared outcome predicates as compute experiments.

References: [BixBench3](https://github.com/EdisonScientific/BixBench3), [Modal Sandboxes](https://modal.com/docs/guide/sandboxes), [DESeq2](https://bioconductor.org/packages/DESeq2).
