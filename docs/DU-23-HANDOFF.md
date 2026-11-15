# DU-23 original local handoff (historical)

Superseded by [DU-23-COMPLETION.md](DU-23-COMPLETION.md). The notes below describe
the original handoff, not the current branch or validation state.

This checkout is on local branch `gavingzhao/du-23-eval-harness`. The work is committed locally as Gavin Zhao, but has not been pushed or changed in Linear. The next chat can inspect the commit, push the branch, and update DU-23 only after reviewing the remaining validation below. Do not add an AI co-author or contributor.

## What changed

- `eval/run_suite.py` adds the Task 9 and Boxing suite command, paired job manifest, five Task 9 presets, Modal `.map()` dispatch, bounded concurrency, separate grading, persistent Volume checkpoints, per-job JSON, `summary.csv`, and `--recover` reconciliation. Boxing rejects the two presets that its numeric adapter cannot distinguish.
- `eval/boxing_study.py` can export held-out predictions without scoring them during execution. The suite coordinator scores those predictions after the Modal job ends. Existing study behavior is unchanged by default.
- `tests/test_run_suite.py` checks paired seeds, evaluator isolation in the job manifest, unsupported Boxing presets, deferred grading, failure scores, and complete summary rows.
- `eval/results/boxing/20261115T042103Z/` retains the first full development run as evidence.

## Checks completed

- Ruff passes on all changed Python files.
- The focused DU-23 and Boxing tests pass (14 tests).
- A Modal preflight with Boxing seed 1000 and `single_agent,crux_random_selection` completed and graded both jobs.
- The full Boxing development split with `single_agent,crux` completed all 10 jobs in about **111 seconds** (directory creation to final summary), below the 20-minute target. All five Crux jobs are recorded as score-zero failures because their shared generated model proposals failed validation. This is the known DU-41 generated-model problem; it does not establish a useful Crux comparison. The proposal cost fields were improved after this run, so this run's failed-proposal costs are incomplete.
- A dry-run manifest followed by `--recover` correctly marked an unstarted job `interrupted` with score zero.
- The full repository suite passes: **354 tests** with `TEMP` and `TMP` pointed to a writable workspace folder. Without that setting, this sandbox produced temporary-file permission and Windows DLL-sharing errors unrelated to DU-23.
- A separate Task 9 Modal smoke job using the local pilot counts and samples completed successfully in about **50 seconds**. It verified remote worker execution and the nested Task 9 sandbox, but did not use the private grader and is not the acceptance run.

## Remaining validation before closing DU-23

The Task 9 suite directory, private references, pinned grader checkout, and grader Python environment were not available in this checkout or its adjacent `work` folder. Run the Task 9 development split with two approaches and reconcile selected `summary.csv` rows against grader output before checking the Task 9 and complete DU-23 acceptance items in Linear. The CLI requires `--suite-dir`, `--references`, `--grader-repo`, and `--grader-python` for that run. Keep all evaluator paths outside agent inputs.

Example:

```powershell
.\.venv\Scripts\python.exe -m eval.run_suite task9 --arms single_agent,crux --cases dev --repeats 1 --concurrency 2 --suite-dir <suite> --references <references> --grader-repo <grader> --grader-python <grader-python>
```

For recovery after an interrupted coordinator, run `python -m eval.run_suite <suite> --recover --output <existing-result-directory>` with the evaluator paths if Task 9 grading remains to be done. Recovery makes no model calls and marks absent worker checkpoints explicitly.
