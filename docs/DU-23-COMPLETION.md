# DU-23 validation, 2026-11-15

## Delivered behavior

The suite CLI runs paired (case, arm, repeat) jobs through Modal `.map()`, bounded
by `max_containers`. Grading runs afterward in the coordinator. Task truth and
reference artifacts never enter execution-worker job payloads. Every scheduled
job has a result row; failed, missing, and timed-out workers receive score zero.

JSON records and CSV include score, worker cost/tokens/calls, elapsed time,
experiment count, fault review, failures, and evidence locations. Fault-review
usage is separate from agent usage. Shared Boxing proposal usage is explicitly
reported per paired case and must be counted once per case, not once per arm.
Unavailable failure-path usage remains unknown rather than becoming a false zero.
Fault detection reviews METHODS.md; missing methods are a miss, clean cases are
not applicable, and evaluator errors remain explicit. This is a methods-based
review, not an assertion about a model's internal reasoning.

Acknowledged events commit to the persistent Volume before returning. Completed
Task 9 sandbox commands collect and commit outputs before returning. Periodic
and final checkpoints provide additional protection. A crash can lose unfinished
command work; the guarantee covers acknowledged checkpoints, not every in-flight
instruction. Recovery retrieves partial evidence, retains interrupted rows, and
can upgrade them when a terminal remote result later becomes available. Empty
artifact directories survive download so the official grader can score a missing
submission rather than incorrectly treating its evidence as unavailable.

## Preset coverage

| Preset | Task 9 | Boxing |
| --- | --- | --- |
| single_agent | supported | supported |
| crux | supported | supported |
| crux_random_selection | supported | supported |
| crux_no_critic | supported | rejected: numeric critic is absent |
| crux_role_only_memory | supported | rejected: numeric role memory is absent |

This implements the approved plan's instruction to make equivalent or unsupported
Boxing settings explicit rather than present them as distinct experiments. All
five Task 9 configurations are verified. The two unsupported Boxing variants
remain a capability limitation; the recorded development runs exercise two
approaches, not all presets. Meaningful numeric critic/memory experiments would
require extending that scientific adapter beyond the agreed harness plan.

## Independent evaluator and inputs

The frozen Task 9 development split contains four opaque cases, generated from
the existing local published counts/sample pilot. This is downstream analysis
validation, not a full FASTQ benchmark. Both approaches receive the same cases,
seeds, and fallback Task 9 prompt. No test cases were used for acceptance.

The public official grader was pinned to EdisonScientific/BixBench3 commit
`d0e0bbb41222335b1b8878f533a58466a3a782dc`. The official public release manifest is
https://storage.googleapis.com/bixbench3-ground-truth/releases/v1.0.0/release.json.
The Task 9 reference manifest is
https://storage.googleapis.com/bixbench3-ground-truth/releases/v1.0.0/manifests/10_1186_s12915_024_01879_0_v1.json,
SHA256 `395b51024226f72b42fd96b2311818ff52b5883a6463f54c85ea2b743404da4b`.
All 15 reference files passed their published SHA256 checks; grader self-test on
the reference artifacts scored 1.0. References and the truth key remain outside
the repository and outside agent inputs. The unavailable teammate evaluator
Volume was not required. Grader dependencies were numpy 2.4.4, pandas 3.0.3,
scipy 1.17.1, and pydantic 2.13.4.

## Recorded acceptance evidence

* `eval/results/boxing/du23-final-dev-20261115`: 10/10 jobs, 10 independent grades,
  520.39 seconds (8.7 minutes), all summary scores reconcile with held-out MSE.
* `eval/results/task9/du23-dev-20261115`: 8/8 jobs, 1007.97 seconds (16.8 minutes).
  Six initial `missing_artifacts` statuses were corrected by preserving empty
  directories and recovering/regrading against the official grader. All eight
  grades reconcile with raw artifact scores. The original timed acceptance file
  retains its pre-recovery grade count; `audit.json` records the reconciled count.
  Seven jobs score zero; baseline case-12 scores 0.4169468466510137. All injected
  faults are missed. These poor agent results are retained, not filtered away.
* `eval/results/harness-recovery/du23-20261115-v2`: a real Modal worker container
  was terminated after committed intermediate evidence, followed by a new
  coordinator subprocess. Both jobs were recovered; the interrupted job scored
  zero and the completed fixture graded successfully. Event logs and partial
  artifact bytes survived. This fixture uses no model calls and is not a
  benchmark result.

`python -m scripts.audit_suite_acceptance <results-directory>` checks every
summary row against per-job JSON and separate grader output, paired seeds,
failure retention, and the under-20-minute timing gate. Archived generated
programs are immutable experiment evidence and excluded from source lint.

Final source verification: Ruff passes. The full repository run produced 792
passes and one Windows `WinError 5` while replacing a temporary budget ledger in
OneDrive. A fresh-directory rerun of that test and all 12 harness tests passed
(13/13). An earlier full run before the last two harness regressions passed
791/791. The temporary-file failure was not hidden or converted into a pass.

Fresh timed Task 9 rerun: `eval/results/task9/du23-final-dev-20261115`, 8/8 jobs
recorded in **755.438 seconds (12.6 minutes)**. Seven completed submissions were
separately graded; one baseline execution-budget limit is retained with grade
status `failed` and score zero. All eight rows pass independent reconciliation.
Baseline case-12 scores 0.3319131109014623; the remaining scores are zero. The
methods-based judge reports the duplicate-library fault caught by Crux and the
gene-version fault caught by the budget-limited baseline. This does not change
the latter's failed benchmark score. Mean scores are 0.08297827772536558 for the
baseline and 0 for Crux. Worker model costs are $0.196044 and $0.1528128;
fault-review calls total $0.00066. These figures exclude Modal compute costs.

The fresh run started from `b9f9e3c` with the empty-directory/download fixes in the
working tree, subsequently committed in `8f593ba`. Its already-running reviewer
still emitted nonfatal async client cleanup warnings. `8f593ba` also closes that
client within its owning event loop. A fresh-process review of saved evidence
passed with no cleanup warning; `lifecycle-review.json` retains this independent
check and its additional $0.0003592 evaluator cost, outside benchmark totals.

Reproduction command (local evaluator paths are intentionally outside the repo):

```powershell
python -m eval.run_suite task9 --arms single_agent,crux --cases dev --repeats 1 --concurrency 4 --timeout 300 --suite-dir ../work/du23-evaluator/suite --references ../work/du23-evaluator/references --grader-repo ../work/du23-evaluator/BixBench3 --grader-python ../work/du23-evaluator/grader-venv/Scripts/python.exe --output eval/results/task9/<new-directory>
```

## Outstanding branch integration review

The completion branch includes the earlier DU-23 and DU-41 branches. DU-44's
`fbf50ea` is patch-equivalent to work already in main. DU-15's `dc50599` was
explicitly ported in main's `58de6b9`, with additional Task 9 wiring, Boolean
search configuration, price validation, blocked-URL metadata, and policy fixes.
Those two old topic branches were already deleted remotely when the final fetch
pruned stale references. No redundant history merge or old implementation was
reintroduced. The only remaining remote topics were the three Gavin branches.
