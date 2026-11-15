# Archived DU-41 issue before the repair study

**Goal:** statistical evidence, tonight, that choosing experiments by information gain beats the alternatives, on a task from an external benchmark. This is the proposal in **Benchmark proposal: BoxingGym + ScienceAgentBench**, made concrete after reading the repo.

**Facts from the repo** ([kanishkg/boxing-gym](<https://github.com/kanishkg/boxing-gym>), MIT, file `src/boxing_gym/envs/location_finding.py`, class `Signal`):

* 3 hidden sources in 2-D, θ \~ N(0, 1).
* Intensity I(x) = b + Σ\_k α / (m + ‖θ\_k − x‖²), with b = 0.1, m = 1e-4, α = 1. Gaussian noise σ = 0.5. Observations rounded to 2 decimals.
* Evaluation: MSE on 10 held-out points (drawn N(0, 1)) against the **noiseless** intensity. The tiny m makes the error **heavy-tailed**.
* Paper budgets: evaluate after 0, 1, 3, 5, 7 and 10 experiments, 5 runs. Reported GPT-4o standardised error after 10 experiments: 0.59 ± 0.55 with the prior description (positive = worse than the prior-mean baseline). So there is room to win.
* Two problems in the official runner: numpy is never seeded, and its PyMC likelihood loops over 2 sources instead of 3. **We don't use their PyMC path.**

**Build**

- [X] Port the environment (same constants, prior and evaluation) to `labs/boxing/location_finding.py`, with credit. Unit-test it against the official formulas on a fixed θ. Seed numpy ourselves.
- [X] Arms, all with the same model where a model is used:
  - [X] Arm 1: Random designs
  - [X] Arm 2: Grid designs
  - [X] Arm 3: **Single agent** choosing designs and predicting in text (BoxingGym's own style)
  - [X] Arm 4: **Crux, random designs:** agents propose executable hypotheses, designs chosen at random
  - [X] Arm 5: **Crux, EIG designs:** numeric orchestrator with total-information selection; artifact critic disabled for scalar forecasts
  - [X] Arm 6: **Numerical known-structure reference**, labeled as such. Gavin explicitly accepted this in place of the originally requested exact oracle. EIG uses Monte Carlo estimates; finite particles and collapse limit accuracy. No mathematical exactness claim is made.
- [X] 20 paired seeds (100–119; same sources and held-out points verified across all six arms), budgets 5, 10 and 20: all 120 episodes and 360 checkpoint rows retained.
- [X] Metrics: held-out **median and mean log10 MSE with seed-bootstrap 95% CIs**; target MSE 1 with checkpoint-resolved, censored times; per-episode costs/tokens/shared proposal usage; failures retained with declared MSE penalty 1e12.
- [X] First chart with arms 1, 2 and 6 only (numpy, no LLM spend): delivered in `eval/results/boxing/pilot-20261114`. **The original Sat 6 PM deadline was missed.** The pilot exercises the numerical EIG path; it does not establish oracle accuracy or EIG superiority.
- [ ] **Optional stretch, deferred:** a second environment (hyperbolic discounting or death process). This does not block the required study deliverables.

**What we can and can't claim**

* Arm 5 vs arm 4 isolates **experiment selection**. That's the headline.
* Arm 5 vs arm 3 compares systems, but arm 5 builds explicit models while arm 3 reasons in text. Say so.
* This is a simulator. It shows experimental-design ability, not scientific discovery.
* ScienceAgentBench can't support an EIG claim (it grades generated programs; most bio tasks need an LLM judge for figures). It's parked unless time remains.

**Done when:** `eval/results/boxing/` holds the error-vs-experiments chart with CIs, the cost table and the per-seed scores.

**Study write-up — November 14, 2026**

**Purpose and what was built**

This study tests whether Crux learns more effectively when Expected Information Gain (EIG) chooses its experiments. EIG estimates which next measurement would reduce uncertainty most.

The task is to locate three hidden signal sources indirectly: take noisy measurements at chosen locations, then predict signal strength at ten unseen locations. The environment was ported from MIT-licensed BoxingGym with attribution, seeded randomness, and tests against the published intensity formula.

Six approaches were implemented: random locations, fixed-grid locations, a single AI agent choosing and predicting directly, Crux with random selection, Crux with EIG selection, and a numerical reference with the correct mathematical model structure. The reference does not know the actual hidden locations. Gavin approved this explicitly labeled numerical reference in place of the original exact-oracle wording.

The two Crux approaches share the same generated executable hypotheses, candidate locations, inference settings, and case seeds. Their intended difference is experiment selection. Crux uses the real numeric orchestrator with total-information scoring, deterministic reporting, and artifact criticism disabled for scalar forecasts. Comparing Crux with the text-based single agent compares systems with different representations.

**What was run**

* A separate free pilot: 20 seeds, three approaches (random, grid, numerical reference), 60 episodes and 180 valid checkpoint scores, with no model calls.
* The main study: 20 fresh paired seeds (100–119), all six approaches, 120 attempted episodes, and 360 checkpoint rows.
* Each main episode had a maximum of 20 measurements, with predictions evaluated after 5, 10, and 20 measurements.
* All approaches used identical hidden sources and held-out evaluation points within each seed. All model calls used gpt-4.1-mini.
* Metrics include median MSE, mean log10 MSE, 95% seed-bootstrap confidence intervals (2,000 resamples), target MSE 1, per-episode cost/tokens/duration, and failure counts. Target times are resolved only at checkpoints and censored if not reached.
* Every failed checkpoint remains in the primary statistics with the declared MSE penalty of 1e12.

**Results**

The study **does not establish an EIG advantage**.

| Approach | Median MSE after 20 measurements | Failed episodes / 20 |
| -- | -- | -- |
| Random locations | 18.94 | 0 |
| Fixed grid | 7.77 | 0 |
| Single AI agent | 7.13 | 0 |
| Crux, random selection | approximately 5 × 10¹¹ | 10 |
| Crux, EIG selection | 10¹² | 11 |
| Numerical known-structure reference | 11.70 | 0 |

Lower MSE is better. The enormous Crux medians come from the failure penalty, rather than measured physical signal errors of that size. These descriptive medians alone do not establish superiority between methods.

The primary paired comparison is mean log10 MSE for Crux EIG minus Crux random. Negative values favor EIG:

| Measurements | Paired difference | Bootstrap 95% CI |
| -- | -- | -- |
| 5 | +0.4841 | \[-0.0703, 1.5407\] |
| 10 | +0.4926 | \[-0.0534, 1.5517\] |
| 20 | +0.5056 | \[-0.0378, 1.5562\] |

All intervals include zero. Neither EIG superiority nor random superiority is established. Failures dominate the Crux aggregates; successful cases have not been substituted for the full primary dataset.

**What went wrong**

1. **Generated model code did not consistently follow the engine's array contract.** For 256 parameter samples and 49 candidate locations, parameter samples must have shape `(256, p)` and predictions must have shape `(256, 49)`. Some models returned `(256, 3, 2)` parameters, `(256, 3, 49)` predictions, or `(256, 256, 49)` predictions. These are parameter-formatting, source-aggregation, or broadcasting errors.
2. **Initial validation was too permissive.** It exercised a small example and checked finite predictions, but did not fully enforce parameter/prediction shapes across different input sizes. Seven generated model sets failed preliminary execution checks; three additional sets failed the production engine's prior or prediction shape requirements. Because hypotheses were shared, those ten bad model sets affected both Crux approaches: 20 failed episodes, or 60 failed checkpoints.
3. **One additional EIG episode failed on Windows file access.** Seed 102 encountered a denied atomic manifest replacement, bringing the total to 21 failed episodes and 63 failed checkpoints. A bounded retry for transient Windows sharing/access errors was added after the run. The archived failed episode remains in the results.
4. **Numerical inference also has limitations.** Particle collapse can produce overconfident, inaccurate uncertainty estimates. Saved effective sample sizes and parameter spreads expose this issue; sample size alone can appear healthy after resampling. The reference uses 256 particles, 16 simulated outcomes, a finite candidate grid, and a continuous Gaussian approximation to rounded measurements. It is not mathematically exact or a proven performance upper bound.

The code failures are fixable implementation problems, not evidence of a fundamental flaw in EIG. EIG nevertheless depends on useful models and credible uncertainty estimates: correcting array shapes alone does not guarantee informative experiment choices or an EIG win. A model can also execute correctly while describing the physics poorly.

**Proposed follow-up — not yet implemented**

* Enforce exact parameter and prediction contracts before accepting generated models.
* Check multiple sample counts and design counts, including finite outputs and deterministic predictions for fixed inputs.
* Add a bounded repair loop that supplies the actual validation error to the model and revalidates the corrected code.
* Retain original proposals, repair attempts, failures, and their costs.
* Improve and validate posterior inference separately, including particle-collapse diagnostics.
* Freeze a new protocol and rerun the paired comparison after the fixes. Preserve this original dataset.

The stricter validation and automatic repair loop discussed after the study have not been implemented. This page update documents the work and proposed follow-up.

**Validation, costs, and evidence**

* **349 repository tests passed; Ruff passed** on the integrated branch. The EIG estimator also agreed within test tolerance with an analytic linear-Gaussian reference. Passing software tests does not establish benchmark superiority.
* The dataset audit verified all paired source/held-out hashes, every saved measurement against the seeded simulator, shared hypothesis sources, and cost reconciliation. Representative Crux random/EIG event replays each checked 20 belief updates.
* Main-study wrapper-accounted cost: **$0.0965012**, 480 calls, 141,601 input tokens and 24,913 output tokens. These are configured list-price estimates, not a provider invoice. Shared hypothesis-generation cost is displayed with both Crux approaches but counted once in actual study spend.
* [GitHub branch](<https://github.com/mian-abd/Crux/tree/gavingzhao/du-41-boxing-location-study>), head `28d53024bc694604236f8b944e5976137f421226`; execution protocol source commit `5d543cceb14fe6ba91546334d4ebf7b63f326681`. Current main was integrated when the branch was pushed.
* [Full results report, charts, scores, costs, and audit](<https://github.com/mian-abd/Crux/blob/gavingzhao/du-41-boxing-location-study/eval/results/boxing/README.md>). Raw generated proposals, validation errors, histories, and Crux event streams/manifests are retained under `eval/results/boxing/`.
* New commits were authored and committed by Gavin Zhao (`gavinzhao30`), without co-author trailers. Configured credential values were absent from all 347 staged deliverable/fix files checked before the final commit.
* The pilot chart missed the original Saturday 6 PM milestone. The optional second environment remains deferred. This simulator study does not demonstrate scientific discovery.
