# DU-41 repair study: frozen protocol

Original results and issue text are retained in `eval/results/boxing/main-20261114`
and `docs/DU-41-original-linear.md`. This rerun changes model reliability and
posterior inference, so before/after performance cannot isolate either change.

Development seeds: 3000–3002. New primary seeds: 4000–4019. All six arms use
paired sources, observation-noise streams and held-out points per seed. Budgets
remain 5/10/20 measurements, ten noiseless held-out targets, and the same grid.
All model calls use gpt-4.1-mini. The primary study has a $5 accounted-spend cap;
output limits and pre-request spend checks include repairs. No settings will be
tuned against primary results.

Generate one set of hypotheses per seed, shared by both Crux arms. Each family
gets strict finite shape, reproducibility, deterministic prediction and input
immutability checks at (particles, designs) sizes (4,2), (7,5), (512,49). Allow
two repair calls per family, retaining every attempted source and validation
error. An unrepairable set fails both arms. Calls that fail to return usable
structured output can end proposal generation; no silent fallback family is used.
Usage includes successful and failed proposal/repair calls with returned usage.
Provider-billed failures without returned usage cannot be fully accounted for.

Use 512 particles, 64 EIG outcome simulations, and four resample/move sweeps.
At ESS below half the particle count, resample then move against the likelihood
of all observations. Generated models use independent proposals from their own
prior sampler; prior factors cancel from the MH ratio. The known-structure
reference alternates symmetric local Normal proposals (SD .15) with independent
prior proposals, including the exact Normal prior density for local acceptance.
Batching independent proposals changes execution cost without changing their
MH acceptance rule. No arbitrary jitter is applied when moves are enabled.
Other labs retain legacy behavior unless numeric_move_steps is explicitly enabled.

This remains finite-particle, approximate inference. Low move acceptance and
particle duplication remain possible and must be reported. Effective sample
sizes, parameter spread, particle uniqueness, pre-resample ESS and move counts
are saved. The reference is numerical, not exact or a guaranteed upper bound.
Likelihood remains a continuous Normal approximation to rounded measurements.

Keep the original failure penalty 1e12, log floor 1e-12, target MSE 1 and 2,000
seed-bootstrap resamples. Report all 120 episodes and 360 checkpoints, including
failures. Primary contrast is paired mean log10 MSE (Crux EIG minus Crux random),
with 95% CI; negative favors EIG. No successful-only replacement analysis.
Target times are checkpoint-resolved and censored when not achieved.

Unit gates include real malformed-model regressions, bounded repair exhaustion,
analytic Gaussian posterior moments for local and independence moves, an
independent Gaussian-mixture quadrature EIG check, analytic Gaussian EIG,
noise sensitivity, stable simple-model rankings at two numerical resolutions,
seed reproducibility and scoring-query purity. Passing these gates does not
prove rankings converge on Location Finding. Crux numerical reporting remains
deterministic and the artifact critic is disabled for scalar forecasts.

The new study's protocol.json records the exact execution source revision.
The local and Modal runners share generation/validation code. A small Modal
preflight checks that integration; it is separate from the six-arm main study.
