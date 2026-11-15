# Location Finding study (DU-41)

The environment is ported from Kanishk Gandhi's MIT-licensed BoxingGym,
`src/boxing_gym/envs/location_finding.py` at
`5df3d32cfa7e4de99b43bb4d5328e888cc0cb7b9`. Its license is retained alongside
the port. Hidden sources and noiseless evaluation answers are evaluator-only.

Run from the repository root:

```shell
uv run python -m eval.boxing_study --arms random,grid,oracle --seeds 20 --output eval/results/boxing/NEW_PILOT
uv run --env-file .env python -m eval.boxing_study --arms random,grid,oracle,single_agent,crux_random,crux_eig --seed-start 100 --seeds 20 --max-usd 5 --output eval/results/boxing/NEW_STUDY
uv run python -m scripts.audit_boxing_study eval/results/boxing/main-20261114
```

Output directories must be fresh. Episodes checkpoint independently; faults and
partial checkpoints are preserved. The simple `run_episode` function is callable
by DU-23 without invoking the CLI. Modal fan-out belongs to DU-23, not this runner.

## Frozen protocol

- Pilot seeds: 0–19. Main study seeds: 100–119, paired across all six arms.
- Three 2-D sources drawn from standard Normal; background .1, softening 1e-4,
  strength 1, observation SD .5, two-decimal measurement rounding.
- Ten held-out standard-Normal coordinates per seed, scored against noiseless
  intensity. Independent streams generate sources, noise and held-out points.
- One 20-measurement trajectory per arm/seed, with checkpoints at 5/10/20.
  Single-agent evaluation requests run after all designs and use frozen history
  prefixes, so held-out coordinates cannot affect subsequent measurement choices.
- Random uses Uniform(-2,2) designs. Grid uses a fixed evenly spaced sequence on
  a 7x7 grid. Oracle and Crux choose from that same 7x7 grid; Crux random and EIG
  have identical candidate sets. Single-agent coordinates are bounded to [-2,2].
- All model calls use gpt-4.1-mini with identical wrapper settings. Source-model
  hypotheses are generated once per seed and shared unchanged by the two Crux
  arms. They are validated/executed in the existing model sandbox. Crux runs the
  real numeric orchestrator, with parameter+family information scoring and no
  early stop at family settlement. Numeric reporting is deterministic; artifact
  criticism is disabled because this study produces scalar forecasts.
- Random, grid and oracle use identical known-structure particle inference;
  only their design policies differ. Oracle has the true functional structure
  and prior, never the actual source coordinates. Its EIG is a finite Monte Carlo
  estimate (256 particles, 16 simulated outcomes), not an exact mathematical
  oracle. Inference uses a continuous Normal approximation to rounded observations.
  Particle collapse and finite-grid error can limit this numerical reference.
  Per-step effective sample sizes and parameter spreads are saved to expose
  collapse; effective sample size alone can look healthy after resampling.
- The single agent chooses measurements and forecasts directly in text/structured
  responses. Comparing it with Crux compares systems with different inference
  representations. Only Crux EIG versus Crux random isolates selection.
- MSE, median MSE, mean log10 MSE, and bootstrap 95% CIs are reported. Resampling
  units are whole seeds (2,000 replicates). A separate paired log-error contrast
  reports EIG minus random, where negative is better.
- Failed/missing checkpoints get MSE penalty 1e12, explicitly marked failed;
  they remain in every aggregate denominator. Log error has a 1e-12 floor.
  Target MSE is 1, fixed before the main study. First target achievement is only
  resolved at evaluated checkpoints; a blank value is censored, not success.
- Per-episode costs count calls actually made. Shared proposal-generation usage
  is reported separately for both Crux arms, and counted once in actual study
  usage. Costs use configured list prices, not an invoice. A conservative request
  estimate is checked before each paid call against the $5 accounted-spend cap;
  output is limited to 2,048 tokens. Provider-billed errors without returned usage
  cannot be fully accounted for. Credentials remain in ignored .env only.
- Model randomness is not guaranteed reproducible from a seed. Case sources,
  held-out points, simulator noise and inference RNG are seeded and reproducible.

The mandatory deliverables are summary/paired comparison JSON, per-seed CSV,
cost CSV/JSON, charts and episode logs under `eval/results/boxing/`. No claim of
EIG superiority is made without the paired results supporting it. This is a
simulator study, not evidence of scientific discovery. The second environment
remains an optional stretch item. The initial pilot chart missed the historical
Saturday 6 PM milestone; its actual creation time is not backdated.
