# Outcome predictor calibration (DU-29)

P(observed bin | true option) on held-out synthetic Task 9 cases. Each case plants zero or one fault; all 12 library diagnostics run on it. Every method scores every diagnostic, with the predictor floor eps = 0.1 applied to all rows. Log loss in nats; lower is better for log loss, Brier and ECE. `option TV` is the mean total-variation distance between a diagnostic's rows under the fork's two options (0 = the predictor ignores the option).

Splits: `test` 200 cases / 2400 records, `shift` 200 cases / 2400 records, `shift:weak_effect` 100 cases / 1200 records, `shift:three_replicates` 100 cases / 1200 records. `test` shares the training knob distribution; `shift` uses knob ranges never seen in training (half weak knockout effect, half three replicates per group).

## All diagnostics

| method | split | n | log loss | Brier | accuracy | ECE | option TV |
|---|---|---|---|---|---|---|---|
| (a) library exact/stated rows | test | 2400 | 0.238 | 0.121 | 0.933 | 0.089 | 0.797 |
| (b) SampledPredictor, gpt-4.1-mini x8 | test | 2400 | 0.302 | 0.133 | 0.933 | 0.068 | 0.776 |
| (c) Qwen3-8B base, logprob scoring | test | 2400 | 0.772 | 0.400 | 0.775 | 0.155 | 0.822 |
| (d) Qwen3-8B LoRA fine-tune | test | 2400 | 0.203 | 0.102 | 0.933 | 0.055 | 0.833 |
| (a) library exact/stated rows | shift | 2400 | 0.245 | 0.128 | 0.910 | 0.077 | 0.797 |
| (b) SampledPredictor, gpt-4.1-mini x8 | shift | 2400 | 0.356 | 0.168 | 0.910 | 0.083 | 0.776 |
| (c) Qwen3-8B base, logprob scoring | shift | 2400 | 0.838 | 0.438 | 0.752 | 0.177 | 0.822 |
| (d) Qwen3-8B LoRA fine-tune | shift | 2400 | 0.235 | 0.129 | 0.910 | 0.072 | 0.833 |
| (a) library exact/stated rows | shift:weak_effect | 1200 | 0.263 | 0.142 | 0.879 | 0.094 | 0.797 |
| (b) SampledPredictor, gpt-4.1-mini x8 | shift:weak_effect | 1200 | 0.428 | 0.215 | 0.879 | 0.113 | 0.776 |
| (c) Qwen3-8B base, logprob scoring | shift:weak_effect | 1200 | 0.909 | 0.480 | 0.724 | 0.205 | 0.822 |
| (d) Qwen3-8B LoRA fine-tune | shift:weak_effect | 1200 | 0.277 | 0.162 | 0.879 | 0.098 | 0.833 |
| (a) library exact/stated rows | shift:three_replicates | 1200 | 0.228 | 0.114 | 0.941 | 0.096 | 0.797 |
| (b) SampledPredictor, gpt-4.1-mini x8 | shift:three_replicates | 1200 | 0.284 | 0.121 | 0.941 | 0.076 | 0.776 |
| (c) Qwen3-8B base, logprob scoring | shift:three_replicates | 1200 | 0.768 | 0.396 | 0.780 | 0.156 | 0.822 |
| (d) Qwen3-8B LoRA fine-tune | shift:three_replicates | 1200 | 0.193 | 0.096 | 0.941 | 0.050 | 0.833 |

## Stated-row forks only (sample_labels, group_direction)

These four diagnostics have no exact rows. Since PR #22 both predictors use the library's stated rows for them by default; with `use_stated_rows=False` TinkerPredictor fills them instead. On the other eight, exact library rows win.

| method | split | n | log loss | Brier | accuracy | ECE | option TV |
|---|---|---|---|---|---|---|---|
| (a) library exact/stated rows | test | 800 | 0.557 | 0.324 | 0.814 | 0.170 | 0.590 |
| (b) SampledPredictor, gpt-4.1-mini x8 | test | 800 | 0.612 | 0.321 | 0.814 | 0.025 | 0.758 |
| (c) Qwen3-8B base, logprob scoring | test | 800 | 0.600 | 0.315 | 0.812 | 0.091 | 0.895 |
| (d) Qwen3-8B LoRA fine-tune | test | 800 | 0.472 | 0.280 | 0.814 | 0.059 | 0.804 |
| (a) library exact/stated rows | shift | 800 | 0.589 | 0.352 | 0.741 | 0.131 | 0.590 |
| (b) SampledPredictor, gpt-4.1-mini x8 | shift | 800 | 0.782 | 0.431 | 0.741 | 0.097 | 0.758 |
| (c) Qwen3-8B base, logprob scoring | shift | 800 | 0.810 | 0.435 | 0.740 | 0.160 | 0.895 |
| (d) Qwen3-8B LoRA fine-tune | shift | 800 | 0.573 | 0.361 | 0.741 | 0.108 | 0.805 |

## Per fork, log loss (test / shift)

| fork | (a) library exact/stated rows | (b) SampledPredictor, gpt-4.1-mini x8 | (c) Qwen3-8B base, logprob scoring | (d) Qwen3-8B LoRA fine-tune |
|---|---|---|---|---|
| sample_labels | 0.363 / 0.402 | 0.340 / 0.398 | 0.302 / 0.386 | 0.282 / 0.346 |
| duplicates | 0.051 / 0.051 | 0.105 / 0.105 | 1.361 / 1.360 | 0.051 / 0.051 |
| sample_quality | 0.140 / 0.118 | 0.171 / 0.155 | 0.140 / 0.118 | 0.094 / 0.086 |
| cell_line | 0.060 / 0.060 | 0.141 / 0.141 | 0.060 / 0.060 | 0.061 / 0.061 |
| gene_ids | 0.065 / 0.065 | 0.171 / 0.171 | 1.870 / 1.870 | 0.065 / 0.065 |
| group_direction | 0.751 / 0.777 | 0.884 / 1.165 | 0.898 / 1.235 | 0.662 / 0.801 |

## Per fork, accuracy (test / shift)

| fork | (a) library exact/stated rows | (b) SampledPredictor, gpt-4.1-mini x8 | (c) Qwen3-8B base, logprob scoring | (d) Qwen3-8B LoRA fine-tune |
|---|---|---|---|---|
| sample_labels | 0.930 / 0.905 | 0.930 / 0.905 | 0.930 / 0.905 | 0.930 / 0.905 |
| duplicates | 1.000 / 1.000 | 1.000 / 1.000 | 0.555 / 0.555 | 1.000 / 1.000 |
| sample_quality | 0.970 / 0.978 | 0.970 / 0.978 | 0.970 / 0.978 | 0.970 / 0.978 |
| cell_line | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| gene_ids | 1.000 / 1.000 | 1.000 / 1.000 | 0.500 / 0.500 | 1.000 / 1.000 |
| group_direction | 0.698 / 0.578 | 0.698 / 0.578 | 0.695 / 0.575 | 0.698 / 0.578 |

Library rows taken literally (no floor): 12 of 2400 test records, 9 of 2400 shift records landed in a bin the row gives probability 0.

## Without the floor (eps = 0), all diagnostics

| method | split | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| (b) SampledPredictor, gpt-4.1-mini x8 | test | 0.279 | 0.126 | 0.933 | 0.029 |
| (c) Qwen3-8B base, logprob scoring | test | 3.197 | 0.435 | 0.775 | 0.212 |
| (d) Qwen3-8B LoRA fine-tune | test | 0.168 | 0.100 | 0.933 | 0.025 |
| (b) SampledPredictor, gpt-4.1-mini x8 | shift | 0.343 | 0.164 | 0.910 | 0.054 |
| (c) Qwen3-8B base, logprob scoring | shift | 3.395 | 0.478 | 0.752 | 0.235 |
| (d) Qwen3-8B LoRA fine-tune | shift | 0.197 | 0.130 | 0.910 | 0.047 |

## Interpretation

On the stated-row forks (sample_labels, group_direction), the only library diagnostics a predictor could fill, the fine-tune (d) has the lowest log loss in distribution: 0.472 on test against 0.557 for the library's stated rows (a), 0.612 for gpt-4.1-mini (b) and 0.600 for base Qwen3-8B (c), with lower ECE (0.059 against 0.170 for (a)). On the shift splits the edge mostly goes away: 0.573 against 0.589 for (a), and with the library floor main now uses (0.02 instead of 0.1) the library rows score 0.560 against 0.569 for (d), so (d) is slightly worse there. It is also worse than (a) on weak knockout effects overall (0.277 against 0.263) and on group_direction under shift (0.801 against 0.777): it learned how often 'unclear' comes up at training effect sizes, and that rate rises when effects are weaker. Argmax accuracy is the same for (a), (b) and (d) on every fork, because every method picks the bin the option names; the gains are in calibration, the mass on 'unclear' or 'other' outcomes that the EIG uses. The base model is overconfident and sometimes ignores the option entirely (duplicates_correlation option TV 0.00, gene_strip_check almost always in the wrong bin: 3.2 nats raw log loss), and fine-tuning fixes that.

Caveats. Everything here is synthetic: train, val and test come from the same simulator, knob ranges and fault appliers, so test is in distribution, and only the shift splits (weaker effects, three replicates) test generalisation, mildly. Real data can differ a lot; PR #22 found PC1 unclear on all four public datasets it tried. Labels exist only for the option that was true in each case, so a row for an option the data contradict is an extrapolation. Where the profile itself reveals a fault (library depth, versioned-ID fraction), the fine-tune partly reads the profile instead of the option: option TV for quality_noise is 0.48 against 0.90 for the library rows. Exact rows win there, so in a run this matters only for options with no library row. These numbers were measured against the library at aae14a5, before PR #22. That PR made cell_line_pc1 stated (as_given 0.62 / mislabelled 0.03 / unclear 0.35) with a new boundary rule, so the cell_line rows above, (a)'s especially, are stale. It also has SampledPredictor use stated rows, so in a sampled run the stated forks get (a)'s rows with a 0.02 floor. The comparison that matters there is (a) against (d). Given the shift result, TinkerPredictor also keeps the stated rows by default (`CRUX_TINKER_OVERRIDE_STATED=1` opts in), so in a run the fine-tune fills only options and experiments the library does not cover, which lie outside its training data.

## Setup

Data: 1,720 synthetic cases from `eval/predictor_data/make_examples.py` (train 1,200, val 120, test 200, shift 200 = 100 weak effect + 100 three replicates), each run through all 12 diagnostics in-process: 20,640 records, none invalid. Knobs per case: genes 2k-12k, depth 0.4M-20M, depth spread, knockout effect sd 0.5-1.2 (shift 0.2-0.45), signature fraction, cell-line and clone effects, dispersion floor 0.01-0.2, shared versus line-specific effects (75% shared, rho 0.4-0.9), knockout target loss, layout and fault severity. A third of the cases are clean; the rest are spread evenly over the six faults and both cell lines. Fine-tune: LoRA rank 16 on Qwen/Qwen3-8B, learning rate 2e-4 with linear decay, batch 64, cross-entropy on the completion tokens only. Early stopping on a 600-example validation subset ended training at step 360 of 675; the best checkpoint, at step 240 (about 1.07 epochs), has validation loss 0.152 nats per example, down from 3.21 at the start. (b) draws 4 independent 8-sample rows per (diagnostic, option) and assigns them to records round-robin, since it never sees the data. Spend (estimated from tokens at list prices): Tinker $5.37 (training $3.01, base scoring $1.18, fine-tune scoring $1.18) and OpenAI $1.05.

**Real-data follow-up (DU-54):** on a real dataset held out from training (COVID, E-ENAD-46, with planted faults), the library rows beat this fine-tune, and training on two other real datasets did not fix that. See [predictor_real_data.md](predictor_real_data.md).
