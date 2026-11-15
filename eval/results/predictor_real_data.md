# Outcome predictors on real data with planted faults (DU-54)

**Question.** The DU-29 predictor (Qwen3-8B, LoRA on Tinker) beat every other predictor on held-out synthetic cases. Does that hold on real RNA-seq data, and does training on real data help? Here we plant the six faults on three public datasets and hold one dataset out completely: COVID (Expression Atlas E-ENAD-46), which no model was trained on.

**Answer: not yet.** On COVID, the library's hand-set rows are still the best predictor: log loss 0.521 over all forks, against 0.681 for the synthetic-only fine-tune. On the two forks where a model may replace the library's stated rows (sample labels and group direction) it is 0.503 against 0.591. The synthetic advantage (0.472 vs 0.557 on those forks) does not carry over to a real dataset. **The library rows stay the default; nothing in the config changes.**

**Training on real data helped where the real data looks like the test data, and hurt where it doesn't.** Adding airway and pasilla cases to the training set (`new`):
- improved the all-forks log loss on COVID from 0.681 to 0.627, with most of the gain on the cell-line fork (1.453 to 0.099);
- cost nothing on the synthetic sets (0.470 vs 0.472 on the stated forks);
- made the two label forks much worse on COVID (0.591 to 1.095).

The reason is in the data. On airway and pasilla the direction diagnostics answer "unclear" in all 255 cases, and the nearest-neighbour label check mostly answers "other". The model learned that real data gives non-answers on those forks, and carried that to COVID, where the same diagnostics work: `labels_nearest_neighbour` reads "consistent" on 104 of 105 unplanted COVID cases, and `new_clean` gives that bin 0.15. With two real training datasets, the model learned their quirks, not when a diagnostic is informative.

**A second model (`new_clean`) ruled out label noise as the cause.** We had listed the airway label forks as doubtful before scoring (the albuterol labels look swapped in donors N052611 and N080611), but `new` still trained on them. `new_clean` drops those records from training. It was trained after `new` had been scored on COVID, so it is a second look at the test set. It did not fix the label forks (sample labels 0.986 vs 0.720), so noise in the airway labels was not the cause.

## Data

Real cases come from `eval/real_data/prepare_cases.py` (public downloads) and `eval/suites/plant_faults.py`: 5 seeds x 4 instances of each fault the design supports, plus one unplanted case per seed. The records are built by `eval/predictor_data/make_real_examples.py`, all 12 diagnostics per case, with the same record schema and profile code as the synthetic generator; 0 of 4,560 runs failed.

The split is by dataset:
- **COVID:** `real_test`, 125 cases with all six faults.
- **Airway (both views of GSE52778) and pasilla:** `real_train`, 204 cases; the last seed is `real_val`, 51 cases.

Airway cannot carry gene-version faults (numeric Entrez ids); the single-line designs cannot carry a cell-line swap; one airway view has no replicated treated group, so it gets no duplicate or low-depth faults.

The new models train on the 14,400 synthetic DU-29 training records plus the real training records repeated twice, with the same LoRA settings as DU-29: rank 16, learning rate 2e-4, batch 64, early stopping on a validation subset that mixes synthetic and real. `new` used 2,448 real training records. `new_clean` used 1,968: it drops 600 records from the doubtful pairs, counting both training and validation.

| Set | Cases | Records |
|---|---|---|
| `real_test` | 125 | 1500 |
| `real_test_trusted` | 125 | 1250 |
| `real_other` | 255 | 3060 |
| `synthetic_test` | 200 | 2400 |
| `synthetic_shift` | 200 | 2400 |

## COVID, held out from every model: forks where the library has only stated rows

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 500 | 0.503 | 0.285 | 0.718 | 0.315 |
| gpt-4.1-mini sampled x8 | 500 | 0.836 | 0.467 | 0.718 | 0.121 |
| Qwen3-8B base | 500 | 1.005 | 0.513 | 0.718 | 0.214 |
| Fine-tune on synthetic only (DU-29) | 500 | 0.591 | 0.394 | 0.718 | 0.215 |
| Fine-tune on synthetic + real (DU-54) | 500 | 1.095 | 0.625 | 0.528 | 0.431 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 500 | 1.230 | 0.721 | 0.528 | 0.398 |

## COVID, held out from every model: all forks

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 1500 | 0.521 | 0.303 | 0.753 | 0.142 |
| gpt-4.1-mini sampled x8 | 1500 | 0.711 | 0.412 | 0.753 | 0.147 |
| Qwen3-8B base | 1500 | 1.104 | 0.600 | 0.670 | 0.270 |
| Fine-tune on synthetic only (DU-29) | 1500 | 0.681 | 0.394 | 0.753 | 0.146 |
| Fine-tune on synthetic + real (DU-54) | 1500 | 0.627 | 0.351 | 0.760 | 0.203 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 1500 | 0.694 | 0.395 | 0.760 | 0.150 |

## COVID without the doubtful forks: forks where the library has only stated rows

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 500 | 0.503 | 0.285 | 0.718 | 0.315 |
| gpt-4.1-mini sampled x8 | 500 | 0.836 | 0.467 | 0.718 | 0.121 |
| Qwen3-8B base | 500 | 1.005 | 0.513 | 0.718 | 0.214 |
| Fine-tune on synthetic only (DU-29) | 500 | 0.591 | 0.394 | 0.718 | 0.215 |
| Fine-tune on synthetic + real (DU-54) | 500 | 1.095 | 0.625 | 0.528 | 0.431 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 500 | 1.230 | 0.721 | 0.528 | 0.398 |

## COVID without the doubtful forks: all forks

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 1250 | 0.323 | 0.184 | 0.803 | 0.136 |
| gpt-4.1-mini sampled x8 | 1250 | 0.614 | 0.332 | 0.803 | 0.177 |
| Qwen3-8B base | 1250 | 1.022 | 0.540 | 0.703 | 0.235 |
| Fine-tune on synthetic only (DU-29) | 1250 | 0.550 | 0.312 | 0.803 | 0.106 |
| Fine-tune on synthetic + real (DU-54) | 1250 | 0.481 | 0.256 | 0.811 | 0.158 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 1250 | 0.571 | 0.320 | 0.795 | 0.112 |

## Airway and pasilla (in the new model's training data): forks where the library has only stated rows

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 1020 | 1.202 | 0.778 | 0.170 | 0.477 |
| gpt-4.1-mini sampled x8 | 1020 | 2.120 | 1.298 | 0.170 | 0.669 |
| Qwen3-8B base | 1020 | 2.507 | 1.375 | 0.214 | 0.695 |
| Fine-tune on synthetic only (DU-29) | 1020 | 1.757 | 1.148 | 0.170 | 0.655 |
| Fine-tune on synthetic + real (DU-54) | trained on this set | | | | |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | trained on this set | | | | |

## Airway and pasilla (in the new model's training data): all forks

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 3060 | 0.573 | 0.360 | 0.623 | 0.224 |
| gpt-4.1-mini sampled x8 | 3060 | 1.025 | 0.604 | 0.623 | 0.344 |
| Qwen3-8B base | 3060 | 1.430 | 0.779 | 0.561 | 0.371 |
| Fine-tune on synthetic only (DU-29) | 3060 | 0.927 | 0.555 | 0.623 | 0.276 |
| Fine-tune on synthetic + real (DU-54) | trained on this set | | | | |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | trained on this set | | | | |

## Synthetic test (DU-29): forks where the library has only stated rows

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 800 | 0.557 | 0.324 | 0.814 | 0.170 |
| gpt-4.1-mini sampled x8 | 800 | 0.612 | 0.321 | 0.814 | 0.025 |
| Qwen3-8B base | 800 | 0.600 | 0.315 | 0.812 | 0.091 |
| Fine-tune on synthetic only (DU-29) | 800 | 0.472 | 0.280 | 0.814 | 0.059 |
| Fine-tune on synthetic + real (DU-54) | 800 | 0.470 | 0.279 | 0.814 | 0.026 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 800 | 0.479 | 0.288 | 0.815 | 0.102 |

## Synthetic test (DU-29): all forks

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 2400 | 0.275 | 0.143 | 0.933 | 0.116 |
| gpt-4.1-mini sampled x8 | 2400 | 0.302 | 0.133 | 0.933 | 0.068 |
| Qwen3-8B base | 2400 | 0.772 | 0.400 | 0.775 | 0.155 |
| Fine-tune on synthetic only (DU-29) | 2400 | 0.203 | 0.102 | 0.933 | 0.055 |
| Fine-tune on synthetic + real (DU-54) | 2400 | 0.202 | 0.102 | 0.933 | 0.046 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 2400 | 0.206 | 0.105 | 0.933 | 0.070 |

## Synthetic shift (DU-29): forks where the library has only stated rows

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 800 | 0.589 | 0.352 | 0.741 | 0.131 |
| gpt-4.1-mini sampled x8 | 800 | 0.782 | 0.431 | 0.741 | 0.097 |
| Qwen3-8B base | 800 | 0.810 | 0.435 | 0.740 | 0.160 |
| Fine-tune on synthetic only (DU-29) | 800 | 0.573 | 0.361 | 0.741 | 0.108 |
| Fine-tune on synthetic + real (DU-54) | 800 | 0.556 | 0.351 | 0.741 | 0.093 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 800 | 0.529 | 0.325 | 0.743 | 0.044 |

## Synthetic shift (DU-29): all forks

| Method | n | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|
| Library rows (exact or stated) | 2400 | 0.282 | 0.150 | 0.910 | 0.093 |
| gpt-4.1-mini sampled x8 | 2400 | 0.356 | 0.168 | 0.910 | 0.083 |
| Qwen3-8B base | 2400 | 0.838 | 0.438 | 0.752 | 0.177 |
| Fine-tune on synthetic only (DU-29) | 2400 | 0.235 | 0.129 | 0.910 | 0.072 |
| Fine-tune on synthetic + real (DU-54) | 2400 | 0.229 | 0.125 | 0.910 | 0.068 |
| Fine-tune on synthetic + real, doubtful forks dropped (DU-54) | 2400 | 0.221 | 0.117 | 0.909 | 0.051 |

## Per fork on COVID (log loss)

| Fork | Library rows (exact or stated) | gpt-4.1-mini sampled x8 | Qwen3-8B base | Fine-tune on synthetic only (DU-29) | Fine-tune on synthetic + real (DU-54) | Fine-tune on synthetic + real, doubtful forks dropped (DU-54) |
|---|---|---|---|---|---|---|
| sample_labels | 0.259 | 0.195 | 0.096 | 0.120 | 0.720 | 0.986 |
| duplicates | 0.051 | 0.105 | 1.287 | 0.051 | 0.051 | 0.051 |
| sample_quality | 1.512 | 1.195 | 1.512 | 1.335 | 1.361 | 1.309 |
| cell_line | 0.494 | 1.124 | 1.460 | 1.453 | 0.099 | 0.262 |
| gene_ids | 0.065 | 0.171 | 0.354 | 0.065 | 0.065 | 0.080 |
| group_direction | 0.748 | 1.478 | 1.914 | 1.061 | 1.469 | 1.475 |

## How often a fork lands in its fault bin with no fault planted

Doubtful pairs, fixed before scoring: `covid/sample_quality` (library SRR12816735 is low complexity); `airway/sample_labels` (albuterol labels look swapped in donors N052611, N080611); `airway/group_direction` (albuterol labels look swapped in donors N052611, N080611).

| Dataset | sample_labels | duplicates | sample_quality | cell_line | gene_ids | group_direction |
|---|---|---|---|---|---|---|
| airway | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| covid | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| pasilla | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Caveats:**
- **One held-out dataset.** COVID's 125 cases come from one study, re-planted over seeds, so they are far from independent.
- **Planted faults only.** The faults are planted by our own code, and the unplanted forks of a public dataset can have real problems. The doubtful pairs are listed above, and the table of unplanted fault-bin rates checks them: COVID's sample-quality fork is in its fault bin half the time with no fault planted, and airway's sample-labels fork a third of the time.
- **Shared gpt-4.1-mini rows.** They come from DU-29's cache: they never see the data, so they are the same rows.

**What would help next:** many more real datasets with leave-one-dataset-out evaluation, so the model learns which data-profile facts make a diagnostic informative instead of each dataset's base rates. Until then, the fine-tune should not replace the library rows on real data.

## Spend

Estimated from tokens at list prices: Tinker scoring (this report) $5.29, Tinker training (new model) $4.64, Tinker training (new_clean model) $3.00.
