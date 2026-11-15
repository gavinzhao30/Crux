# Task 9 outcome calibration (DU-31)

Observed bins come from executing the library diagnostics locally on the real-count cases. The matched comparison uses diagnostics with library-stated rows and conditions on the evaluator's known option. All arms use the same 0.1 probability floor. Lower log loss and Brier are better.

Stated probability tables are available for `cell_line_pc1`, `direction_marker`, `direction_signature`, `labels_nearest_neighbour`, `labels_swap_search`.

Offline rankings replay the first choice within each eligible case and fork, with the same priors, candidate costs and minimum EIG. They count changes relative to Stated, including decisions to skip every candidate.

## Dev

48 diagnostic executions, 0 invalid, 20 matched with stated rows.

| Predictor | n | Log loss | Brier | ECE | First choices changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stated | 20 | 0.5084 | 0.2886 | 0.3079 | 0/8 |
| Sampled | 20 | 0.7937 | 0.4685 | 0.2268 | 4/8 |
| Jev | 0 | — | — | — | — |

| Predictor | Confidence bucket | n | Mean confidence | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Stated | 0.4–0.5 | 4 | 0.483 | 0.000 |
| Stated | 0.5–0.6 | 8 | 0.560 | 0.875 |
| Stated | 0.7–0.8 | 7 | 0.779 | 1.000 |
| Stated | 0.8–0.9 | 1 | 0.843 | 1.000 |
| Sampled | 0.4–0.5 | 1 | 0.460 | 1.000 |
| Sampled | 0.5–0.6 | 1 | 0.554 | 0.000 |
| Sampled | 0.6–0.7 | 3 | 0.649 | 1.000 |
| Sampled | 0.7–0.8 | 2 | 0.744 | 0.000 |
| Sampled | 0.8–0.9 | 13 | 0.839 | 0.769 |

## Test

108 diagnostic executions, 0 invalid, 45 matched with stated rows.

| Predictor | n | Log loss | Brier | ECE | First choices changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stated | 45 | 0.5948 | 0.3352 | 0.2877 | 0/18 |
| Sampled | 45 | 0.8414 | 0.4901 | 0.1549 | 9/18 |
| Jev | 0 | — | — | — | — |

| Predictor | Confidence bucket | n | Mean confidence | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Stated | 0.4–0.5 | 9 | 0.483 | 0.000 |
| Stated | 0.5–0.6 | 16 | 0.556 | 0.938 |
| Stated | 0.7–0.8 | 19 | 0.772 | 0.895 |
| Stated | 0.8–0.9 | 1 | 0.843 | 1.000 |
| Sampled | 0.6–0.7 | 9 | 0.649 | 0.778 |
| Sampled | 0.7–0.8 | 4 | 0.744 | 0.250 |
| Sampled | 0.8–0.9 | 32 | 0.839 | 0.719 |

- Stated — Library-authored approximate probability tables
- Sampled — gpt-4.1-mini, eight draws per option, 0 fresh calls, $0.000 estimated
- Jev — Pending after TypeSafe HTTP 401

The three-way decision requires full matched held-out coverage.
