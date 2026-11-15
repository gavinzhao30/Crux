# BixBench3 task 9: swarm runs, officially graded

**Protocol:** the downstream pilot, with the published count matrix as input (Edison's artifact 1) and the 9 required artifacts graded by Edison's deterministic grader at commit `d0e0bbb`. This is one task out of 20, with counts as input, so it is **not** comparable to the 0.523 full-suite record, which starts from raw data. The fair comparison is the single-agent baseline (DU-22) on the same input, budget and grader.

**Where everything lives:**
- **Inputs:** staged on Modal from Edison's public buckets (bundle `160253fe…`).
- **References:** in evaluator-only storage, outside the repo and every run folder. The grader self-test scores the references 1.0 on all 9 artifacts.
- **Prompt:** the fallback prompt from `load_prompt`. The official `agent/prompt.md` is in the gated HF dataset, which we can't read yet.

## Run 1: r-20261114-175926-6050 (Nov 14)

The full swarm: 3 proposers plus the DU-42 library, the sampled predictor, the referee, the gpt-5.5 deliverer and critic, and a Modal sandbox. It cost $1.03, 20 model calls and 293 s.

- **Questions:** 18 proposed. The 5 library forks settled on real evidence: no swapped labels, duplicates, degraded libraries or cell-line errors, and uniform gene IDs.
- **Unresolved:** the 12 questions the proposers raised (design formula, pre-filtering, how shared and D4-specific sets are defined, and so on) went unresolved. The designer stand-in has no diagnostics for them (DU-10).
- **Critic:** it blocked versioned gene IDs in 7 DE tables. One repair round fixed all but one `_PAR_Y` id in 4 tables.

| Artifact | Score | Checks |
| --- | --- | --- |
| t98g_c10_vs_control_degs | 0.549 | F1 0.549 · CCC 1.0 |
| t98g_f7_vs_control_degs | 0.847 | F1 0.847 · CCC 1.0 |
| t98g_g2_vs_control_degs | 0.890 | F1 0.890 · CCC 1.0 |
| t98g_slow_shared_degs | 0.000 | F1 0 · label F1 0 |
| hek293_d10_vs_control_degs | 0.863 | F1 0.863 · CCC 1.0 |
| hek293_e6_vs_control_degs | 0.844 | F1 0.844 · CCC 1.0 |
| hek293_shared_degs | 0.000 | F1 0 · label F1 0 |
| t98g_d4_vs_control_degs | 0.847 | F1 0.847 · CCC 1.0 |
| t98g_d4_specific_upregulated_genes | 0.436 | F1 0.436 |
| **Mean** | **0.586** | **5 of 9 at ≥ 0.80** |

**What the scores say:**
- The fold changes are exactly right (CCC 1.0 everywhere). The significant-gene sets differ, which points to the unresolved DESeq2 design and pre-filtering choices.
- The two shared-gene lists score 0. Their definition question was unresolved and the fallback prompt doesn't define them. The official prompt, or a designer that can test definitions, is the fix.
