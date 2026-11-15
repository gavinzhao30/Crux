# Task 9 batch-01: swarm repeat runs

The protocol is in `task9_protocol.md`, committed before this batch ran, at code commit `2d7736a`. Grading used Edison's grader on the 9 required artifacts.

**Change from the protocol:** the 3 baseline runs were stopped about 100 s in, before producing anything, because a teammate had already run the baseline separately. The comparison with the baseline will use his runs once they're posted, after checking that his inputs, prompt, budget and grader match. Only the swarm results are reported here.

| Run | Score | Artifacts ≥ 0.80 | Cost | Model calls | Wall time |
| --- | --- | --- | --- | --- | --- |
| batch-01-swarm-1 | 0.650 | 6/9 | $1.05 | 20 | 508 s |
| batch-01-swarm-2 | 0.576 | 5/9 | $0.94 | 20 | 397 s |
| batch-01-swarm-3 | 0.659 | 6/9 | $1.08 | 20 | 528 s |
| **Mean ± SD** | **0.628 ± 0.046** | | **$1.02** | | |

Together with the first graded run (0.586, `task9_runs.md`), the four swarm runs average 0.618.

| Artifact | swarm-1 | swarm-2 | swarm-3 |
| --- | --- | --- | --- |
| t98g_c10_vs_control_degs | 0.891 | 0.530 | 0.903 |
| t98g_f7_vs_control_degs | 0.890 | 0.817 | 0.906 |
| t98g_g2_vs_control_degs | 0.898 | 0.884 | 0.905 |
| t98g_slow_shared_degs | 0.000 | 0.000 | 0.000 |
| hek293_d10_vs_control_degs | 0.933 | 0.823 | 0.964 |
| hek293_e6_vs_control_degs | 0.954 | 0.898 | 0.964 |
| hek293_shared_degs | 0.000 | 0.000 | 0.000 |
| t98g_d4_vs_control_degs | 0.898 | 0.822 | 0.897 |
| t98g_d4_specific_upregulated_genes | 0.383 | 0.409 | 0.395 |

**Why the two shared-gene lists score 0 in every run:** they are keyed by `gene_symbol`, and our staged inputs had no gene annotation, only the count matrix and sample sheet. The official task inputs include `gencode.v38.annotation.gtf.gz`, which maps Ensembl IDs to symbols, but our staging didn't copy it. Every run's own METHODS.md says the symbols were unavailable and that Ensembl IDs were used instead; it said so before any grading. This is an input gap in our setup, not a reasoning failure. It applies equally to any baseline run on the same staged bundle.
