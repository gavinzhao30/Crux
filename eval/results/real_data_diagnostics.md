# The DE diagnostics on real public data (DU-48)

Until now the 12 diagnostics in `diagnostics/de_from_counts/` were tuned on synthetic counts only. This is their first run on real data: three public datasets, which Oriental Magic found and vetted on DU-48, converted into four cases.

Reproduce:

```
uv run python -m eval.real_data.prepare_cases --out ../data
uv run python -m diagnostics.de_from_counts.run_all ../data/cases/<case>
uv run python -m eval.real_data.airway_label_check ../data/cases/airway-donor-x-treatment
```

The data stays out of the repo.

## Datasets

| Case | Source | Design | Notes |
| --- | --- | --- | --- |
| `covid-lung-colon` | Expression Atlas E-ENAD-46 | 58,735 genes × 38 samples. Lung and colon (as `cell_line`) × normal / fatal COVID-19, 9–10 per group | Post-mortem tissue. CC0 through the Kaggle mirror |
| `airway-treatment-groups` | GEO GSE52778, NCBI raw counts | 39,376 genes × 16 samples. 4 treatments (untreated, dex, albuterol, both) with the 4 donors as replicates | Donor is a hidden batch. Entrez IDs |
| `airway-donor-x-treatment` | same data | donor (as `cell_line`) × treatment, 1 sample per group | The true design. No within-group replicates |
| `pasilla-knockdown` | Bioconductor pasilla, GEO GSE18508 | 14,599 genes × 7 samples. Drosophila S2, 4 untreated vs 3 knockdown | Library type (single-read or paired-end) is a batch. LGPL |

## Outcomes

These are the outcomes after the PC1 fix described below. In the table, ⚠ marks a confident call that differs from what clean data should give, and · marks an honest abstention (`unclear` or `other`).

| Diagnostic (clean outcome) | COVID lung/colon | Airway, treatment groups | Airway, donor × treatment | Pasilla |
| --- | --- | --- | --- | --- |
| sample_labels / `labels_swap_search` (as_given) | as_given | **swapped ⚠** | unclear · | as_given |
| sample_labels / `labels_nearest_neighbour` (consistent) | consistent | other · | consistent | other · |
| duplicates / `duplicates_identical` (none) | none | none | none | none |
| duplicates / `duplicates_correlation` (none) | none | none | none | none |
| sample_quality / `quality_depth` (all_usable) | **degraded ⚠** | all_usable | all_usable | all_usable |
| sample_quality / `quality_noise` (all_usable) | all_usable | all_usable | all_usable | all_usable |
| cell_line / `cell_line_centroid` (as_given) | as_given | as_given | as_given | as_given |
| cell_line / `cell_line_pc1` (as_given) | unclear · | unclear · | unclear · | unclear · |
| gene_ids / `gene_versions` (uniform) | uniform | uniform | uniform | uniform |
| gene_ids / `gene_strip_check` (uniform) | uniform | uniform | uniform | uniform |
| group_direction / `direction_signature` (consistent) | consistent | unclear · | unclear · | unclear · |
| group_direction / `direction_marker` (consistent) | unclear · | unclear · | unclear · | unclear · |
| **Lands where clean data should** | 9/12 | 7/12 | 8/12 | 8/12 |

## What the confident calls were

### 1. Airway: the albuterol labels look swapped in two of four donors (a real finding)

`labels_swap_search` flagged GSM1275868 (labelled N052611 albuterol) and GSM1275873 (labelled N080611 albuterol + dex). An independent check uses only the samples whose labels aren't in question. It builds a genome-wide dexamethasone signature (200 up and 200 down genes from Dex vs untreated in all four donors) and scores each albuterol sample against its donor's untreated sample (`eval/real_data/airway_label_check.py`):

| Donor | Labelled albuterol | Labelled albuterol + dex | Dex, for scale |
| --- | --- | --- | --- |
| N052611 | GSM1275868: **2.86** | GSM1275869: **0.36** | 3.39 |
| N061011 | GSM1275876: 0.00 | GSM1275877: 3.47 | 3.72 |
| N080611 | GSM1275872: **3.11** | GSM1275873: **0.55** | 3.21 |
| N61311 | GSM1275864: −0.14 | GSM1275865: 3.16 | 3.31 |

In N052611 and N080611, the library labelled albuterol carries the full dexamethasone response, and the one labelled albuterol + dex has none. The canonical glucocorticoid targets agree: FKBP5, ZBTB16, TSC22D3, PER1 and KLF15 are high in the "albuterol" libraries of those two donors and low in their "albuterol + dex" libraries.

So in two donors the albuterol and albuterol + dex labels are most likely swapped. That could be in the GEO sample records, or in how runs were mapped to samples. I found no public report of it. The widely used Bioconductor `airway` package uses only the untreated and dex samples, so it never touches these labels.

The diagnostic's single-swap model found one cross-donor exchange that explains part of the problem. The underlying error is two within-donor swaps, so delivery should relabel by donor and not apply that exchange literally.

### 2. COVID: SRR12816735 is a low-complexity library (real, borderline)

`quality_depth` flagged SRR12816735, a fatal-COVID lung sample. It detects 14,041 genes against a median of 28,805 (49%, just under the 50% threshold), and its correlation with every other library is low (Spearman about 0.69, against about 0.8 for its peers). Depth itself is fine, at 53% of the median.

For post-mortem lung tissue, this looks like real RNA degradation, not a false alarm. `quality_noise` passes it, so in the loop the two diagnostics disagree and the question stays open, which is the right outcome for a borderline library.

### 3. COVID: `cell_line_pc1` called two lung samples colon (a false alarm, now fixed)

Before the fix, PC1 split the libraries with separation 5.05 and put SRR12816720 and SRR12816735 (both lung) on the colon side. Both are slightly closer to the lung samples, and the centroid check correctly said the labels are as given. In damaged lung tissue, PC1 picks up disease and quality rather than tissue identity.

**Fix:** a mismatched library now counts only when it sits deep in the other group, with a PC1 score at least 0.75 of that group's mean. Libraries nearer the boundary are reported as `boundary_samples` and make the check `unclear`. The two COVID samples sat at 0.06 and 0.54. Planted cell-line swaps on synthetic data sit at 0.97–1.0 (24 of 24 still caught).

PC1's likelihood rows changed from exact to measured, because "unclear" happens under either option on real data. The predictor now floors every library row, so an "unclear" outcome can't reject every real option at once.

## What the abstentions mean

- **The direction checks** need a perturbation signature shared across two cell lines. None of these datasets has one, so `unclear` is the right answer. Their rows already treat `unclear` as uninformative.
- **`labels_nearest_neighbour` = `other`:**
  - On pasilla, the library-type batch moves libraries without a clean swap.
  - On airway treatment groups, the real swap shows up this way, since two swaps can't be fixed by one exchange.
- **PC1 `unclear` on all four:** tissues and donors with strong secondary structure don't split cleanly on PC1. The centroid check carries the cell-line question on real data.

## Limits

- **Not a detection rate.** This shows how the diagnostics behave on real data with no planted faults. Detection on real counts needs the generic fault planter (DU-49).
- **No dataset matches task 9's shape**, a replicated perturbation in two or more cell lines. The ELEM-1200RS candidate on DU-48 is gated and needs a subset screen.
- **Unconfirmed:** the airway swap comes from the data alone. The dataset's authors haven't confirmed it.
