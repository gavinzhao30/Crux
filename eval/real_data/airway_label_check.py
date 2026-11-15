"""Check the albuterol labels in GEO GSE52778 against a genome-wide dexamethasone signature.

    uv run python -m eval.real_data.airway_label_check ../data/cases/airway-donor-x-treatment

The Crux diagnostic ``labels_swap_search`` flagged a swap in the airway data. This checks
it independently of the diagnostic, using only samples whose labels are not in question:

1. Dex signature: the 200 genes most raised and the 200 most lowered by dexamethasone,
   from Dex vs untreated in all four donors (log2 CPM, genes with 10+ reads in 4+ samples).
2. Score each albuterol sample: mean(up) - mean(down) relative to its donor's untreated
   sample. A dex-treated library scores about as high as that donor's Dex sample; a library
   without dex scores near 0.

A sample labelled "albuterol" that scores like Dex, next to an "albuterol + dex" sample
from the same donor that scores near 0, means the two labels are swapped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COUNTS, SAMPLES = "tagseq_featurecounts_matrix.tsv", "samples.tsv"


def dex_scores(case: Path, n_genes: int = 200) -> pd.DataFrame:
    counts = pd.read_csv(case / COUNTS, sep="\t", dtype={"Geneid": str}).set_index(
        "Geneid"
    )
    sheet = pd.read_csv(case / SAMPLES, sep="\t").set_index("sample_id")
    counts = counts[(counts >= 10).sum(axis=1) >= 4]
    log = np.log2(counts / counts.sum() * 1e6 + 1)

    def sample(donor: str, condition: str) -> str:
        hit = sheet[(sheet.cell_line == donor) & (sheet.condition == condition)]
        return hit.index[0]

    donors = sorted(sheet.cell_line.unique())
    lfc = pd.concat(
        [log[sample(d, "Dex")] - log[sample(d, "Ctrl")] for d in donors], axis=1
    ).mean(axis=1)
    up, down = lfc.nlargest(n_genes).index, lfc.nsmallest(n_genes).index
    rows = []
    for d in donors:
        ctrl = log[sample(d, "Ctrl")]

        def score(sample_id: str, ctrl: pd.Series = ctrl) -> float:
            delta = log[sample_id] - ctrl
            return round(float(delta[up].mean() - delta[down].mean()), 2)

        alb, both = sample(d, "Alb"), sample(d, "AlbDex")
        rows.append(
            {
                "donor": d,
                "labelled_alb": f"{alb} {score(alb)}",
                "labelled_alb_dex": f"{both} {score(both)}",
                "dex": score(sample(d, "Dex")),
                "verdict": "labels look swapped"
                if score(alb) > score(both)
                else "labels agree",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "case", type=Path, help="the airway-donor-x-treatment case folder"
    )
    args = parser.parse_args()
    print(dex_scores(args.case).to_string(index=False))


if __name__ == "__main__":
    main()
