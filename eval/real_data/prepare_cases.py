"""Download three public RNA-seq count datasets and convert them into Crux case folders.

    uv run python -m eval.real_data.prepare_cases --out ../data

Writes ``<out>/raw/<dataset>/`` (the downloads, untouched) and ``<out>/cases/<case>/`` with
``tagseq_featurecounts_matrix.tsv``, ``samples.tsv`` and ``SOURCE.md``. Run the diagnostics
on a case with ``python -m diagnostics.de_from_counts.run_all <out>/cases/<case>``. Keep the
data out of the repo. Results and caveats: ``eval/results/real_data_diagnostics.md``.

* covid-lung-colon: Expression Atlas E-ENAD-46, fatal COVID-19 vs normal lung and colon,
  38 samples (CC0 via the Kaggle mirror; cite the study).
* airway-*: GEO GSE52778, 4 airway smooth-muscle donors x untreated/dex/albuterol/both,
  NCBI-generated raw counts (Entrez ids). Two views of the same data.
* pasilla-knockdown: Bioconductor pasilla (GEO GSE18508), Drosophila S2 knockdown vs
  untreated, 7 samples (LGPL).
"""

import argparse
import gzip
import urllib.request
from pathlib import Path

import pandas as pd

COUNTS, SAMPLES = "tagseq_featurecounts_matrix.tsv", "samples.tsv"
DOWNLOADS = {
    "covid-E-ENAD-46": {
        name: f"https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/E-ENAD-46/{name}"
        for name in ("E-ENAD-46-raw-counts.tsv", "E-ENAD-46.condensed-sdrf.tsv")
    },
    "airway-GSE52778": {
        "GSE52778_raw_counts_GRCh38.p13_NCBI.tsv.gz": (
            "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE52778"
            "&format=file&file=GSE52778_raw_counts_GRCh38.p13_NCBI.tsv.gz"
        ),
        "GSE52778_series_matrix.txt.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778/matrix/"
            "GSE52778_series_matrix.txt.gz"
        ),
    },
    "pasilla-GSE18508": {
        name: f"https://raw.githubusercontent.com/bioc/pasilla/devel/inst/extdata/{name}"
        for name in ("pasilla_gene_counts.tsv", "pasilla_sample_annotation.csv")
    },
}


def download(raw: Path) -> None:
    for folder, files in DOWNLOADS.items():
        (raw / folder).mkdir(parents=True, exist_ok=True)
        for name, url in files.items():
            target = raw / folder / name
            if not target.exists():
                print(f"downloading {folder}/{name}")
                urllib.request.urlretrieve(url, target)


def write(
    OUT: Path, name: str, counts: pd.DataFrame, samples: pd.DataFrame, note: str
) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    data_cols = [col for col in counts.columns[1:] if col != "gene_symbol"]
    assert data_cols == list(samples["count_column"]), name
    vals = counts[data_cols]
    assert (vals >= 0).all().all() and (vals == vals.round()).all().all(), (
        f"{name}: non-integer counts"
    )
    counts.to_csv(d / COUNTS, sep="\t", index=False, lineterminator="\n")
    samples.to_csv(d / SAMPLES, sep="\t", index=False, lineterminator="\n")
    (d / "SOURCE.md").write_text(note + "\n", encoding="utf-8")
    groups = samples.groupby(["cell_line", "condition"]).size().to_dict()
    print(f"{name}: {counts.shape[0]} genes x {len(samples)} samples; groups {groups}")


def convert(RAW: Path, OUT: Path) -> None:
    # COVID-19 lung/colon (Expression Atlas E-ENAD-46, CC0 via the Kaggle mirror)
    src = RAW / "covid-E-ENAD-46"
    c = pd.read_csv(src / "E-ENAD-46-raw-counts.tsv", sep="\t")
    c = c.rename(columns={"Gene ID": "Geneid", "Gene Name": "gene_symbol"})
    sd = pd.read_csv(
        src / "E-ENAD-46.condensed-sdrf.tsv",
        sep="\t",
        header=None,
        names=["exp", "x", "run", "kind", "attr", "value", "onto"],
    )
    sd = sd[sd["kind"] == "characteristic"].pivot_table(
        index="run", columns="attr", values="value", aggfunc="first"
    )
    runs = [col for col in c.columns if col.startswith("SRR")]
    s = pd.DataFrame(
        {
            "sample_id": runs,
            "cell_line": [sd.loc[r, "organism part"] for r in runs],
            "condition": [
                "Ctrl" if sd.loc[r, "disease"] == "normal" else "COVID" for r in runs
            ],
            "individual": [sd.loc[r, "individual"] for r in runs],
            "count_column": runs,
        }
    )
    write(
        OUT,
        "covid-lung-colon",
        c[["Geneid", "gene_symbol", *runs]],
        s,
        "Expression Atlas E-ENAD-46 (fatal COVID-19 lung and colon vs normal). Raw counts. CC0 per the Kaggle mirror; cite the original study. "
        "cell_line = tissue (organism part); condition: normal -> Ctrl, COVID-19 -> COVID.",
    )

    # airway (GEO GSE52778, NCBI-generated raw counts on GRCh38.p13, Entrez gene ids)
    src = RAW / "airway-GSE52778"
    c = pd.read_csv(
        src / "GSE52778_raw_counts_GRCh38.p13_NCBI.tsv.gz", sep="\t"
    ).rename(columns={"GeneID": "Geneid"})
    meta = {}
    with gzip.open(src / "GSE52778_series_matrix.txt.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip('"\n') for p in line.split("\t")]
            if parts[0] == "!Sample_geo_accession":
                meta["gsm"] = parts[1:]
            elif parts[0] == "!Sample_characteristics_ch1" and parts[1].split(":")[
                0
            ] in ("treatment", "cell line"):
                meta[parts[1].split(":")[0]] = [p.split(": ", 1)[1] for p in parts[1:]]
    gsm = meta["gsm"]
    treat = {
        "Untreated": "Ctrl",
        "Dexamethasone": "Dex",
        "Albuterol": "Alb",
        "Albuterol_Dexamethasone": "AlbDex",
    }
    cond = [treat[t] for t in meta["treatment"]]
    c["Geneid"] = c["Geneid"].astype(str)
    note = (
        "GEO GSE52778 (airway smooth muscle, 4 donor cell lines x untreated/dex/albuterol/both, 1 sample each). "
        "NCBI-generated raw counts (GRCh38.p13, Entrez gene ids, so no Ensembl versions). "
    )
    write(
        OUT,
        "airway-donor-x-treatment",
        c[["Geneid", *gsm]],
        pd.DataFrame(
            {
                "sample_id": gsm,
                "cell_line": meta["cell line"],
                "condition": cond,
                "count_column": gsm,
            }
        ),
        note
        + "True design: cell_line = donor, condition = treatment; no within-group replicates.",
    )
    write(
        OUT,
        "airway-treatment-groups",
        c[["Geneid", *gsm]],
        pd.DataFrame(
            {
                "sample_id": gsm,
                "cell_line": ["ASM"] * len(gsm),
                "condition": cond,
                "count_column": gsm,
            }
        ),
        note
        + "Replicate view: one background, 4 treatment groups with the 4 donors as replicates (donor is a hidden batch).",
    )

    # pasilla (Bioconductor pasilla, GEO GSE18508, Drosophila, LGPL)
    src = RAW / "pasilla-GSE18508"
    c = pd.read_csv(src / "pasilla_gene_counts.tsv", sep="\t").rename(
        columns={"gene_id": "Geneid"}
    )
    ann = pd.read_csv(src / "pasilla_sample_annotation.csv")
    ann["count_column"] = ann["file"].str.replace("fb$", "", regex=True)
    cols = [col for col in c.columns if col != "Geneid"]
    ann = ann.set_index("count_column").loc[cols].reset_index()
    write(
        OUT,
        "pasilla-knockdown",
        c,
        pd.DataFrame(
            {
                "sample_id": cols,
                "cell_line": ["S2"] * len(cols),
                "condition": [
                    "Ctrl" if x == "untreated" else "KD" for x in ann["condition"]
                ],
                "library_type": ann["type"].tolist(),
                "count_column": cols,
            }
        ),
        "Bioconductor pasilla (GEO GSE18508): Drosophila S2 cells, pasilla RNAi knockdown vs untreated. Gene-level raw counts. LGPL. "
        "Library type (single-read vs paired-end) is a known batch.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("../data"))
    args = parser.parse_args()
    download(args.out / "raw")
    convert(args.out / "raw", args.out / "cases")


if __name__ == "__main__":
    main()
