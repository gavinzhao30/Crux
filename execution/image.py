"""Build the Task 9 analysis image before running agent code."""

import modal


def task9_image(*, minimal: bool = False) -> modal.Image:
    packages = ["r-base=4.5", "bioconductor-deseq2", "r-data.table", "pandas", "numpy"]
    if not minimal:
        packages += [
            "bioconductor-edger",
            "bioconductor-apeglm",
            "scipy",
            "pydeseq2",
            "gseapy",
        ]
    return (
        modal.Image.micromamba(python_version="3.12")
        .micromamba_install(*packages, channels=["conda-forge", "bioconda"])
        .run_commands(
            "mkdir -p /work /outputs /opt/crux",
            "Rscript -e 'suppressPackageStartupMessages(library(DESeq2))'",
            "python -c 'import pandas, numpy" + ("" if minimal else ", pydeseq2") + "'",
        )
    )
