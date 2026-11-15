"""Gene IDs, check 1: the share of Ensembl gene IDs that carry a version suffix.

Counts Ensembl gene IDs (``ENSG00000141510`` and the other species' ``ENS...G`` prefixes)
with and without a ``.N`` version (``_PAR_Y`` style tails allowed). A matrix in one
namespace is all versioned or all unversioned; a mixture means some IDs were edited.
"""

from _common import emit, gene_column, read_matrix

ENSEMBL = r"^ENS[A-Z]*G\d+"
VERSIONED = r"^ENS[A-Z]*G\d+\.\d+(?:_[A-Za-z0-9_]+)?$"


def main():
    frame = read_matrix()
    genes = frame[gene_column(frame)].astype(str).str.strip()
    ensembl = genes.str.match(ENSEMBL)
    versioned = ensembl & genes.str.match(VERSIONED)
    unversioned = ensembl & ~versioned
    n_ens, n_ver, n_unver = int(ensembl.sum()), int(versioned.sum()), int(unversioned.sum())
    minority = versioned if 0 < n_ver <= n_unver else unversioned
    return {
        "n_genes": len(genes),
        "n_ensembl": n_ens,
        "n_versioned": n_ver,
        "n_unversioned": n_unver,
        "frac_versioned": n_ver / n_ens if n_ens else 0.0,
        "n_minority": min(n_ver, n_unver),
        "minority_examples": genes[minority].head(5).tolist() if min(n_ver, n_unver) else [],
        "example_ids": genes.head(3).tolist(),
        "flagged_samples": [],
    }


if __name__ == "__main__":
    emit(main())
