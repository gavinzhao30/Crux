"""Gene IDs, check 2: what stripping version suffixes would change.

Strips a trailing ``.N`` version (keeping ``_PAR_Y`` style tails) from every ID. Reports how
many IDs change and how many do not, among IDs that look like stable gene IDs, and how many
stripped IDs collide with another row. One namespace means either every ID changes or none
does; collisions mean the matrix holds two rows for one gene.
"""

import re

from _common import emit, gene_column, read_matrix

STABLE_ID = re.compile(r"^[A-Za-z]+\d{5,}")
VERSION = re.compile(r"\.\d+(?=$|_)")


def main():
    frame = read_matrix()
    genes = frame[gene_column(frame)].astype(str).str.strip().tolist()
    stable = [g for g in genes if STABLE_ID.match(g)]
    stripped = [VERSION.sub("", g, count=1) for g in stable]
    changed = sum(a != b for a, b in zip(stable, stripped))
    unchanged = len(stable) - changed
    collisions = len(stripped) - len(set(stripped))
    seen, colliding = set(), []
    for g in stripped:
        if g in seen and len(colliding) < 5:
            colliding.append(g)
        seen.add(g)
    return {
        "n_stable_ids": len(stable),
        "n_changed": changed,
        "n_unchanged": unchanged,
        "n_minority": min(changed, unchanged),
        "n_collisions": collisions,
        "collision_examples": colliding,
        "flagged_samples": [],
    }


if __name__ == "__main__":
    emit(main())
