"""HBB reference, the known-variant library and the standard genetic code for One Letter.

Every sequence here is cited; nothing is invented (docs: tools/ONE_LETTER_CONTRACT.md).

* HBB coding sequence: NCBI RefSeq NM_000518.5 (CDS 51..494). Its first 27 nucleotides are
  ``ATG GTG CAT CTG ACT CCT GAG GAG AAG``. HGVS numbers codons from the Met start codon;
  the legacy haemoglobin numbering leaves Met out, so legacy codon 6 is nt 19-21 (``GAG``,
  Glu) and HGVS calls it codon 7.
* HbS: c.20A>T, dbSNP rs334, p.Glu7Val (legacy Glu6Val). Window 3 reads ``GTGGAGAAG``.
* HbC: c.19G>A, dbSNP rs33930165, p.Glu7Lys (legacy Glu6Lys). Window 3 reads ``AAGGAGAAG``.
* Structures, checked on the RCSB REST API (``https://data.rcsb.org/rest/v1/core/entry/<ID>``)
  on 2026-11-14: 2HBS "The high resolution crystal structure of deoxyhemoglobin S" (X-ray,
  2.05 A, beta Val6); 1A3N "Deoxy human hemoglobin" (X-ray, 1.8 A), the normal-haemoglobin
  entry used for HbA, HbC and unlisted variants. Residue 6 is on the beta chains.

The strip is 27 nt in three 9-nt windows: W1 = nt 1-9, W2 = nt 10-18, W3 = nt 19-27.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REFSEQ = "NM_000518.5"
REFERENCE = "ATGGTGCATCTGACTCCTGAGGAGAAG"  # NM_000518.5 CDS nt 1-27
WINDOW_LEN = 9
WINDOWS: dict[int, tuple[int, int]] = {
    1: (1, 9),
    2: (10, 18),
    3: (19, 27),
}  # 1-based, inclusive
CODON6_NT = (19, 21)  # legacy codon 6 (HGVS codon 7)
BASES = "ACGT"

PDB_HBS = "2HBS"
PDB_NORMAL = "1A3N"
# Anything farther than this from the reference window is not treated as a variant: a strip
# that differs at more positions than one codon holds is a wrong strip or an unreadable one.
MAX_NOVEL_MISMATCHES = 3

# Standard genetic code (NCBI translation table 1), codons in TCAG order.
_AA1 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_ORDER = "TCAG"
CODON_TABLE: dict[str, str] = {
    a + b + c: _AA1[16 * i + 4 * j + k]
    for i, a in enumerate(_ORDER)
    for j, b in enumerate(_ORDER)
    for k, c in enumerate(_ORDER)
}
AA3: dict[str, str] = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "E": "Glu",
    "Q": "Gln",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "*": "Ter",
}


def translate(codon: str) -> str:
    """Three-letter amino acid for a DNA codon (``Ter`` for a stop)."""
    return AA3[CODON_TABLE[codon.upper()]]


def window_of(sequence: str, window: int) -> str:
    start, end = WINDOWS[window]
    return sequence[start - 1 : end]


def reference_window(window: int) -> str:
    return window_of(REFERENCE, window)


def is_bases(text: object) -> bool:
    """True for a 9-letter string over A, C, G and T."""
    return isinstance(text, str) and len(text) == WINDOW_LEN and set(text) <= set(BASES)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b, strict=True))


def mismatches(a: str, b: str) -> list[int]:
    """0-based positions where ``a`` and ``b`` differ."""
    return [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]


@dataclass(frozen=True)
class Change:
    """One codon that differs from the reference."""

    hgvs_codon: int  # counted from Met = 1
    ref_codon: str
    alt_codon: str
    substitutions: tuple[tuple[int, str, str], ...]  # (c. position, ref base, alt base)

    @property
    def legacy_codon(self) -> int:
        return self.hgvs_codon - 1

    @property
    def ref_aa(self) -> str:
        return translate(self.ref_codon)

    @property
    def alt_aa(self) -> str:
        return translate(self.alt_codon)

    @property
    def legacy(self) -> str:  # Glu6Lys; Glu6= for a synonymous change
        alt = "=" if self.alt_aa == self.ref_aa else self.alt_aa
        return f"{self.ref_aa}{self.legacy_codon}{alt}"

    @property
    def protein(self) -> str:  # p.Glu7Lys
        alt = "=" if self.alt_aa == self.ref_aa else self.alt_aa
        return f"p.{self.ref_aa}{self.hgvs_codon}{alt}"


def changes(sequence: str) -> list[Change]:
    """Codon-level differences between a 27-nt strip sequence and the reference."""
    out = []
    for c0 in range(0, len(REFERENCE), 3):
        ref, alt = REFERENCE[c0 : c0 + 3], sequence[c0 : c0 + 3]
        if ref != alt:
            subs = tuple(
                (c0 + i + 1, r, a) for i, (r, a) in enumerate(zip(ref, alt, strict=True)) if r != a
            )
            out.append(Change(c0 // 3 + 1, ref, alt, subs))
    return out


def hgvs_c(chs: list[Change]) -> str:
    subs = [s for ch in chs for s in ch.substitutions]
    parts = [f"{pos}{ref}>{alt}" for pos, ref, alt in subs]
    if not parts:
        return f"{REFSEQ}:c.="
    if len(parts) == 1:
        return f"{REFSEQ}:c.{parts[0]}"
    return f"{REFSEQ}:c.[{';'.join(parts)}]"


@dataclass(frozen=True)
class Variant:
    """One explanation of the strip: what it predicts for each window and what to show."""

    key: str
    hypothesis_id: str
    name: str
    hgvs: str
    legacy: str
    rsid: str
    codon: str  # legacy codon 6 on this sequence
    amino_acid: str  # three-letter residue 6
    sequence: str  # the 27 nt the strip carries under this hypothesis
    pdb_id: str
    known: bool
    source: str

    def window(self, window: int) -> str:
        return window_of(self.sequence, window)

    @property
    def short(self) -> str:
        """Display name with the legacy change, e.g. ``HbS (sickle), Glu6Val``."""
        return self.name if self.key == "hba" else f"{self.name}, {self.legacy}"

    def statement(self) -> str:
        if self.key == "hba":
            return (
                f"HbA (normal): codon 6 is GAG (Glu), as in {REFSEQ}; "
                f"window 3 reads {self.window(3)}"
            )
        ids = ", ".join(x for x in (self.hgvs.split(":", 1)[-1], self.rsid) if x)
        return (
            f"{self.name}: {ids}, {self.legacy}; codon 6 is {self.codon} ({self.amino_acid}); "
            f"window 3 reads {self.window(3)}"
        )

    def dna_variant(self) -> dict:
        """The ``dna.variant`` payload (contract fields, plus ``source``)."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "hgvs": self.hgvs,
            "legacy": self.legacy,
            "rsid": self.rsid,
            "codon": self.codon,
            "amino_acid": self.amino_acid,
            "source": self.source,
        }


def _point(
    key: str,
    hid: str,
    name: str,
    pos: int,
    ref: str,
    alt: str,
    rsid: str,
    pdb: str,
    source: str,
) -> Variant:
    if REFERENCE[pos - 1] != ref:
        raise AssertionError(f"{name}: reference base at c.{pos} is not {ref}")
    seq = REFERENCE[: pos - 1] + alt + REFERENCE[pos:]
    chs = changes(seq)
    codon = window_of(seq, 3)[:3]
    return Variant(
        key=key,
        hypothesis_id=hid,
        name=name,
        hgvs=f"{hgvs_c(chs)} ({chs[0].protein})",
        legacy=chs[0].legacy,
        rsid=rsid,
        codon=codon,
        amino_acid=translate(codon),
        sequence=seq,
        pdb_id=pdb,
        known=True,
        source=source,
    )


HBA = Variant(
    key="hba",
    hypothesis_id="h-hba",
    name="HbA (normal)",
    hgvs=f"{REFSEQ}:c.=",
    legacy="Glu6 (normal)",
    rsid="",
    codon=REFERENCE[18:21],
    amino_acid=translate(REFERENCE[18:21]),
    sequence=REFERENCE,
    pdb_id=PDB_NORMAL,
    known=True,
    source=f"NCBI RefSeq {REFSEQ} (CDS 51..494), nt 1-27",
)
HBS = _point(
    "hbs",
    "h-hbs",
    "HbS (sickle)",
    20,
    "A",
    "T",
    "rs334",
    PDB_HBS,
    f"HBB c.20A>T, dbSNP rs334, on NCBI RefSeq {REFSEQ}",
)
HBC = _point(
    "hbc",
    "h-hbc",
    "HbC",
    19,
    "G",
    "A",
    "rs33930165",
    PDB_NORMAL,
    f"HBB c.19G>A, dbSNP rs33930165, on NCBI RefSeq {REFSEQ}",
)

VARIANTS: dict[str, Variant] = {v.key: v for v in (HBA, HBS, HBC)}
LIBRARY: tuple[Variant, ...] = (HBS, HBC)  # what regeneration may propose
START_KEYS: tuple[str, ...] = ("hba", "hbs")

assert window_of(REFERENCE, 1) == "ATGGTGCAT" and window_of(REFERENCE, 2) == "CTGACTCCT"
assert HBA.window(3) == "GAGGAGAAG" and HBS.window(3) == "GTGGAGAAG"
assert HBC.window(3) == "AAGGAGAAG"
assert HBS.legacy == "Glu6Val" and HBC.legacy == "Glu6Lys"


def strip_sequence(window: int, bases: str) -> str:
    """The 27-nt sequence that reads ``bases`` in ``window`` and the reference elsewhere."""
    start, end = WINDOWS[window]
    return REFERENCE[: start - 1] + bases + REFERENCE[end:]


def match_library(window: int, bases: str, exclude: set[str] = frozenset()) -> Variant | None:
    """The library variant (not in ``exclude``, by key) that predicts ``bases`` in ``window``."""
    for v in LIBRARY:
        if v.key not in exclude and v.window(window) == bases:
            return v
    return None


def novel_variant(window: int, bases: str) -> Variant | None:
    """Name an unlisted change from the reads, or ``None`` when it is too far from HBB.

    The name comes from translating the observed codons with the standard genetic code; it
    has no dbSNP id because it is not in the library. It is reported as an unknown variant.
    """
    if hamming(bases, reference_window(window)) > MAX_NOVEL_MISMATCHES:
        return None
    seq = strip_sequence(window, bases)
    chs = changes(seq)
    if not chs:
        return None
    legacy = ", ".join(ch.legacy for ch in chs)
    proteins = ", ".join(ch.protein for ch in chs)
    slug = "-".join(ch.legacy.replace("=", "syn").lower() for ch in chs)
    codon = window_of(seq, 3)[:3]
    return Variant(
        key=f"novel-{slug}",
        hypothesis_id=f"h-{slug}",
        name="Unknown variant",
        hgvs=f"{hgvs_c(chs)} ({proteins})",
        legacy=legacy,
        rsid="",
        codon=codon,
        amino_acid=translate(codon),
        sequence=seq,
        pdb_id=PDB_NORMAL,
        known=False,
        source=(
            f"derived from the camera reads of window {window} ({bases}); "
            "not in the known-variant library (HbS, HbC)"
        ),
    )


def variant_from_dict(d: dict) -> Variant:
    fields = {k: d[k] for k in Variant.__dataclass_fields__}
    v = Variant(**fields)
    if len(v.sequence) != len(REFERENCE) or set(v.sequence) - set(BASES):
        raise ValueError(f"lesson {v.key}: bad sequence")
    return v


# --- lessons ---------------------------------------------------------------------------------


def load_lessons(path: Path) -> list[Variant]:
    """Variants saved by earlier runs, first occurrence of each key, skipping bad lines."""
    out: dict[str, Variant] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            v = variant_from_dict(json.loads(line)["variant"])
        except (ValueError, KeyError, TypeError):
            continue
        out.setdefault(v.key, v)
    return list(out.values())


def append_lesson(path: Path, variant: Variant, **meta: object) -> int:
    """Append one lesson line; returns the number of distinct variants in the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"variant": asdict(variant), **meta}) + "\n")
    return len(load_lessons(path))
