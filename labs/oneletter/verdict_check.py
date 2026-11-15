"""Check a model-written verdict against the facts the engine gave the model.

The Jetson's small model (qwen2.5:3b) gets facts wrong about one time in four even with the
facts in its prompt: it pairs a codon with the wrong amino acid ("Val (GAG)"), garbles a read
("GCAGAAG"), or adds a claim the facts never make ("leading to sickle cell disease"). A wrong
sentence on stage is worse than the plain template, so the engine shows the model's sentence
only when ``check_verdict`` finds nothing in it that the facts do not support.

The rule is simple: every checkable token in the reply (a DNA sequence, a codon and its amino
acid, an amino acid, an HGVS or dbSNP code, a variant name, a number) must come from the facts.
"""

from __future__ import annotations

import re

from labs.oneletter import variants as V

MAX_WORDS = 50  # the prompt asks for 30; the model often runs over, and MAX_CHARS is the real limit
MAX_CHARS = 300

FULL_NAMES: dict[str, str] = {
    "alanine": "Ala",
    "arginine": "Arg",
    "asparagine": "Asn",
    "aspartic acid": "Asp",
    "aspartate": "Asp",
    "cysteine": "Cys",
    "glutamine": "Gln",
    "glutamic acid": "Glu",
    "glutamate": "Glu",
    "glycine": "Gly",
    "histidine": "His",
    "isoleucine": "Ile",
    "leucine": "Leu",
    "lysine": "Lys",
    "methionine": "Met",
    "phenylalanine": "Phe",
    "proline": "Pro",
    "serine": "Ser",
    "threonine": "Thr",
    "tryptophan": "Trp",
    "tyrosine": "Tyr",
    "valine": "Val",
}
_AA3 = sorted(set(V.AA3.values()) - {"Ter"})
_FULL = sorted(FULL_NAMES, key=len, reverse=True)
_AA = "(?:" + "|".join(_AA3) + "|(?i:" + "|".join(_FULL) + "))"
_AA3_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(_AA3) + r")(?![a-z])")
_FULL_RE = re.compile(r"\b(" + "|".join(_FULL) + r")\b", re.IGNORECASE)
_CODON = r"(?<![A-Za-z])(?P<c>[ACGT]{3})(?![A-Za-z])"
_PAIRS = (
    re.compile(_CODON + r"\s*\(\s*(?P<a>" + _AA + r")(?![a-z])\s*\)"),
    re.compile(r"(?<![A-Za-z])(?P<a>" + _AA + r")(?![a-z])\s*\(\s*" + _CODON + r"\s*\)"),
    re.compile(  # also "GCG (Ala), which codes for Glu": the claim after the bracket counts
        _CODON
        + r"(?:\s*\([^)]*\))?,?\s+(?:which\s+)?(?:codes|coding|encodes|encoding)\s+"
        + r"(?:for\s+)?(?P<a>"
        + _AA
        + r")(?![a-z])"
    ),
    re.compile(_CODON + r"\s+(?:for|=|->|→)\s+(?P<a>" + _AA + r")(?![a-z])"),
)


def _item(name: str) -> str:
    """An amino acid or a codon, captured as ``name``."""
    return (
        rf"(?P<{name}>(?<![A-Za-z])(?:{_AA})(?![a-z])|(?<![A-Za-z])[ACGT]{{3}}(?![A-Za-z]))"
    )


# "Glu to Val", "GAG to GTG", "changes Ala at codon 6 to Glu": normal first, then the variant
_CHANGE_RE = re.compile(
    _item("x")
    + r"(?:\s+at\s+(?:codon|position)\s+\d+)?\s+(?:to|into|->|→)\s+(?:the\s+)?"
    + _item("y")
)
# "Val instead of Glu", "GTG instead of the normal GAG": the variant first
_INSTEAD_RE = re.compile(
    _item("x") + r"\s+instead\s+of\s+(?:the\s+)?(?:normal\s+)?(?:codon\s+)?" + _item("y")
)
_DNA_RE = re.compile(r"(?<![A-Za-z])[ACGT]{3,}(?![A-Za-z])")
_CODES = (  # order matters: each match is blanked before the next pattern runs
    re.compile(r"NM_\d+\.\d+"),
    re.compile(r"c\.\d+[ACGT]>[ACGT]"),
    re.compile(r"p\.[A-Z][a-z]{2}\d+(?:[A-Z][a-z]{2}|=)"),
    re.compile(r"\brs\d+\b"),
    re.compile(r"(?<![\w.])[A-Z][a-z]{2}\d+(?:[A-Z][a-z]{2}|=)"),
)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"(?<![\w.])(\d*\.\d+)(?!\d)")
_WINDOW_RE = re.compile(r"\bwindow\s+(\d+)\b(?!\s+times)", re.IGNORECASE)
_TIMES_RE = re.compile(r"\b(\d+)\s+times\b", re.IGNORECASE)
_OF_RE = re.compile(r"\b(\d+)\s+of\s+(\d+)\b", re.IGNORECASE)
_INT_RE = re.compile(r"\d+")
_LABEL_RE = re.compile(r"\bHb[A-Z]\b|\bsickle\b", re.IGNORECASE)
_COUNT_WORDS = {"once": 1, "twice": 2, "thrice": 3}
_CLAIMS = re.compile(
    r"\b(?:disease\w*|disorder\w*|diagnos\w*|treat\w*|therap\w*|symptom\w*|cure\w*|"
    r"carrier\w*|trait\w*|an(?:a)?emi\w*|you\s+have|your|should\s+be)\b",
    re.IGNORECASE,
)
_NORMAL_CLAIMS = re.compile(r"\b(?:instead\s+of|mutat\w*|mutant|abnormal)\b", re.IGNORECASE)


def _aa3(name: str) -> str:
    return name if name in _AA3 else FULL_NAMES[name.lower()]


def _amino_acids(text: str) -> set[str]:
    found = set(_AA3_RE.findall(text))
    return found | {FULL_NAMES[m.lower()] for m in _FULL_RE.findall(text)}


def _norm(token: str) -> tuple[str, str]:
    """("codon", "GTG") or ("aa", "Val")."""
    if re.fullmatch(r"[ACGT]{3}", token):
        return "codon", token
    return "aa", _aa3(token)


def _direction(text: str, variant: V.Variant) -> list[str]:
    """Changes stated the wrong way round ("Ala to Glu" for Glu6Ala)."""
    normal = {"codon": V.REFERENCE[18:21], "aa": V.translate(V.REFERENCE[18:21])}
    alt = {"codon": variant.codon, "aa": variant.amino_acid}
    out = []
    for pattern, first, second in ((_CHANGE_RE, normal, alt), (_INSTEAD_RE, alt, normal)):
        for m in pattern.finditer(text):
            (kx, x), (ky, y) = _norm(m.group("x")), _norm(m.group("y"))
            if kx == ky and (x, y) != (first[kx], second[kx]):
                out.append(f"states the change as {m.group(0)!r}; it is {normal[kx]} to {alt[kx]}")
    return out


def _allowed_dna(prompt: str) -> set[str]:
    allowed = set()
    for token in _DNA_RE.findall(prompt):
        allowed.add(token)
        if len(token) % 3 == 0:
            allowed.update(token[i : i + 3] for i in range(0, len(token), 3))
    return allowed


def check_verdict(
    text: str,
    *,
    variant: V.Variant,
    posterior: float,
    window: int,
    reads: list[str],
    prompt: str,
) -> list[str]:
    """What in ``text`` the facts do not support; an empty list means it can be shown."""
    problems: list[str] = []
    if not text:
        return ["the reply is empty"]
    if len(text) > MAX_CHARS or len(text.split()) > MAX_WORDS:
        problems.append(f"too long ({len(text.split())} words)")
    if not re.search(r"[.!?][\"')]*$", text):
        problems.append("the sentence is cut off")

    # the answer has to be named, and no other variant may be
    if variant.key == "hba":
        label = re.compile(r"\bHbA\b|\bnormal\b", re.IGNORECASE)
    elif variant.known:
        label = re.compile(r"\b" + re.escape(variant.name.split()[0]) + r"\b", re.IGNORECASE)
    else:
        label = re.compile(r"\bunknown\b", re.IGNORECASE)
    if not label.search(text):
        problems.append(f"does not name the answer ({variant.name})")
    prompt_labels = {m.lower() for m in _LABEL_RE.findall(prompt)}
    for m in _LABEL_RE.findall(text):
        if m.lower() not in prompt_labels:
            problems.append(f"names {m}, which the facts do not")

    # sequences and codons, and each codon's amino acid by the genetic code
    allowed_dna = _allowed_dna(prompt)
    for token in _DNA_RE.findall(text):
        if token not in allowed_dna:
            problems.append(f"sequence {token} is not in the facts")
    for pattern in _PAIRS:
        for m in pattern.finditer(text):
            codon, aa = m.group("c"), _aa3(m.group("a"))
            if V.translate(codon) != aa:
                problems.append(f"pairs {codon} with {aa}; {codon} codes for {V.translate(codon)}")
    problems.extend(_direction(text, variant))
    allowed_aa = _amino_acids(prompt)
    for aa in sorted(_amino_acids(text) - allowed_aa):
        problems.append(f"mentions {aa}, which the facts do not")

    # codes and numbers: blank each checked token so the integer check sees only the rest
    rest = text
    for pattern in _CODES:
        for m in pattern.finditer(rest):
            if m.group(0) not in prompt:
                problems.append(f"{m.group(0)} is not in the facts")
        rest = pattern.sub(" ", rest)
    for m in _PERCENT_RE.finditer(rest):
        if abs(float(m.group(1)) - posterior * 100.0) > 0.5:
            problems.append(f"says {m.group(0)}; the posterior is {posterior:.2f}")
    rest = _PERCENT_RE.sub(" ", rest)
    for m in _DECIMAL_RE.finditer(rest):
        if abs(float(m.group(1)) - posterior) > 0.005:
            problems.append(f"says {m.group(1)}; the posterior is {posterior:.2f}")
    rest = _DECIMAL_RE.sub(" ", rest)
    for m in _WINDOW_RE.finditer(rest):
        if int(m.group(1)) != window:
            problems.append(f"says window {m.group(1)}; the camera read window {window}")
    for m in _TIMES_RE.finditer(rest):
        if int(m.group(1)) != len(reads):
            problems.append(f"says {m.group(0)}; there were {len(reads)} reads")
    for word, n in _COUNT_WORDS.items():
        if re.search(rf"\b{word}\b", rest, re.IGNORECASE) and n != len(reads):
            problems.append(f"says {word}; there were {len(reads)} reads")
    agree = sum(b == variant.window(window) for b in reads)
    for m in _OF_RE.finditer(rest):
        if (int(m.group(1)), int(m.group(2))) != (agree, len(reads)):
            problems.append(f"says {m.group(0)}; {agree} of {len(reads)} reads match")
    prompt_ints = set(_INT_RE.findall(prompt))
    for n in _INT_RE.findall(rest):
        if n not in prompt_ints:
            problems.append(f"the number {n} is not in the facts")

    # claims the facts never make
    for m in _CLAIMS.finditer(text):
        problems.append(f"adds a claim ({m.group(0)})")
    if variant.key == "hba":
        for m in _NORMAL_CLAIMS.finditer(text):
            problems.append(f"calls the normal codon a change ({m.group(0)})")
    return list(dict.fromkeys(problems))
