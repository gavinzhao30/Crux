"""The fact check on the Jetson model's verdict sentence.

The replies below are real: qwen2.5:3b on the Jetson, answering the engine's own prompt
(posterior 0.97, two reads of window 3). The ones marked bad have a wrong fact or a claim the
facts never make; the check must reject those and let the others through.
"""

from __future__ import annotations

import pytest

from labs.oneletter import variants as V
from labs.oneletter.lab import verdict_prompt
from labs.oneletter.verdict_check import check_verdict

P = 0.97
NOVEL = V.novel_variant(3, "GCGGAGAAG")
CASES = {"hba": V.HBA, "hbs": V.HBS, "hbc": V.HBC, "novel": NOVEL}

# (case, reply, ok) - replies copied from the Jetson model
REAL = [
    (
        "hba",
        ('The camera read "GAGGAGAAG" twice from a demo strip of the human HBB gene; codon 6 '
        "reads GAG for Glu instead of normal GAG, indicating HbA (normal haemoglobin) with a "
        "high posterior probability of 97%."),
        False,  # "instead of": the normal codon is not a change
    ),
    (
        "hba",
        ('The camera read "GAGGAGAAG" twice from a demo strip of the human HBB gene, matching '
        "the normal codon for Glu (GAG), indicating HbA or normal haemoglobin with a high "
        "posterior probability of 0.97."),
        True,
    ),
    (
        "hba",
        ('The camera read "GAGGAGAAG" twice from a demo strip of the human HBB gene, matching '
        "the normal codon for Glu (GAG), indicating a result consistent with HbA, or normal "
        "haemoglobin. Posterior probability 0.97."),
        True,
    ),
    (
        "hbs",
        ('The camera read "GTGGAGAAG" twice from the HBB gene codon 6, which codes for Val '
        "(GAG), matching the answer of HbS with a posterior probability of 0.97, indicating a "
        "likely mutation to Glu6Val in this demo sample."),
        False,  # GAG codes for Glu, not Val
    ),
    (
        "hbs",
        ("The camera read GTGGAGAAG twice from the HBB gene, matching the answer of HbS "
        "(sickle), with codon 6 reading Val instead of Glu as normal, indicating a genetic "
        "mutation at NM_000518.5:c.20A>T (p.Glu7Val). Posterior probability is 97%."),
        True,
    ),
    (
        "hbs",
        ('The camera read "GTGGAGAAG" twice from the HBB gene codon 6 (Val) which matches the '
        "answer of HbS with a posterior probability of 0.97, indicating a Glu6Val mutation "
        "leading to sickle cell disease. This is for demonstration purposes only."),
        False,  # a disease claim the facts never make
    ),
    (
        "hbc",
        ("The camera read window 2 times the correct HbC sequence (AAGGAGAAG) matching the "
        "answer: NM_000518.5:c.19G>A (p.Glu7Lys), with a high posterior probability of 0.97, "
        "in this demo but not for patients."),
        True,
    ),
    (
        "hbc",
        ("The camera read window 2 times the correct AAGGAGAAG sequence matching HbC mutation "
        "at codon 6 (p.Glu7Lys) with high posterior probability of 0.97 in the demo screen."),
        True,
    ),
    (
        "hbc",
        ("The camera read the correct HbC codon (AAG) matching the answer, indicating a "
        "genetic mutation at position 6 from glutamic acid (GAG) to lysine (AAG), with a high "
        "posterior probability of 0.97. This is for demonstration purposes only."),
        True,
    ),
    (
        "novel",
        ('The camera read "GCAGAAG" twice from codon 6 of the HBB gene on the screen, '
        "indicating an unknown variant (Glu to Ala at position 6) with a high posterior "
        "probability of 0.97, different from common disorders like HbS and HbC in this demo "
        "strip."),
        False,  # GCAGAAG is not what the camera read; "disorders" is a claim
    ),
    (
        "novel",
        ('The camera read "GCGGAGAAG" twice from codon 6 of the HBB gene on the screen, '
        "indicating an unknown variant (Ala at position 6 instead of Glu), with a high "
        "posterior probability of 0.97 that this is not present in common hemoglobin "
        "variants like HbS or HbC."),
        True,
    ),
    (
        "novel",
        ('The camera read "GCAGAAG" twice from codon 6 of the HBB gene; this variant changes '
        "Glu to Ala at position 6 and has a posterior probability of 97%. No match found in "
        "the demo's library for known conditions like HbS or HbC."),
        False,  # GCAGAAG again
    ),
    # second round, after "copy every sequence, codon and code exactly" was added
    (
        "novel",
        ("The camera read window 3 of the human HBB gene sequence twice: GCGGAGAAG. Codon 6 "
         "reads GCG (Ala), which codes for Glu (normal codon is GAG). The posterior "
         "probability suggests an unknown variant (Glu6Ala) not found in demo library (HbS, "
         "HbC)."),
        False,  # GCG codes for Ala, not Glu
    ),
    (
        "novel",
        ("The camera read window 3 of the human HBB gene sequence twice: GCGGAGAAG. Codon 6 "
         "reads GCG (Ala), which codes for Glu in this case (normal codon is GAG, Glu). This "
         "unknown variant (Glu6Ala) does not match any variants in the demo's library (HbS, "
         "HbC"),
        False,  # wrong pairing, and cut off at the token limit
    ),
    (
        "hba",
        ("The camera read window 3 of the human HBB gene on the demo strip as GAGGAGAAG "
         "twice, matching the answer; codon 6 codes for Glu, but should be GAG (Glu) "
         "indicating HbA, normal haemoglobin with a posterior probability of 0.97."),
        False,  # "should be": says the normal codon is an error
    ),
    (
        "novel",
        ("The camera read window 3 of the human HBB gene on the screen as GCGGAGAAG twice; "
         "this variant changes Ala at codon 6 to Glu (normal is Glu), indicating an unknown "
         "genetic variation with a high probability (0.97). No match found in demo library "
         "for HbS or HbC variants."),
        False,  # backwards: the change is Glu to Ala
    ),
    (
        "novel",
        ("The camera read window 3 of the human HBB gene sequence twice: GCGGAGAAG. Codon 6 "
         "shows mutation from Glu to Ala (GAG to GCG), indicating an unknown variant "
         "(Glu6Ala) not found in common hemoglobin variants like HbS or HbC. Posterior "
         "probability is 0.97."),
        True,
    ),
    (
        "hbs",
        ("HbS (sickle) variant detected at NM_000518.5:c.20A>T (p.Glu7Val), corresponding "
         "to codon GTG in sequence GTGGAGAAG, with a high posterior probability of 97%."),
        True,
    ),
    (
        "hbc",
        ("The camera read window 3 of the HBB gene on the demo strip: AAGGAGAAG, matching "
         "the answer which states HbC (NM_000518.5:c.19G>A, p.Glu7Lys), with a posterior "
         "probability of 0.97."),
        True,
    ),
]


def check(case: str, text: str, reads: list[str] | None = None) -> list[str]:
    v = CASES[case]
    mine = reads if reads is not None else [v.window(3)] * 2
    prompt = verdict_prompt(v, P, 3, mine)
    return check_verdict(text, variant=v, posterior=P, window=3, reads=mine, prompt=prompt)


@pytest.mark.parametrize(("case", "text", "ok"), REAL)
def test_real_jetson_replies_are_judged_on_their_facts(case, text, ok):
    problems = check(case, text)
    assert (problems == []) == ok, problems


def test_the_prompt_carries_the_facts_the_check_relies_on():
    prompt = verdict_prompt(V.HBS, P, 3, ["GTGGAGAAG", "GTGGAGAAG"])
    for fact in ("GTGGAGAAG", "GTG", "Val", "GAG (Glu)", "c.20A>T", "rs334", "Glu6Val", "0.97"):
        assert fact in prompt
    assert "2 of them match" in prompt and "copy every sequence" in prompt


@pytest.mark.parametrize(
    ("case", "text", "why"),
    [
        ("hbc", "Codon 6 reads AAG (Lys), from valine to glutamine: HbC, posterior 0.97.", "Val"),
        ("hbs", "Codon 6 reads GTG (Val): HbS, rs33930165, posterior 0.97.", "rs33930165"),
        ("hbs", "Codon 6 reads GTG (Val): HbS (sickle), c.19G>A, posterior 0.97.", "c.19G>A"),
        ("hbs", "Codon 6 reads GTG (Val): HbS, Glu6Lys, posterior 0.97.", "Glu6Lys"),
        ("hbs", "Codon 6 reads GTG (Val): HbS, posterior 0.79.", "0.79"),
        ("hbs", "Codon 6 reads GTG (Val): HbS with 99% confidence.", "99%"),
        ("hbs", "Window 2 reads GTG (Val): HbS, posterior 0.97.", "window 2"),
        ("hbs", "Codon 17 reads GTG (Val): HbS, posterior 0.97.", "17"),
        ("hbs", "Codon 6 reads GTG (Val), read 3 times: HbS, posterior 0.97.", "3 times"),
        ("hbs", "Codon 6 reads GTG (Val), read once: HbS, posterior 0.97.", "once"),
        ("hbs", "Codon 6 reads GTG (Val): HbC, posterior 0.97.", "HbS"),
        ("hba", "Codon 6 reads GAG (Glu): normal, not sickle, posterior 0.97.", "sickle"),
        ("hbs", "Codon 6 reads GTG (Val): HbS, the sickle cell trait, posterior 0.97.", "trait"),
        ("hbs", "Codon 6 reads GTG (Val): you have HbS, posterior 0.97.", "you have"),
        ("hbs", "GTG codes for Glu: HbS, posterior 0.97.", "GTG with Glu"),
        ("hbs", "Codon 6 reads ATG (Met): HbS, posterior 0.97.", "ATG"),
        ("hbs", "Codon 6 reads GTG (Val): HbS. " + "word " * 45, "too long"),
        ("hbs", "", "empty"),
        ("hbs", "Codon 6 reads GTG (Val): HbS (sickle), posterior 0.97, which", "cut off"),
        ("novel", "GCG (Ala), which codes for Glu: an unknown variant, Glu6Ala.", "GCG with Glu"),
        ("hbs", "Codon 6 changes from Val to Glu: HbS (sickle), posterior 0.97.", "Glu to Val"),
        ("hbs", "Codon 6 changes GTG to GAG: HbS (sickle), posterior 0.97.", "GAG to GTG"),
        ("hbs", "Codon 6 reads GAG instead of GTG: HbS (sickle), posterior 0.97.", "GAG to GTG"),
        ("hba", "Codon 6 reads GAG (Glu), not GTG: HbA, posterior 0.97.", "GTG"),
    ],
)
def test_wrong_facts_are_rejected(case, text, why):
    problems = check(case, text)
    assert problems, text
    assert any(why in p for p in problems), problems


@pytest.mark.parametrize(
    ("case", "text"),
    [
        ("hbs", "Codon 6 reads GTG (Val) instead of GAG (Glu): HbS (sickle), rs334, posterior 0.97."),
        ("hbs", "GTG, which codes for valine, replaces GAG (glutamic acid): HbS, 97% posterior."),
        ("hbc", "Both reads show AAG, lysine, at codon 6: HbC, Glu6Lys, posterior 0.97."),
        ("hba", "Both reads show GAG (Glu), the normal codon: HbA, posterior 0.97."),
        ("hbs", "Codon 6 changes from GAG to GTG, Glu to Val: HbS (sickle), posterior 0.97."),
        ("hbs", "Codon 6 reads GTG instead of the normal GAG: HbS, posterior 0.97."),
        ("novel", "GCG (Ala) at codon 6 matches no variant in the library: unknown, Glu6Ala."),
    ],
)
def test_sentences_that_only_restate_the_facts_pass(case, text):
    assert check(case, text) == []


def test_a_count_that_disagrees_with_the_reads_is_rejected():
    reads = ["GTGGAGAAG", "GTGGAGAAG", "GTGGAGAAC"]
    assert check("hbs", "2 of 3 reads show GTG (Val): HbS, posterior 0.97.", reads) == []
    problems = check("hbs", "3 of 3 reads show GTG (Val): HbS, posterior 0.97.", reads)
    assert any("3 of 3" in p for p in problems)
