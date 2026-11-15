"""The real-data helpers: the all-diagnostics runner and the predictor's library rows."""

from __future__ import annotations

import pytest

from agents.predictor import library_row
from core.schemas import Hypothesis
from diagnostics.de_from_counts.library import library_questions
from diagnostics.de_from_counts.run_all import clean_bin, run_all
from tests.test_task9_wiring import make_case


def test_run_all_reports_every_diagnostic(tmp_path):
    case = make_case(tmp_path / "case", duplicate=True)
    rows = run_all(case)
    assert len(rows) == sum(len(q.diagnostics) for q in library_questions()) == 12
    assert all(r["error"] == "" and r["outcome"] for r in rows)
    duplicates = [r for r in rows if r["fork"] == "duplicates"]
    assert {r["outcome"] for r in duplicates} == {"duplicate"}
    assert all(set(r["flagged_samples"]) == {"S03", "S04"} for r in duplicates)


def test_clean_bin_is_the_null_options_likeliest_outcome():
    for question in library_questions():
        null_key = question.proposal.options[0].answer_key
        for exp in question.diagnostics:
            assert clean_bin(exp.design, null_key) in {b.id for b in exp.bins}


def test_library_rows_never_call_an_outcome_impossible():
    """PC1 is measured, not exact: 'unclear' must keep weight under both options."""
    pc1 = next(
        exp
        for q in library_questions()
        for exp in q.diagnostics
        if exp.design["diagnostic"] == "cell_line_pc1"
    )
    for key in ("as_given", "mislabelled"):
        h = Hypothesis(
            id=key,
            question_id="q",
            statement=key,
            author="a",
            rationale="-",
            answer_key=key,
            prior=0.5,
            posterior=0.5,
        )
        row = library_row(pc1, h)
        assert row is not None and sum(row.values()) == pytest.approx(1.0)
        assert min(row.values()) > 0 and row["unclear"] >= 0.2
