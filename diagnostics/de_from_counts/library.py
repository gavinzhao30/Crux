"""DU-42: analysis forks and ready diagnostics for differential expression from counts.

``library_questions()`` returns one :class:`LibraryQuestion` per fork. Each question has
stable option ``answer_key``s and at least two independent diagnostics (different code), so
two agreeing results can settle a question past the orchestrator's 0.9 posterior bar even
with the "none of these" option in play.

Every diagnostic is a pandas/numpy script under ``scripts/`` that reads ``samples.tsv`` and
``tagseq_featurecounts_matrix.tsv`` from ``$CRUX_DATA_DIR`` (default ``/data``) and prints
one ``RESULT: {json}`` line with flat values. The bins are exhaustive and mutually
exclusive rules over those values (``labs.bixbench3.bins.choose_bin``). Results name the
implicated samples in ``flagged_samples`` so delivery can exclude or relabel them.

``design`` carries ``library`` (the fork key) and ``diagnostic`` (the script name). When an
option fully determines the bin, ``design["exact"]`` maps ``answer_key -> {bin_id: p}``.
For best-effort diagnostics, ``design["stated"]`` holds approximate rows in the same shape,
measured on synthetic data; a predictor may use them as a prior but they are not exact.
The proposer re-ids each diagnostic and sets ``question_ids``.
"""

import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from agents.proposer import LibraryQuestion, Option, Proposal
from core.schemas import Experiment, OutcomeBin

SCRIPTS = Path(__file__).parent / "scripts"
_IMPORT = re.compile(r"^from _common import (?:\([^)]*\)|[^\n]*)\n", re.MULTILINE)


@cache
def script_code(name: str) -> str:
    """The runnable source of one diagnostic, with the shared helpers inlined."""
    source = (SCRIPTS / f"{name}.py").read_text(encoding="utf-8")
    common = (SCRIPTS / "_common.py").read_text(encoding="utf-8")
    if len(_IMPORT.findall(source)) != 1:
        raise ValueError(f"{name}.py must import from _common exactly once")
    return _IMPORT.sub(lambda _: common + "\n", source, count=1)


@dataclass(frozen=True)
class _Diagnostic:
    name: str
    goal: str
    bins: tuple[tuple[str, str, str], ...]  # (id, description, rule)
    est_cost_s: float
    exact: dict[str, dict[str, float]] | None = None
    stated: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class _Fork:
    key: str
    text: str
    impact: float
    options: tuple[tuple[str, str, str], ...]  # (answer_key, statement, rationale)
    diagnostics: tuple[_Diagnostic, ...] = field(default_factory=tuple)


def _complement(*rules: str) -> str:
    return " and ".join(f"not ({rule})" for rule in rules)


_SWAP = "swap_ratio >= 1.0 and candidate_gap >= 0.1"
_NO_SWAP = "swap_ratio < 0.5 and min_line_gap >= 0.1"
_SIGNATURE_OK = "min_cross_corr > 0.0 and mean_cross_corr >= 0.1"
_SIGNATURE_FLIP = (
    "min_cross_corr < 0.0 and worst_group_max_corr < -0.05 and worst_group_score <= -0.1"
)
_MARKER_OK = "n_ctrl_markers >= 1 and n_discordant_markers == 0"
_MARKER_FLIP = "n_discordant_markers >= 1 and n_ctrl_markers == 0"

FORKS: tuple[_Fork, ...] = (
    _Fork(
        key="sample_labels",
        text=(
            "Do the condition labels in the sample sheet match the libraries, or were two "
            "samples (a control and a treated library in one cell line) swapped?"
        ),
        impact=0.95,
        options=(
            (
                "as_given",
                "The condition labels are correct as given.",
                (
                    "Replicates of every condition resemble each other more than other "
                    "conditions, so DE can use the sample sheet unchanged."
                ),
            ),
            (
                "two_swapped",
                (
                    "Two libraries in one cell line carry each other's condition labels; "
                    "exchange them before DE."
                ),
                (
                    "A swap puts a control library in a treated group and vice versa, which "
                    "shrinks fold changes and inflates dispersion for both groups."
                ),
            ),
        ),
        diagnostics=(
            _Diagnostic(
                name="labels_swap_search",
                goal=(
                    "Find the single label exchange that most raises replicate "
                    "correlation within a cell line, relative to the line's "
                    "replicate-versus-condition gap"
                ),
                bins=(
                    ("as_given", "no exchange helps (swap_ratio < 0.5)", _NO_SWAP),
                    ("swapped", "one exchange restores replicate structure", _SWAP),
                    (
                        "unclear",
                        "a weak candidate, or too little structure to judge",
                        _complement(_NO_SWAP, _SWAP),
                    ),
                ),
                est_cost_s=5,
                stated={
                    "as_given": {"as_given": 0.85, "swapped": 0.03, "unclear": 0.12},
                    "two_swapped": {"as_given": 0.05, "swapped": 0.85, "unclear": 0.10},
                },
            ),
            _Diagnostic(
                name="labels_nearest_neighbour",
                goal=(
                    "Check that every library's nearest neighbour (Spearman, within its "
                    "cell line) carries its condition label, and whether one exchange "
                    "fixes all misplaced libraries"
                ),
                bins=(
                    (
                        "consistent",
                        "every nearest neighbour shares the condition",
                        "n_misplaced == 0",
                    ),
                    (
                        "swap_pattern",
                        "misplaced libraries, fixed by one label exchange",
                        "n_misplaced > 0 and swap_resolves",
                    ),
                    (
                        "other",
                        "misplaced libraries that no single exchange fixes",
                        "n_misplaced > 0 and not swap_resolves",
                    ),
                ),
                est_cost_s=5,
                stated={
                    "as_given": {"consistent": 0.8, "swap_pattern": 0.05, "other": 0.15},
                    "two_swapped": {"consistent": 0.02, "swap_pattern": 0.9, "other": 0.08},
                },
            ),
        ),
    ),
    _Fork(
        key="duplicates",
        text="Is every library distinct, or is one library present under two sample names?",
        impact=0.9,
        options=(
            (
                "none",
                "Every library is distinct.",
                "Independent libraries always differ by sampling noise.",
            ),
            (
                "duplicate_library",
                (
                    "One library appears under two sample names; keep one copy and drop the "
                    "other, leaving that group with one replicate."
                ),
                (
                    "A copied library fakes a perfectly concordant replicate, which deflates "
                    "dispersion and inflates significance."
                ),
            ),
        ),
        diagnostics=(
            _Diagnostic(
                name="duplicates_identical",
                goal="Find pairs of count columns identical on at least 99% of detected genes",
                bins=(
                    ("none", "no identical pair", "n_duplicate_pairs == 0"),
                    ("duplicate", "at least one identical pair", "n_duplicate_pairs >= 1"),
                ),
                est_cost_s=5,
                exact={
                    "none": {"none": 1.0, "duplicate": 0.0},
                    "duplicate_library": {"none": 0.0, "duplicate": 1.0},
                },
            ),
            _Diagnostic(
                name="duplicates_correlation",
                goal="Find library pairs whose log2 CPM correlate at 0.9999 or above",
                bins=(
                    ("none", "no pair at 0.9999", "n_duplicate_pairs == 0"),
                    ("duplicate", "at least one pair at 0.9999", "n_duplicate_pairs >= 1"),
                ),
                est_cost_s=5,
                exact={
                    "none": {"none": 1.0, "duplicate": 0.0},
                    "duplicate_library": {"none": 0.0, "duplicate": 1.0},
                },
            ),
        ),
    ),
    _Fork(
        key="sample_quality",
        text="Are all libraries usable, or is at least one library degraded?",
        impact=0.8,
        options=(
            (
                "all_usable",
                (
                    "Every library is usable: depth within ten-fold of the median library, at "
                    "least half the median detected genes, and noise in line with the rest."
                ),
                "Ordinary depth spread within one experiment is two- to four-fold.",
            ),
            (
                "degraded_library",
                (
                    "At least one library is degraded (over ten-fold below median depth, under "
                    "half the median detected genes, or over 2.5 times its cell line's "
                    "typical noise); exclude it from DE."
                ),
                (
                    "A failed or heavily down-sampled library adds noise to its group and "
                    "distorts size factors and dispersion estimates."
                ),
            ),
        ),
        diagnostics=(
            _Diagnostic(
                name="quality_depth",
                goal="Flag libraries below 10% of median depth or 50% of median detected genes",
                bins=(
                    ("all_usable", "no library flagged", "n_flagged == 0"),
                    ("degraded", "at least one library flagged", "n_flagged >= 1"),
                ),
                est_cost_s=5,
                exact={
                    "all_usable": {"all_usable": 1.0, "degraded": 0.0},
                    "degraded_library": {"all_usable": 0.0, "degraded": 1.0},
                },
            ),
            _Diagnostic(
                name="quality_noise",
                goal=(
                    "Flag libraries whose noise (one minus median correlation with their "
                    "cell line) exceeds 2.5 times the line's median"
                ),
                bins=(
                    ("all_usable", "no library flagged", "n_flagged == 0"),
                    ("degraded", "at least one library flagged", "n_flagged >= 1"),
                ),
                est_cost_s=5,
                exact={
                    "all_usable": {"all_usable": 1.0, "degraded": 0.0},
                    "degraded_library": {"all_usable": 0.0, "degraded": 1.0},
                },
            ),
        ),
    ),
    _Fork(
        key="cell_line",
        text=(
            "Does the cell_line column match the libraries' expression, or are some "
            "libraries assigned to the wrong cell line?"
        ),
        impact=0.75,
        options=(
            (
                "as_given",
                "The cell_line column is correct as given.",
                "Each library's expression profile matches its labelled cell line.",
            ),
            (
                "mislabelled",
                (
                    "Some libraries carry the wrong cell line; reassign them to the line their "
                    "expression matches before splitting the analysis by cell line."
                ),
                (
                    "A library in the wrong cell line dominates that line's variance and "
                    "corrupts every contrast that uses it."
                ),
            ),
        ),
        diagnostics=(
            _Diagnostic(
                name="cell_line_centroid",
                goal=(
                    "Assign each library to the nearest leave-one-out cell line centroid "
                    "and compare with the cell_line column"
                ),
                bins=(
                    ("as_given", "every library is nearest its own line", "n_mismatched == 0"),
                    ("mislabelled", "a library is nearer another line", "n_mismatched >= 1"),
                ),
                est_cost_s=5,
                exact={
                    "as_given": {"as_given": 1.0, "mislabelled": 0.0},
                    "mislabelled": {"as_given": 0.0, "mislabelled": 1.0},
                },
            ),
            _Diagnostic(
                name="cell_line_pc1",
                goal=(
                    "Split libraries in two on PC1 without labels and compare each group "
                    "with its majority cell line"
                ),
                bins=(
                    (
                        "as_given",
                        "PC1 groups match the labels",
                        "n_mismatched == 0 and separation >= 4",
                    ),
                    (
                        "mislabelled",
                        "a library sits in the other line's PC1 group",
                        "n_mismatched >= 1 and separation >= 4",
                    ),
                    ("unclear", "cell lines do not separate on PC1", "separation < 4"),
                ),
                est_cost_s=5,
                exact={
                    "as_given": {"as_given": 1.0, "mislabelled": 0.0, "unclear": 0.0},
                    "mislabelled": {"as_given": 0.0, "mislabelled": 1.0, "unclear": 0.0},
                },
            ),
        ),
    ),
    _Fork(
        key="gene_ids",
        text=(
            "Are the gene IDs in one namespace as in the matrix, or do versioned and "
            "unversioned Ensembl IDs mix?"
        ),
        impact=0.6,
        options=(
            (
                "as_given",
                (
                    "Gene IDs share one namespace (all versioned or all unversioned); use them "
                    "as given and strip versions uniformly only for the output table."
                ),
                "A matrix from one annotation release uses one ID form throughout.",
            ),
            (
                "mixed_versions",
                (
                    "Versioned and unversioned Ensembl IDs are mixed; strip versions from every "
                    "ID, checking for collisions, before joining annotation or reporting."
                ),
                (
                    "Mixed forms break joins with annotation and the reference DE table, "
                    "silently dropping genes."
                ),
            ),
        ),
        diagnostics=(
            _Diagnostic(
                name="gene_versions",
                goal="Count Ensembl gene IDs with and without a version suffix",
                bins=(
                    (
                        "uniform",
                        "all versioned or all unversioned",
                        "n_versioned == 0 or n_unversioned == 0",
                    ),
                    ("mixed", "both forms present", "n_versioned > 0 and n_unversioned > 0"),
                ),
                est_cost_s=3,
                exact={
                    "as_given": {"uniform": 1.0, "mixed": 0.0},
                    "mixed_versions": {"uniform": 0.0, "mixed": 1.0},
                },
            ),
            _Diagnostic(
                name="gene_strip_check",
                goal=("Strip version suffixes and count changed, unchanged and colliding gene IDs"),
                bins=(
                    (
                        "uniform",
                        "every ID or no ID changes, no collisions",
                        "n_minority == 0 and n_collisions == 0",
                    ),
                    (
                        "mixed",
                        "some IDs change and some do not, no collisions",
                        "n_minority > 0 and n_collisions == 0",
                    ),
                    (
                        "uniform_collisions",
                        "one form, but stripping creates duplicate IDs",
                        "n_minority == 0 and n_collisions > 0",
                    ),
                    (
                        "mixed_collisions",
                        "mixed forms and duplicate IDs after stripping",
                        "n_minority > 0 and n_collisions > 0",
                    ),
                ),
                est_cost_s=3,
                exact={
                    "as_given": {
                        "uniform": 1.0,
                        "mixed": 0.0,
                        "uniform_collisions": 0.0,
                        "mixed_collisions": 0.0,
                    },
                    "mixed_versions": {
                        "uniform": 0.0,
                        "mixed": 1.0,
                        "uniform_collisions": 0.0,
                        "mixed_collisions": 0.0,
                    },
                },
            ),
        ),
    ),
    _Fork(
        key="group_direction",
        text=(
            "Are control and treated labels the right way round in each cell line, or are "
            "the control group and one treated group exchanged?"
        ),
        impact=0.55,
        options=(
            (
                "as_given",
                "Control and treated groups are labelled the right way round.",
                "The treated groups share the knockout's signature relative to the control.",
            ),
            (
                "ctrl_inverted",
                (
                    "In one cell line the control group and one treated group carry each other's "
                    "labels; exchange them before DE."
                ),
                (
                    "An inversion flips the sign of that contrast and contaminates every other "
                    "contrast in the line through the wrong reference."
                ),
            ),
        ),
        diagnostics=(
            _Diagnostic(
                name="direction_signature",
                goal=(
                    "Correlate each treated group's fold change over control with the "
                    "other cell line's treated groups (best effort: needs a shared "
                    "treatment signature)"
                ),
                bins=(
                    ("consistent", "every cross-line correlation is positive", _SIGNATURE_OK),
                    ("inverted", "one group anti-correlates with the other line", _SIGNATURE_FLIP),
                    (
                        "unclear",
                        "no decisive shared signature",
                        _complement(_SIGNATURE_OK, _SIGNATURE_FLIP),
                    ),
                ),
                est_cost_s=5,
                stated={
                    "as_given": {"consistent": 0.55, "inverted": 0.05, "unclear": 0.40},
                    "ctrl_inverted": {"consistent": 0.05, "inverted": 0.55, "unclear": 0.40},
                },
            ),
            _Diagnostic(
                name="direction_marker",
                goal=(
                    "Find genes lost in every group but one in both cell lines (the knockout "
                    "target) and check that the group keeping them is the control (best "
                    "effort)"
                ),
                bins=(
                    ("consistent", "markers stay on in the control of every line", _MARKER_OK),
                    ("inverted", "markers stay on in a treated group of one line", _MARKER_FLIP),
                    (
                        "unclear",
                        "no markers, or markers that disagree",
                        _complement(_MARKER_OK, _MARKER_FLIP),
                    ),
                ),
                est_cost_s=5,
                stated={
                    "as_given": {"consistent": 0.5, "inverted": 0.05, "unclear": 0.45},
                    "ctrl_inverted": {"consistent": 0.05, "inverted": 0.5, "unclear": 0.45},
                },
            ),
        ),
    ),
)


def _experiment(fork: _Fork, diagnostic: _Diagnostic) -> Experiment:
    design: dict = {"library": fork.key, "diagnostic": diagnostic.name}
    if diagnostic.exact is not None:
        design["exact"] = {key: dict(row) for key, row in diagnostic.exact.items()}
    if diagnostic.stated is not None:
        design["stated"] = {key: dict(row) for key, row in diagnostic.stated.items()}
    return Experiment(
        id=f"library-{fork.key}-{diagnostic.name}",
        question_ids=[],
        kind="compute",
        goal=diagnostic.goal,
        language="python",
        code=script_code(diagnostic.name),
        design=design,
        bins=[OutcomeBin(id=i, description=d, rule=r) for i, d, r in diagnostic.bins],
        est_cost_s=diagnostic.est_cost_s,
    )


def library_questions(brief: dict | None = None) -> list[LibraryQuestion]:
    """The v0 DE-from-counts forks, highest priority first, each with its diagnostics.

    ``brief`` is accepted for interface stability; the v0 library is the same for every
    Task 9 brief. Each call returns fresh model instances.
    """
    del brief
    return [
        LibraryQuestion(
            key=fork.key,
            proposal=Proposal(
                text=fork.text,
                kind="categorical",
                impact=fork.impact,
                options=[
                    Option(statement=statement, rationale=rationale, answer_key=key)
                    for key, statement, rationale in fork.options
                ],
            ),
            diagnostics=[_experiment(fork, d) for d in fork.diagnostics],
        )
        for fork in FORKS
    ]
