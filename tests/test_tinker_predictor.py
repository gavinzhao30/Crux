"""DU-29 learned predictor: prompts, scoring math, fallback, wiring and training labels.

No network: Tinker clients and tokenizers are fakes.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import numpy as np
import pytest

from agents.predictor import LIBRARY_FLOOR, SampledPredictor, library_row
from agents.tinker_predictor import (
    TinkerPredictor,
    bin_completion,
    build_prompt,
    completion_logprob,
    logprobs_to_row,
    profile_summary,
)
from core import scoring
from core.config import RunConfig, load_config
from core.fakes_engine import FakeClock, FakeLab, fake_deps
from core.orchestrator import run
from core.schemas import Experiment, Hypothesis, OutcomeBin
from diagnostics.de_from_counts.synth import (
    FAULT_FORK,
    Knobs,
    featurecounts_case,
    plant,
    simulate,
    suite_case,
    write_case,
)
from eval.predictor_data.dataset import library_index, record_example, record_prompt
from eval.predictor_data.make_examples import case_specs, make_case, profile_case
from eval.train_tinker_predictor import sft_arrays

PROFILE = {
    "files": ["samples.tsv", "tagseq_featurecounts_matrix.tsv"],
    "shape": [3000, 4],
    "example_gene_ids": ["ENSG00000100000.1"],
    "versioned_gene_ids": 2999,
    "library_sizes": {"S1": 1_000_000, "S2": 1_200_000, "S3": 20_000, "S4": 900_000},
    "samples": [
        {"sample_id": "S1", "cell_line": "A", "condition": "Ctrl"},
        {"sample_id": "S2", "cell_line": "A", "condition": "Ctrl"},
        {"sample_id": "S3", "cell_line": "A", "condition": "KO"},
        {"sample_id": "S4", "cell_line": "A", "condition": "KO"},
    ],
}


class CharTokenizer:
    """Deterministic fake tokenizer: one token per character."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class FakeSamplingClient:
    """Scores a completion by the bin id it spells; records every request."""

    def __init__(
        self, scores: dict[str, float] | None = None, error: Exception | None = None
    ):
        self.scores = scores or {}
        self.error = error
        self.requests: list[tuple[list[int], int]] = []
        self.tokenizer = CharTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    async def compute_logprobs(self, prompt, prompt_logprobs_last_n=None):
        if self.error is not None:
            raise self.error
        tokens = prompt.to_ints()
        self.requests.append((tokens, prompt_logprobs_last_n))
        text = self.tokenizer.decode(tokens[-prompt_logprobs_last_n:])
        bin_id = text.removesuffix("<|im_end|>")
        total = self.scores.get(bin_id, -5.0)
        per = total / prompt_logprobs_last_n
        return [None] * (len(tokens) - prompt_logprobs_last_n) + [
            per
        ] * prompt_logprobs_last_n


class FakeFallback:
    def __init__(self):
        self.asked: list[str] = []

    async def predict(self, exp, hypotheses, context):
        self.asked += [h.id for h in hypotheses]
        return {h.id: {b.id: 1.0 / len(exp.bins) for b in exp.bins} for h in hypotheses}


def _exp(exact: dict | None = None) -> Experiment:
    design = {"library": "fork", "diagnostic": "check"}
    if exact is not None:
        design["exact"] = exact
    return Experiment(
        id="e1",
        question_ids=["q1"],
        kind="compute",
        goal="Count something",
        language="python",
        code="print('RESULT: {}')",
        design=design,
        bins=[
            OutcomeBin(id="low", description="below one", rule="x < 1"),
            OutcomeBin(id="high", description="one or more", rule="x >= 1"),
            OutcomeBin(id="high_x", description="prefix of high", rule="x > 9"),
        ],
    )


def _hyp(hid: str, key: str) -> Hypothesis:
    return Hypothesis(
        id=hid,
        question_id="q1",
        statement=f"Option {key} is true.",
        author="test",
        rationale="",
        answer_key=key,
        prior=0.5,
        posterior=0.5,
    )


# --- prompt -----------------------------------------------------------------------------


def test_prompt_has_goal_bins_option_and_profile_but_no_code():
    exp, hyp = _exp(), _hyp("h1", "a")
    prompt = build_prompt(exp, hyp, [PROFILE])
    assert exp.goal in prompt and hyp.statement in prompt
    assert all(f"- {b.id}: {b.description}" in prompt for b in exp.bins)
    assert exp.code not in prompt and "x >= 1" not in prompt
    summary = profile_summary([PROFILE])
    assert summary in prompt
    assert "3000 genes x 4 libraries" in summary
    assert "libraries per group: 2" in summary
    assert "min/median 0.02" in summary
    assert "versioned gene IDs 100.0%" in summary
    assert (
        profile_summary([]) == "unavailable" and profile_summary(None) == "unavailable"
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert bin_completion("high") == "high<|im_end|>"


def test_training_and_inference_use_the_same_text_and_tokens():
    entry = library_index()["labels_swap_search"]
    record = {
        "case_id": "c1",
        "diagnostic": "labels_swap_search",
        "answer_key": entry.fault_key,
        "observed_bin": "swapped",
        "profile": PROFILE,
    }
    prompt, completion = record_example(record)
    option = entry.options[entry.fault_key]
    assert prompt == build_prompt(entry.experiment, option, [PROFILE])
    assert completion == bin_completion("swapped")
    assert record_prompt(record, entry.null_key) == build_prompt(
        entry.experiment, entry.options[entry.null_key], [PROFILE]
    )

    client = FakeSamplingClient()
    predictor = TinkerPredictor(client, profile=lambda: [PROFILE], eps=0.0)
    exp = entry.experiment.model_copy(deep=True)
    exp.design = {}
    asyncio.run(predictor.predict(exp, [option], context="ignored"))
    tok = CharTokenizer()
    arrays = sft_arrays(tok.encode(prompt), tok.encode(completion))
    trained = arrays["input"] + arrays["targets"][-1:]
    # The inference request for the observed bin is exactly the trained token sequence.
    assert trained in [tokens for tokens, _ in client.requests]
    assert len(client.requests) == len(exp.bins)
    # Weights cover exactly the completion tokens.
    assert sum(arrays["weights"]) == len(tok.encode(completion))
    assert arrays["weights"][-len(tok.encode(completion)) :] == [1.0] * len(
        tok.encode(completion)
    )


# --- scoring math -------------------------------------------------------------------------


def test_logprobs_to_row_is_a_floored_softmax():
    logprobs = {"low": -0.1, "high": -3.0, "high_x": -9.0}
    raw = logprobs_to_row(logprobs, eps=0.0)
    total = sum(math.exp(v) for v in logprobs.values())
    assert raw == pytest.approx({b: math.exp(v) / total for b, v in logprobs.items()})
    row = logprobs_to_row(logprobs, eps=0.1)
    assert sum(row.values()) == pytest.approx(1.0)
    assert min(row.values()) >= 0.1 / 3
    assert row == pytest.approx({b: 0.9 * p + 0.1 / 3 for b, p in raw.items()})
    # Same floor as the sampled predictor's smoothing: a one-hot count matches a one-hot row.
    counts = scoring.likelihood_from_samples({"low": 8, "high": 0, "high_x": 0}, 3, 0.1)
    assert counts["high"] >= 0.1 / 3 and row["high"] >= 0.1 / 3
    assert logprobs_to_row({"a": -math.inf, "b": -math.inf}) == {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError):
        logprobs_to_row({})
    assert completion_logprob([None, -1.0, -0.5, -0.25], 2) == pytest.approx(-0.75)
    with pytest.raises(ValueError):
        completion_logprob([None, None, -1.0], 2)


def test_predict_scores_bins_and_exact_rows_win():
    exact = {"a": {"low": 0.0, "high": 1.0, "high_x": 0.0}}
    exp = _exp(exact)
    ha, hb = _hyp("h1", "a"), _hyp("h2", "b")
    client = FakeSamplingClient({"low": -0.2, "high": -2.0, "high_x": -4.0})
    predictor = TinkerPredictor(client, profile=lambda: [PROFILE], eps=0.1)
    table = asyncio.run(predictor.predict(exp, [ha, hb], context=""))
    assert table["h1"] == library_row(exp, ha)  # floored exact row, no model call
    assert table["h1"]["low"] == pytest.approx(LIBRARY_FLOOR / 3)
    assert len(client.requests) == len(exp.bins)  # one prefill per bin, for h2 only
    row = table["h2"]
    assert set(row) == {"low", "high", "high_x"}
    assert sum(row.values()) == pytest.approx(1.0)
    assert row["low"] > row["high"] > row["high_x"] >= 0.1 / 3
    # The completion is scored with prompt_logprobs_last_n = its own length.
    for tokens, last_n in client.requests:
        tail = CharTokenizer().decode(tokens[-last_n:])
        assert tail.endswith("<|im_end|>")
    # Prefix-safe bins: "high" and "high_x" are different completions.
    tails = {CharTokenizer().decode(t[-n:]) for t, n in client.requests}
    assert tails == {"low<|im_end|>", "high<|im_end|>", "high_x<|im_end|>"}


def test_tinker_errors_fall_back_to_the_injected_predictor():
    exact = {"a": {"low": 1.0, "high": 0.0, "high_x": 0.0}}
    exp = _exp(exact)
    fallback = FakeFallback()
    client = FakeSamplingClient(error=RuntimeError("tinker is down"))
    predictor = TinkerPredictor(client, fallback=fallback)
    table = asyncio.run(
        predictor.predict(exp, [_hyp("h1", "a"), _hyp("h2", "b")], context="")
    )
    assert fallback.asked == ["h2"]  # only the options without exact rows
    assert table["h1"]["low"] == pytest.approx(1 - LIBRARY_FLOOR + LIBRARY_FLOOR / 3)
    assert table["h2"] == pytest.approx({"low": 1 / 3, "high": 1 / 3, "high_x": 1 / 3})
    assert exp.likelihood_source == "sampled"
    with pytest.raises(RuntimeError):
        asyncio.run(
            TinkerPredictor(client).predict(_exp(), [_hyp("h2", "b")], context="")
        )


def test_stated_rows_win_unless_the_model_may_replace_them():
    stated = {"b": {"low": 0.2, "high": 0.7, "high_x": 0.1}}
    exp = _exp()
    exp.design["stated"] = stated
    hb = _hyp("h2", "b")
    client = FakeSamplingClient({"low": -0.1, "high": -3.0, "high_x": -3.0})
    keep = TinkerPredictor(client, eps=0.1)
    table = asyncio.run(keep.predict(exp, [hb], context=""))
    assert table["h2"] == library_row(exp, hb) and not client.requests
    replace = TinkerPredictor(client, eps=0.1, use_stated_rows=False)
    table = asyncio.run(replace.predict(exp, [hb], context=""))
    assert len(client.requests) == 3 and table["h2"]["low"] > table["h2"]["high"]


def test_tinker_path_opens_lazily():
    predictor = TinkerPredictor("tinker://run/sampler_weights/x", profile=None)
    assert predictor._client is None  # nothing opened, no SDK needed until predict()
    with pytest.raises(ValueError):
        TinkerPredictor(FakeSamplingClient(), eps=1.5)


# --- config, schema and wiring ------------------------------------------------------------


def test_tinker_is_a_predictor_mode_and_a_likelihood_source():
    assert RunConfig(predictor="tinker").predictor == "tinker"
    assert load_config("default", predictor="tinker").predictor == "tinker"
    assert _exp().model_copy(update={"likelihood_source": "tinker"}).likelihood_source
    Experiment.model_validate({**_exp().model_dump(), "likelihood_source": "tinker"})


def test_orchestrator_requests_tinker_rows(tmp_path: Path):
    lab = FakeLab()
    deps = fake_deps(lab, clock=FakeClock())
    inner = deps.predictor
    seen: list[str] = []

    class Recording:
        async def predict(self, exp, hypotheses, context):
            seen.append(exp.likelihood_source)
            return await inner.predict(exp, hypotheses, context)

    deps.predictor = Recording()
    cfg = load_config("default", runs_dir=str(tmp_path), predictor="tinker")
    asyncio.run(run(lab, cfg, deps, run_id="r-tinker"))
    assert seen and set(seen) == {"tinker"}


def test_build_task9_run_uses_tinker_predictor(tmp_path: Path, monkeypatch):
    from labs.bixbench3.swarm_lab import build_task9_run

    counts, genes, design = simulate(0, n_genes=200)
    write_case(tmp_path / "case", *suite_case(counts, genes, design, 0))
    cfg = load_config("default", runs_dir=str(tmp_path / "runs"), predictor="tinker")
    monkeypatch.delenv("CRUX_TINKER_PREDICTOR", raising=False)
    with pytest.raises(ValueError, match="CRUX_TINKER_PREDICTOR"):
        build_task9_run(cfg, case_dir=tmp_path / "case", llm=object())
    monkeypatch.setenv("CRUX_TINKER_PREDICTOR", "tinker://run-1/sampler_weights/best")
    lab, deps = build_task9_run(cfg, case_dir=tmp_path / "case", llm=object())
    assert isinstance(deps.predictor, TinkerPredictor)
    assert isinstance(deps.predictor.fallback, SampledPredictor)
    assert deps.predictor.use_stated_rows
    monkeypatch.setenv("CRUX_TINKER_OVERRIDE_STATED", "1")
    _, override = build_task9_run(cfg, case_dir=tmp_path / "case", llm=object())
    assert not override.predictor.use_stated_rows
    assert deps.predictor._target == "tinker://run-1/sampler_weights/best"
    lab.profile_facts.append(PROFILE)
    assert deps.predictor.profile() == [PROFILE]
    sampled_cfg = load_config("default", runs_dir=str(tmp_path / "runs"))
    _, sampled = build_task9_run(sampled_cfg, case_dir=tmp_path / "case", llm=object())
    assert isinstance(sampled.predictor, SampledPredictor)


# --- synthetic data and the example generator ----------------------------------------------


def test_knob_defaults_reproduce_the_original_generator_and_new_knobs_work(
    tmp_path: Path,
):
    a, genes_a, _ = simulate(5, n_genes=300)
    b, genes_b, _ = simulate(5, knobs=Knobs(n_genes=300))
    assert np.array_equal(a, b) and genes_a == genes_b
    counts, genes, design = simulate(
        1, knobs=Knobs(n_genes=300, replicates=3, **Knobs.sharing(False))
    )
    assert counts.shape == (300, 24) and len(design) == 24
    sheet, frame = featurecounts_case(counts, genes, design)
    for fault in FAULT_FORK:
        rng = np.random.default_rng(3)
        _, new_frame, implicated = plant(fault, sheet, frame, "HEK293", rng)
        assert new_frame.shape == frame.shape
        if fault == "duplicate_library":
            source, target = (
                sheet.set_index("sample_id").loc[s, "count_column"] for s in implicated
            )
            assert new_frame[source].equals(new_frame[target])
    assert Knobs.from_dict(Knobs().to_dict()) == Knobs()


def test_example_generator_labels_each_fork_with_its_true_option():
    specs = [dict(s) for s in case_specs("train", 9, base_seed=11)[:9]]
    clean = next(s for s in specs if s["fault"] is None)
    planted = dict(clean, case_id="train-x", fault="duplicate_library")
    for spec in (clean, planted):
        records = make_case(spec)
        assert len(records) == 12
        by_fork = {}
        for record in records:
            entry = library_index()[record["diagnostic"]]
            by_fork.setdefault(record["fork"], set()).add(record["answer_key"])
            assert record["observed_bin"] in {b.id for b in entry.experiment.bins}
            assert "shape" in record["profile"]
            assert record["profile_summary"] == profile_summary([record["profile"]])
        for fork, keys in by_fork.items():
            entry = next(e for e in library_index().values() if e.fork == fork)
            expected = (
                entry.fault_key
                if spec["fault"] and FAULT_FORK[spec["fault"]] == fork
                else entry.null_key
            )
            assert keys == {expected}, (spec["fault"], fork)
        if spec["fault"] == "duplicate_library":
            observed = {r["diagnostic"]: r["observed_bin"] for r in records}
            assert observed["duplicates_identical"] == "duplicate"
            assert observed["duplicates_correlation"] == "duplicate"
    faults = [s["fault"] for s in case_specs("val", 90, base_seed=1)]
    assert faults.count(None) == 30 and all(faults.count(f) == 10 for f in FAULT_FORK)
    shift = case_specs("shift", 4, base_seed=1)
    assert [s["shift_kind"] for s in shift] == ["weak_effect", "three_replicates"] * 2


def test_profile_case_runs_the_lab_profile_code(tmp_path: Path):
    counts, genes, design = simulate(2, n_genes=150)
    write_case(tmp_path, *suite_case(counts, genes, design, 2))
    facts = profile_case(tmp_path)
    assert facts["shape"] == [150, 16]
    assert set(facts["library_sizes"]) == {f"S{i:02d}" for i in range(1, 17)}
    assert facts["versioned_gene_ids"] == 149  # the _PAR_Y tail does not match


def test_calibration_metrics():
    from eval.eval_predictor import ece, floor, metrics, total_variation

    perfect = metrics([({"a": 1.0, "b": 0.0}, "a")] * 4)
    assert perfect["log_loss"] == pytest.approx(0.0) and perfect["brier"] == 0.0
    assert perfect["accuracy"] == 1.0 and perfect["ece"] == pytest.approx(0.0)
    coin = metrics([({"a": 0.5, "b": 0.5}, "a"), ({"a": 0.5, "b": 0.5}, "b")])
    assert coin["log_loss"] == pytest.approx(math.log(2)) and coin["brier"] == 0.5
    wrong = metrics([({"a": 1.0, "b": 0.0}, "b")])
    assert wrong["impossible"] == 1 and wrong["ece"] == pytest.approx(1.0)
    assert floor({"a": 1.0, "b": 0.0}, 0.1) == pytest.approx({"a": 0.95, "b": 0.05})
    assert ece([0.8, 0.8], [True, False]) == pytest.approx(0.3)
    assert total_variation({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == 1.0
