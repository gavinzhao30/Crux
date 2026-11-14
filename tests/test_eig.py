from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from core.eig import (
    e_value,
    expected_info_gain,
    is_surprise,
    likelihood_from_samples,
    question_status,
    rank,
    score,
    update,
)


class TestCategoricalScoring(TestCase):
    def test_exact_distinct_predictions_with_null(self):
        prior = [0.425, 0.425, 0.15]
        rows = [[1, 0], [0, 1], [0.5, 0.5]]
        self.assertAlmostEqual(expected_info_gain(prior, rows), 0.85)

    def test_unanimous_samples_are_smoothed(self):
        rows = [
            likelihood_from_samples([8, 0], 2),
            likelihood_from_samples([0, 8], 2),
            [0.5, 0.5],
        ]
        np.testing.assert_allclose(rows[0], [0.9, 0.1])
        np.testing.assert_allclose(likelihood_from_samples([0, 0], 2), [0.5, 0.5])
        self.assertAlmostEqual(expected_info_gain([0.425, 0.425, 0.15], rows), 0.451354, places=6)

    def test_identical_likelihoods_are_uninformative(self):
        self.assertAlmostEqual(expected_info_gain([1, 1, 1], [[0.7, 0.3]] * 3), 0)

    def test_merging_uses_prior_weighted_likelihoods(self):
        prior = [0.1, 0.3, 0.6]
        rows = [[1, 0], [0, 1], [0, 1]]
        merged = expected_info_gain(prior, rows, ["a", "a", "b"])
        self.assertAlmostEqual(merged, expected_info_gain([0.4, 0.6], [[0.25, 0.75], [0, 1]]))
        self.assertEqual(expected_info_gain(prior, rows, ["a", "a", "a"]), 0)
        self.assertAlmostEqual(
            expected_info_gain(prior, rows, [None] * 3), expected_info_gain(prior, rows)
        )

    def test_update_matches_hand_calculation(self):
        np.testing.assert_allclose(update([0.25, 0.75], [[0.9, 0.1], [0.1, 0.9]], 0), [0.75, 0.25])
        with self.assertRaisesRegex(ValueError, "zero mixture probability"):
            update([0.5, 0.5], [[1, 0], [1, 0]], 1)

    def test_rare_outcome_raises_null_posterior(self):
        prior = [0.425, 0.425, 0.15]
        rows = [[0.99, 0.01], [0.98, 0.02], [0.5, 0.5]]
        self.assertTrue(is_surprise(prior, rows, 1))
        self.assertGreater(update(prior, rows, 1)[2], 0.4)

    def test_null_threshold_can_trigger_on_its_own(self):
        prior = [0.6, 0.4]
        rows = [[0.8, 0.2], [0.5, 0.5]]
        self.assertFalse(is_surprise(prior, rows, 1))
        self.assertTrue(is_surprise(prior, rows, 1, null_index=1))

    def test_stop_rules(self):
        self.assertEqual(question_status([0.45, 0.55], 0, 0, ["same", "same"]), "robust")
        self.assertEqual(question_status([0.9, 0.1], 0.2, 4), "settled")
        self.assertEqual(question_status([0.6, 0.4], 0.049, 0), "unresolved")
        self.assertEqual(question_status([0.6, 0.4], 0.2, 4), "unresolved")
        self.assertEqual(question_status([0.6, 0.4], None, 0), "unresolved")
        self.assertEqual(question_status([0.6, 0.4], 0.05, 3, [None, None]), "open")

    def test_score_and_e_value(self):
        self.assertAlmostEqual(score(0.6, 60, impact=0.5), 0.25)
        self.assertAlmostEqual(e_value(0.001), 15.8113883)

    def test_rank_uses_ids_current_posteriors_cost_and_uniform_null(self):
        question = SimpleNamespace(id="question", impact=0.5)
        hypotheses = [
            SimpleNamespace(
                id=hypothesis_id,
                question_id=question.id,
                status=status,
                answer_key=None,
                is_null=hypothesis_id == "null",
                prior=0.25,
                posterior=posterior,
            )
            for hypothesis_id, status, posterior in (
                ("b", "active", 0.25),
                ("a", "active", 0.60),
                ("null", "active", 0.15),
                ("rejected", "rejected", 0),
            )
        ]
        slow = SimpleNamespace(
            id="slow",
            bins=[SimpleNamespace(id="yes"), SimpleNamespace(id="no")],
            likelihood={"a": {"no": 0, "yes": 1}, "b": {"yes": 0, "no": 1}},
            est_cost_s=120,
        )
        fast = deepcopy(slow)
        fast.id = "fast"
        fast.est_cost_s = 30
        tied = deepcopy(fast)
        tied.id = "first"

        self.assertEqual(rank(question, hypotheses, [slow, tied, fast]), [fast, tied, slow])
        expected = expected_info_gain([0.25, 0.60, 0.15], [[0, 1], [1, 0], [0.5, 0.5]])
        self.assertAlmostEqual(fast.eig_bits, expected)
        self.assertAlmostEqual(fast.score, score(expected, 30, impact=0.5))
        self.assertGreater(fast.score, slow.score)
