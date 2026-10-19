
import unittest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    latency_summary,
    evaluate_ranking
)


class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        self.retrieved = [
            "C001", "C002", "C003", "C004", "C005"
        ]

        self.relevant = {
            "C002", "C003", "C006"
        }

    def test_precision_at_5(self):
        self.assertAlmostEqual(
            precision_at_k(
                self.retrieved,
                self.relevant,
                5
            ),
            0.4
        )

    def test_recall_at_5(self):
        self.assertAlmostEqual(
            recall_at_k(
                self.retrieved,
                self.relevant,
                5
            ),
            2 / 3
        )

    def test_hit_rate(self):
        self.assertEqual(
            hit_rate_at_k(
                self.retrieved,
                self.relevant,
                1
            ),
            0
        )

        self.assertEqual(
            hit_rate_at_k(
                self.retrieved,
                self.relevant,
                5
            ),
            1
        )

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(
            reciprocal_rank(
                self.retrieved,
                self.relevant
            ),
            0.5
        )

    def test_mrr(self):
        rankings = [
            ["C001", "C002"],
            ["C003", "C004"]
        ]

        relevant_sets = [
            {"C002"},
            {"C003"}
        ]

        self.assertAlmostEqual(
            mean_reciprocal_rank(
                rankings,
                relevant_sets
            ),
            0.75
        )

    def test_empty_ground_truth(self):
        self.assertIsNone(
            recall_at_k(
                self.retrieved, set(), 5
            )
        )

        self.assertIsNone(
            reciprocal_rank(
                self.retrieved, set()
            )
        )

    def test_no_relevant_results(self):
        self.assertEqual(
            reciprocal_rank(
                ["C001", "C002"],
                {"C010"}
            ),
            0
        )

    def test_precision_short_ranking(self):
        self.assertAlmostEqual(
            precision_at_k(
                ["C001"],
                {"C001"},
                5
            ),
            0.2
        )

    def test_duplicate_ranking_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_ranking(
                ["C001", "C001"],
                {"C001"}
            )

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            precision_at_k(
                self.retrieved,
                self.relevant,
                0
            )

    def test_latency_summary(self):
        summary = latency_summary(
            [10, 20, 30, 40, 50]
        )

        self.assertEqual(
            summary["mean_ms"], 30
        )

        self.assertEqual(
            summary["median_ms"], 30
        )

        self.assertEqual(
            summary["p95_ms"], 50
        )


if __name__ == "__main__":
    unittest.main()
