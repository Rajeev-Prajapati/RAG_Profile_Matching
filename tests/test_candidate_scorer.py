
import unittest

from matching.candidate_scorer import (
    CandidateScorer,
    ScoringWeights
)


class TestCandidateScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = CandidateScorer()

        self.requirements = {
            "must_have_skills": ["Python", "FastAPI"],
            "preferred_skills": [
                "Docker", "PostgreSQL", "AWS"
            ],
            "min_experience": 5
        }

        self.candidate = {
            "candidate_name": "John Doe",
            "eligible": True,
            "semantic_score": 0.85,
            "experience_years": 6,
            "matched_must_have": ["Python", "FastAPI"],
            "matched_preferred": ["Docker", "PostgreSQL"],
            "missing_preferred": ["AWS"]
        }

    def test_weighted_score(self):

        result = self.scorer.calculate_score(
            self.candidate,
            self.requirements
        )

        self.assertAlmostEqual(
            result["match_score"],
            87.33,
            places=2
        )

    def test_score_within_range(self):

        result = self.scorer.calculate_score(
            self.candidate,
            self.requirements
        )

        self.assertGreaterEqual(
            result["match_score"], 0
        )

        self.assertLessEqual(
            result["match_score"], 100
        )

    def test_experience_capped_at_100(self):

        self.assertEqual(
            self.scorer.experience_score(12, 5),
            100.0
        )

    def test_no_preferred_skills(self):

        requirements = {
            **self.requirements,
            "preferred_skills": []
        }

        result = self.scorer.calculate_score(
            self.candidate,
            requirements
        )

        self.assertIsNone(
            result["score_breakdown"]["preferred_skills"]
        )

        self.assertAlmostEqual(
            sum(result["effective_weights"].values()),
            1.0,
            places=3
        )

    def test_invalid_weights(self):

        with self.assertRaises(ValueError):
            ScoringWeights(
                semantic=0.50,
                mandatory_skills=0.30,
                preferred_skills=0.30,
                experience=0.10
            )

    def test_ineligible_candidate_rejected(self):

        candidate = {
            **self.candidate,
            "eligible": False
        }

        with self.assertRaises(ValueError):
            self.scorer.calculate_score(
                candidate,
                self.requirements
            )

    def test_unknown_experience(self):

        self.assertEqual(
            self.scorer.experience_score(None, 5),
            0.0
        )

    def test_semantic_clamping(self):

        self.assertEqual(
            self.scorer.semantic_score(-0.2),
            0.0
        )

        self.assertEqual(
            self.scorer.semantic_score(1.2),
            100.0
        )


if __name__ == "__main__":
    unittest.main()
