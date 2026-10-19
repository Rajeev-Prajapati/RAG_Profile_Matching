
import unittest

from matching.hybrid_matcher import HybridMatcher
from matching.job_requirements import JobRequirementExtractor


class TestHybridMatching(unittest.TestCase):

    def setUp(self):

        self.matcher = HybridMatcher()
        self.extractor = JobRequirementExtractor()

        self.requirements = {
            "must_have_skills": ["Python", "FastAPI"],
            "preferred_skills": ["Docker", "AWS"],
            "min_experience": 5
        }

    def make_candidate(self, skills, experience):
        return {
            "candidate_name": "Test Candidate",
            "resume_path": "resumes/test.pdf",
            "skills": skills,
            "experience_years": experience,
            "semantic_score": 0.85,
            "relevant_chunks": []
        }

    def test_candidate_meets_all_requirements(self):

        candidate = self.make_candidate(
            "Python, FastAPI, Docker",
            6
        )

        result = self.matcher.evaluate_candidate(
            candidate, self.requirements
        )

        self.assertTrue(result["eligible"])
        self.assertEqual(result["missing_must_have"], [])
        self.assertEqual(
            result["matched_preferred"], ["Docker"]
        )

    def test_missing_mandatory_skill(self):

        candidate = self.make_candidate(
            "Python, Django, Docker",
            7
        )

        result = self.matcher.evaluate_candidate(
            candidate, self.requirements
        )

        self.assertFalse(result["eligible"])
        self.assertIn(
            "FastAPI", result["missing_must_have"]
        )

    def test_insufficient_experience(self):

        candidate = self.make_candidate(
            "Python, FastAPI, Docker",
            3
        )

        result = self.matcher.evaluate_candidate(
            candidate, self.requirements
        )

        self.assertFalse(result["eligible"])
        self.assertFalse(result["experience_eligible"])

    def test_unknown_experience(self):

        candidate = self.make_candidate(
            "Python, FastAPI",
            None
        )

        result = self.matcher.evaluate_candidate(
            candidate, self.requirements
        )

        self.assertFalse(result["eligible"])

    def test_preferred_skill_not_mandatory(self):

        candidate = self.make_candidate(
            "Python, FastAPI",
            5
        )

        result = self.matcher.evaluate_candidate(
            candidate, self.requirements
        )

        self.assertTrue(result["eligible"])
        self.assertEqual(
            result["missing_preferred"],
            ["Docker", "AWS"]
        )

    def test_skill_word_boundaries(self):

        self.assertFalse(
            self.matcher.has_skill("JavaScript", "Java")
        )

        self.assertTrue(
            self.matcher.has_skill("C# Developer", "C#")
        )

    def test_extract_minimum_experience(self):

        requirements = self.extractor.extract_requirements(
            "Python developer with at least 5 years experience",
            must_have_skills=["Python"]
        )

        self.assertEqual(
            requirements["min_experience"], 5.0
        )

    def test_explicit_requirements_override_defaults(self):

        requirements = self.extractor.extract_requirements(
            "Python and FastAPI development",
            must_have_skills=["Python"],
            preferred_skills=["FastAPI"],
            min_experience=4
        )

        self.assertEqual(
            requirements["must_have_skills"], ["Python"]
        )

        self.assertEqual(
            requirements["preferred_skills"], ["FastAPI"]
        )

        self.assertEqual(
            requirements["min_experience"], 4.0
        )


if __name__ == "__main__":
    unittest.main()
