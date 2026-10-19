
import json
import unittest
from pathlib import Path


class TestOutput(unittest.TestCase):

    def test_generated_json(self):

        output_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "matches.json"
        )

        if not output_path.exists():
            self.skipTest(
                "Run job_matcher.py to generate matches.json"
            )

        with output_path.open(
            "r", encoding="utf-8"
        ) as file:
            data = json.load(file)

        self.assertIn("job_description", data)
        self.assertIn("top_matches", data)

        self.assertIsInstance(
            data["top_matches"], list
        )

        self.assertLessEqual(
            len(data["top_matches"]), 10
        )

        previous_score = float("inf")
        seen_paths = set()

        required_fields = {
            "candidate_name",
            "resume_path",
            "match_score",
            "matched_skills",
            "relevant_excerpts",
            "reasoning"
        }

        for candidate in data["top_matches"]:

            self.assertTrue(
                required_fields.issubset(candidate)
            )

            score = candidate["match_score"]

            self.assertIsInstance(
                score, (int, float)
            )

            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            self.assertLessEqual(score, previous_score)

            previous_score = score

            path = candidate["resume_path"]

            self.assertNotIn(path, seen_paths)
            seen_paths.add(path)

            self.assertIsInstance(
                candidate["matched_skills"], list
            )

            self.assertIsInstance(
                candidate["relevant_excerpts"], list
            )

            self.assertIsInstance(
                candidate["reasoning"], list
            )


if __name__ == "__main__":
    unittest.main()
