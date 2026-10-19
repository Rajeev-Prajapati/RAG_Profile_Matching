
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate import (
    candidate_id_from_path,
    macro_average,
    run_evaluation
)


class FakeRetriever:

    def search_candidates(
        self,
        job_description,
        retrieve_all=True
    ):
        return [
            {
                "candidate_name": "Candidate One",
                "resume_path": "resumes/C001_python.pdf"
            },
            {
                "candidate_name": "Candidate Two",
                "resume_path": "resumes/C002_python.pdf"
            }
        ]


class FakeMatcher:

    def match_candidates(
        self,
        job_description,
        must_have_skills,
        preferred_skills,
        min_experience,
        top_n
    ):
        return {
            "top_matches": [
                {
                    "candidate_name": "Candidate Two",
                    "resume_path": "resumes/C002_python.pdf",
                    "match_score": 95
                }
            ],
            "eligible_count": 1
        }


class TestEvaluationPipeline(unittest.TestCase):

    def test_candidate_id_extraction(self):
        self.assertEqual(
            candidate_id_from_path(
                r"E:\dataset\resumes\C003_backend.pdf"
            ),
            "C003"
        )

    def test_macro_average(self):
        rows = [
            {
                "method": "semantic_only",
                "precision_at_5": 0.4,
                "recall_at_5": 0.5,
                "hit_rate_at_5": 1,
                "precision_at_10": 0.2,
                "recall_at_10": 0.5,
                "hit_rate_at_10": 1,
                "reciprocal_rank": 0.5,
                "mean_ms": 10,
                "median_ms": 10,
                "p95_ms": 12
            },
            {
                "method": "semantic_only",
                "precision_at_5": 0.6,
                "recall_at_5": 0.75,
                "hit_rate_at_5": 1,
                "precision_at_10": 0.3,
                "recall_at_10": 0.75,
                "hit_rate_at_10": 1,
                "reciprocal_rank": 1.0,
                "mean_ms": 20,
                "median_ms": 20,
                "p95_ms": 24
            }
        ]

        summary = macro_average(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(
            summary[0]["precision_at_5"], 0.5
        )
        self.assertEqual(
            summary[0]["mrr"], 0.75
        )

    def test_full_evaluation_with_fakes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_dir = root / "dataset"
            results_dir = root / "results"

            dataset_dir.mkdir()

            candidates = [
                {"candidate_id": "C001"},
                {"candidate_id": "C002"}
            ]

            jobs = [{
                "job_id": "JD001",
                "title": "Python Developer",
                "job_description": "Python developer",
                "must_have_skills": ["Python"],
                "preferred_skills": [],
                "min_experience": 2
            }]

            labels = {
                "JD001": {
                    "judgments": [
                        {
                            "candidate_id": "C001",
                            "relevant": False
                        },
                        {
                            "candidate_id": "C002",
                            "relevant": True
                        }
                    ]
                }
            }

            for name, data in [
                ("candidates.json", candidates),
                ("job_descriptions.json", jobs),
                ("relevance_labels.json", labels)
            ]:
                (dataset_dir / name).write_text(
                    json.dumps(data),
                    encoding="utf-8"
                )

            report = run_evaluation(
                dataset_dir=dataset_dir,
                results_dir=results_dir,
                repetitions=2,
                warmups=0,
                semantic_retriever=FakeRetriever(),
                job_matcher=FakeMatcher()
            )

            self.assertEqual(
                len(report["per_job"]), 2
            )

            semantic = report["per_job"][0]
            hybrid = report["per_job"][1]

            self.assertEqual(
                semantic["reciprocal_rank"], 0.5
            )

            self.assertEqual(
                hybrid["reciprocal_rank"], 1.0
            )

            self.assertEqual(
                hybrid["precision_at_5"], 0.2
            )

            self.assertTrue(
                (
                    results_dir /
                    "performance_metrics.json"
                ).exists()
            )

            self.assertTrue(
                (
                    results_dir /
                    "per_job_metrics.csv"
                ).exists()
            )

            self.assertTrue(
                (
                    results_dir /
                    "summary_metrics.csv"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
