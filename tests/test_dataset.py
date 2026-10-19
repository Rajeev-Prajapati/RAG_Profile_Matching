
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from dataset.generate_dataset import (
    generate_dataset,
    label_candidate
)


class TestSyntheticDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)

        cls.result = generate_dataset(
            output_dir=cls.root,
            seed=42
        )

        cls.candidates = cls.result["candidates"]
        cls.jobs = cls.result["jobs"]
        cls.relevance = cls.result["relevance"]

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_resume_count(self):
        self.assertEqual(len(self.candidates), 35)

        files = [
            path
            for path in (self.root / "resumes").iterdir()
            if path.is_file()
        ]

        self.assertEqual(len(files), 35)

    def test_file_format_distribution(self):
        files = list((self.root / "resumes").iterdir())

        counts = Counter(
            path.suffix.lower() for path in files
        )

        self.assertEqual(counts[".pdf"], 15)
        self.assertEqual(counts[".docx"], 10)
        self.assertEqual(counts[".txt"], 10)

    def test_seven_roles(self):
        roles = Counter(
            candidate["role"]
            for candidate in self.candidates
        )

        self.assertEqual(len(roles), 7)

        for count in roles.values():
            self.assertEqual(count, 5)

    def test_unique_candidate_ids(self):
        ids = [
            candidate["candidate_id"]
            for candidate in self.candidates
        ]

        self.assertEqual(len(ids), len(set(ids)))

    def test_resume_files_are_readable(self):
        for candidate in self.candidates:
            path = self.root / candidate["resume_path"]

            self.assertTrue(path.exists())

            if path.suffix == ".pdf":
                reader = PdfReader(str(path))

                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

            elif path.suffix == ".docx":
                document = Document(path)

                text = "\n".join(
                    paragraph.text
                    for paragraph in document.paragraphs
                )

            else:
                text = path.read_text(
                    encoding="utf-8"
                )

            self.assertIn(
                candidate["candidate_name"],
                text
            )

            self.assertIn(
                "SKILLS",
                text.upper()
            )

            self.assertIn(
                "EXPERIENCE",
                text.upper()
            )

    def test_job_count(self):
        self.assertEqual(len(self.jobs), 5)

        job_ids = [
            job["job_id"]
            for job in self.jobs
        ]

        self.assertEqual(len(job_ids), len(set(job_ids)))

    def test_ground_truth_completeness(self):
        total_judgments = 0

        for job in self.jobs:
            judgments = self.relevance[
                job["job_id"]
            ]["judgments"]

            self.assertEqual(len(judgments), 35)

            total_judgments += len(judgments)

        self.assertEqual(total_judgments, 175)

    def test_ground_truth_consistency(self):
        for job in self.jobs:
            judgments = self.relevance[
                job["job_id"]
            ]["judgments"]

            by_id = {
                item["candidate_id"]: item
                for item in judgments
            }

            for candidate in self.candidates:
                expected = label_candidate(
                    candidate, job
                )

                actual = by_id[
                    candidate["candidate_id"]
                ]

                self.assertEqual(
                    actual["relevant"],
                    expected["relevant"]
                )

    def test_relevant_and_non_relevant_examples(self):
        for job in self.jobs:
            judgments = self.relevance[
                job["job_id"]
            ]["judgments"]

            relevant_count = sum(
                item["relevant"]
                for item in judgments
            )

            self.assertGreater(relevant_count, 0)
            self.assertLess(relevant_count, 35)

    def test_json_files_exist(self):
        for filename in [
            "candidates.json",
            "job_descriptions.json",
            "relevance_labels.json"
        ]:
            path = self.root / filename

            self.assertTrue(path.exists())

            with path.open(
                encoding="utf-8"
            ) as file:
                json.load(file)


if __name__ == "__main__":
    unittest.main()
