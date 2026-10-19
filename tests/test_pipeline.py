
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vector_store.chroma_store import ChromaStore
from retrieval.semantic_retriever import SemanticRetriever
from matching.hybrid_matcher import HybridMatcher
from matching.candidate_scorer import CandidateScorer


class FakeEmbeddingService:

    def generate_embedding(self, text):
        vector = np.zeros(384, dtype=float)

        text = text.lower()

        if "python" in text:
            vector[0] = 1.0

        if "fastapi" in text:
            vector[1] = 1.0

        if "docker" in text:
            vector[2] = 1.0

        if not vector.any():
            vector[3] = 1.0

        return vector / np.linalg.norm(vector)


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.store = ChromaStore(
            persist_directory=self.temp_dir.name,
            collection_name="test_resumes"
        )

        self.embedding_service = FakeEmbeddingService()

        self.retriever = SemanticRetriever.__new__(
            SemanticRetriever
        )

        self.retriever.embedding_service = (
            self.embedding_service
        )

        self.retriever.vector_store = self.store

        self.matcher = HybridMatcher()
        self.scorer = CandidateScorer()

        self.resume_path = str(
            Path(self.temp_dir.name) / "candidate.pdf"
        )

        self.chunk = {
            "resume_path": self.resume_path,
            "section": "Experience",
            "text": (
                "Developed Python FastAPI services "
                "and Docker deployments."
            ),
            "metadata": {
                "candidate_name": "Test Candidate",
                "skills": ["Python", "FastAPI", "Docker"],
                "experience_years": 6.0,
                "education": "B.Tech",
                "chunk_index": 1
            }
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def index_sample(self):
        text = (
            f"Section: {self.chunk['section']}\n"
            f"{self.chunk['text']}"
        )

        vector = self.embedding_service.generate_embedding(
            text
        )

        self.store.store_chunks(
            [self.chunk],
            np.array([vector])
        )

    def test_store_and_retrieve(self):
        self.index_sample()

        candidates = self.retriever.search_candidates(
            "Python FastAPI developer",
            retrieve_all=True
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["candidate_name"],
            "Test Candidate"
        )

    def test_no_duplicate_on_upsert(self):
        self.index_sample()
        self.index_sample()

        self.assertEqual(
            self.store.count_chunks(), 1
        )

    def test_persistence(self):
        self.index_sample()

        reopened = ChromaStore(
            persist_directory=self.temp_dir.name,
            collection_name="test_resumes"
        )

        self.assertEqual(
            reopened.count_chunks(), 1
        )

    def test_complete_matching_flow(self):
        self.index_sample()

        candidates = self.retriever.search_candidates(
            "Python FastAPI developer",
            retrieve_all=True
        )

        requirements = {
            "must_have_skills": ["Python", "FastAPI"],
            "preferred_skills": ["Docker"],
            "min_experience": 5
        }

        evaluated = self.matcher.evaluate_candidate(
            candidates[0],
            requirements
        )

        self.assertTrue(evaluated["eligible"])

        scored = self.scorer.calculate_score(
            evaluated,
            requirements
        )

        self.assertGreaterEqual(
            scored["match_score"], 0
        )

        self.assertLessEqual(
            scored["match_score"], 100
        )

    def test_missing_mandatory_skill(self):
        self.index_sample()

        candidates = self.retriever.search_candidates(
            "Python developer",
            retrieve_all=True
        )

        requirements = {
            "must_have_skills": ["Python", "Kubernetes"],
            "preferred_skills": [],
            "min_experience": 5
        }

        evaluated = self.matcher.evaluate_candidate(
            candidates[0],
            requirements
        )

        self.assertFalse(evaluated["eligible"])
        self.assertIn(
            "Kubernetes",
            evaluated["missing_must_have"]
        )


if __name__ == "__main__":
    unittest.main()
