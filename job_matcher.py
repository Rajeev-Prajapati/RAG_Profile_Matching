import json
from matching.candidate_scorer import (
    CandidateScorer,
    ScoringWeights
)
from matching.job_requirements import JobRequirementExtractor
from matching.hybrid_matcher import HybridMatcher
from retrieval.semantic_retriever import SemanticRetriever
from pathlib import Path


class JobMatcher:

    def __init__(self, weights=None):
        self.retriever = SemanticRetriever()
        self.requirement_extractor = JobRequirementExtractor()
        self.hybrid_matcher = HybridMatcher()
        self.retriever = SemanticRetriever()
        self.requirement_extractor = JobRequirementExtractor()
        self.hybrid_matcher = HybridMatcher()
        self.scorer = CandidateScorer(weights=weights)

    
    def match_candidates(
        self,
        job_description,
        must_have_skills=None,
        preferred_skills=None,
        min_experience=None,
        top_n=10
    ):
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        # Step 1: Extract job requirements
        requirements = (
            self.requirement_extractor.extract_requirements(
                job_description=job_description,
                must_have_skills=must_have_skills,
                preferred_skills=preferred_skills,
                min_experience=min_experience
            )
        )

        # Step 2: Retrieve candidates from ChromaDB
        candidates = self.retriever.search_candidates(
            job_description=job_description,
            retrieve_all=True
        )

        scored_candidates = []
        rejected_candidates = []

        # Step 3: Check eligibility and calculate scores
        for candidate in candidates:

            evaluation = self.hybrid_matcher.evaluate_candidate(
                candidate,
                requirements
            )

            if evaluation["eligible"]:
                scored = self.scorer.calculate_score(
                    evaluation,
                    requirements
                )

                scored_candidates.append(scored)

            else:
                rejected_candidates.append(evaluation)

        # Step 4: Rank by final weighted score
        scored_candidates.sort(
            key=lambda candidate: candidate["match_score"],
            reverse=True
        )

        # Step 5: Build assignment-compatible output
        top_matches = []

        for candidate in scored_candidates[:top_n]:

            excerpts = [
                {
                    "section": chunk["metadata"]["section"],
                    "text": chunk["text"],
                    "similarity": round(
                        chunk["similarity"], 4
                    )
                }
                for chunk in candidate["relevant_chunks"][:3]
            ]

            matched_skills = list(dict.fromkeys(
                candidate["matched_must_have"]
                + candidate["matched_preferred"]
            ))

            top_matches.append({
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "match_score": candidate["match_score"],
                "matched_skills": matched_skills,
                "relevant_excerpts": excerpts,
                "reasoning": candidate["scoring_reasoning"],
                "score_breakdown": candidate["score_breakdown"],
                "effective_weights": candidate["effective_weights"]
            })

        rejected = [
            {
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "missing_must_have": candidate["missing_must_have"],
                "experience_eligible": candidate["experience_eligible"],
                "reasons": candidate["reasons"]
            }
            for candidate in rejected_candidates
        ]

        return {
            "job_description": requirements["job_description"],
            "requirements": {
                "must_have_skills": requirements["must_have_skills"],
                "preferred_skills": requirements["preferred_skills"],
                "min_experience": requirements["min_experience"]
            },
            "total_candidates_evaluated": len(candidates),
            "eligible_count": len(scored_candidates),
            "rejected_count": len(rejected_candidates),
            "top_matches": top_matches,
            "rejected_candidates": rejected
        }

        
    def format_submission_output(self, results):
        """
        Convert internal matching results into the
        assignment's required JSON structure.
        """

        top_matches = []

        for candidate in results.get("top_matches", []):

            excerpts = candidate.get("relevant_excerpts", [])

            # Convert passage objects into plain strings.
            excerpt_texts = [
                excerpt.get("text", "")
                if isinstance(excerpt, dict)
                else str(excerpt)
                for excerpt in excerpts
            ]

            top_matches.append({
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "match_score": candidate["match_score"],
                "matched_skills": candidate.get(
                    "matched_skills", []
                ),
                "relevant_excerpts": excerpt_texts,
                "reasoning": candidate.get("reasoning", [])
            })

        return {
            "job_description": results["job_description"],
            "top_matches": top_matches
        }


    def save_results(self, results, output_path="outputs/matches.json"):
        """
        Save the assignment-compatible JSON output.
        """

        project_root = Path(__file__).resolve().parent
        destination = Path(output_path)

        if not destination.is_absolute():
            destination = project_root / destination

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        formatted = self.format_submission_output(results)

        with destination.open("w", encoding="utf-8") as file:
            json.dump(
                formatted,
                file,
                indent=4,
                ensure_ascii=False
            )

        return destination





if __name__ == "__main__":

    weights = ScoringWeights(
        semantic=0.40,
        mandatory_skills=0.30,
        preferred_skills=0.20,
        experience=0.10
    )

    matcher = JobMatcher(weights=weights)

    job_description = """
    Hiring a Python Backend Developer with at least
    5 years of experience.

    Python and FastAPI are mandatory.
    Docker, PostgreSQL and AWS are preferred.
    """

    results = matcher.match_candidates(
        job_description=job_description,
        must_have_skills=["Python", "FastAPI"],
        preferred_skills=["Docker", "PostgreSQL", "AWS"],
        min_experience=5,
        top_n=10
    )

    formatted = matcher.format_submission_output(results)

    print(json.dumps(formatted, indent=4))

    output_path = matcher.save_results(results)

    print(f"\nResults saved to: {output_path}")

