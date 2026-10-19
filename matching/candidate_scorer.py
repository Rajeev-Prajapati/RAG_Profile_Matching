
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ScoringWeights:
    semantic: float = 0.40
    mandatory_skills: float = 0.30
    preferred_skills: float = 0.20
    experience: float = 0.10

    def __post_init__(self):
        weights = [
            self.semantic,
            self.mandatory_skills,
            self.preferred_skills,
            self.experience
        ]

        if any(
            not isfinite(weight) or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Weights must be finite, non-negative numbers"
            )

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(
                "Scoring weights must add up to 1.0"
            )


class CandidateScorer:

    def __init__(self, weights=None):
        self.weights = weights or ScoringWeights()

    @staticmethod
    def coverage_score(matched, required):
        """
        Percentage of skills covered.

        Returns None when there are no skills
        in the requirement category.
        """
        if not required:
            return None

        required_keys = {
            skill.casefold() for skill in required
        }

        matched_keys = {
            skill.casefold() for skill in matched
        }

        covered = len(required_keys & matched_keys)

        return 100.0 * covered / len(required_keys)

    @staticmethod
    def semantic_score(similarity):
        """
        Convert cosine similarity to a 0-100 scale.

        Negative similarities are clamped to zero.
        """
        value = float(similarity)

        if not isfinite(value):
            raise ValueError(
                "Semantic similarity must be finite"
            )

        return max(0.0, min(100.0, value * 100.0))

    @staticmethod
    def experience_score(years, minimum):
        """
        Experience requirement score.

        When a minimum is specified:
        - 100 if the minimum is met
        - proportional score if below the minimum
        - 0 if experience is unknown

        No extra points for exceeding the minimum.
        """
        if minimum is None:
            return None

        minimum = float(minimum)

        if minimum < 0 or not isfinite(minimum):
            raise ValueError(
                "Minimum experience must be non-negative"
            )

        if years is None:
            return 0.0

        years = float(years)

        if years < 0 or not isfinite(years):
            raise ValueError(
                "Experience years must be non-negative"
            )

        if minimum == 0:
            return 100.0

        return min(100.0, years / minimum * 100.0)

    def calculate_score(self, candidate, requirements):
        """
        Score a candidate evaluated by HybridMatcher.
        """

        if not candidate.get("eligible", False):
            raise ValueError(
                "Only eligible candidates can be scored"
            )

        semantic = self.semantic_score(
            candidate["semantic_score"]
        )

        mandatory = self.coverage_score(
            candidate["matched_must_have"],
            requirements["must_have_skills"]
        )

        preferred = self.coverage_score(
            candidate["matched_preferred"],
            requirements["preferred_skills"]
        )

        experience = self.experience_score(
            candidate.get("experience_years"),
            requirements["min_experience"]
        )

        component_scores = {
            "semantic": semantic,
            "mandatory_skills": mandatory,
            "preferred_skills": preferred,
            "experience": experience
        }

        configured_weights = {
            "semantic": self.weights.semantic,
            "mandatory_skills": self.weights.mandatory_skills,
            "preferred_skills": self.weights.preferred_skills,
            "experience": self.weights.experience
        }

        # Requirements that do not exist in the job
        # should not unfairly reward or penalize anyone.
        active_weights = {
            name: weight
            for name, weight in configured_weights.items()
            if component_scores[name] is not None
        }

        total_active_weight = sum(active_weights.values())

        if total_active_weight <= 0:
            raise ValueError(
                "At least one active scoring weight is required"
            )

        effective_weights = {
            name: weight / total_active_weight
            for name, weight in active_weights.items()
        }

        final_score = sum(
            component_scores[name] * weight
            for name, weight in effective_weights.items()
        )

        score_breakdown = {
            name: (
                round(component_scores[name], 2)
                if component_scores[name] is not None
                else None
            )
            for name in component_scores
        }

        reasoning = self.generate_reasoning(
            candidate,
            requirements,
            score_breakdown
        )

        return {
            **candidate,
            "match_score": round(final_score, 2),
            "score_breakdown": score_breakdown,
            "effective_weights": {
                name: round(weight, 4)
                for name, weight in effective_weights.items()
            },
            "scoring_reasoning": reasoning
        }

    @staticmethod
    def generate_reasoning(
        candidate,
        requirements,
        scores
    ):
        reasons = []

        reasons.append(
            f"Semantic relevance: {scores['semantic']:.2f}/100"
        )

        if scores["mandatory_skills"] is not None:
            matched = candidate["matched_must_have"]

            reasons.append(
                "Mandatory skills matched: "
                + ", ".join(matched)
            )

        if scores["preferred_skills"] is not None:
            matched = candidate["matched_preferred"]
            missing = candidate["missing_preferred"]

            if matched:
                reasons.append(
                    "Preferred skills evidenced: "
                    + ", ".join(matched)
                )

            if missing:
                reasons.append(
                    "Preferred skills not evidenced: "
                    + ", ".join(missing)
                )

        minimum = requirements["min_experience"]

        if minimum is not None:
            years = candidate.get("experience_years")

            reasons.append(
                f"Experience: {years} years "
                f"against {minimum} required years"
            )

        return reasons
