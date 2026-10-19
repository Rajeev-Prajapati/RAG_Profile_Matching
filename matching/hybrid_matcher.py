
import re


class HybridMatcher:

    @staticmethod
    def has_skill(text, skill):
        """
        Match a skill as a complete term.
        Prevents Java from matching JavaScript.
        """
        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        return bool(
            re.search(pattern, text, re.IGNORECASE)
        )

    def evaluate_candidate(self, candidate, requirements):

        # Skills extracted from resume metadata
        skills = candidate.get("skills", "")

        if isinstance(skills, list):
            skills = ", ".join(skills)

        # Include retrieved evidence passages
        excerpts = [
            chunk["text"]
            for chunk in candidate.get("relevant_chunks", [])
        ]

        searchable_text = "\n".join(
            [skills] + excerpts
        )

        must_have = requirements["must_have_skills"]
        preferred = requirements["preferred_skills"]

        matched_must_have = [
            skill for skill in must_have
            if self.has_skill(searchable_text, skill)
        ]

        missing_must_have = [
            skill for skill in must_have
            if skill not in matched_must_have
        ]

        matched_preferred = [
            skill for skill in preferred
            if self.has_skill(searchable_text, skill)
        ]

        missing_preferred = [
            skill for skill in preferred
            if skill not in matched_preferred
        ]

        minimum = requirements["min_experience"]
        experience = candidate.get("experience_years")

        if experience is not None:
            experience = float(experience)

        experience_eligible = (
            minimum is None
            or (
                experience is not None
                and experience >= minimum
            )
        )

        eligible = (
            not missing_must_have
            and experience_eligible
        )

        reasons = []

        if missing_must_have:
            reasons.append(
                "Mandatory skills not evidenced: "
                + ", ".join(missing_must_have)
            )

        if not experience_eligible:
            if experience is None:
                reasons.append(
                    "Experience information unavailable"
                )
            else:
                reasons.append(
                    f"Experience {experience} years "
                    f"is below required {minimum} years"
                )

        if eligible:
            reasons.append(
                "All mandatory requirements are evidenced"
            )

        return {
            **candidate,
            "matched_must_have": matched_must_have,
            "missing_must_have": missing_must_have,
            "matched_preferred": matched_preferred,
            "missing_preferred": missing_preferred,
            "experience_eligible": experience_eligible,
            "eligible": eligible,
            "reasons": reasons
        }
