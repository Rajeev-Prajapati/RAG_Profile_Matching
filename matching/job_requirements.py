
import re

from utils.metadata_extractor import MetadataExtractor


class JobRequirementExtractor:

    def __init__(self):
        self.known_skills = MetadataExtractor.SKILLS

    @staticmethod
    def find_skills(text, known_skills):
        found = []

        for skill in known_skills:
            pattern = (
                r"(?<!\w)"
                + re.escape(skill)
                + r"(?!\w)"
            )

            if re.search(pattern, text, re.IGNORECASE):
                found.append(skill)

        return found

    def extract_min_experience(self, text):
        patterns = [
            r"(?i)(?:at least|minimum|min\.?)\s*"
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?",

            r"(?i)(\d+(?:\.\d+)?)\s*\+\s*years?",

            r"(?i)(\d+(?:\.\d+)?)\s*years?\s+"
            r"(?:of\s+)?(?:experience\s+)?required"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                return float(match.group(1))

        return None

    def extract_requirements(
        self,
        job_description,
        must_have_skills=None,
        preferred_skills=None,
        min_experience=None
    ):
        if not isinstance(job_description, str) or not job_description.strip():
            raise ValueError("Job description cannot be empty")

        detected_skills = self.find_skills(
            job_description, self.known_skills
        )

        # Explicit skill requirements take priority.
        # Automatic parsing does not guess which skills
        # are mandatory from arbitrary job-description prose.
        mandatory = (
            list(must_have_skills)
            if must_have_skills is not None
            else []
        )

        preferred = (
            list(preferred_skills)
            if preferred_skills is not None
            else [
                skill for skill in detected_skills
                if skill.casefold() not in {
                    item.casefold() for item in mandatory
                }
            ]
        )

        minimum_years = (
            float(min_experience)
            if min_experience is not None
            else self.extract_min_experience(job_description)
        )

        if minimum_years is not None and minimum_years < 0:
            raise ValueError("Minimum experience cannot be negative")

        return {
            "job_description": job_description.strip(),
            "must_have_skills": mandatory,
            "preferred_skills": preferred,
            "min_experience": minimum_years,
            "detected_skills": detected_skills
        }
