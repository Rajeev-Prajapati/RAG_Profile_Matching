
import re


class MetadataExtractor:

    SKILLS = [
        "Python", "Java", "C#", "JavaScript",
        "TypeScript", "React", "Angular",
        "Django", "FastAPI", "Flask",
        "SQL", "PostgreSQL", "MySQL",
        "SQL Server", "MongoDB", "Redis",
        "Docker", "Kubernetes", "AWS",
        "Azure", "Git", "Machine Learning",
        "Deep Learning", "TensorFlow",
        "PyTorch", "REST API"
    ]

    EDUCATION = [
        "B.Tech", "B.E.", "M.Tech", "M.E.",
        "BCA", "MCA", "B.Sc", "M.Sc",
        "MBA", "PhD", "Bachelor of Technology",
        "Master of Technology",
        "Bachelor of Engineering",
        "Master of Computer Applications"
    ]

    def extract_name(self, text):

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        # Prefer an explicitly labelled name
        match = re.search(
            r"(?im)^name\s*:\s*([A-Za-z][A-Za-z .'-]+)$",
            text
        )

        if match:
            return match.group(1).strip()

        # Fallback: first plausible line
        for line in lines[:5]:

            if (
                2 <= len(line.split()) <= 4
                and re.fullmatch(r"[A-Za-z .'-]+", line)
                and not re.search(
                    r"resume|curriculum vitae|developer|engineer",
                    line,
                    re.IGNORECASE
                )
            ):
                return line

        return None

    def extract_skills(self, text):

        found_skills = []

        for skill in self.SKILLS:

            pattern = (
                r"(?<!\w)"
                + re.escape(skill)
                + r"(?!\w)"
            )

            if re.search(pattern, text, re.IGNORECASE):
                found_skills.append(skill)

        return found_skills

    def extract_experience(self, text):

        patterns = [
            r"(?i)(?:total\s+)?experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:\+?\s*)years?",
            r"(?i)(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+(?:professional\s+)?experience",
            r"(?i)(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:in|of)\s+(?:software|IT|development)"
        ]

        for pattern in patterns:

            match = re.search(pattern, text)

            if match:
                return float(match.group(1))

        return None

    def extract_education(self, text):

        for degree in self.EDUCATION:

            pattern = (
                r"(?<!\w)"
                + re.escape(degree)
                + r"(?!\w)"
            )

            if re.search(pattern, text, re.IGNORECASE):
                return degree

        return None

    def extract_metadata(self, text):

        return {
            "candidate_name": self.extract_name(text),
            "skills": self.extract_skills(text),
            "experience_years": self.extract_experience(text),
            "education": self.extract_education(text)
        }


if __name__ == "__main__":

    sample_resume = """
    Name: John Doe

    Python Backend Developer

    Skills:
    Python, FastAPI, Django, PostgreSQL, Docker

    Experience:
    5 years of professional experience developing
    Python backend applications.

    Education:
    B.Tech in Computer Science
    """

    extractor = MetadataExtractor()

    metadata = extractor.extract_metadata(sample_resume)

    print(metadata)
