
import re
from pathlib import Path


class ResumeChunker:

    SECTION_ALIASES = {
        "Summary": {
            "summary", "professional summary",
            "profile", "objective", "career objective"
        },
        "Skills": {
            "skills", "technical skills",
            "core competencies", "technologies",
            "technical expertise"
        },
        "Experience": {
            "experience", "work experience",
            "professional experience",
            "employment history", "work history"
        },
        "Projects": {
            "projects", "personal projects",
            "academic projects", "key projects"
        },
        "Education": {
            "education", "academic qualifications",
            "educational qualifications",
            "academic background"
        },
        "Certifications": {
            "certifications", "certificates",
            "professional certifications"
        }
    }

    def __init__(self, chunk_size=600, chunk_overlap=100):

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and < chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def identify_section(self, line):
        """
        Detect standalone resume section headings.
        """

        normalized = re.sub(
            r"\s+", " ", line.strip().rstrip(":")
        ).casefold()

        for section, aliases in self.SECTION_ALIASES.items():
            if normalized in aliases:
                return section

        return None

    def split_into_sections(self, text):
        """
        Group text by resume section.
        """

        sections = []
        current_section = "General"
        current_lines = []

        def save_section():
            content = "\n".join(current_lines).strip()

            if content:
                sections.append({
                    "section": current_section,
                    "text": content
                })

        for line in text.splitlines():

            detected = self.identify_section(line)

            if detected:
                save_section()

                current_section = detected
                current_lines = []
            else:
                current_lines.append(line)

        save_section()

        return sections

    def split_long_section(self, text):
        """
        Split long text into overlapping character chunks.
        Prefer breaking at whitespace.
        """

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text)
            )

            # Prefer a whitespace boundary
            if end < len(text):
                boundary = text.rfind(
                    " ", start + self.chunk_size // 2, end
                )

                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(text):
                break

            # Keep overlapping context
            next_start = end - self.chunk_overlap

            # Guarantee forward progress
            start = max(start + 1, next_start)

        return chunks

    def chunk_resume(self, document):
        """
        Convert a loaded resume document into
        section-aware chunks with metadata.
        """

        text = document["text"]
        metadata = document.get("metadata", {})

        sections = self.split_into_sections(text)

        chunks = []
        chunk_index = 0

        for section_data in sections:

            section_name = section_data["section"]
            section_text = section_data["text"]

            parts = self.split_long_section(section_text)

            for part in parts:

                chunk_index += 1

                chunk = {
                    "chunk_id": (
                        f"{Path(document['resume_path']).stem}"
                        f"_{chunk_index}"
                    ),
                    "resume_path": document["resume_path"],
                    "section": section_name,
                    "text": part,
                    "metadata": {
                        **metadata,
                        "section": section_name,
                        "chunk_index": chunk_index
                    }
                }

                chunks.append(chunk)

        return chunks


if __name__ == "__main__":

    sample_resume = """
John Doe
Python Backend Developer

Skills
Python, FastAPI, Django, PostgreSQL, Docker

Experience
5 years of Python backend development.
Built REST APIs using FastAPI and PostgreSQL.
Developed microservices and deployed them using Docker.

Projects
Payment Processing API
Built a secure payment service using Python.

Education
B.Tech in Computer Science
"""

    document = {
        "resume_path": "resumes/john_doe.pdf",
        "text": sample_resume,
        "metadata": {
            "candidate_name": "John Doe",
            "skills": ["Python", "FastAPI", "Docker"],
            "experience_years": 5.0,
            "education": "B.Tech"
        }
    }

    chunker = ResumeChunker(
        chunk_size=120,
        chunk_overlap=20
    )

    chunks = chunker.chunk_resume(document)

    for chunk in chunks:
        print("\n" + "=" * 50)
        print("ID:", chunk["chunk_id"])
        print("Section:", chunk["section"])
        print("Text:", chunk["text"])
