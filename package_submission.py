
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "RAG_Profile_Matching_Submission.zip"

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".vscode",
    ".idea",
    ".ipynb_checkpoints",
    ".pytest_cache",
    "chroma_db",
    "chroma_db_evaluation"
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp"
}

INCLUDE_DIRS = {
    "milestone1",
    "chunking",
    "embeddings",
    "vector_store",
    "retrieval",
    "utils",
    "evaluation",
    "dataset",
    "notebooks",
    "results",
    "tests",
    "demo"
}

INCLUDE_FILES = {
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    ".gitignore",
    "resume_rag.py",
    "job_matcher.py",
    "package_submission.py"
}


def should_include(path):
    relative = path.relative_to(ROOT)

    if any(
        part in EXCLUDE_DIRS
        for part in relative.parts
    ):
        return False

    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False

    if path.name.startswith(".env"):
        return False

    if len(relative.parts) == 1:
        return path.name in INCLUDE_FILES

    return relative.parts[0] in INCLUDE_DIRS


files = sorted(
    path for path in ROOT.rglob("*")
    if path.is_file() and should_include(path)
)

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(
            path,
            arcname=Path(ROOT.name) / path.relative_to(ROOT)
        )

print("Submission created:", OUTPUT)
print("Total files:", len(files))
print(
    "ZIP size:",
    round(OUTPUT.stat().st_size / (1024 * 1024), 2),
    "MB"
)
