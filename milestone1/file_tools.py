
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

from pypdf import PdfReader
from docx import Document


# Only allow file operations inside this project directory.
WORKSPACE_DIR = Path(__file__).resolve().parent.parent

SUPPORTED_READ_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _safe_path(filepath: str) -> Path:
    """Resolve a path and reject paths outside the project workspace."""
    if not isinstance(filepath, str) or not filepath.strip():
        raise ValueError("File path must be a non-empty string.")

    requested_path = Path(filepath).expanduser()

    if not requested_path.is_absolute():
        requested_path = WORKSPACE_DIR / requested_path

    resolved_path = requested_path.resolve()
    
    if not resolved_path.is_relative_to(WORKSPACE_DIR):
        raise ValueError("Access denied: path is outside the project workspace.")

    return resolved_path


def _get_metadata(path: Path) -> dict:
    """Return common metadata for an existing file."""
    stat = path.stat()

    return {
        "name": path.name,
        "path": str(path),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_date": datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(timespec="seconds"),
    }


def read_file(filepath: str) -> dict:
    """
    Read a PDF, TXT, or DOCX file.

    Returns text content and file metadata.
    """
    try:
        path = _safe_path(filepath)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {filepath}",
            }

        if not path.is_file():
            return {
                "success": False,
                "error": "The specified path is not a file.",
            }

        extension = path.suffix.lower()

        if extension not in SUPPORTED_READ_EXTENSIONS:
            return {
                "success": False,
                "error": (
                    f"Unsupported file type: {extension}. "
                    "Supported types: PDF, TXT, DOCX."
                ),
            }

        if extension == ".txt":
            content = path.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )

        elif extension == ".pdf":
            reader = PdfReader(str(path))

            if reader.is_encrypted:
                return {
                    "success": False,
                    "error": "The PDF is encrypted and cannot be read.",
                }

            content = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        else:  # .docx
            document = Document(str(path))
            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]
            content = "\n".join(paragraphs)

        return {
            "success": True,
            "content": content,
            "metadata": _get_metadata(path),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"Could not read file: {exc}",
        }


def list_files(
    directory: str,
    extension: Optional[str] = None,
) -> list:
    """
    List files in a directory, optionally filtering by extension.
    """
    try:
        path = _safe_path(directory)

        if not path.exists() or not path.is_dir():
            return [{
                "success": False,
                "error": f"Directory not found: {directory}",
            }]

        if extension:
            extension = extension.strip().lower()
            if not extension.startswith("."):
                extension = "." + extension

        results = []

        # List files directly in this directory, not recursively.
        for file_path in sorted(path.iterdir()):
            if not file_path.is_file():
                continue

            if extension and file_path.suffix.lower() != extension:
                continue

            try:
                results.append(_get_metadata(file_path))
            except OSError:
                continue

        return results

    except Exception as exc:
        return [{
            "success": False,
            "error": f"Could not list files: {exc}",
        }]


def write_file(filepath: str, content: str) -> dict:
    """
    Write text content to a file, creating parent directories if needed.
    Existing files can be overwritten, so use this tool deliberately.
    """
    try:
        path = _safe_path(filepath)

        if path.exists():
            return {
            "success": False,
            "error": (
                "File already exists. Choose a new filename "
                "to avoid overwriting existing content."
            ),
        }

        if not isinstance(content, str):
            return {
                "success": False,
                "error": "Content must be a string.",
            }

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "message": "File written successfully.",
            "metadata": _get_metadata(path),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"Could not write file: {exc}",
        }


def search_in_file(filepath: str, keyword: str) -> dict:
    """
    Search file content case-insensitively and return matching context.
    """
    try:
        if not isinstance(keyword, str) or not keyword.strip():
            return {
                "success": False,
                "error": "Keyword must be a non-empty string.",
            }

        result = read_file(filepath)

        if not result.get("success"):
            return result

        content = result["content"]
        keyword_lower = keyword.casefold()
        content_lower = content.casefold()

        matches = []
        start = 0
        context_size = 80

        while True:
            index = content_lower.find(keyword_lower, start)

            if index == -1:
                break

            context_start = max(0, index - context_size)
            context_end = min(
                len(content),
                index + len(keyword) + context_size,
            )

            matches.append({
                "position": index,
                "match": content[index:index + len(keyword)],
                "context": content[context_start:context_end],
            })

            # Advance past this match, avoiding an infinite loop.
            start = index + len(keyword)

        return {
            "success": True,
            "filepath": result["metadata"]["path"],
            "keyword": keyword,
            "match_count": len(matches),
            "matches": matches,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"Search failed: {exc}",
        }