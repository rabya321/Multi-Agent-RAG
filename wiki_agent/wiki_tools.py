"""Wiki file read helper used by the retrieval agent."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def read_file(path: str) -> str:
    """Read a UTF-8 text file relative to project root.

    Returns the file content, or an error string if the path is empty,
    escapes the project root, or does not exist. Never raises, so it is
    safe to use as a tool result.
    """
    if not path or not path.strip():
        return "File path is required."
    rel_path = Path(path.strip())
    target = (PROJECT_ROOT / rel_path).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return f"Path escapes project root: {path}"
    if not target.exists() or not target.is_file():
        return f"File does not exist: {path}"
    try:
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return f"Failed to read file: {path} ({exc})"
