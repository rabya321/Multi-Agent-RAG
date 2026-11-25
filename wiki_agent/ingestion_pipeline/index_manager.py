import re
from datetime import datetime
from pathlib import Path

from config import (
    INDEX_PATH,
    CONCEPTS_DIR,
)
from models import IndexUpdate, ModuleMetadata


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _title_slug(title: str) -> str:
    """Convert title to a URL-friendly slug."""
    slug = re.sub(r"[^\w\s-]", "", title).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug


def _slugify_module_id(module_id: str, title: str) -> str:
    return f"{module_id}__{_title_slug(title)}"


def _md_table_cell(text: str) -> str:
    """Escape markdown table delimiters and normalize whitespace."""
    return text.replace("|", r"\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Global index.md
# ---------------------------------------------------------------------------

def _ensure_index() -> Path:
    """
    Ensure wiki/index.md exists with the required template.
    The global index contains the complete Modules and Concepts tables.
    """
    path = Path(INDEX_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        today = _today()
        content = (
            f"## E-learning Wiki Index\n"
            f"last_updated: {today}\n"
            "modules_ingested: 0\n"
            "concepts_created: 0\n"
            "\n---\n\n"
            "### Modules\n\n"
            "| ID | Title | Track | Subtrack | Status |\n"
            "|----|-------|-------|----------|--------|\n\n"
            "---\n\n"
            "### Concepts\n\n"
            "| ID | Title | Description | Keywords | Primary Source |\n"
            "|----|-------|-------------|----------|----------------|\n\n"
            "---\n\n"
        )
        path.write_text(content, encoding="utf-8")
    return path


def _section_bounds(content: str, heading: str) -> tuple[int, int, int] | None:
    """
    Return table header end, table body end, and body start for a section.
    """
    tracks_section = re.search(
        rf"(### {re.escape(heading)}\n\n.*?\n)(.*?)(?=\n---|\Z)",
        content,
        re.DOTALL
    )
    if not tracks_section:
        return None

    return (
        tracks_section.start(2),
        tracks_section.end(2),
        tracks_section.end(1),
    )


def _update_modules_table(
    content: str,
    metadata: ModuleMetadata
) -> tuple[str, bool]:
    """Add the module to the global Modules table if it is new."""
    bounds = _section_bounds(content, "Modules")
    if bounds is None:
        return content, False

    body_start, body_end, _ = bounds
    table_text = content[body_start:body_end]
    module_slug = _slugify_module_id(metadata.id, metadata.title)

    if re.search(rf"\[\[{re.escape(module_slug)}(?:\||\]\])", table_text):
        return content, False

    title = _md_table_cell(metadata.title) if metadata.title else metadata.id
    track = _md_table_cell(metadata.track) if metadata.track else "Unknown"
    subtrack = _md_table_cell(metadata.subtrack) if metadata.subtrack else ""
    row = (
        f"| [[{module_slug}]] | {title} | {track} | "
        f"{subtrack} | ingested |\n"
    )
    content = content[:body_end] + row + content[body_end:]
    content = re.sub(
        r"modules_ingested: (\d+)",
        lambda m: f"modules_ingested: {int(m.group(1)) + 1}",
        content,
        count=1,
    )
    return content, True


def _read_concept_metadata(slug: str) -> tuple[str, str, str]:
    """Read title, description, and keywords from concept frontmatter."""
    concept_path = Path(CONCEPTS_DIR) / f"{slug}.md"
    if not concept_path.exists():
        return slug.replace("_", " ").title(), "", ""

    page_content = concept_path.read_text(encoding="utf-8")
    title = slug.replace("_", " ").title()
    description = ""
    keywords = ""
    if not page_content.startswith("---"):
        return title, description, keywords

    try:
        end = page_content.index("\n---", 3)
    except ValueError:
        return title, description, keywords

    frontmatter = page_content[3:end]
    title_match = re.search(r"^title:\s*(.+)$", frontmatter, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip().strip('"')

    description_match = re.search(
        r"^description:\s*(.+)$",
        frontmatter,
        re.MULTILINE,
    )
    if description_match:
        description = description_match.group(1).strip().strip('"')

    keywords_match = re.search(
        r"keywords:\s*\n((?:\s+-[^\n]+\n?)+)",
        frontmatter,
    )
    if keywords_match:
        items = re.findall(r"-\s*(.+)", keywords_match.group(1))
        keywords = ", ".join(item.strip() for item in items)

    return title, description, keywords


def _normalise_index_schema(content: str) -> str:
    """
    Enforce expected index headings and Concepts table columns.
    Keeps existing table rows intact while normalizing the template.
    """
    content = re.sub(
        r"^#{1,6}\s+.*?Index\s*$",
        "## E-learning Wiki Index",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^##\s+Modules\s*$",
        "### Modules",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^##\s+Concepts\s*$",
        "### Concepts",
        content,
        flags=re.MULTILINE,
    )
    content = content.replace(
        "| ID | Title | Keywords | Primary Source |",
        "| ID | Title | Description | Keywords | Primary Source |",
    )
    content = content.replace(
        "|----|-------|----------|----------------|",
        "|----|-------|-------------|----------|----------------|",
    )
    return content


def _update_concepts_table(
    content: str,
    concepts_added: list[str],
    module_id: str,
) -> tuple[str, int]:
    """Add new concepts to the global Concepts table."""
    if not concepts_added:
        return content, 0

    bounds = _section_bounds(content, "Concepts")
    if bounds is None:
        return content, 0

    body_start, body_end, _ = bounds
    table_text = content[body_start:body_end]
    added = 0

    for slug in concepts_added:
        if re.search(rf"\[\[{re.escape(slug)}(?:\||\]\])", table_text):
            continue

        title, description, keywords = _read_concept_metadata(slug)
        table_text += (
            f"| [[{slug}]] | {_md_table_cell(title)} | "
            f"{_md_table_cell(description)} | {_md_table_cell(keywords)} | "
            f"{_md_table_cell(module_id)} |\n"
        )
        added += 1

    if added == 0:
        return content, 0

    content = content[:body_start] + table_text + content[body_end:]
    content = re.sub(
        r"concepts_created: (\d+)",
        lambda m: f"concepts_created: {int(m.group(1)) + added}",
        content,
        count=1,
    )
    return content, added

# ---------------------------------------------------------------------------
# Main apply function
# ---------------------------------------------------------------------------

def apply_updates(
    updates: IndexUpdate,
    module_metadata: ModuleMetadata
) -> None:
    """
        Apply all index updates to the single global wiki/index.md.

                The global index contains:
                        - a Modules table with ID, title, track, subtrack, and status
                        - a Concepts table with ID, title, description, keywords,
                            and primary source

    NOT handled here:
      errors           — removed from pipeline
      synthesis triggers — removed from pipeline
      pending concepts   — removed from pipeline
    """
    # update global index
    path    = _ensure_index()
    content = path.read_text(encoding="utf-8")
    content = _normalise_index_schema(content)

    content, _ = _update_modules_table(content, module_metadata)
    content, _ = _update_concepts_table(
        content,
        updates.concepts_added,
        module_metadata.id,
    )
    content = re.sub(
        r"last_updated: [^\n]*",
        f"last_updated: {_today()}",
        content
    )
    path.write_text(content, encoding="utf-8")


