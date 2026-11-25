import re
from datetime import datetime
from pathlib import Path

from config import (
    MODULES_DIR,
    CONCEPTS_DIR,
    INDEX_PATH,
    LOG_PATH,
)
from models import WikiEnvelope, ModuleMetadata
from ingestion_pipeline import index_manager


# ---------------------------------------------------------------------------
# Embedded log utility (replaces utils.append_log)
# ---------------------------------------------------------------------------

def _append_log(entry: str) -> None:
    """
    Append a timestamped entry to wiki/log.md.
    Creates log.md if it does not exist.
    """
    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {entry}\n"
        )


# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------

def _normalise_paths(content: str) -> str:
    """
    Normalise backslashes to forward slashes in image paths.
    Only replaces backslashes inside markdown image syntax.
    """
    def replace_path(match):
        alt  = match.group(1)
        path = match.group(2).replace("\\", "/")
        return f"![{alt}]({path})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_path, content)


# ---------------------------------------------------------------------------
# Page writing
# ---------------------------------------------------------------------------

def _write_module_page(envelope: WikiEnvelope) -> bool:
    """
    Write module page if it does not already exist.
    Never overwrites — delete manually to re-ingest.
    Normalises image path backslashes before writing.
    Module pages do not embed images — no image copy needed.
    """
    Path(MODULES_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(MODULES_DIR) / envelope.module_page.filename
    if path.exists():
        print(
            f"module page {envelope.module_page.filename} "
            f"already exists — skipping"
        )
        return False
    path.write_text(
        _normalise_paths(envelope.module_page.content),
        encoding="utf-8"
    )
    return True


def _write_concept_pages(envelope: WikiEnvelope) -> tuple[int, int]:
    """
    Write new concept pages or overwrite existing with LLM-merged content.
    LLM read existing pages via read_file tool and produced
    complete merged versions — writer.py trusts this output entirely.
    No section matching, no heading comparison.
    Normalises backslashes in image paths before writing.
    Image embedding removed — concept pages no longer copy images.
    """
    Path(CONCEPTS_DIR).mkdir(parents=True, exist_ok=True)
    created  = 0
    extended = 0

    for page in envelope.concept_pages:
        path = Path(CONCEPTS_DIR) / page.filename
        extended += 1 if path.exists() else 0
        created  += 0 if path.exists() else 1
        content = _normalise_paths(page.content)
        path.write_text(content, encoding="utf-8")

    return created, extended


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_envelope(
    envelope: WikiEnvelope,
    module_id: str,
    module_metadata: ModuleMetadata
) -> dict:
    """
    Write all WikiEnvelope contents to disk.

    6a.  module page — new file only, never overwrite
    6b.  concept pages — LLM-merged, overwrite existing
         normalise image paths only (no image copying)
        6c.  index update via index_manager
         includes track index update
        6d.  log entries

    Returns summary dict with actual operation counts.
    """
    # 6a — module page
    module_page_written = _write_module_page(envelope)

    # 6b — concept pages
    # image embedding removed — _copy_embedded_images removed
    # LLM-merged content written directly after path normalisation
    concept_pages_written, concept_pages_extended = _write_concept_pages(envelope)

    # 6c — index update
    # index_manager handles global index.md and track _index.md
    # does NOT trigger maintain automatically
    index_manager.apply_updates(envelope.index_updates, module_metadata)

    # 6d — log entries from LLM output
    for entry in envelope.log_entries:
        _append_log(entry)

    return {
        "module_page_written":    module_page_written,
        "concept_pages_written":  concept_pages_written,
        "concept_pages_extended": concept_pages_extended,
    }