import csv
from datetime import datetime
from pathlib import Path

from config import INGESTION_LOG_PATH
from models import TokenUsage


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


HEADERS = [
    "module_id",
    "title",
    "track",
    "subtrack",
    "ingestion_date",
    "slides_total",
    "slides_navigation",
    "slides_text_only",
    "slides_image",
    "concept_pages_created",
    "concept_pages_extended",
    "input_tokens",
    "output_tokens",
    "finish_reason",
    "tool_calls",
]



def _ensure_log() -> Path:
    """Create wiki/ingestion_log.csv with headers if it does not exist."""
    path = Path(INGESTION_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
    return path


def write_module_log(
    module_id: str,
    title: str,
    track: str,
    subtrack: str,
    slides: list,
    summary: dict,
    token_usage: TokenUsage,
) -> None:
    """
    Append one row to wiki/ingestion_log.csv for a completed
    module ingestion.

    Per-module only — no cumulative totals since errors and
    synthesis triggers have been removed from the pipeline.

    Parameters:
        module_id:    module identifier
        title:        extracted module title
        track:        from CLI --track
        subtrack:     effective subtrack
        slides:       categorized slides list
        summary:      dict returned by write_envelope()
        token_usage:  TokenUsage from write_wiki()
    """
    path = _ensure_log()

    nav   = sum(1 for s in slides if s.category == "NAVIGATION")
    text  = sum(1 for s in slides if s.category == "TEXT_ONLY")
    image = sum(1 for s in slides if s.category == "IMAGE")
    total = nav + text + image

    row = [
        module_id,
        title,
        track,
        subtrack,
        _today(),
        total,
        nav,
        text,
        image,
        summary.get("concept_pages_written", 0),
        summary.get("concept_pages_extended", 0),
        token_usage.input_tokens,
        token_usage.output_tokens,
        token_usage.finish_reason,
        token_usage.tool_calls_made,
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"  logged → wiki/ingestion_log.csv")