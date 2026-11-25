import argparse
from datetime import datetime
from pathlib import Path

from config import (
    PPTS_DIR,
    IMAGES_DIR,
    WIKI_DIR,
    MODULES_DIR,
    CONCEPTS_DIR,
    LOG_PATH,
    LOG_TOKEN_USAGE,
    INGESTION_LOG_PATH,
)
from models import ModuleMetadata, SlideData
from ingestion_pipeline.ppt_parser_step1 import parse_pptx, extract_slide1_metadata
from ingestion_pipeline.slide_categorizer_step2 import categorize_slides
from ingestion_pipeline.slide_assembler_step3 import assemble_module
from ingestion_pipeline.wiki_agent_step4 import write_wiki
from ingestion_pipeline.writer_step5 import write_envelope
from ingestion_pipeline.ingestion_log import write_module_log
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
# Wiki initialisation
# ---------------------------------------------------------------------------

def init_wiki(track: str = None) -> None:
    """
    Create wiki/ directory structure and initial files if absent.
    index.md creation delegated to index_manager._ensure_index().
    log.md created here if absent.
    Track local index directory created if track provided.
    No errors.md, no synthesis_triggers.md, no wiki/images/.
    """
    dirs = [
        PPTS_DIR, IMAGES_DIR, WIKI_DIR,
        MODULES_DIR, CONCEPTS_DIR,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # global index.md owned by index_manager
    index_manager._ensure_index()

    log_path = Path(LOG_PATH)
    if not log_path.exists():
        log_path.write_text(
            "# SCA Wiki Log\n\n---\n",
            encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(
    module_id: str,
    track: str,
    subtrack: str,
    partial_metadata: dict,
) -> ModuleMetadata:
    """
    Build ModuleMetadata from PPT-extracted fields and CLI arguments.
    subtrack already resolved by caller before this is called.
    """
    return ModuleMetadata(
        id=module_id,
        title=partial_metadata["title"],
        contact=partial_metadata["contact"],
        date=partial_metadata.get("date", ""),
        track=track,
        subtrack=subtrack,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    module_id: str,
    track: str,
    subtrack: str,
) -> None:
    """
    Run the full ingestion pipeline for one module.

    1.   parse PPT → slides + partial_metadata
    1b.  deterministic metadata extraction from slide 1
    2.   categorize slides
    3.   load full metadata
    4.   assemble
    5.   wiki writer LLM call → WikiEnvelope + TokenUsage
    6.   write files
    7.   write ingestion log row
    8.   print summary
    """
    init_wiki(track=track)

    # step 1 — parse
    slides, partial_metadata = parse_pptx(module_id)

    # step 1b — deterministic metadata extraction from slide 1
    print("  Extracting metadata from slide 1 (deterministic rules)...")
    slide1_meta = extract_slide1_metadata(slides[0])
    partial_metadata["title"] = slide1_meta.module_title or module_id
    partial_metadata["contact"] = slide1_meta.contact
    effective_subtrack = subtrack or slide1_meta.series_label

    # step 2 — categorize
    slides = categorize_slides(
        slides,
        render_height=partial_metadata.get("render_height")
    )

    # step 3 — load full metadata
    metadata = load_metadata(
        module_id,
        track,
        effective_subtrack,
        partial_metadata,
    )

    # step 4 — assemble
    assembly = assemble_module(slides, metadata)

    # step 5 — wiki writer LLM call
    print(f"  Calling wiki writer (LLM call)...")
    envelope, token_usage = write_wiki(assembly)

    # step 6 — write files
    summary = write_envelope(envelope, module_id, metadata)

    # step 7 — write ingestion log row
    write_module_log(
        module_id=module_id,
        title=metadata.title,
        track=track,
        subtrack=effective_subtrack,
        slides=slides,
        summary=summary,
        token_usage=token_usage,
    )

    # step 8 — print summary
    nav_count   = sum(1 for s in slides if s.category == "NAVIGATION")
    text_count  = sum(1 for s in slides if s.category == "TEXT_ONLY")
    image_count = sum(1 for s in slides if s.category == "IMAGE")

    print(f"\n--- {module_id} ---")
    print(f"  Track: {track} | Subtrack: {effective_subtrack}")
    print(f"  Slides: NAVIGATION={nav_count} TEXT_ONLY={text_count} IMAGE={image_count}")
    print(f"  Module page written:    {summary['module_page_written']}")
    print(f"  Concept pages created:  {summary['concept_pages_written']}")
    print(f"  Concept pages extended: {summary['concept_pages_extended']}")
    print(f"  Input tokens:    {token_usage.input_tokens}")
    print(f"  Output tokens:   {token_usage.output_tokens}")
    print(f"  Total tokens:    {token_usage.total_tokens}")
    print(f"  Finish reason:   {token_usage.finish_reason}")
    print(f"  Tool calls:      {token_usage.tool_calls_made}")

    if token_usage.finish_reason == "length":
        print(
            f"  WARNING: response truncated — "
            f"increase WIKI_MAX_TOKENS in config.py"
        )

    if LOG_TOKEN_USAGE:
        _append_log(
            f"{module_id} — "
            f"input: {token_usage.input_tokens} tokens, "
            f"output: {token_usage.output_tokens} tokens, "
            f"finish: {token_usage.finish_reason}, "
            f"tool_calls: {token_usage.tool_calls_made}"
        )


def run_batch(
    module_ids: list[str],
    track: str,
    subtrack: str
) -> None:
    """
    Run ingestion for a list of modules in order.
    Continues on single module failure — does not abort batch.
    All modules in a batch share the same track and subtrack.
    subtrack may be overridden per module by deterministic extraction
    if not provided via CLI.
    """
    total = len(module_ids)
    for i, module_id in enumerate(module_ids, 1):
        print(f"\n[{i}/{total}] Ingesting {module_id}...")
        try:
            run_pipeline(
                module_id=module_id,
                track=track,
                subtrack=subtrack,
            )
        except Exception as e:
            print(f"  ERROR ingesting {module_id}: {e}")
            _append_log(f"ERROR: {module_id} ingestion failed — {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SCA Wiki Ingestion Pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--module",
        type=str,
        help="Single module ID e.g. 00_05_02_Resource_Description_Framework"
    )
    group.add_argument(
        "--batch",
        nargs="+",
        type=str,
        help="List of module IDs in order"
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        help="Track: General, Plan, Make, Deliver, Enabler etc."
    )
    parser.add_argument(
        "--subtrack",
        type=str,
        required=False,
        default="",
        help="Subtrack e.g. 'Semantic Web' — falls back to slide-1 extraction"
    )
    args = parser.parse_args()

    if args.module:
        run_pipeline(
            module_id=args.module,
            track=args.track,
            subtrack=args.subtrack,
        )
    elif args.batch:
        run_batch(
            module_ids=args.batch,
            track=args.track,
            subtrack=args.subtrack
        )