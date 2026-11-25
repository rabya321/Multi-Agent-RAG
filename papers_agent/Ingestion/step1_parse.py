"""
Step 1: PDF Parsing with Docling
========================================================
Converts a PDF into a structured ParsedDocument.
Docling features used:
  PdfPipelineOptions              : controls pipeline
  do_ocr=True                     : handles scanned PDFs
  iterate_items()                 : walks document tree
  DocItemLabel                    : semantic label per element
  element.prov                    : page number and bbox
  TableItem.export_to_markdown()  : structured table text
  TableItem.export_to_dataframe() : structured table data
  element.caption_text()          : gets associated caption
  docling_doc.export_to_markdown(): full document markdown

Post-processing:
    Figure caption extraction      : keeps caption text only
    Token counting                 : exact cl100k_base (tiktoken)
  Reference stripping        : removes references section
                               for clean token count
  Corpus summary table       : CSV + terminal report
USAGE:
    python Ingestion/step1_parse.py --file paper.pdf
    python Ingestion/step1_parse.py --file paper.pdf --output_dir ./out
    python Ingestion/step1_parse.py --folder ./papers
   
"""
import os
import json
import argparse
import hashlib
import re
import time
import csv
import tiktoken
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict


from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
try:
    from docling_core.types.doc import (
        TextItem, PictureItem,
        TableItem, ListItem, DocItemLabel
    )
except ImportError:
    from docling.datamodel.document import (
        TextItem, PictureItem,
        TableItem, ListItem
    )
    from docling.datamodel.base_models import DocItemLabel

# ─────────────────────────────────────────────────────────
# TOKEN COUNT HELPER
# ─────────────────────────────────────────────────────────
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Returns exact token count using cl100k_base encoding.
    """
    if not text or not text.strip():
        return 0
    return len(TOKENIZER.encode(text))

# ─────────────────────────────────────────────────────────
# REFERENCE STRIPPING
# ─────────────────────────────────────────────────────────
REFERENCES_PATTERNS = [
    re.compile(r'^references?$',               re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+references?$',     re.IGNORECASE),
    re.compile(r'^bibliography$',              re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+bibliography$',    re.IGNORECASE),
    re.compile(r'^works\s+cited$',             re.IGNORECASE),
    re.compile(r'^author\s+biograph',          re.IGNORECASE),
    re.compile(r'^acknowledgem',               re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+acknowledgem',     re.IGNORECASE),
]

def is_references_heading(text: str) -> bool:
    t = text.strip()
    return any(p.match(t) for p in REFERENCES_PATTERNS)

# ─────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────
@dataclass
class ParsedElement:
    """
    One extracted element from the document.
    element_type: text | header | table | list_item | figure_caption
    section_id:
      block_id of the nearest preceding header element.
      Empty string for elements before the first header.
      Populated in _post_process_elements().
      Used by Step 3 to group elements into sections
      without re-scanning the flat elements list.
    parent_header_block_id:
      Same value as section_id. Kept as a separate
      semantically-named alias for Step 3 clarity.
    label values:
      section_header    : Docling section heading
      title             : Docling title heading
      """
    block_id:               str
    element_type:           str
    label:                  str
    content:                str
    page:                   int
    bbox:                   Optional[list] = None  # Bounding box on page: [left, top, right, bottom].
    heading_level:          Optional[int]  = None
    token_count:            int            = 0
    table_md:               Optional[str]  = None  # for retrieval
    table_df:               Optional[list] = None
    table_rows:             int            = 0
    table_cols:             int            = 0
    caption:                Optional[str]  = None
    section_id:             str            = ""    # Explicit hierarchy fields for downstream chunk grouping.
    parent_header_block_id: str            = ""

@dataclass
class ParsedDocument:
    """
    Complete output of Step 1 for one document.
    title:
      Extracted from the first H1 on page 1.
      Stored as top-level field so Step 2 does not
      need to search the elements list.
    abstract:
      Text of the abstract section if found.
      Stored as top-level field to anchor Step 2
      summary generation.
    front_matter:
            Text assembled from parsed content on pages 1-2,
            ordered by page position.
            Step 2 reads this as metadata context.
    """
    filename:         str
    doc_id:           str
    pages:            int
    # Top-level fields consumed by Step 2 metadata generation.
    title:            str   = ""
    abstract:         str   = ""
    elements:         list  = field(default_factory=list)
    front_matter:     str   = ""
    full_text:        str   = ""
    full_markdown:    str   = ""
    full_text_tokens: int   = 0
    stripped_tokens:  int   = 0
    # released to None after save() in run_batch()
    docling_doc:      object = field(default=None, repr=False)

# ─────────────────────────────────────────────────────────
# FRONT MATTER BUILDER
# ─────────────────────────────────────────────────────────
def build_front_matter(elements: list) -> str:
    """
    Collects all page 1-2 content sorted by bbox position.
    """
    front = [e for e in elements if e.page <= 2]

    def get_text(e) -> str:
        if e.element_type == "table":
            return e.table_md or e.content or ""
        return e.content or ""

    def sort_key(e):
        return (
            e.page or 0,
            -(e.bbox[1] if e.bbox else -1)
        )

    parts = []
    for e in sorted(front, key=sort_key):
        t = get_text(e).strip()
        if t:
            parts.append(t)
    return "\n".join(parts)

# ─────────────────────────────────────────────────────────
# ABSTRACT DETECTION
# ─────────────────────────────────────────────────────────
ABSTRACT_HEADER_PATTERN = re.compile(
    r'^abstract$', re.IGNORECASE
)

def extract_abstract(elements: list) -> str:
    """
    Finds abstract text using two strategies:
    Strategy A: text block following a header whose
                content matches 'abstract' (case-insensitive).
    Strategy B: first substantial text block on page 1
                if no abstract header is found (some papers
                have no explicit Abstract heading).
    Returns empty string if not found.
    """
    # Strategy A: explicit Abstract header
    found_abstract_header = False
    for e in elements:
        if (
            e.element_type == "header"
            and ABSTRACT_HEADER_PATTERN.match(
                e.content.strip()
            )
        ):
            found_abstract_header = True
            continue
        if (
            found_abstract_header
            and e.element_type == "text"
            and len(e.content.strip()) > 80
        ):
            return e.content.strip()
        # stop looking if we hit the next header
        if found_abstract_header and e.element_type == "header":
            break

    # Strategy B: first long text block on page 1
    for e in elements:
        if (
            e.page == 1
            and e.element_type == "text"
            and len(e.content.strip()) > 150
        ):
            return e.content.strip()

    return ""

# ─────────────────────────────────────────────────────────
# TITLE EXTRACTION
# ─────────────────────────────────────────────────────────
# Normalized to work with ParsedElement objects and dicts.
def _to_dict(e) -> dict:
    """
    Converts a ParsedElement dataclass OR an already-dict
    element to a plain dict.
    Fixes the silent failure where extract_title_from_elements
    used .get() which only works on dicts, returning 'unknown'
    when called with dataclass objects.
    """
    if isinstance(e, dict):
        return e
    return e.__dict__

def extract_title_from_elements(elements: list) -> str:
    """
    Extracts paper title from the elements list.
    Accepts both ParsedElement dataclasses (used during
    parse/preview) and plain dicts (used when reading
    saved JSON in save_corpus_summary).
    Priority: page-1 H1 → any page-1 header → first header.
    """
    dicts = [_to_dict(e) for e in elements]
    headers = [
        e for e in dicts
        if e.get("element_type") == "header"
    ]
    if not headers:
        return "unknown"
    page1_h1 = [
        h for h in headers
        if h.get("page") == 1
        and h.get("heading_level") == 1
    ]
    if page1_h1:
        return page1_h1[0]["content"].strip()
    page1 = [h for h in headers if h.get("page") == 1]
    if page1:
        return page1[0]["content"].strip()
    return headers[0]["content"].strip()

# ─────────────────────────────────────────────────────────
# Single post-processing pass that computes all derived fields.
# ─────────────────────────────────────────────────────────
def _post_process_elements(
    elements:    list,
    counters:    dict
) -> dict:
    """
    Single pass over elements after the main parse loop.
    Computes and returns:
      full_text_parts      : list of text strings for join
        stripped_parts       : list of text strings (no refs)
      front_matter_elements: elements on pages 1-2 for sorting
      section assignments  : sets section_id and
                             parent_header_block_id on each
                             element in-place
    Also resets counters from live parse values to match
    element labels exactly (single source of truth).
    Counters are recomputed here so parse() print statements
    and save() JSON counts always agree.
    """
    full_text_parts       = []
    stripped_parts        = []
    front_matter_elements = []
    past_references       = False
    current_section_id    = ""

    # reset all counters to recompute from element truth
    for key in counters:
        counters[key] = 0

    for e in elements:
        et = e.element_type

        # ── recompute counters ────────────────────────────
        if et == "header":
            counters["header"] += 1
        elif et == "text":
            counters["text"] += 1
        elif et == "list_item":
            counters["list_item"] += 1
        elif et == "table":
            counters["table"] += 1

        # ── section hierarchy assignment ──────────────────
        if et == "header":
            current_section_id = e.block_id
        e.section_id             = current_section_id
        e.parent_header_block_id = current_section_id

        # ── full_text accumulation ────────────────────────
        if (
            et in ("text", "header", "list_item", "table")
            and (e.content or e.table_md or "").strip()
        ):
            content = (
                e.table_md
                if et == "table" and e.table_md
                else e.content
            )
            if content and content.strip():
                full_text_parts.append(content)

        # ── stripped_tokens accumulation (no refs) ───────
        if et == "header":
            if is_references_heading(e.content or ""):
                past_references = True
            if not past_references:
                stripped_parts.append(e.content or "")
            continue
        if past_references:
            continue
        if et in ("text", "list_item"):
            if (e.content or "").strip():
                stripped_parts.append(e.content)
        elif et == "table" and e.table_md:
            stripped_parts.append(e.table_md)

        # ── front matter candidates ───────────────────────
        if e.page <= 2:
            front_matter_elements.append(e)

    return {
        "full_text_parts"       : full_text_parts,
        "stripped_parts"        : stripped_parts,
        "front_matter_elements" : front_matter_elements,
        "counters"              : counters,
    }

# ─────────────────────────────────────────────────────────
# BUILD CONVERTER  (once per batch)
# ─────────────────────────────────────────────────────────
def build_converter() -> DocumentConverter:
    """Builds a reusable Docling PDF converter with project pipeline options."""
    options = PdfPipelineOptions()
    options.do_ocr                  = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def make_doc_id(file_path: str) -> str:
    """Creates a stable short document ID from the PDF filename stem."""
    name = Path(file_path).stem.encode("utf-8")
    return "doc_" + hashlib.md5(name).hexdigest()[:12]

def get_output_path(output_dir: str, doc_id: str) -> str:
    """Returns the Step 1 JSON output path for one parsed document."""
    return os.path.join(output_dir, f"step1_{doc_id}.json")

def already_processed(
    output_dir: str, doc_id: str
) -> bool:
    """Checks whether Step 1 output already exists for this document ID."""
    return os.path.exists(
        get_output_path(output_dir, doc_id)
    )

def collect_pdfs(folder: str) -> List[Path]:
    """Recursively collects and sorts all PDF files from a folder."""
    return sorted(Path(folder).rglob("*.pdf"))

# ─────────────────────────────────────────────────────────
# CORE PARSE  (one document)
# ─────────────────────────────────────────────────────────
def parse(
    file_path:  str,
    output_dir: str,
    converter:  DocumentConverter
) -> ParsedDocument:
    """
    Parses one PDF into a ParsedDocument.
    Uses Docling structural extraction, figure-caption extraction,
    exact token counts, and reference stripping.
    """
    doc_id     = make_doc_id(file_path)
    tables_dir = os.path.join(output_dir, "tables",  doc_id)
    os.makedirs(tables_dir, exist_ok=True)
    print(f"  File  : {Path(file_path).name}")
    print(f"  doc_id: {doc_id}")

    result      = converter.convert(file_path)
    docling_doc = result.document
    total_pages = len(docling_doc.pages)

    parsed = ParsedDocument(
        filename    = Path(file_path).name,
        doc_id      = doc_id,
        pages       = total_pages,
        docling_doc = docling_doc
    )

    # Single counters dict: updated during iteration and reused in save().
    counters: Dict[str, int] = {
        "text": 0, "header": 0, "table": 0,
        "list_item": 0
    }

    for i, (element, level) in enumerate(
        docling_doc.iterate_items()
    ):
        block_id = f"{doc_id}_block_{i:04d}"
        page = 0
        bbox = None
        if element.prov:
            page = element.prov[0].page_no or 0
            b    = element.prov[0].bbox
            if b:
                bbox = [b.l, b.t, b.r, b.b]

        # ── TEXT and HEADER ───────────────────────────────
        if isinstance(element, TextItem):
            is_header = element.label in (
                DocItemLabel.SECTION_HEADER,
                DocItemLabel.TITLE
            )
            etype = "header" if is_header else "text"
            if len(element.text.strip()) < 3:
                continue

            parsed.elements.append(ParsedElement(
                block_id      = block_id,
                element_type  = etype,
                label         = str(element.label),
                content       = element.text,
                page          = page,
                bbox          = bbox,
                heading_level = level if is_header else None,
                token_count   = count_tokens(element.text),
            ))

        # ── FIGURE CAPTION (caption text only) ────────────
        elif isinstance(element, PictureItem):
            caption = element.caption_text(docling_doc)

            parsed.elements.append(ParsedElement(
                block_id     = block_id,
                element_type = "figure_caption",
                label        = "figure_caption",
                content      = caption or "",
                page         = page,
                bbox         = bbox,
                caption      = caption,
                token_count  = count_tokens(caption or ""),
            ))

        # ── TABLE ─────────────────────────────────────────
        elif isinstance(element, TableItem):
            table_md = element.export_to_markdown(
                docling_doc
            )
            caption  = element.caption_text(docling_doc)

            # Keep native list-of-dicts, not a JSON-encoded string.
            table_df = None
            try:
                df       = element.export_to_dataframe(
                    docling_doc
                )
                table_df = df.to_dict(orient="records")
            except Exception:
                pass

            table_path = os.path.join(
                tables_dir,
                f"{block_id}_p{page}.md"
            )
            with open(
                table_path, "w", encoding="utf-8"
            ) as f:
                if caption:
                    f.write(f"Caption: {caption}\n\n")
                f.write(table_md)

            parsed.elements.append(ParsedElement(
                block_id      = block_id,
                element_type  = "table",
                label         = "table",
                content       = table_md,
                page          = page,
                bbox          = bbox,
                caption       = caption,
                table_md      = table_md,
                table_df      = table_df,
                table_rows    = element.data.num_rows,
                table_cols    = element.data.num_cols,
                token_count   = count_tokens(table_md),
            ))

        # ── LIST ITEM ─────────────────────────────────────
        elif isinstance(element, ListItem):
            if len(element.text.strip()) < 3:
                continue
            parsed.elements.append(ParsedElement(
                block_id     = block_id,
                element_type = "list_item",
                label        = "list_item",
                content      = element.text,
                page         = page,
                bbox         = bbox,
                token_count  = count_tokens(element.text),
            ))

    # ── Single post-processing pass ───────────────────────
    # Computes full_text, stripped_tokens,
    # front_matter candidates, section_id assignments,
    # and final counter values in one loop.
    pp = _post_process_elements(parsed.elements, counters)

    # ── Full document exports ─────────────────────────────
    parsed.full_markdown = docling_doc.export_to_markdown()
    parsed.full_text     = "\n\n".join(pp["full_text_parts"])
    parsed.full_text_tokens = count_tokens(parsed.full_text)
    parsed.stripped_tokens  = count_tokens(
        "\n\n".join(pp["stripped_parts"])
    )

    # ── Top-level title and abstract ──────────────────────
    # Fallback title extraction: keep a stable top-level
    # title even when Docling TITLE labeling is missing/noisy.
    parsed.title    = extract_title_from_elements(
        parsed.elements
    )
    # Fallback abstract extraction: prefer explicit
    # "Abstract" heading, then page-1 heuristic.
    parsed.abstract = extract_abstract(parsed.elements)
    if parsed.abstract:
        print(
            f"  [ABSTRACT] Found "
            f"({count_tokens(parsed.abstract)} tokens)"
        )

    # ── Front matter ──────────────────────────────────────
    parsed.front_matter = build_front_matter(
        pp["front_matter_elements"]
    )

    # ── Print summary using final counters ────────────────
    print(f"\n  Pages           : {parsed.pages}")
    print(f"  Headers         : {counters['header']}")
    print(f"  Text blocks     : {counters['text']}")
    print(f"  Tables          : {counters['table']}")
    print(f"  Raw tokens      : {parsed.full_text_tokens} "
            f"(cl100k exact)")
    print(f"  Stripped tokens : {parsed.stripped_tokens} "
            f"(references removed)")
    print(f"  Title           : {parsed.title[:60]}")
    print(
        f"  Abstract found  : "
        f"{'yes' if parsed.abstract else 'no'}"
    )

    # attach final counters to parsed for save()
    parsed._counters = counters
    return parsed

# ─────────────────────────────────────────────────────────
# SAVE
# Receives counters from parse() and writes table/title fields in native form.
# ─────────────────────────────────────────────────────────
def save(parsed: ParsedDocument, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # Use counters built during parse + post-process.
    counters = getattr(parsed, "_counters", {})

    def serialise_element(e: ParsedElement) -> dict:
        d = e.__dict__.copy()
        # table_df is already a native list, json.dump handles it
        return d

    output = {
        "step"               : "step1_parse",
        "filename"           : parsed.filename,
        "doc_id"             : parsed.doc_id,
        "pages"              : parsed.pages,
        # Top-level fields used by downstream steps.
        "title"              : parsed.title,
        "abstract"           : parsed.abstract,
        "total_elements"     : len(parsed.elements),
        # Counters from the parse-time source of truth.
        "counts"             : counters,
        "full_text_tokens"   : parsed.full_text_tokens,
        "stripped_tokens"    : parsed.stripped_tokens,
        "front_matter"       : parsed.front_matter,
        "elements"           : [
            serialise_element(e) for e in parsed.elements
        ],
        "full_markdown"      : parsed.full_markdown,
        "full_text"          : parsed.full_text,
    }

    out_path = get_output_path(output_dir, parsed.doc_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            output, f, indent=2,
            ensure_ascii=False, default=str
        )
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"  Saved : {out_path}  ({size_kb:.1f} KB)")
    return out_path

# ─────────────────────────────────────────────────────────
# CORPUS SUMMARY TABLE
# ─────────────────────────────────────────────────────────
def build_document_summary(step1_data: dict) -> dict:
    counts   = step1_data.get("counts", {})
    elements = step1_data.get("elements", [])

    heading_levels = sorted(set(
        e["heading_level"]
        for e in elements
        if e.get("element_type") == "header"
        and e.get("heading_level") is not None
    ))

    raw_tokens      = step1_data.get("full_text_tokens", 0)
    stripped_tokens = step1_data.get("stripped_tokens",  0)

    return {
        "doc_id"          : step1_data.get("doc_id",   ""),
        "filename"        : step1_data.get("filename", ""),
        # Prefer top-level title; fallback to header extraction.
        "title"           : step1_data.get(
                                "title", ""
                            ) or extract_title_from_elements(
                                elements
                            ),
        "abstract_found"  : "yes" if step1_data.get(
                                "abstract", ""
                            ) else "no",
        "pages"           : step1_data.get("pages", 0),
        "raw_tokens"      : raw_tokens,
        "stripped_tokens" : stripped_tokens,
        "headers"         : counts.get("header",         0),
        "heading_levels"  : str(heading_levels),
        "text_blocks"     : counts.get("text",            0),
        "tables"          : counts.get("table",           0),
    }

def _format_summary_table(rows: list) -> list:
    display_cols = [
        ("filename",         "File",         20),
        ("title",            "Title",        28),
        ("abstract_found",   "Abs",           4),
        ("pages",            "Pg",            3),
        ("raw_tokens",       "RawTok",        7),
        ("stripped_tokens",  "StrTok",        7),
        ("headers",          "Hdrs",          5),
        ("heading_levels",   "Lvls",          6),
        ("tables",           "Tbl",           4),
    ]
    lines     = []
    header    = " | ".join(
        f"{n:<{w}}" for _, n, w in display_cols
    )
    separator = "-+-".join(
        "-" * w for _, _, w in display_cols
    )
    lines.append(header)
    lines.append(separator)
    for row in rows:
        cells = []
        for key, _, width in display_cols:
            val = str(row.get(key, ""))
            if len(val) > width:
                val = val[:width-3] + "..."
            cells.append(f"{val:<{width}}")
        lines.append(" | ".join(cells))
    return lines

def _print_corpus_stats(rows: list):
    if not rows:
        return
    total        = len(rows)
    total_raw    = sum(r["raw_tokens"]      for r in rows)
    total_str    = sum(r["stripped_tokens"] for r in rows)
    avg_raw      = total_raw   // total if total else 0
    avg_stripped = total_str   // total if total else 0
    avg_pages    = sum(r["pages"] for r in rows) // total \
                   if total else 0
    has_abstract = sum(
        1 for r in rows if r.get("abstract_found") == "yes"
    )
    total_tables = sum(r["tables"]          for r in rows)
    print(f"\n{'─'*60}")
    print(f"CORPUS STATISTICS ({total} documents)")
    print(f"{'─'*60}")
    print(f"  Avg pages per paper      : {avg_pages}")
    print(f"  Avg raw tokens           : {avg_raw:,}"
            f"  (full doc, cl100k exact)")
    print(f"  Avg stripped tokens      : {avg_stripped:,}"
            f"  (references removed, LLM input)")
    print(f"  Total corpus raw tokens  : {total_raw:,}")
    print(f"  Total stripped tokens    : {total_str:,}")
    print(f"")
    print(f"  Abstracts found          : "
          f"{has_abstract}/{total}")
    print(f"")
    print(f"  Extracted elements:")
    print(f"    Tables           : {total_tables}")
    print(f"{'─'*60}")

def save_corpus_summary(output_dir: str) -> str:
    step1_files = sorted(
        Path(output_dir).glob("step1_*.json")
    )
    if not step1_files:
        print("No step1_*.json files found.")
        return ""
    rows = []
    for path in step1_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(build_document_summary(data))
        except Exception as e:
            print(
                f"  Warning: could not read "
                f"{path.name}: {e}"
            )
    if not rows:
        return ""
    csv_path   = os.path.join(
        output_dir, "corpus_summary.csv"
    )
    fieldnames = list(rows[0].keys())
    try:
        with open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        print(
            f"\n  Warning: Cannot write {csv_path}"
            f"\n  File may be open in Excel."
            f"\n  Close it and rerun your Step 1 command."
        )
    txt_path    = os.path.join(
        output_dir, "corpus_summary.txt"
    )
    table_lines = _format_summary_table(rows)
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(table_lines))
    except PermissionError:
        print(f"  Warning: Cannot write {txt_path}")
    print(f"\n{'='*70}")
    print(f"CORPUS PARSING SUMMARY")
    print(f"{'='*70}")
    for line in table_lines:
        print(line)
    _print_corpus_stats(rows)
    print(f"\n  CSV : {csv_path}")
    print(f"  TXT : {txt_path}")
    return csv_path

# ─────────────────────────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────────────────────────
def preview(parsed: ParsedDocument):
    print(f"\n{'─'*55}")
    print(f"PREVIEW: Section structure (first 8 headers)")
    print(f"{'─'*55}")
    headers = [
        e for e in parsed.elements
        if e.element_type == "header"
    ]
    for h in headers[:8]:
        indent = "  " * (h.heading_level or 1)
        print(
            f"  p{h.page:>3} | L{h.heading_level} | "
            f"~{h.token_count:>4}tok |"
            f"{indent}{h.content[:55]}"
        )
    print(f"\n{'─'*55}")
    print(f"PREVIEW: Document size")
    print(f"{'─'*55}")
    print(f"  Title           : {parsed.title[:60]}")
    print(
        f"  Abstract        : "
        f"{'found' if parsed.abstract else 'not found'}"
    )
    print(f"  Pages           : {parsed.pages}")
    print(f"  Raw tokens      : {parsed.full_text_tokens:,}"
            f"  (cl100k exact, full doc)")
    print(f"  Stripped tokens : {parsed.stripped_tokens:,}"
            f"  (references removed)")
    print(f"\n{'─'*55}")
    print(f"PREVIEW: Captions from picture items (first 6)")
    print(f"{'─'*55}")
    figure_captions = [
        e for e in parsed.elements
        if e.element_type == "figure_caption" and e.content.strip()
    ]
    for e in figure_captions[:6]:
        print(
            f"  p{e.page:>3} | "
            f"sec: {e.section_id[-12:] or 'none':<12} | "
            f"{e.content[:80]}"
        )
    if not figure_captions:
        print("  None")
    print(f"\n{'─'*55}")
    print(f"PREVIEW: Front matter (first 400 chars)")
    print(f"{'─'*55}")
    print(parsed.front_matter[:400])
    print(f"\n{'─'*55}")
    print(f"PREVIEW: Section hierarchy sample (first 5 elements)")
    print(f"{'─'*55}")
    for e in parsed.elements[:5]:
        print(
            f"  p{e.page:>3} | {e.element_type:<10} | "
            f"sec: {e.section_id[-12:] or '<before first header>':<14} | "
            f"{e.content[:50]}"
        )
    print(f"\n{'─'*55}")
    print(f"PREVIEW: Tables")
    print(f"{'─'*55}")
    tables = [
        e for e in parsed.elements
        if e.element_type == "table"
    ]
    if tables:
        for t in tables[:3]:
            print(
                f"  p{t.page:>3} | "
                f"{t.table_rows}x{t.table_cols} | "
                f"~{t.token_count}tok | "
                f"sec: {t.section_id[-12:] or 'none'} | "
                f"caption: {bool(t.caption)}"
            )
    else:
        print("  None")

# ─────────────────────────────────────────────────────────
# BATCH RUNNER
# Release parsed.docling_doc after save() to keep batch memory stable.
# ─────────────────────────────────────────────────────────
def run_batch(
    pdf_files:  List[Path],
    output_dir: str,
    overwrite:  bool
) -> List[ParsedDocument]:
    total   = len(pdf_files)
    skipped = failed = done = 0

    print(f"\n{'='*55}")
    print(f"STEP 1: BATCH PDF PARSING WITH DOCLING")
    print(f"{'='*55}")
    print(f"  Documents : {total}")
    print(f"  Output    : {os.path.abspath(output_dir)}")
    print(f"  Overwrite : {overwrite}")
    print(f"  Tokens    : cl100k_base exact (tiktoken)")
    print(f"\n  Loading Docling models...")
    converter = build_converter()
    print(f"  Docling ready.\n")

    results = []
    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_id = make_doc_id(str(pdf_path))
        print(f"[{idx}/{total}] {pdf_path.name}")
        if not overwrite and already_processed(
            output_dir, doc_id
        ):
            print(
                f"  SKIPPED (output exists). "
                f"Use --overwrite to redo.\n"
            )
            skipped += 1
            continue
        try:
            parsed = parse(
                str(pdf_path), output_dir, converter
            )
            preview(parsed)
            save(parsed, output_dir)
            # Release large Docling object after save for stable memory use.
            parsed.docling_doc = None
            results.append(parsed)
            done += 1
            print()
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print(f"\n{'='*55}")
    print(f"BATCH COMPLETE")
    print(f"  Processed : {done}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Total     : {total}")
    print(f"{'='*55}")
    save_corpus_summary(output_dir)
    return results

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Step 1: PDF Parsing with Docling"
    )
    group = parser.add_mutually_exclusive_group(
        required=True
    )
    group.add_argument(
        "--file",
        help="Path to a single PDF file"
    )
    group.add_argument(
        "--folder",
        help="Folder of PDFs searched recursively"
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Root output directory (default: output)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process even if output exists"
    )
    args  = parser.parse_args()
    start = time.time()
    if args.file:
        pdf_files = [Path(args.file)]
    else:
        pdf_files = collect_pdfs(args.folder)
        if not pdf_files:
            print(f"No PDFs found in: {args.folder}")
            return
    run_batch(pdf_files, args.output_dir, args.overwrite)
    print(f"\n  Total time : {time.time() - start:.1f}s")
    print(
        f"  Outputs    : "
        f"{os.path.abspath(args.output_dir)}"
    )
    print(f"  Next step  : step2_summarize_metadata.py")

if __name__ == "__main__":
    main()
