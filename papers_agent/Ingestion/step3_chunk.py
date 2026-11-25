"""
Step 3: Hybrid Chunking with Docling
========================================================
Converts PDFs using Docling HybridChunker for structure
and semantically aware chunking.


Key behaviours:
  Tables  : treated as atomic, never split across chunks.
            flagged with is_oversized=True if token count
            exceeds MAX_TOKENS. Embedding model handles
            up to 8191 tokens so oversized is informational
            not an error.
  Figures : caption text retained as figure_caption chunks.
            page and section context are preserved.
  Text    : chunked by HybridChunker within section
                        boundaries, preserving heading context.

content_type field on every chunk:
  "text"           : paragraph, list, or prose content
  "table"          : full table serialised as markdown
  "figure_caption" : figure caption text

Tokenizer:
  tiktoken cl100k_base. Matches text-embedding-3-small
  token counting exactly. 

Output structure:
  output/step3_chunk/chunks/{doc_id}/
      step3_chunks_{doc_id}.json
  output/step3_chunk/chunks_summary/
      corpus_chunk_stats.csv

USAGE:
  python step3_chunk.py --pdf_folder ./papers
                        --input_dir  output/
                        --output_dir output/

  python step3_chunk.py --pdf_file   papers/paper.pdf
                        --input_dir  output/
                        --output_dir output/
                        --overwrite
"""
import os
import re
import json
import hashlib
import argparse
import time
import tiktoken
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from docling.document_converter import (
    DocumentConverter, PdfFormatOption
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

try:
    from docling_core.types.doc import (
        TableItem, PictureItem
    )
    _DOCLING_TYPES_AVAILABLE = True
except ImportError:
    _DOCLING_TYPES_AVAILABLE = False

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIGURATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TIKTOKEN_ENCODING = "cl100k_base"
MAX_TOKENS        = 512
MERGE_PEERS       = True
EMBED_MODEL       = "text-embedding-3-small"
TOKENIZER         = tiktoken.get_encoding(TIKTOKEN_ENCODING)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# OUTPUT PATH HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_chunk_dir(output_dir: str, doc_id: str) -> str:
    return os.path.join(
        output_dir, "step3_chunk", "chunks", doc_id
    )


def get_chunk_output_path(
    output_dir: str, doc_id: str
) -> str:
    return os.path.join(
        get_chunk_dir(output_dir, doc_id),
        f"step3_chunks_{doc_id}.json"
    )


def get_summary_dir(output_dir: str) -> str:
    return os.path.join(
        output_dir, "step3_chunk", "chunks_summary"
    )


def get_stats_csv_path(output_dir: str) -> str:
    return os.path.join(
        get_summary_dir(output_dir),
        "corpus_chunk_stats.csv"
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TIKTOKEN SETUP
# Used for both HybridChunker token limit enforcement
# and chunk token count logging.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def count_tokens(text: str) -> int:
    """
    Returns exact token count using cl100k_base encoding.
    """
    if not text or not text.strip():
        return 0
    return len(TOKENIZER.encode(text))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TOKENIZER WRAPPER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _build_tokenizer_wrapper():
    """
    Uses docling_core's built-in OpenAITokenizer.
    Designed specifically for OpenAI tokenizers
    including cl100k_base used by text-embedding-3-small.
    Proper BaseTokenizer subclass so Pydantic accepts it.
    Uses tiktoken internally.
    """
    from docling_core.transforms.chunker.tokenizer.openai \
        import OpenAITokenizer
    wrapper  = OpenAITokenizer(
        tokenizer  = TOKENIZER,
        max_tokens = MAX_TOKENS,
    )
    print(
        f"  [TOKENIZER] OpenAITokenizer loaded with "
        f"{TIKTOKEN_ENCODING} (cl100k_base, no HF download)"
    )
    return wrapper

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_doc_id(file_path: str) -> str:
    """Reproduces Step 1 doc_id so files link correctly."""
    name = Path(file_path).stem.encode("utf-8")
    return "doc_" + hashlib.md5(name).hexdigest()[:12]


def already_processed(
    output_dir: str, doc_id: str
) -> bool:
    return os.path.exists(
        get_chunk_output_path(output_dir, doc_id)
    )


def load_step1(
    input_dir: str, doc_id: str
) -> Optional[dict]:
    candidates = [
        os.path.join(input_dir, f"step1_{doc_id}.json"),
        os.path.join(
            input_dir, "step1_parsed", "pdfs_markdown",
            f"step1_{doc_id}.json"
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_step2_metadata(
    input_dir: str, doc_id: str
) -> Optional[dict]:
    """
    Loads Step 2 metadata for filter field enrichment.
    Returns None if not found. Chunking continues without
    it but filter fields in chunk payload will be empty.
    """
    path = os.path.join(
        input_dir, "step2_summarize_metadata", "metadata",
        f"step2_metadata_{doc_id}.json"
    )
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# REFERENCES DETECTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
REFERENCES_PATTERNS = [
    re.compile(r'^references?$',            re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+references?$',  re.IGNORECASE),
    re.compile(r'^bibliography$',           re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+bibliography$', re.IGNORECASE),
    re.compile(r'^works\s+cited$',          re.IGNORECASE),
    re.compile(r'^author\s+biograph',       re.IGNORECASE),
    re.compile(r'^acknowledgem',            re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+acknowledgem',  re.IGNORECASE),
]

def is_in_references_section(
    headings: List[str]
) -> bool:
    return any(
        any(
            p.match(h.strip())
            for p in REFERENCES_PATTERNS
        )
        for h in headings
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONTENT TYPE DETECTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def detect_content_type(raw_chunk) -> str:
    """
    Inspects raw_chunk.meta.doc_items element types.
    Table detection takes priority over figure.
    Returns: "table" | "figure_caption" | "text"
    """
    if (
        not _DOCLING_TYPES_AVAILABLE
        or not hasattr(raw_chunk, "meta")
        or not raw_chunk.meta
        or not raw_chunk.meta.doc_items
    ):
        return "text"

    has_table   = False
    has_picture = False

    for item in raw_chunk.meta.doc_items:
        if isinstance(item, TableItem):
            has_table = True
        elif isinstance(item, PictureItem):
            has_picture = True

    if has_table:
        return "table"
    if has_picture:
        return "figure_caption"
    return "text"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DATA MODEL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class Chunk:
    """
    One chunk from a paper.

    content_type:
      "text"           : paragraph or prose content
      "table"          : full table as markdown, atomic,
                         never split even if oversized
      "figure_caption" : caption text for a figure

    is_oversized:
      True if token_count > MAX_TOKENS.
      Always True for large tables (kept intact).
      Embedding model handles these fine.
      Informational flag only.

    method_family and methodology_type excluded by design.
    These remain in summary store payload only.
    """
    chunk_id:      str
    doc_id:        str
    chunk_index:   int
    embed_text:    str
    chunk_text:    str
    heading_path:  str
    heading_level: int
    page:          int
    token_count:   int
    content_type:  str           = "text"
    is_oversized:  bool          = False
    title:         str           = ""
    authors:       List[str]     = field(
        default_factory=list
    )
    year:          Optional[int] = None
    venue:         str           = ""
    domain:        str           = ""
    source_file:   str           = ""

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HEADING PATH AND EMBED TEXT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_heading_path(headings: List[str]) -> str:
    """
    Joins parent headings into a readable path string.
    [] â†’ ""
    ["Methodology"] â†’ "Methodology"
    ["Methodology", "RL Setup"] â†’ "Methodology > RL Setup"
    """
    cleaned = [h.strip() for h in headings if h.strip()]
    return " > ".join(cleaned)


def build_embed_text(
    heading_path: str,
    chunk_text:   str,
) -> str:
    """
    Prepends section heading as context anchor so the
    chunk is semantically grounded even when the text
    alone lacks positional context clues.
    """
    if heading_path:
        return f"Section: {heading_path}\n\n{chunk_text}"
    return chunk_text

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DOCLING SETUP
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_converter() -> DocumentConverter:
    """
    do_ocr=False: OCR already ran in Step 1.
    generate_picture_images=False: not needed for chunking.
    """
    options = PdfPipelineOptions()
    options.do_ocr                  = False
    options.generate_picture_images = False
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )


def build_chunker() -> HybridChunker:
    """
    Builds HybridChunker using OpenAITokenizer with
    cl100k_base. Matches text-embedding-3-small token
    counting exactly. No HuggingFace download needed.
    """
    tokenizer_wrapper = _build_tokenizer_wrapper()
    return HybridChunker(
        tokenizer   = tokenizer_wrapper,
        max_tokens  = MAX_TOKENS,
        merge_peers = MERGE_PEERS,
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORE: CHUNK ONE DOCUMENT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def chunk_document(
    pdf_path:      str,
    doc_id:        str,
    converter:     DocumentConverter,
    chunker:       HybridChunker,
    step2_meta:    Optional[dict],
) -> List[Chunk]:
    """
    Re-converts PDF and applies HybridChunker.

    For each raw chunk:
      - detects content_type (text/table/figure_caption)
      - builds embed_text with heading context prefix
      - counts tokens with tiktoken cl100k_base
      - flags is_oversized if token_count > MAX_TOKENS

    Reference and acknowledgement chunks excluded.
    Returns list of Chunk objects ready for Step 4.
    """
    result      = converter.convert(pdf_path)
    docling_doc = result.document

    title            = ""
    authors: List[str] = []
    year             = None
    venue            = ""
    domain           = ""

    if step2_meta:
        title   = step2_meta.get("title",   "")
        authors = step2_meta.get("authors", [])
        year    = step2_meta.get("year")
        venue   = step2_meta.get("venue",   "")
        domain  = step2_meta.get("domain",  "")

    chunks:      List[Chunk] = []
    chunk_index: int         = 0
    skipped_ref: int         = 0

    for raw_chunk in chunker.chunk(docling_doc):
        chunk_text = raw_chunk.text.strip()
        if not chunk_text:
            continue

        headings: List[str] = []
        try:
            if (
                hasattr(raw_chunk, "meta")
                and raw_chunk.meta
            ):
                headings = raw_chunk.meta.headings or []
        except Exception:
            headings = []

        if is_in_references_section(headings):
            skipped_ref += 1
            continue

        content_type = detect_content_type(raw_chunk)
        heading_path = build_heading_path(headings)
        embed_text   = build_embed_text(
            heading_path, chunk_text
        )
        token_count  = count_tokens(embed_text)

        page = 0
        try:
            if (
                hasattr(raw_chunk, "meta")
                and raw_chunk.meta
                and raw_chunk.meta.doc_items
            ):
                first_item = raw_chunk.meta.doc_items[0]
                if first_item.prov:
                    page = (
                        first_item.prov[0].page_no or 0
                    )
        except Exception:
            page = 0

        chunks.append(Chunk(
            chunk_id      = f"{doc_id}_chunk_{chunk_index:04d}",
            doc_id        = doc_id,
            chunk_index   = chunk_index,
            embed_text    = embed_text,
            chunk_text    = chunk_text,
            heading_path  = heading_path,
            heading_level = len(headings),
            page          = page,
            token_count   = token_count,
            content_type  = content_type,
            is_oversized  = token_count > MAX_TOKENS,
            title         = title,
            authors       = authors,
            year          = year,
            venue         = venue,
            domain        = domain,
            source_file   = Path(pdf_path).name,
        ))
        chunk_index += 1

    # â”€â”€ post-chunk reporting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if skipped_ref:
        print(
            f"  [SKIP]     {skipped_ref} reference/"
            f"acknowledgement chunks excluded"
        )

    oversized = [c for c in chunks if c.is_oversized]
    if oversized:
        print(
            f"  [OVERSIZE] {len(oversized)} chunks exceed "
            f"{MAX_TOKENS} tokens (kept, not split)"
        )
        for c in oversized[:3]:
            print(
                f"             {c.chunk_id} | "
                f"{c.token_count} tok | "
                f"{c.content_type} | "
                f"{c.heading_path[:40]}"
            )

    type_counts: dict = {}
    for c in chunks:
        type_counts[c.content_type] = (
            type_counts.get(c.content_type, 0) + 1
        )
    print(
        f"  [TYPES]    "
        + "  ".join(
            f"{t}: {n}" for t, n in type_counts.items()
        )
    )

    return chunks

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SAVE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_chunks(
    chunks:     List[Chunk],
    doc_id:     str,
    step1_data: dict,
    output_dir: str,
) -> str:
    chunk_dir = get_chunk_dir(output_dir, doc_id)
    os.makedirs(chunk_dir, exist_ok=True)

    token_counts = [c.token_count for c in chunks]
    type_counts: dict = {}
    for c in chunks:
        type_counts[c.content_type] = (
            type_counts.get(c.content_type, 0) + 1
        )

    output = {
        "step"               : "step3_chunks",
        "doc_id"             : doc_id,
        "filename"           : step1_data.get(
            "filename", ""
        ),
        "title"              : step1_data.get(
            "title", ""
        ),
        "pages"              : step1_data.get("pages", 0),
        "tokenizer"          : TIKTOKEN_ENCODING,
        "max_tokens_target"  : MAX_TOKENS,
        "embed_model"        : EMBED_MODEL,
        "total_chunks"       : len(chunks),
        "total_tokens"       : sum(token_counts),
        "avg_tokens"         : (
            round(
                sum(token_counts) / len(token_counts)
            )
            if token_counts else 0
        ),
        "min_tokens"         : (
            min(token_counts) if token_counts else 0
        ),
        "max_tokens_obs"     : (
            max(token_counts) if token_counts else 0
        ),
        "oversized_chunks"   : sum(
            1 for c in chunks if c.is_oversized
        ),
        "content_type_counts": type_counts,
        "chunks"             : [
            c.__dict__ for c in chunks
        ],
    }

    out_path = get_chunk_output_path(output_dir, doc_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            output, f, indent=2, ensure_ascii=False
        )

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"  Saved : {out_path}  ({size_kb:.1f} KB)")
    return out_path

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PREVIEW
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def preview(chunks: List[Chunk], doc_id: str):
    if not chunks:
        print("  No chunks produced.")
        return

    token_counts = [c.token_count for c in chunks]
    n            = len(chunks)

    print(f"\n{'â”€'*55}")
    print(f"CHUNKS: {doc_id}")
    print(f"{'â”€'*55}")
    print(f"  Total chunks    : {n}")
    print(
        f"  Total tokens    : {sum(token_counts):,}"
        f"  (cl100k_base)"
    )
    print(
        f"  Avg / Min / Max : "
        f"{round(sum(token_counts)/n)} / "
        f"{min(token_counts)} / "
        f"{max(token_counts)}"
    )

    type_counts: dict = {}
    for c in chunks:
        type_counts[c.content_type] = (
            type_counts.get(c.content_type, 0) + 1
        )
    print(f"\n  Content types:")
    for t, count in type_counts.items():
        print(f"    {count:>4} | {t}")

    sections: dict = {}
    for c in chunks:
        top = (
            c.heading_path.split(" > ")[0]
            if c.heading_path else "(pre-heading)"
        )
        sections[top] = sections.get(top, 0) + 1
    print(f"\n  Top-level sections:")
    for sec, count in list(sections.items())[:8]:
        print(f"    {count:>4} chunks | {sec[:48]}")

    print(f"\n  First 3 chunks:")
    for c in chunks[:3]:
        print(
            f"\n  [{c.chunk_index:04d}] "
            f"p{c.page} | "
            f"{c.token_count} tok | "
            f"{c.content_type}"
            f"{'*' if c.is_oversized else ''}"
        )
        print(f"  heading: {c.heading_path[:55]}")
        print(f"  text   : {c.chunk_text[:100]}...")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORPUS CHUNK STATS CSV
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_corpus_chunk_stats(
    stats_rows: List[dict],
    output_dir: str,
):
    if not stats_rows:
        return

    summary_dir = get_summary_dir(output_dir)
    os.makedirs(summary_dir, exist_ok=True)

    csv_path = get_stats_csv_path(output_dir)
    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(stats_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    n            = len(stats_rows)
    total_chunks = sum(
        r["total_chunks"] for r in stats_rows
    )
    total_tokens = sum(
        r["total_tokens"] for r in stats_rows
    )
    total_over   = sum(
        r["oversized_chunks"] for r in stats_rows
    )
    print(f"\n{'â”€'*55}")
    print(f"CORPUS CHUNK STATISTICS ({n} documents)")
    print(f"{'â”€'*55}")
    print(f"  Total chunks          : {total_chunks:,}")
    print(
        f"  Total tokens          : {total_tokens:,}"
        f"  (cl100k_base)"
    )
    print(
        f"  Avg chunks / paper    : {total_chunks // n}"
    )
    print(
        f"  Avg tokens / paper    : {total_tokens // n:,}"
    )
    print(
        f"  Oversized chunks      : {total_over}"
        f"  (tables, kept intact)"
    )
    print(f"  CSV                   : {csv_path}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# BATCH RUNNER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_batch(
    pdf_files:  List[Path],
    input_dir:  str,
    output_dir: str,
    overwrite:  bool,
) -> None:
    total   = len(pdf_files)
    done    = skipped = failed = 0
    stats_rows: List[dict] = []

    print(f"\n{'='*55}")
    print(f"STEP 3: HYBRID CHUNKING WITH DOCLING")
    print(f"{'='*55}")
    print(f"  PDFs         : {total}")
    print(f"  Input dir    : {os.path.abspath(input_dir)}")
    print(
        f"  Chunks out   : "
        f"{os.path.abspath(os.path.join(output_dir, 'step3_chunk', 'chunks'))}"
    )
    print(
        f"  Summary out  : "
        f"{os.path.abspath(get_summary_dir(output_dir))}"
    )
    print(
        f"  Tokenizer    : tiktoken {TIKTOKEN_ENCODING}"
        f"  (no HF download)"
    )
    print(f"  Embed model  : {EMBED_MODEL}")
    print(f"  Max tokens   : {MAX_TOKENS}")
    print(f"  Merge peers  : {MERGE_PEERS}")
    print(f"  Overwrite    : {overwrite}")
    print(
        f"\n  Loading Docling converter and chunker..."
    )

    converter = build_converter()
    chunker   = build_chunker()
    print(f"  Ready.\n")

    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_id = make_doc_id(str(pdf_path))
        print(
            f"[{idx}/{total}] "
            f"{pdf_path.name}  ({doc_id})"
        )

        if not overwrite and already_processed(
            output_dir, doc_id
        ):
            print(
                f"  SKIPPED (output exists). "
                f"Use --overwrite to redo.\n"
            )
            skipped += 1
            continue

        step1_data = load_step1(input_dir, doc_id)
        if step1_data is None:
            print(
                f"  FAILED: step1_{doc_id}.json not found."
                f" Run step1 first.\n"
            )
            failed += 1
            continue

        step2_meta = load_step2_metadata(
            input_dir, doc_id
        )
        if step2_meta is None:
            print(
                f"  [WARN] step2_metadata_{doc_id}.json "
                f"not found. Filter fields will be empty."
            )

        try:
            chunks = chunk_document(
                str(pdf_path), doc_id,
                converter, chunker,
                step2_meta,
            )

            if not chunks:
                print(
                    f"  WARN: zero chunks produced.\n"
                )
                failed += 1
                continue

            preview(chunks, doc_id)
            save_chunks(
                chunks, doc_id, step1_data, output_dir
            )

            token_counts = [c.token_count for c in chunks]
            stats_rows.append({
                "doc_id"             : doc_id,
                "filename"           : pdf_path.name,
                "title"              : step1_data.get(
                    "title", ""
                ),
                "total_chunks"       : len(chunks),
                "total_tokens"       : sum(token_counts),
                "avg_tokens"         : round(
                    sum(token_counts) / len(token_counts)
                ),
                "min_tokens"         : min(token_counts),
                "max_tokens"         : max(token_counts),
                "oversized_chunks"   : sum(
                    1 for c in chunks if c.is_oversized
                ),
                "text_chunks"        : sum(
                    1 for c in chunks
                    if c.content_type == "text"
                ),
                "table_chunks"       : sum(
                    1 for c in chunks
                    if c.content_type == "table"
                ),
                "figure_chunks"      : sum(
                    1 for c in chunks
                    if c.content_type == "figure_caption"
                ),
                "has_metadata"       : (
                    step2_meta is not None
                ),
            })
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

    save_corpus_chunk_stats(stats_rows, output_dir)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: Hybrid Chunking with Docling. "
            "Tokenizer: tiktoken cl100k_base wrapper "
            "(no HuggingFace download)."
        )
    )
    source = parser.add_mutually_exclusive_group(
        required=True
    )
    source.add_argument(
        "--pdf_folder",
        help="Folder of PDFs (searched recursively)"
    )
    source.add_argument(
        "--pdf_file",
        help="Path to a single PDF"
    )
    parser.add_argument(
        "--input_dir",  default="output",
        help="Directory with step1_*.json and "
             "step2_summarize_metadata/metadata/ files "
             "(default: output)"
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Root output directory. Chunks saved to "
             "output/step3_chunk/chunks/{doc_id}/ "
             "(default: output)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-chunk even if output exists"
    )

    args  = parser.parse_args()
    start = time.time()

    if args.pdf_file:
        pdf_files = [Path(args.pdf_file)]
    else:
        pdf_files = sorted(
            Path(args.pdf_folder).rglob("*.pdf")
        )
        if not pdf_files:
            print(
                f"No PDFs found in: {args.pdf_folder}"
            )
            return

    run_batch(
        pdf_files,
        args.input_dir,
        args.output_dir,
        args.overwrite,
    )

    print(
        f"\n  Total time   : {time.time() - start:.1f}s"
    )
    print(
        f"  Chunks       : "
        f"{os.path.abspath(os.path.join(args.output_dir, 'step3_chunk', 'chunks'))}"
    )
    print(
        f"  Stats        : "
        f"{os.path.abspath(get_summary_dir(args.output_dir))}"
    )
    print(f"  Next step    : step4_embed_index.py")


if __name__ == "__main__":
    main()
