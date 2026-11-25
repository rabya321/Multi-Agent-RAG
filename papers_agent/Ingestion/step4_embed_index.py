"""
Step 4: Embed & Index
========================================================
Reads Step 2 embed_text / metadata files and Step 3
chunk files, embeds them via LiteLLM, and upserts into
two Qdrant local vector stores:

  summary_store : one point per paper
                  vector     = embed of structured summary
                  payload    = all metadata fields

  chunk_store   : one point per chunk
                  vector     = embed of heading-prefixed text
                  payload    = all chunk fields 

Collections are created on first run and reused across
runs.  Use --overwrite to delete and rebuild.

Environment variables:
  API_KEY   : company LiteLLM gateway API key
  CERT_PATH : path to CA bundle .crt file

USAGE:
  uv run step4_embed_index.py --output_dir output/
  uv run step4_embed_index.py --output_dir output/ --overwrite
  uv run step4_embed_index.py --output_dir output/ --inspect
  uv run step4_embed_index.py --output_dir output/ --doc_id doc_9808ee598042
  uv run step4_embed_index.py --summaries_only
  uv run step4_embed_index.py --chunks_only
"""

import os
import json
import csv
import hashlib
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import litellm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
EMBED_MODEL  = "openai/text-embedding-3-small"
VECTOR_SIZE  = 1536
BATCH_SIZE   = 50
MAX_RETRIES  = 3
RETRY_WAIT   = 2   # seconds, doubles each retry

QDRANT_PATH        = "./qdrant_storage"
SUMMARY_COLLECTION = "summary_store"
CHUNK_COLLECTION   = "chunk_store"

EMBED_TEXT_SUBDIR    = "step2_summarize_metadata/embed_text"
METADATA_SUBDIR      = "step2_summarize_metadata/metadata"
CHUNKS_SUBDIR        = "step3_chunk/chunks"
STEP4_SUBDIR         = "step4_embed"
SUMMARY_LOG_FILENAME = "summary_index_log.csv"
CHUNK_LOG_FILENAME   = "chunk_index_log.csv"

LITELLM_BASE_URL = "https://litellm.icp.infineon.com/"

# ─────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────
def load_env() -> dict:
    """
    Loads and validates required environment variables.
    Returns dict with api_key and cert_path.
    Raises clear error if either is missing or cert
    file does not exist on disk.
    """
    api_key   = os.getenv("API_KEY")
    cert_path = os.getenv("CERT_PATH")

    if not api_key:
        raise ValueError(
            "API_KEY not set.\n"
            "Windows: $env:API_KEY='your_key'\n"
            "Linux:   export API_KEY=your_key"
        )
    if not cert_path:
        raise ValueError(
            "CERT_PATH not set.\n"
            "Windows: $env:CERT_PATH="
            "'C:/path/ca-bundle.crt'\n"
            "Linux:   export CERT_PATH="
            "/path/ca-bundle.crt"
        )
    if not Path(cert_path).exists():
        raise FileNotFoundError(
            f"Certificate not found: {cert_path}\n"
            f"Check CERT_PATH is correct."
        )
    return {
        "api_key"  : api_key,
        "cert_path": cert_path,
    }

# ─────────────────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────────────────
def embed_batch(
    texts:     list,
    api_key:   str,
    cert_path: str,
) -> list:
    """
    Embeds one batch of texts via LiteLLM.
    Retries up to MAX_RETRIES times on failure.
    Waits RETRY_WAIT * 2^attempt seconds between retries.
    Raises RuntimeError if all retries exhausted.
    Returns list of float vectors.
    """
    litellm.ssl_verify = cert_path
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = litellm.embedding(
                model    = EMBED_MODEL,
                input    = texts,
                api_base = LITELLM_BASE_URL,
                api_key  = api_key,
            )
            return [
                item["embedding"]
                for item in response.data
            ]
        except Exception as e:
            last_error = str(e)
            wait = RETRY_WAIT * (2 ** (attempt - 1))
            print(
                f"  [EMBED] Attempt {attempt}/"
                f"{MAX_RETRIES} failed: {e}"
            )
            if attempt < MAX_RETRIES:
                print(f"  [EMBED] Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Embedding failed after {MAX_RETRIES} attempts."
        f" Last error: {last_error}"
    )


def embed_all(
    texts:     list,
    api_key:   str,
    cert_path: str,
    label:     str = "texts",
) -> list:
    """
    Embeds all texts in batches of BATCH_SIZE.
    Returns list of vectors in the same order as input.
    label is used for progress print only.

    If a batch fails after all retries:
      logs the failure
      inserts None for each text in that batch
      continues with remaining batches
      caller checks for None values in result
    """
    total   = len(texts)
    vectors = []
    batches = [
        texts[i: i + BATCH_SIZE]
        for i in range(0, total, BATCH_SIZE)
    ]

    print(
        f"  [EMBED] {total} {label} in "
        f"{len(batches)} batch(es) "
        f"(batch_size={BATCH_SIZE})"
    )

    for idx, batch in enumerate(batches, 1):
        print(
            f"  [EMBED] Batch {idx}/{len(batches)} "
            f"({len(batch)} texts)..."
        )
        try:
            batch_vectors = embed_batch(
                batch, api_key, cert_path
            )
            vectors.extend(batch_vectors)
            print(
                f"  [EMBED] Batch {idx} done. "
                f"Vector dim: {len(batch_vectors[0])}"
            )
        except RuntimeError as e:
            print(f"  [EMBED] Batch {idx} FAILED: {e}")
            vectors.extend([None] * len(batch))

    successful = sum(1 for v in vectors if v is not None)
    failed     = total - successful
    print(
        f"  [EMBED] Complete: "
        f"{successful} succeeded, {failed} failed"
    )
    return vectors

# ─────────────────────────────────────────────────────────
# QDRANT CLIENT
# ─────────────────────────────────────────────────────────
def build_client() -> QdrantClient:
    """
    Opens or creates local Qdrant database at QDRANT_PATH.
    No server or Docker needed.
    Persists to disk across runs.
    """
    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)
    print(
        f"  [QDRANT] Local DB: "
        f"{os.path.abspath(QDRANT_PATH)}"
    )
    return client

# ─────────────────────────────────────────────────────────
# COLLECTION CREATION
# ─────────────────────────────────────────────────────────
def _create_collection(
    client:    QdrantClient,
    name:      str,
    overwrite: bool,
    hnsw:      bool,
):
    """
    Internal helper. Creates one collection.
    If overwrite=True, deletes existing collection first.
    If overwrite=False and collection exists, skips.
    hnsw=False → flat exact search (summary store).
    hnsw=True  → HNSW configured for chunk store.

    Note on retrieval mode:
    Qdrant keeps exact full-scan search while vector data
    size is below full_scan_threshold (default here: 10000).
    HNSW traversal is used only after that threshold is
    exceeded.
    """
    existing = [
        c.name
        for c in client.get_collections().collections
    ]

    if name in existing:
        if overwrite:
            client.delete_collection(name)
            print(f"  [QDRANT] Deleted collection: {name}")
        else:
            print(
                f"  [QDRANT] Collection exists: {name} "
                f"(use --overwrite to rebuild)"
            )
            return

    vector_params = VectorParams(
        size     = VECTOR_SIZE,
        distance = Distance.COSINE,
        on_disk  = False,
    )

    if hnsw:
        client.create_collection(
            collection_name = name,
            vectors_config  = vector_params,
            hnsw_config     = HnswConfigDiff(
                m                   = 16, #only for HNSW, maximum number of outgoing connections per node
                ef_construct        = 100, #only for HNSW, search breadth during graph construction
                full_scan_threshold = 10000
            ),
        )
        print(
            f"  [QDRANT] Created: {name} "
            f"(HNSW configured, exact full scan below threshold)"
        )
    else:
        client.create_collection(
            collection_name   = name,
            vectors_config    = vector_params,
            optimizers_config = OptimizersConfigDiff(
                indexing_threshold = 0
            ),
        )
        print(
            f"  [QDRANT] Created: {name} "
            f"(flat exact search, HNSW disabled)"
        )


def create_summary_collection(
    client:    QdrantClient,
    overwrite: bool = False,
):
    """
    Creates summary_store collection with flat exact search.
    indexing_threshold=0 disables HNSW entirely.
    Every query does brute force cosine similarity.
    Guarantees 100% recall on summary retrieval.
    """
    _create_collection(
        client    = client,
        name      = SUMMARY_COLLECTION,
        overwrite = overwrite,
        hnsw      = False,
    )


def create_chunk_collection(
    client:    QdrantClient,
    overwrite: bool = False,
):
    """
    Creates chunk_store collection with HNSW indexing.
    full_scan_threshold=10000 keeps exact full-scan
    retrieval while vector data remains below threshold;
    HNSW is used when the collection grows beyond it.

    In the evaluated system, chunk-store data volume is
    below this threshold, so retrieval remains exact full
    scan and the default threshold is retained.
    """
    _create_collection(
        client    = client,
        name      = CHUNK_COLLECTION,
        overwrite = overwrite,
        hnsw      = True,
    )

# ─────────────────────────────────────────────────────────
# QDRANT HELPERS
# ─────────────────────────────────────────────────────────
def get_indexed_doc_ids(
    client:          QdrantClient,
    collection_name: str,
) -> set:
    """
    Scrolls entire collection and returns set of
    doc_id values found in payloads.
    Used at startup to skip already indexed documents.
    """
    indexed = set()
    offset  = None

    while True:
        records, next_offset = client.scroll(
            collection_name = collection_name,
            limit           = 100,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
        )
        for record in records:
            doc_id = record.payload.get("doc_id")
            if doc_id:
                indexed.add(doc_id)
        if next_offset is None:
            break
        offset = next_offset

    return indexed


def delete_doc(
    client:          QdrantClient,
    collection_name: str,
    doc_id:          str,
):
    """
    Deletes all points where payload["doc_id"] == doc_id.
    Used when --overwrite is passed for a specific doc.
    """
    client.delete(
        collection_name = collection_name,
        points_selector = FilterSelector(
            filter = Filter(
                must = [
                    FieldCondition(
                        key   = "doc_id",
                        match = MatchValue(value=doc_id)
                    )
                ]
            )
        )
    )
    print(
        f"  [QDRANT] Deleted existing points for "
        f"{doc_id} from {collection_name}"
    )


def upsert_points(
    client:          QdrantClient,
    collection_name: str,
    points:          list,
):
    """
    Upserts a list of PointStruct objects into collection.
    Uses upsert so re-runs are safe: existing point_id
    is updated not duplicated.
    """
    client.upsert(
        collection_name = collection_name,
        points          = points,
    )


def build_point(
    point_id: int,
    vector:   list,
    payload:  dict,
) -> PointStruct:
    """Constructs one PointStruct for upsert."""
    return PointStruct(
        id      = point_id,
        vector  = vector,
        payload = payload,
    )


def make_point_id(identifier: str) -> int:
    """
    Converts a string identifier to a deterministic
    integer point ID via MD5 hash mod 10^15.
    For summaries: identifier = doc_id
    For chunks:    identifier = chunk_id
    Same identifier always produces same integer.
    Safe for upsert re-runs.
    """
    return (
        int(hashlib.md5(
            identifier.encode("utf-8")
        ).hexdigest(), 16)
        % (10 ** 15)
    )


def print_collection_info(
    client: QdrantClient,
    name:   str,
):
    """Prints collection statistics for verification."""
    try:
        info  = client.get_collection(name)
        count = client.count(name).count
        print(f"\n  {'─'*45}")
        print(f"  COLLECTION : {name}")
        print(f"  Documents  : {count}")
        print(
            f"  Vector size: "
            f"{info.config.params.vectors.size}"
        )
        print(
            f"  Distance   : "
            f"{info.config.params.vectors.distance}"
        )
        print(f"  Status     : {info.status}")
    except Exception as e:
        print(f"  [QDRANT] Could not get info: {e}")

# ─────────────────────────────────────────────────────────
# INPUT LOADERS
# ─────────────────────────────────────────────────────────
def load_summary_inputs(output_dir: str) -> list:
    """
    Loads all summary inputs by pairing embed_text
    .txt files with their metadata JSON files.

    Each returned dict has:
      doc_id        : string identifier
      summary_text  : full flat prose string
                      (from embed_text .txt file)
                      used as: vector source + LLM context
      metadata      : dict from metadata JSON
                      used as: payload filter fields

    Pairing is by doc_id extracted from filename stem.
    Skips any doc_id where either file is missing.
    """
    embed_dir = Path(output_dir) / EMBED_TEXT_SUBDIR
    meta_dir  = Path(output_dir) / METADATA_SUBDIR

    inputs  = []
    missing = []

    for txt_path in sorted(
        embed_dir.glob("step2_embed_*.txt")
    ):
        doc_id    = txt_path.stem.replace(
            "step2_embed_", ""
        )
        meta_path = (
            meta_dir / f"step2_metadata_{doc_id}.json"
        )

        if not meta_path.exists():
            missing.append(doc_id)
            continue

        summary_text = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        with open(
            meta_path, "r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        inputs.append({
            "doc_id"       : doc_id,
            "summary_text" : summary_text,
            "metadata"     : metadata,
        })

    if missing:
        print(
            f"  [LOAD] {len(missing)} summaries skipped "
            f"(no matching metadata): {missing}"
        )
    print(
        f"  [LOAD] {len(inputs)} summary inputs loaded"
    )
    return inputs


def load_chunk_inputs(output_dir: str) -> list:
    """
    Loads all chunk inputs from step3 output.
    Each step3_chunks_{doc_id}.json contains a list
    of chunk dicts already carrying all needed fields.

    Returns flat list of chunk dicts across all papers.
    Each chunk dict has embed_text, chunk_text,
    heading_path, page, content_type, and document
    metadata fields (title, authors, year, venue,
    domain, doc_id, etc.).
    """
    chunks_root   = Path(output_dir) / CHUNKS_SUBDIR
    all_chunks    = []
    papers_loaded = 0

    for doc_dir in sorted(chunks_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id     = doc_dir.name
        chunk_file = (
            doc_dir / f"step3_chunks_{doc_id}.json"
        )
        if not chunk_file.exists():
            print(
                f"  [LOAD] Missing chunk file for "
                f"{doc_id}, skipping"
            )
            continue
        with open(
            chunk_file, "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        all_chunks.extend(chunks)
        papers_loaded += 1

    print(
        f"  [LOAD] {len(all_chunks)} chunks loaded "
        f"from {papers_loaded} papers"
    )
    return all_chunks

# ─────────────────────────────────────────────────────────
# PAYLOAD BUILDERS
# ─────────────────────────────────────────────────────────
def build_summary_payload(
    doc_id:       str,
    summary_text: str,
    metadata:     dict,
) -> dict:
    """
    Builds flat Qdrant payload for one summary.
    Contains identity fields, filter fields,
    and summary_text for LLM context.
    No nested dicts. Qdrant requires flat payload.
    """
    return {
        # identity
        "doc_id"           : doc_id,
        "title"            : metadata.get("title",   ""),
        "authors"          : metadata.get("authors", []),
        "year"             : metadata.get("year"),
        "venue"            : metadata.get("venue",   ""),
        # filter fields
        "domain"           : metadata.get("domain",  ""),
        "keywords_explicit": metadata.get(
            "keywords_explicit", []
        ),
        "keywords_inferred": metadata.get(
            "keywords_inferred", []
        ),
        "all_keywords"     : metadata.get(
            "all_keywords", []
        ),
        # LLM context text
        # same string that was embedded as the vector
        "summary_text"     : summary_text,
    }


def build_chunk_payload(chunk: dict) -> dict:
    """
    Builds flat Qdrant payload for one chunk.
    chunk dict from step3 already carries all fields.
    embed_text excluded from payload:
      it was used to produce the vector.
      LLM reads chunk_text + heading context instead.
    """
    return {
        # identity
        "chunk_id"     : chunk.get("chunk_id",     ""),
        "doc_id"       : chunk.get("doc_id",        ""),
        "title"        : chunk.get("title",         ""),
        "authors"      : chunk.get("authors",       []),
        "year"         : chunk.get("year"),
        "venue"        : chunk.get("venue",         ""),
        # filter fields
        "domain"       : chunk.get("domain",        ""),
        "content_type" : chunk.get("content_type",  "text"),
        # LLM context fields
        "chunk_text"   : chunk.get("chunk_text",    ""),
        "heading_path" : chunk.get("heading_path",  ""),
        "page"         : chunk.get("page",          0),
        "image_path"   : chunk.get("image_path"),
        "token_count"  : chunk.get("token_count",   0),
        "source_file"  : chunk.get("source_file",   ""),
        "chunk_index"  : chunk.get("chunk_index",   0),
    }

# ─────────────────────────────────────────────────────────
# INDEX LOG
# ─────────────────────────────────────────────────────────
def save_index_log(
    rows:       list,
    output_dir: str,
    filename:   str,
):
    """
    Writes index log CSV to output/step4_embed/.
    Creates directory if it does not exist.
    """
    if not rows:
        return

    log_dir  = Path(output_dir) / STEP4_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    with open(
        log_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [LOG] Saved: {log_path}")

# ─────────────────────────────────────────────────────────
# SUMMARY INDEXING PIPELINE
# ─────────────────────────────────────────────────────────
def index_summaries(
    client:        QdrantClient,
    env:           dict,
    output_dir:    str,
    overwrite:     bool,
    doc_id_filter: Optional[str] = None,
):
    """
    Full summary indexing pipeline.
    Load → detect already indexed → embed → upsert → log.

    doc_id_filter: if provided, only indexes that
    one document. Used with --doc_id CLI argument.
    """
    print(f"\n{'='*55}")
    print(f"INDEXING SUMMARIES → {SUMMARY_COLLECTION}")
    print(f"{'='*55}")

    inputs = load_summary_inputs(output_dir)

    if doc_id_filter:
        inputs = [
            i for i in inputs
            if i["doc_id"] == doc_id_filter
        ]
        if not inputs:
            print(
                f"  [SKIP] doc_id {doc_id_filter} "
                f"not found in summary inputs"
            )
            return

    indexed_ids = get_indexed_doc_ids(
        client, SUMMARY_COLLECTION
    )
    print(
        f"  Already indexed: {len(indexed_ids)} summaries"
    )

    to_index = []
    for item in inputs:
        if item["doc_id"] in indexed_ids and not overwrite:
            print(
                f"  [SKIP] {item['doc_id']} already indexed"
            )
            continue
        if item["doc_id"] in indexed_ids and overwrite:
            delete_doc(
                client, SUMMARY_COLLECTION, item["doc_id"]
            )
        to_index.append(item)

    if not to_index:
        print("  Nothing to index.")
        return

    texts   = [i["summary_text"] for i in to_index]
    vectors = embed_all(
        texts     = texts,
        api_key   = env["api_key"],
        cert_path = env["cert_path"],
        label     = "summaries",
    )

    log_rows = []
    points   = []

    for item, vector in zip(to_index, vectors):
        if vector is None:
            log_rows.append({
                "doc_id"         : item["doc_id"],
                "title"          : item["metadata"].get(
                    "title", ""
                ),
                "point_id"       : None,
                "summary_tokens" : len(
                    item["summary_text"].split()
                ),
                "indexed_at"     : datetime.now().isoformat(),
                "status"         : "FAILED",
            })
            continue

        payload  = build_summary_payload(
            item["doc_id"],
            item["summary_text"],
            item["metadata"],
        )
        point_id = make_point_id(item["doc_id"])
        points.append(build_point(point_id, vector, payload))
        log_rows.append({
            "doc_id"         : item["doc_id"],
            "title"          : item["metadata"].get(
                "title", ""
            ),
            "point_id"       : point_id,
            "summary_tokens" : len(
                item["summary_text"].split()
            ),
            "indexed_at"     : datetime.now().isoformat(),
            "status"         : "OK",
        })

    if points:
        upsert_points(client, SUMMARY_COLLECTION, points)
        print(
            f"  [QDRANT] Upserted {len(points)} "
            f"summaries into {SUMMARY_COLLECTION}"
        )

    save_index_log(
        log_rows, output_dir, SUMMARY_LOG_FILENAME
    )
    print_collection_info(client, SUMMARY_COLLECTION)

# ─────────────────────────────────────────────────────────
# CHUNK INDEXING PIPELINE
# ─────────────────────────────────────────────────────────
def index_chunks(
    client:        QdrantClient,
    env:           dict,
    output_dir:    str,
    overwrite:     bool,
    doc_id_filter: Optional[str] = None,
):
    """
    Full chunk indexing pipeline.
    Load → detect already indexed → embed → upsert → log.

    Processes in batches of BATCH_SIZE.
    Builds one log row per paper (not per chunk)
    for a readable audit trail.
    """
    print(f"\n{'='*55}")
    print(f"INDEXING CHUNKS → {CHUNK_COLLECTION}")
    print(f"{'='*55}")

    all_chunks = load_chunk_inputs(output_dir)

    if doc_id_filter:
        all_chunks = [
            c for c in all_chunks
            if c.get("doc_id") == doc_id_filter
        ]
        if not all_chunks:
            print(
                f"  [SKIP] doc_id {doc_id_filter} "
                f"not found in chunk inputs"
            )
            return

    indexed_ids = get_indexed_doc_ids(
        client, CHUNK_COLLECTION
    )
    print(
        f"  Already indexed: "
        f"{len(indexed_ids)} documents in chunk store"
    )

    seen_for_delete = set()
    to_index        = []

    for chunk in all_chunks:
        doc_id = chunk.get("doc_id", "")
        if doc_id in indexed_ids and not overwrite:
            continue
        if (
            doc_id in indexed_ids
            and overwrite
            and doc_id not in seen_for_delete
        ):
            delete_doc(client, CHUNK_COLLECTION, doc_id)
            seen_for_delete.add(doc_id)
        to_index.append(chunk)

    if not to_index:
        print("  Nothing to index.")
        return

    print(f"  Chunks to index: {len(to_index)}")

    texts   = [c["embed_text"] for c in to_index]
    vectors = embed_all(
        texts     = texts,
        api_key   = env["api_key"],
        cert_path = env["cert_path"],
        label     = "chunks",
    )

    points    = []
    per_paper : dict = {}

    for chunk, vector in zip(to_index, vectors):
        doc_id = chunk.get("doc_id", "")
        if doc_id not in per_paper:
            per_paper[doc_id] = {
                "doc_id"        : doc_id,
                "title"         : chunk.get("title", ""),
                "total_chunks"  : 0,
                "failed_chunks" : 0,
                "indexed_at"    : datetime.now().isoformat(),
            }

        if vector is None:
            per_paper[doc_id]["failed_chunks"] += 1
            continue

        payload  = build_chunk_payload(chunk)
        point_id = make_point_id(
            chunk.get("chunk_id", doc_id)
        )
        points.append(build_point(point_id, vector, payload))
        per_paper[doc_id]["total_chunks"] += 1

    if points:
        upsert_points(client, CHUNK_COLLECTION, points)
        print(
            f"  [QDRANT] Upserted {len(points)} chunks "
            f"into {CHUNK_COLLECTION}"
        )

    log_rows = []
    for doc_id, stats in per_paper.items():
        status = (
            "PARTIAL" if stats["failed_chunks"] > 0
            else "OK"
        )
        log_rows.append({
            "doc_id"        : stats["doc_id"],
            "title"         : stats["title"],
            "total_chunks"  : stats["total_chunks"],
            "failed_chunks" : stats["failed_chunks"],
            "indexed_at"    : stats["indexed_at"],
            "status"        : status,
        })

    save_index_log(
        log_rows, output_dir, CHUNK_LOG_FILENAME
    )
    print_collection_info(client, CHUNK_COLLECTION)

# ─────────────────────────────────────────────────────────
# INSPECT MODE
# ─────────────────────────────────────────────────────────
def inspect_collections(client: QdrantClient):
    """Prints stats for both collections."""
    print(f"\n{'='*55}")
    print(f"COLLECTION INSPECTION")
    print(f"{'='*55}")
    print_collection_info(client, SUMMARY_COLLECTION)
    print_collection_info(client, CHUNK_COLLECTION)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: Embed and index summaries and chunks. "
            "Model: openai/text-embedding-3-small. "
            "Collections: summary_store (flat), "
            "chunk_store (HNSW)."
        )
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Root output directory containing step2 "
             "and step3 outputs (default: output)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-index even if document already indexed. "
             "Deletes existing points for that doc_id first."
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print collection statistics and exit. "
             "No embedding or indexing performed."
    )
    parser.add_argument(
        "--doc_id", default=None,
        help="Index a single document by doc_id. "
             "e.g. --doc_id doc_9808ee598042"
    )
    parser.add_argument(
        "--summaries_only", action="store_true",
        help="Index summaries only, skip chunks."
    )
    parser.add_argument(
        "--chunks_only", action="store_true",
        help="Index chunks only, skip summaries."
    )

    args  = parser.parse_args()
    start = time.time()

    client = build_client()

    if args.inspect:
        inspect_collections(client)
        return

    try:
        env = load_env()
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return

    print(f"\n{'='*55}")
    print(f"STEP 4: EMBEDDING AND INDEXING")
    print(f"{'='*55}")
    print(
        f"  Output dir   : "
        f"{os.path.abspath(args.output_dir)}"
    )
    print(
        f"  Qdrant path  : "
        f"{os.path.abspath(QDRANT_PATH)}"
    )
    print(f"  Embed model  : {EMBED_MODEL}")
    print(f"  Vector size  : {VECTOR_SIZE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Certificate  : {env['cert_path']}")
    print(f"  Overwrite    : {args.overwrite}")

    create_summary_collection(
        client,
        overwrite=args.overwrite,
    )
    create_chunk_collection(
        client,
        overwrite=args.overwrite,
    )

    if not args.chunks_only:
        index_summaries(
            client        = client,
            env           = env,
            output_dir    = args.output_dir,
            overwrite     = args.overwrite,
            doc_id_filter = args.doc_id,
        )

    if not args.summaries_only:
        index_chunks(
            client        = client,
            env           = env,
            output_dir    = args.output_dir,
            overwrite     = args.overwrite,
            doc_id_filter = args.doc_id,
        )

    print(f"\n{'='*55}")
    print(f"STEP 4 COMPLETE")
    print(f"  Total time : {time.time() - start:.1f}s")
    print(
        f"  Qdrant     : "
        f"{os.path.abspath(QDRANT_PATH)}"
    )
    print(
        f"  Logs       : "
        f"{os.path.abspath(os.path.join(args.output_dir, STEP4_SUBDIR))}"
    )
    print(f"  Next step  : step5_retrieval.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
