import json
import time
import argparse
import base64
import re
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from config import QUERY_MODEL, WIKI_MODEL, API_KEY, BASE_URL, CERT_PATH, ANSWER_LENGTH_INSTRUCTION
import httpx

DEFAULT_TOP_K      = 7
DEFAULT_CHUNK      = 512
DEFAULT_COLLECTION = "naive_rag_compiled_slides"
EMBEDDING_DIM      = 1536

IMAGE_TO_TEXT_PROMPT = """
You are extracting factual slide content for a retrieval corpus.
Return plain text only (no markdown fences, no JSON).

Extraction priorities (same as wiki ingestion image policy):
1) Hierarchy/layer structures: list all levels in order with labels.
2) Flows/sequences: capture every step and connector direction.
3) Entity relationships: include entity names and relationship labels.
4) Numeric content: extract numbers, percentages, thresholds, table values.
5) Component diagrams: list every labeled component.
6) Swimlanes/process roles: capture who does what and in what order.

Do not include decorative or non-informational elements:
- branding logos
- slide numbers
- decorative icons/background art

Requirements:
- Keep wording concise and factual.
- Preserve domain terms exactly when legible.
- If text is unclear, mark as [unclear] rather than guessing.
""".strip()

def _module_stem_from_image_path(image_path: Path) -> str:
    """Infer module stem from image path or fallback to file stem."""
    name = image_path.stem
    slide_pattern = re.search(r"^(.*?)(?:[_-]slide[_-]?\d+)$", name, re.IGNORECASE)
    if slide_pattern and slide_pattern.group(1):
        return slide_pattern.group(1)

    parent = image_path.parent.name
    if parent and parent.lower() not in {"images", "compiled", "slides"}:
        return parent

    return name


def _transcribe_slide_image(image_path: Path, vision_llm: ChatOpenAI) -> str:
    """Convert one compiled slide image to text using Claude Sonnet 5.5."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    message = vision_llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_TO_TEXT_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }
    ])
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts).strip()
    return str(content).strip()


def extract_content_from_compiled_slides(compiled_slides_path: str) -> list[dict]:
    """Extract module text by transcribing compiled PNG slides with Claude Sonnet 5.5."""
    folder = Path(compiled_slides_path)
    image_files = sorted(folder.rglob("*.png"))
    if not image_files:
        raise FileNotFoundError(
            f"No .png compiled slides found in folder: {compiled_slides_path}"
        )

    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    vision_llm = ChatOpenAI(
        model=WIKI_MODEL,
        temperature=0.0,
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
    )

    grouped: dict[str, list[Path]] = {}
    for image_path in image_files:
        stem = _module_stem_from_image_path(image_path)
        grouped.setdefault(stem, []).append(image_path)

    results: list[dict] = []
    print(f"Found {len(image_files)} compiled slide images in {compiled_slides_path}")
    for stem in sorted(grouped.keys()):
        module_text_parts: list[str] = []
        for image_path in sorted(grouped[stem]):
            extracted = _transcribe_slide_image(image_path, vision_llm)
            if extracted:
                module_text_parts.append(f"[{image_path.name}]\n{extracted}")

        module_text = "\n\n".join(module_text_parts).strip()
        if not module_text:
            print(f"  Skipped (no extracted text): {stem}")
            continue

        print(f"  Transcribed: {stem} ({len(module_text)} chars)")
        results.append({
            "filename": f"{stem}.pngset",
            "stem": stem,
            "content": module_text,
        })

    print(f"\nExtracted content from {len(results)} compiled slide sets.")
    return results


# -------------------------
# BUILD VECTOR STORE
# -------------------------

def build_vector_store(
    compiled_slides_path: str,
    chunk_size: int,
    collection_name: str,
    store_path: str,
) -> QdrantVectorStore:
    """
    Extract content from compiled slide images, then chunk, embed and
    index into a local Qdrant store at store_path.
    """
    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    embeddings  = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
    )

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separator="",
    )

    extracted = extract_content_from_compiled_slides(compiled_slides_path)
    documents = []
    for item in extracted:
        chunks = splitter.split_text(item["content"])
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "filename":    item["filename"],
                    "stem":        item["stem"],
                    "chunk_index": idx,
                },
            ))

    print(f"Built {len(documents)} chunks from {len(extracted)} files.")

    client = QdrantClient(path=store_path)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    store.add_documents(documents)
    print(f"Indexed {len(documents)} chunks. Store saved to {store_path}")
    return store


# -------------------------
# LOAD VECTOR STORE
# -------------------------

def load_vector_store(
    store_path: str,
    collection_name: str,
) -> QdrantVectorStore:
    """Load existing Qdrant store from disk."""
    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    embeddings  = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
    )
    client = QdrantClient(path=store_path)
    store  = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    print(f"Loaded existing vector store from {store_path}")
    return store


# -------------------------
# ANSWER ONE QUESTION
# -------------------------

def answer_question(
    question: str,
    store: QdrantVectorStore,
    llm,
    top_k: int,
) -> dict:
    start_time = time.monotonic()

    docs    = store.similarity_search(question, k=top_k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a supply chain knowledge assistant for Infineon.
Answer the question using only the provided context.
Answer format: {ANSWER_LENGTH_INSTRUCTION}

Context:
{context}

Question: {question}

Answer:"""

    message    = llm.invoke(prompt)
    response   = message.content.strip()
    usage      = message.usage_metadata or {}
    latency_ms = int((time.monotonic() - start_time) * 1000)

    return {
        "answer":             response,
        "retrieved_contexts": [d.page_content for d in docs],
        "retrieved_stems":    [d.metadata.get("stem", "unknown") for d in docs],
        "latency_ms":         latency_ms,
        "total_tokens":       usage.get("total_tokens", 0),
        "input_tokens":       usage.get("input_tokens", 0),
        "output_tokens":      usage.get("output_tokens", 0),
        "model_calls":        1,
        "tool_calls":         0,
    }


# -------------------------
# LOAD QUESTIONS
# -------------------------

def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError(
        f"Unrecognized format in {path}. "
        f"Expected a list or a dict with a 'questions' key."
    )


# -------------------------
# MAIN RUN
# -------------------------

def run_naive_rag(
    questions_paths: list[str],
    compiled_slides_path: str | None,
    store_path: str,
    top_k: int              = DEFAULT_TOP_K,
    chunk_size: int         = DEFAULT_CHUNK,
    collection_name: str    = DEFAULT_COLLECTION,
    output_path: str | None = None,
) -> None:
    """
    If store_path exists load it. If not build it from compiled slides.
    Then answer all questions from every questions file and write
    the results to a single JSONL file.
    """
    if Path(store_path).exists():
        store = load_vector_store(store_path, collection_name)
    else:
        if not compiled_slides_path:
            raise ValueError(
                "--compiled-slides is required when building a new store."
            )
        print("No existing store found. Building from compiled slide images...")
        store = build_vector_store(
            compiled_slides_path=compiled_slides_path,
            chunk_size=chunk_size,
            collection_name=collection_name,
            store_path=store_path,
        )

    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    llm = ChatOpenAI(
        model=QUERY_MODEL,
        temperature=0.0,
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
    )

    questions = []
    for path in questions_paths:
        questions.extend(load_questions(path))

    ids = [q.get("id") for q in questions]
    duplicates = {i for i in ids if i is not None and ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids across files: {sorted(duplicates)}")

    if output_path is None:
        timestamp   = int(time.time())
        stem        = (
            Path(questions_paths[0]).stem
            if len(questions_paths) == 1 else "combined"
        )
        output_path = f"naive_rag_{stem}_{timestamp}.jsonl"

    out_path = Path(output_path)
    print(f"\nRunning {len(questions)} questions.")
    print(f"Output: {out_path}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            q_id   = q.get("id", f"Q{i}")
            q_type = q.get("type", "unknown")

            print(
                f"[{i}/{len(questions)}] {q_id} ({q_type})...",
                end=" ", flush=True,
            )

            result = answer_question(q["question"], store, llm, top_k)

            gt = q.get("ground_truth", {})
            record = {
                "id":       q_id,
                "type":     q_type,
                "evaluation_mode": q.get("evaluation_mode"),
                "track":    q.get("track"),
                "question": q["question"],
                "required_modules": q.get("required_modules", []),
                **result,
                "retrieved_modules": sorted(set(result.get("retrieved_stems", []))),
                "ground_truth": {"answer": gt.get("answer", "")},
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"latency={result['latency_ms']}ms")

    print(f"\nDone. {len(questions)} questions written to {out_path}")


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
      description="Naive RAG baseline using compiled slide images (OCR -> chunk -> retrieve).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  First run — builds store then queries:
      python results.naive_rag.py --questions evaluation/output/standard_questions.json --compiled-slides raw_sources/general/images --store qdrant_store

  Second run — loads existing store and queries:
      python results.naive_rag.py --questions evaluation/output/cross_track_questions.json --compiled-slides raw_sources/general/images --store qdrant_store

  Standard + cross-track into one results file:
      uv run python -m results.naive_rag --questions evaluation/output/standard_questions.json evaluation/output/cross_track_questions.json --compiled-slides raw_sources/general/images --store qdrant_store --output results/naive_all.jsonl
        """,
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more questions JSON files (standard and/or "
            "cross-track). All are combined into one results file."
        ),
    )
    parser.add_argument(
        "--compiled-slides",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to compiled slide PNGs from ingestion pipeline output. "
            "Required only when building a new store; optional when loading "
            "an existing store."
        ),
    )
    parser.add_argument(
        "--store",
        type=str,
        default="qdrant_store",
        help="Path to Qdrant store folder. Default: qdrant_store",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve. Default: {DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Chunk size in characters. Default: {DEFAULT_CHUNK}",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file. Auto-generated if not set.",
    )

    args = parser.parse_args()
    run_naive_rag(
        questions_paths=args.questions,
        compiled_slides_path=args.compiled_slides,
        store_path=args.store,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        collection_name=args.collection,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()