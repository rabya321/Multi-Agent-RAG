"""
naive_rag.py
========================================================
Simple baseline RAG for comparison with agent_rag.
ALL questions routed directly to chunk_store.
No planning node, no routing, no fallback, no validation.
Just: embed query → search chunks → generate answer.

Used to demonstrate the value of agent_rag's architecture.


Changes:
- All questions are routed directly to the chunk_store, bypassing any planning or routing logic.
- top_k
- no metadata filtering
- no reranking by formula score
- no validation

"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import litellm
from qdrant_client import QdrantClient

#-------------------
# PATH SETUP
#-------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rag.retrieval import set_embedder, CHUNK_COLLECTION, QDRANT_PATH
from agent_rag.run       import LiteLLMEmbedder

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
LITELLM_BASE_URL = os.getenv("BASE_URL")
GENERATION_MODEL = "openai/gpt-5.2"
TOP_K            = 15
SLEEP_BETWEEN    = 0.5

# ─────────────────────────────────────────────────────────
# MINIMAL GENERATION PROMPT
# No routing instructions, no domain context,
# no citation rules, no length guidance
# ─────────────────────────────────────────────────────────
NAIVE_SYSTEM = """\
Answer the question using only the provided context. Donot exceed 200 tokens
\
"""

NAIVE_USER = """\
Context:
{context}

Question: {query}

Answer:\
"""


# ─────────────────────────────────────────────────────────
# QDRANT CLIENT
# ─────────────────────────────────────────────────────────
_qdrant_client = None

def get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


# ─────────────────────────────────────────────────────────
# Bypasses all retrieval.py logic:
#   NO metadata filtering
#   NO content_type filtering
#   NO filter relaxation variants
#   NO formula reranking
#   Pure cosine similarity only
# ─────────────────────────────────────────────────────────
def direct_chunk_search(query: str) -> list:
    """
    Embeds query and searches chunk_store directly.
    NO filters of any kind — pure cosine similarity.
    Returns list of (payload, cosine_score) tuples.
    """
    from agent_rag.retrieval import get_embedder
    vector = get_embedder().embed_query(query)
    client = get_client()

    response = client.query_points(
        collection_name = CHUNK_COLLECTION,
        query           = vector,
        query_filter    = None,      # ← NO filter at all
        limit           = TOP_K,
        with_payload    = True,
    )
    return [(hit.payload, hit.score) for hit in response.points]


# ─────────────────────────────────────────────────────────
# CALL LLM
# ─────────────────────────────────────────────────────────
def call_llm(system: str, user: str, max_retries: int = 2) -> str:
    litellm.ssl_verify = os.getenv("CERT_PATH")
    for attempt in range(max_retries + 1):
        try:
            response = litellm.completion(
                model    = GENERATION_MODEL,
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                api_base = LITELLM_BASE_URL,
                api_key  = os.getenv("API_KEY"),
            )
            return response.choices[0].message.content
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"LLM call failed: {exc}") from exc

# ─────────────────────────────────────────────────────────
# BUILD CONTEXT
# ─────────────────────────────────────────────────────────
def _build_context(payloads_with_scores: list) -> tuple:
    """
    Builds plain text context from chunk payloads.
    Returns (context_str, retrieved_contexts_list).
    No XML formatting, no source attribution — minimal.
    """
    context_parts      = []
    retrieved_contexts = []

    for payload, score in payloads_with_scores:
        text = (
            payload.get("chunk_text",   "")
            or payload.get("summary_text", "")
        )
        if text:
            context_parts.append(text)
            retrieved_contexts.append(text)

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, retrieved_contexts

# ─────────────────────────────────────────────────────────
# NAIVE RAG QUERY
# ─────────────────────────────────────────────────────────
def naive_rag_query(query: str) -> dict:
    """
    Maximally simple RAG:
      1. Embed query
      2. Search chunk_store — NO filters, NO reranking
      3. Build plain text context
      4. Generate with minimal prompt
            5. Return — NO validation

    Every query type gets the same treatment.
    Reasoning/multi_hop/comparative get chunk content
    instead of summaries → degraded answers for these types.
    """
    # Step 1: Direct chunk search — no filters, no reranking
    results = direct_chunk_search(query)

    # Step 2: Build minimal context
    context_str, retrieved_contexts = _build_context(results)

    # Step 3: Generate with minimal prompt
    if not context_str.strip():
        answer = "No relevant context found."
    else:
        user_prompt = NAIVE_USER.format(
            context = context_str,
            query   = query,
        )
        answer = call_llm(NAIVE_SYSTEM, user_prompt)

    # Step 4: Collect retrieval metadata
    retrieval_scores = [round(float(s), 4) for _, s in results]
    source_ids       = [
        p.get("chunk_id") or p.get("doc_id") or ""
        for p, _ in results
    ]
    top_retrieval_score = retrieval_scores[0] if retrieval_scores else 0.0

    return {
        "answer"             : answer,
        "retrieval_path"     : f"direct ({CHUNK_COLLECTION})",
        "top_retrieval_score": top_retrieval_score,
        "query_type"         : "naive_rag",
        "answer_tokens"      : round(len(answer.split()) * 1.3),
        "retrieved_contexts" : retrieved_contexts,
        "retrieval_scores"   : retrieval_scores,
        "source_ids"         : source_ids,
        "quality_flags"      : {
            "confidence"  : "MEDIUM",
            "completeness": "COMPLETE",
            "flagged"     : False,
        },
        "source_title"       : "",
        "source_page"        : None,
        "source_section"     : "",
        "citation_string"    : "",
    }

# ─────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────
def save_results(results: list, output_path: Path, total_qs: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at"   : datetime.now().isoformat(),
                "total_questions": total_qs,
                "total_generated": len(results),
                "pipeline_type"  : "naive_rag_chunk_only_no_filter",
                "config": {
                    "top_k"              : TOP_K,
                    "metadata_filtering" : False,
                    "content_filtering"  : False,
                    "reranking"          : False,
                    "query_enrichment"   : False,
                    "routing"            : False,
                    "validation"         : False,
                    "store"              : CHUNK_COLLECTION,
                },
                "results": results,
            },
            f, indent=2, ensure_ascii=False,
        )

# ─────────────────────────────────────────────────────────
# RUN ON TESTSET
# ─────────────────────────────────────────────────────────
def run_naive_on_testset(questions: list, output_path: Path):
    total      = len(questions)
    done       = 0
    errors     = 0
    results    = []
    start_time = time.time()

    print(f"\n  Processing {total} questions with naive RAG...")
    print(f"  Store       : chunk_store only (all query types)")
    print(f"  Top-k       : {TOP_K}")
    print(f"  Filters     : NONE")
    print(f"  Reranking   : NONE")
    print(f"  OOS detect  : NONE")

    for idx, q in enumerate(questions, 1):
        q_id   = q.get("question_id", f"Q{idx:04d}")
        q_type = q.get("question_type", "")

        print(
            f"\n  [{idx:>4}/{total}] "
            f"{q_type:<30} "
            f"{q.get('question', '')[:60]}"
        )

        try:
            response = naive_rag_query(q.get("question", ""))
            result = {
                # From test set
                "question_id"        : q_id,
                "question"           : q.get("question",        ""),
                "question_type"      : q_type,
                "expected_answer"    : q.get("expected_answer",  ""),
                "expected_winner"    : q.get("expected_winner",  ""),
                "source_titles"      : q.get("source_titles",   []),
                # From naive RAG
                "generated_answer"   : response.get("answer",          ""),
                "retrieval_path"     : response.get("retrieval_path",  ""),
                "top_retrieval_score": response.get("top_retrieval_score", 0.0),
                "query_type_pred"    : "naive_rag",
                "answer_tokens"      : response.get("answer_tokens",    0),
                "answer_flagged"     : False,
                "confidence"         : "MEDIUM",
                "completeness"       : "COMPLETE",
                "source_title"       : "",
                "source_page"        : None,
                "source_section"     : "",
                "citation_string"    : "",
                "retrieved_contexts" : response.get("retrieved_contexts", []),
                "retrieval_scores"   : response.get("retrieval_scores",   []),
                "source_ids"         : response.get("source_ids",         []),
                "generated_at"       : datetime.now().isoformat(),
            }
        except Exception as e:
                print(f"  [Error] {e}")
                errors += 1
                result = {
                "question_id"        : q_id,
                "question"           : q.get("question",       ""),
                "question_type"      : q_type,
                "expected_answer"    : q.get("expected_answer", ""),
                "expected_winner"    : q.get("expected_winner", ""),
                "source_titles"      : q.get("source_titles",  []),
                "generated_answer"   : f"[ERROR] {e}",
                "retrieval_path"     : "error",
                "top_retrieval_score": 0.0,
                "query_type_pred"    : "naive_rag",
                "answer_tokens"      : 0,
                "answer_flagged"     : True,
                "confidence"         : "LOW",
                "completeness"       : "EMPTY",
                "source_title"       : "",
                "source_page"        : None,
                "source_section"     : "",
                "citation_string"    : "",
                "retrieved_contexts" : [],
                "retrieval_scores"   : [],
                "source_ids"         : [],
                "generated_at"       : datetime.now().isoformat(),
            }

        results.append(result)
        done += 1

        print(
            f"    top_score={result['top_retrieval_score']:.3f} | "
            f"tokens={result['answer_tokens']}"
        )

        if done % 10 == 0:
            save_results(results, output_path, total)
            elapsed   = time.time() - start_time
            per_q     = elapsed / done
            remaining = (total - done) * per_q
            print(
                f"\n  [CHECKPOINT] {done} saved. "
                f"~{remaining/60:.1f} min remaining.\n"
            )

        if idx < total:
            time.sleep(SLEEP_BETWEEN)

    save_results(results, output_path, total)
    return done, errors


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Minimal naive RAG baseline — maximum degradation."
    )
    parser.add_argument("--test_set",   required=True)
    parser.add_argument("--output_dir", default="Evaluation")
    parser.add_argument("--sample",     type=int, default=None)
    args = parser.parse_args()

    missing = [v for v in ("API_KEY", "CERT_PATH") if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    with open(args.test_set, "r", encoding="utf-8") as f:
        questions = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        questions   = questions[:args.sample]
        output_path = out_dir / "naive_rag_results_sample.json"
        print(f"\n[SAMPLE MODE] {args.sample} questions.")
    else:
        output_path = out_dir / "naive_rag_results.json"

    print(f"\n{'='*60}")
    print(f"NAIVE RAG BASELINE")
    print(f"{'='*60}")
    print(f"  Test set         : {args.test_set}")
    print(f"  Questions        : {len(questions)}")
    print(f"  Output           : {output_path}")
    print(f"  Store            : chunk_store (ALL query types)")
    print(f"  Top-k            : {TOP_K}")
    print(f"  Metadata filter  : DISABLED")
    print(f"  Reranking        : DISABLED")
    print(f"  Query enrichment : DISABLED")
    print(f"  Validation       : DISABLED")
    print(f"  Model            : {GENERATION_MODEL}")

    print("\n  Loading embedder...")
    embedder = LiteLLMEmbedder()
    set_embedder(embedder)
    print("  Ready.\n")

    start_time = time.time()
    done, errors = run_naive_on_testset(
        questions   = questions,
        output_path = output_path,
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"COMPLETE — NAIVE RAG")
    print(f"{'='*60}")
    print(f"  Generated : {done}")
    print(f"  Errors    : {errors}")
    print(f"  Time      : {elapsed/60:.1f} minutes")
    print(f"  Saved to  : {output_path}")
    

if __name__ == "__main__":
    main()


