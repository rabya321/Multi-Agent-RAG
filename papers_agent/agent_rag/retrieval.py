"""
retrieval.py
========================================================
All Qdrant operations and formula confidence scoring.
No nodes imports. No prompts imports.
"""
import os
import re as _re
import logging
from pathlib import Path

import litellm
import tiktoken
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
)

load_dotenv()

logger = logging.getLogger(__name__)
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
QDRANT_PATH          = str(Path(__file__).parent.parent / "qdrant_storage")
SUMMARY_COLLECTION   = "summary_store"
CHUNK_COLLECTION     = "chunk_store"
CONFIDENCE_THRESHOLD = 0.45   # fixed retrieval control threshold

# ─────────────────────────────────────────────────────────
# LITELLM CONFIGURATION
# Same .env as wiki_agent (API_KEY, CERT_PATH shared).
# ─────────────────────────────────────────────────────────
BASE_URL        = os.getenv("BASE_URL")
API_KEY         = os.getenv("API_KEY")
CERT_PATH       = os.getenv("CERT_PATH")
EMBEDDING_MODEL = "openai/text-embedding-3-small"

if CERT_PATH:
    litellm.ssl_verify = CERT_PATH


# ─────────────────────────────────────────────────────────
# STOPWORDS 
# ─────────────────────────────────────────────────────────
_STOPWORDS = frozenset({
    # Question words
    "what", "why", "how", "when", "where", "which", "who",
    "whom", "whose",
    # Auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "shall", "can",
    # Articles
    "the", "a", "an",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "into", "onto", "about", "above", "below",
    "between", "through", "during", "before", "after",
    "within", "without", "across", "under", "via",
    "toward", "towards", "versus",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet",
    "although", "because", "while", "whereas",
    "however", "therefore", "thus", "also",
    "moreover", "furthermore", "either", "neither",
    "both", "whether",
    # Pronouns
    "this", "that", "these", "those",
    "it", "its", "they", "them", "their",
    "we", "our", "you", "your",
    # Question / request verbs
    "tell", "explain", "describe", "compare", "list",
    "give", "show", "provide", "define", "state",
    "name", "identify", "discuss",
    # Generic academic / paper words
    "paper", "study", "research", "work", "article",
    "according", "based", "using", "used",
    "presented", "proposed", "described",
    "referred", "cited", "noted", "found", "shown",
    # Generic document structure
    # NOTE: "table" and "figure" deliberately EXCLUDED
    "section", "chapter", "equation",
    "page", "paragraph", "appendix",
    # Generic filler / connectors
    "generally", "typically", "usually", "often",
    "always", "never", "sometimes", "then", "than",
    "such", "more", "most", "less", "least",
    "very", "quite", "rather", "mainly", "primarily",
    "specifically", "particularly", "especially",
    "commonly", "widely", "largely",
    # Generic methodology words
    # NOTE: "level" and "scenario" deliberately EXCLUDED
    "approach", "method", "model",
    # Citation noise
    "et", "al",
    # Miscellaneous                      ← FIX: removed duplicate "me"
    "me",
})


# ─────────────────────────────────────────────────────────
# EMBEDDER
# Built the same way search_agent.py builds its LLM client:
# a thin client around litellm using the centralized config
# above. set_embedder()/get_embedder() are kept so callers can
# still override with a mock/alternate embedder, but nothing
# needs to call set_embedder() anymore — get_embedder() lazily
# builds this default LiteLLM embedder on first use.
# ─────────────────────────────────────────────────────────
class LiteLLMEmbedder:
    """Wrapper around litellm.embedding(), sharing config with search_agent.py."""

    def embed_query(self, text: str) -> list:
        response = litellm.embedding(
            model    = EMBEDDING_MODEL,
            input    = [text],
            api_base = BASE_URL,
            api_key  = API_KEY,
        )
        return response.data[0]["embedding"]


_embedder = None


def set_embedder(embedder) -> None:
    """Override the default embedder (e.g. for tests/mocks). Optional."""
    global _embedder
    _embedder = embedder


def get_embedder():
    """Return the active embedder, building the default LiteLLM embedder lazily."""
    global _embedder
    if _embedder is None:
        _embedder = LiteLLMEmbedder()
    return _embedder


def exact_tokens(text: str) -> int:
    """Return exact token count using cl100k_base."""
    if not text:
        return 0
    return len(TOKENIZER.encode(text))


# ─────────────────────────────────────────────────────────
# QDRANT CLIENT 
# ─────────────────────────────────────────────────────────
_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    """Open QdrantClient once and reuse across calls."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


# ─────────────────────────────────────────────────────────
# FUNCTION: build_qdrant_filter
# ─────────────────────────────────────────────────────────
def build_qdrant_filter(metadata_filters: dict):
    if not metadata_filters:
        return None
    conditions = []
    for key, value in metadata_filters.items():
        if value is None:
            continue
        # Ignore legacy hint fields if planner still emits them.
        if key in ("title_hint", "title_hints"):    
            continue
        if key == "author":
            conditions.append(
                FieldCondition(key="authors", match=MatchText(text=value))
            )
            continue
        conditions.append(
            FieldCondition(key=key, match=MatchValue(value=value))
        )
    if not conditions:
        return None
    return Filter(must=conditions)


# ─────────────────────────────────────────────────────────
# FUNCTION: _build_filter_variants
# ─────────────────────────────────────────────────────────
def _build_filter_variants(metadata_filters: dict) -> list:
    """
    Builds list of (label, Filter) tuples for progressive relaxation.
    Order:
      ("full_filter",      full Filter object)
      ("no_author_filter", Filter without author)
      ("no_year_filter",   Filter without author or year)
      ("no_filter",        None)
    Removes duplicate variants keeping order.
    """
    variants     = []
    seen_filters = []

    full_filter = build_qdrant_filter(metadata_filters)
    variants.append(("full_filter", full_filter))
    seen_filters.append(full_filter)

    no_author   = {k: v for k, v in metadata_filters.items() if k != "author"}
    f_no_author = build_qdrant_filter(no_author)
    if f_no_author not in seen_filters:
        variants.append(("no_author_filter", f_no_author))
        seen_filters.append(f_no_author)

    no_author_year   = {k: v for k, v in metadata_filters.items()
                        if k not in ("author", "year")}
    f_no_author_year = build_qdrant_filter(no_author_year)
    if f_no_author_year not in seen_filters:
        variants.append(("no_year_filter", f_no_author_year))
        seen_filters.append(f_no_author_year)

    if None not in seen_filters:
        variants.append(("no_filter", None))

    return variants


# ─────────────────────────────────────────────────────────
# FUNCTION: semantic_search
# ─────────────────────────────────────────────────────────
def semantic_search(
    query: str,
    collection_name: str,
    metadata_filters: dict,
    top_k: int,
) -> list:
    """
    Embeds query and searches the given collection with progressive
    filter relaxation.
    Returns list of (payload_dict, cosine_score) tuples.
    Returns empty list if no results after all variants.
    """
    vector = get_embedder().embed_query(query)
    client = get_qdrant_client()

    filter_variants = _build_filter_variants(metadata_filters)

    for label, qdrant_filter in filter_variants:
        try:
            response = client.query_points(
                collection_name = collection_name,
                query           = vector,
                query_filter    = qdrant_filter,
                limit           = top_k,
                with_payload    = True,
            )
            results = [(hit.payload, hit.score) for hit in response.points]
            if results:
                logger.info(
                    "[semantic_search] %s: %d results at step '%s'",
                    collection_name, len(results), label,
                )
                return results
        except Exception as exc:
            logger.warning(
                "[semantic_search] %s step '%s' error: %s",
                collection_name, label, exc,
            )
            continue

    logger.warning(
        "[semantic_search] %s: no results after all filter variants",
        collection_name,
    )
    return []


# ─────────────────────────────────────────────────────────
# FORMULA CONFIDENCE SCORE
# Uses combined semantic + lexical score:
#   S(c,q) = w_c * S_cos(c,q) + w_k * S_key(c,q)
# where S_key is keyword overlap in [0,1] after removing stopwords.
# ─────────────────────────────────────────────────────────
def formula_confidence_score(
    query: str,
    retrieved_text: str,
    cosine_sim: float,
    query_type: str,
) -> float:
    """
    Retrieval confidence score in [0, 1].

    score = w_cosine * cosine_similarity + w_keyword * keyword_overlap

    Weights sum to 1 and depend on query type:
      document-level questions: cosine weighted higher
      factual/table lookup questions: keyword weighted higher
    """
    qt = (query_type or "").lower()

    # Document-level routing gets stronger semantic weighting.
    if qt in ("reasoning", "comparative", "multi_hop", "superficial"):
        w_cosine  = 0.60
        w_keyword = 0.40
    # Fact-focused questions rely more on lexical cues
    # (acronyms, method names, table labels, numeric terms).
    elif qt in ("table_lookup", "factual", "section_specific"):
        w_cosine  = 0.40
        w_keyword = 0.60
    else:
        w_cosine  = 0.55
        w_keyword = 0.45

    # ── Component 1: Keyword Overlap ──────────────────────
    query_words = [
        w.lower()
        for w in _re.findall(r"[a-zA-Z0-9_\-]+", query)
        if w.lower() not in _STOPWORDS
    ]

    if query_words:
        text_words      = {
            w.lower()
            for w in _re.findall(r"[a-zA-Z0-9_\-]+", retrieved_text)
        }
        hits            = sum(1 for w in query_words if w in text_words)
        keyword_overlap = hits / len(query_words)
    else:
        keyword_overlap = 0.5   # neutral when no meaningful words remain

    score = w_cosine * cosine_sim + w_keyword * keyword_overlap
    return round(max(0.0, min(1.0, score)), 4)


# ─────────────────────────────────────────────────────────
# FUNCTION: _score_and_rerank
#  Scores ALL chunks then reranks by formula score.
# ─────────────────────────────────────────────────────────
def _score_and_rerank(
    results: list,
    query: str,
    query_type: str,
) -> list:
    """
    Applies formula_confidence_score() to ALL retrieved chunks.
    Returns list of (payload, cosine_score, formula_score) tuples
    sorted by formula_score descending.

    Args:
        results:    list of (payload, cosine_score) from semantic_search
        query:      query string
        query_type: planning node output (reasoning, table_lookup, etc.)

    Returns:
        list of (payload, cosine_score, formula_score) sorted descending
        by formula_score
    """
    if not results:
        return []

    scored = []
    for payload, cosine in results:
        text = (
            payload.get("chunk_text", "")
            or payload.get("summary_text", "")
        )

        f_score = formula_confidence_score(
            query,
            text,
            cosine,
            query_type,
        )

        scored.append((payload, cosine, f_score))

    # Sort descending by formula_score
    # Best formula chunk → top-1 → first seen by LLM
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ─────────────────────────────────────────────────────────
# FUNCTION: build_context
# ─────────────────────────────────────────────────────────

def build_context(
    payloads_with_scores: list,
    retrieval_path: str,
    formula_score: float,
    primary_score: float = 0.0,
    fallback_score: float = 0.0,
) -> tuple:
    source_tags = []

    for i, (payload, _score) in enumerate(payloads_with_scores, start=1):
        if "chunk_text" in payload:
            title      = payload.get("title", "")
            page       = payload.get("page", "")
            heading    = payload.get("heading_path", "")
            chunk_text = payload.get("chunk_text", "")
            name = f"{title} | p.{page} | {heading}"
            source_tags.append(
                f'<source id="{i}" name="{name}">\n'
                f"  {chunk_text}\n"
                f"</source>"
            )

        else:
            # Summary payload
            title        = payload.get("title", "")
            summary_text = payload.get("summary_text", "")

            source_tags.append(
                f'<source id="{i}" name="{title}">\n'
                f"  {summary_text}\n"
                f"</source>"
            )

    context_str = "\n\n".join(source_tags)

    if "direct" in retrieval_path:
        retrieval_status = (
            f"NORMAL: Retrieved from {retrieval_path}. "
            f"Confidence score: {formula_score:.2f}. "
            f"Context is reliable."
        )
    elif "fallback" in retrieval_path:
        retrieval_status = (
            f"FALLBACK: {retrieval_path}. "
            f"Primary score was below threshold. "
            f"Fallback score: {formula_score:.2f}. "
            f"Answer quality may vary."
        )
    else:
        retrieval_status = (
            f"DEGRADED: Both stores returned low confidence. "
            f"Primary store score: {primary_score:.2f}. "
            f"Fallback store score: {fallback_score:.2f}. "
            f"Confidence threshold: {CONFIDENCE_THRESHOLD}.\n\n"
            f"YOU MUST DO THE FOLLOWING:\n"
            f"1. Start response with: ⚠ DEGRADED RETRIEVAL\n"
            f"2. State both stores failed with these scores.\n"
            f"3. If answering from context: note accuracy limited.\n"
            f"4. If answering from own knowledge: state explicitly:\n"
            f"   ⚠ This answer is based on the model's general "
            f"knowledge and is NOT sourced from the research "
            f"corpus. It has not been verified against the "
            f"indexed papers.\n"
            f"5. End with the validation block."
        )

    return context_str, retrieval_status


# ─────────────────────────────────────────────────────────
# FUNCTION: _dedupe_results
# ─────────────────────────────────────────────────────────
def _dedupe_results(results: list) -> list:
    """
    Remove repeated payloads by stable source/text fingerprint.
    Accepts list of (payload, score) tuples.
    Preserves input order (formula_score order after reranking).
    """
    deduped = []
    seen    = set()

    for payload, score in results:
        key = (
            payload.get("chunk_id")
            or payload.get("doc_id")
            or payload.get("summary_id")
            or "",
            payload.get("title", ""),
            payload.get("page", ""),
            payload.get("heading_path", ""),
            (payload.get("chunk_text", "") or payload.get("summary_text", ""))[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((payload, score))

    return deduped


# ─────────────────────────────────────────────────────────
# FUNCTION: _max_context_items
# 
# ─────────────────────────────────────────────────────────
def _max_context_items(query_type: str, route: str) -> int:
    """
    How many chunks to send to the LLM after reranking and dedup.

    SUMMARY STORE: always 3 summaries.
    CHUNK STORE:   always 5 chunks.
    """
    if route == "summary_store" or route == SUMMARY_COLLECTION:
        return 3

    return 5   # chunk_store


# ─────────────────────────────────────────────────────────
# FUNCTION: run_retrieval
# ─────────────────────────────────────────────────────────



def run_retrieval(
    query: str,
    route: str,
    metadata_filters: dict,
    query_type: str,
) -> dict:
    """
    Full retrieval pipeline. Zero LLM calls.
    Returns partial AgentState dict.

    Pipeline:
            1. Determine primary collection + top_k values
            2. Semantic search primary collection
            3. Score ALL primary results + rerank by formula score
            4. Threshold check on best formula score (top-1 after rerank)
            5. If below threshold: search secondary, score all, rerank
            6. Deduplicate
            7. Trim to max_context_items (type-specific limits)
            8. Build XML context
            9. Return state dict
    """

    # ── Step 1: Normalize filters ─────────────────────────
    filters = dict(metadata_filters)
    
    # ── Step 2: Determine collections + top_k ────────────
    if route == "summary_store":
        primary_collection = SUMMARY_COLLECTION
        other_collection   = CHUNK_COLLECTION
        primary_top_k      = 5   # all summary types uniform now
        other_top_k        = 3
    else:
        primary_collection = CHUNK_COLLECTION
        other_collection   = SUMMARY_COLLECTION
        primary_top_k      = 5
        other_top_k        = 3

    # ── Step 3: Semantic search — primary ─────────────────
    primary_results = semantic_search(
        query, primary_collection, filters, primary_top_k
    )

    # ── Step 4: Score ALL + rerank ───────────────
    # Returns list of (payload, cosine, formula_score) sorted by formula desc
    primary_scored = _score_and_rerank(
        primary_results,
        query,
        query_type,
    )

    # ── Step 5: Threshold on best formula score [CHG-4] ──
    # top-1 after rerank = chunk with BEST formula score
    primary_score = primary_scored[0][2] if primary_scored else 0.0
    fallback_score = 0.0

    if primary_score >= CONFIDENCE_THRESHOLD:
        # ── DIRECT PATH ───────────────────────────────────
        # Convert back to (payload, score) for downstream functions
        final_results  = [(p, c) for p, c, _ in primary_scored]
        final_score    = primary_score
        retrieval_path = f"direct ({primary_collection})"
        is_degraded    = False

    else:
        # ── FALLBACK PATH — search secondary ──────────────
        fallback_results = semantic_search(
            query, other_collection, filters, other_top_k
        )

        # Score ALL fallback chunks + rerank
        fallback_scored = _score_and_rerank(
            fallback_results, query, query_type,
        )
        fallback_score = fallback_scored[0][2] if fallback_scored else 0.0

        if fallback_score >= CONFIDENCE_THRESHOLD:
            # ── FALLBACK RECOVERED ─────────────────────────
            final_results  = [(p, c) for p, c, _ in fallback_scored]
            final_score    = fallback_score
            retrieval_path = (
                f"fallback ({primary_collection} to {other_collection})"
            )
            is_degraded = False

        else:
            # ── DEGRADED — use best available ─────────────
            # Prefer fallback if it returned anything, else primary
            if fallback_scored:
                final_results = [(p, c) for p, c, _ in fallback_scored]
                final_score   = fallback_score
            else:
                final_results = [(p, c) for p, c, _ in primary_scored]
                final_score   = primary_score

            retrieval_path = "degraded"
            is_degraded    = True

    # ── Step 6: Deduplicate ───────────────────────────────
    # Preserves formula_score order from reranking
    deduped_results = _dedupe_results(final_results)

    # ── Step 7: Trim to context limit ────────────────────
    max_items       = _max_context_items(query_type, route)
    context_results = deduped_results[:max_items]

    # ── Step 8: Build context ─────────────────────────
    context_str, retrieval_status = build_context(
        context_results,
        retrieval_path,
        final_score,
        primary_score  = primary_score,
        fallback_score = fallback_score,
    )
    context_tokens = exact_tokens(context_str)

    logger.info(
        "[retrieval] route=%s qtype=%s path=%s "
        "primary=%.4f fallback=%.4f final=%.4f "
        "retrieved=%d deduped=%d sent=%d context_tokens=%d",
        route, query_type, retrieval_path,
        primary_score, fallback_score, final_score,
        len(primary_results), len(deduped_results), len(context_results),
        context_tokens,
    )

    # ── Step 9: Return state dict ─────────────────────────
    return {
        "retrieved_payloads": [p for p, _ in context_results],
        "retrieval_scores":   [round(float(s), 4) for _, s in context_results],
        "source_ids": [
            p.get("chunk_id")
            or p.get("doc_id")
            or p.get("filename")
            or ""
            for p, _ in context_results
        ],
        "formula_score":    final_score,
        "retrieval_path":   retrieval_path,
        "context":          context_str,
        "context_tokens":   context_tokens,
        "retrieval_status": retrieval_status,
        "is_degraded":      is_degraded,
    }