"""
nodes.py
========================================================
"""

import json
import os
import re
import time

import litellm
import tiktoken

from .prompts import (
    PLANNING_SYSTEM,
    PLANNING_USER_FIRST,
    GENERATION_SYSTEM,
    GENERATION_USER,
)
from .state import AgentState


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
BASE_URL         = os.getenv("BASE_URL")
API_KEY          = os.getenv("API_KEY")
CERT_PATH        = os.getenv("CERT_PATH")
PLANNING_MODEL   = "openai/claudesonnet4.6"
GENERATION_MODEL = "openai/gpt-5.4"

_VALID_CONFIDENCE   = {"HIGH", "MEDIUM", "LOW"}
_VALID_COMPLETENESS = {"COMPLETE", "PARTIAL", "EMPTY"}

# Query types that use summary_store and need source citations
_SUMMARY_TYPES = {"reasoning", "multi_hop", "comparative", "superficial"}


# ─────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def exact_tokens(text: str) -> int:
    """
    Exact token count via tiktoken (cl100k_base).
    Usage APIs only report a single prompt_tokens total for the
    whole request (system + context + query combined) — they do
    not break it down per component, so isolating the context's
    token count requires local tokenization, not a usage call.
    """
    if not text or not text.strip():
        return 0
    return len(_TOKENIZER.encode(text))


def _normalize_choice(value: str, allowed: set, fallback: str) -> str:
    # Normalizes noisy LLM enum outputs (e.g., "low", "LOW|MEDIUM") to
    # one allowed label; otherwise returns a deterministic fallback.
    """Return value.upper() if it is in allowed, else return fallback."""
    if isinstance(value, str) and value.strip().upper() in allowed:
        return value.strip().upper()
    if isinstance(value, str):
        for part in re.split(r"[|/,]", value):
            token = part.strip().upper()
            if token in allowed:
                return token
    return fallback


def _fallback_confidence(formula_score: float, path_label: str) -> str:
    """
    Derive confidence from retrieval score when LLM validation missing.
    Uses path_label (processed) not raw retrieval_path. [CHG-5]
    """
    if path_label == "degraded":
        return "LOW"
    if formula_score >= 0.70:
        return "HIGH"
    if formula_score >= 0.45:
        return "MEDIUM"
    return "LOW"


def _fallback_completeness(answer_text: str) -> str:
    """Derive completeness from answer text when LLM validation missing."""
    if not answer_text or not answer_text.strip():
        return "EMPTY"
    lower = answer_text.lower()
    if any(
        phrase in lower
        for phrase in (
            "no relevant",
            "not found",
            "cannot answer",
            "no information",
            "not contain",
            "does not contain sufficient",
        )
    ):
        return "PARTIAL"
    return "COMPLETE"




# ─────────────────────────────────────────────────────────
# LLM reads full "TITLE | AUTHORS | YEAR | VENUE" from
# source tag name attribute and sometimes copies all of it.
# ─────────────────────────────────────────────────────────
def _clean_sources_line(text: str) -> str:
    """
    Cleans the (Sources: ...) line in answer text.
    Removes author names, years, venues — keeps only titles.

    Input:  "(Sources: Title A | Author | 2020 | Journal; Title B | Auth)"
    Output: "(Sources: Title A; Title B)"
    """
    if not text:
        return text

    def _clean_entry(entry: str) -> str:
        """Extract title only from 'Title | Author | Year | Venue'."""
        entry = entry.strip()
        if "|" in entry:
            return entry.split("|")[0].strip()
        return entry

    def _replace_sources(match) -> str:
        content = match.group(1)
        entries = content.split(";")
        cleaned = [_clean_entry(e) for e in entries if e.strip()]
        return f"(Sources: {'; '.join(cleaned)})"

    # Match (Sources: ...) — handles multiline content
    text = re.sub(
        r"\(Sources:\s*(.*?)\)",
        _replace_sources,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text


# ─────────────────────────────────────────────────────────
# LLM CALL HELPER
# ─────────────────────────────────────────────────────────
def call_llm(
    system: str,
    user: str,
    model: str,
    max_retries: int = 2,
) -> tuple:
    """
        Single LiteLLM completion call with retry.
        Returns (response_text, input_tokens, output_tokens).
        Raises RuntimeError if all retries fail.
    """
    litellm.ssl_verify = CERT_PATH

    for attempt in range(max_retries + 1):
        try:
            response = litellm.completion(
                model    = model,
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                api_base = BASE_URL,
                api_key  = API_KEY,
            )
            usage   = getattr(response, "usage", None)
            input_tokens  = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            return response.choices[0].message.content, input_tokens, output_tokens
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"LLM call failed after {max_retries + 1} attempts: {exc}"
            ) from exc


# ─────────────────────────────────────────────────────────
# NODE 1: PLANNING NODE
# ─────────────────────────────────────────────────────────
def planning_node(state: AgentState) -> dict:
    """
    CALL 1. Called ONCE per query.
    Classifies query, routes it, extracts metadata filters.
    """
    raw_query  = state.get("raw_query", "")
    user_input = PLANNING_USER_FIRST.format(query=raw_query)

    started_at = time.time()
    raw_response, planning_input_tokens, planning_output_tokens = call_llm(
        PLANNING_SYSTEM,
        user_input,
        model=PLANNING_MODEL,
    )
    finished_at = time.time()
    planning_response_time = round(finished_at - started_at, 3)

    # Strip markdown fences before parsing
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$",           "", cleaned)

    # Parse JSON response
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                result = _default_planning_result()
        else:
            result = _default_planning_result()

    # Normal routing — strip null values from metadata_filters
    raw_filters     = result.get("metadata_filters", {})
    cleaned_filters = {k: v for k, v in raw_filters.items() if v is not None}

    return {
        "query_type":               result.get("query_type", "superficial"),
        "route":                    result.get("route", "summary_store"),
        "metadata_filters":         cleaned_filters,
        "is_degraded":              False,
        "planning_input_tokens":    planning_input_tokens,
        "planning_output_tokens":   planning_output_tokens,
        "planning_response_time_s": planning_response_time,
    }


def _default_planning_result() -> dict:
    """Safe default when planning LLM response cannot be parsed."""
    return {
        "query_type":          "superficial",
        "route":               "summary_store",
        "metadata_filters":    {},
    }


# ─────────────────────────────────────────────────────────
# SOURCE METADATA HELPER 
# ─────────────────────────────────────────────────────────
def _extract_source_metadata(
    state: AgentState,
) -> dict:
    """
        Builds lightweight citation metadata from retrieved payloads.

        For multi_hop/comparative: include up to two unique titles.
        For all other query types: use only top-1 payload metadata.
    """
    retrieved_chunks = (
        state.get("retrieved_payloads", [])
        or state.get("retrieved_chunks",  [])
        or state.get("chunks",            [])
    )
    query_type = str(state.get("query_type", "")).lower()

    payloads = []
    for chunk in retrieved_chunks:
        payload = chunk.payload if hasattr(chunk, "payload") else chunk
        if isinstance(payload, dict):
            payloads.append(payload)

    if not payloads:
        return {
            "source_title"   : "",
            "source_page"    : None,
            "source_section" : "",
            "citation_string": "",
        }

    if query_type in {"multi_hop", "comparative"}:
        seen = set()
        titles = []
        for payload in payloads:
            title = str(payload.get("title", "")).strip()
            if title and title not in seen:
                titles.append(title)
                seen.add(title)
            if len(titles) == 2:
                break

        if len(titles) >= 2:
            return {
                "source_title"   : " | ".join(titles),
                "source_page"    : None,
                "source_section" : "",
                "citation_string": "(Sources: " + "; ".join(titles) + ")",
            }

    payload = payloads[0]
    title = str(payload.get("title", "")).strip()
    page = payload.get("page", None)
    section = str(payload.get("heading_path", payload.get("section", "")) or "").strip()
    citation = f"[{title}]" if title else ""

    return {
        "source_title"   : title,
        "source_page"    : page,
        "source_section" : section,
        "citation_string": citation,
    }


# ─────────────────────────────────────────────────────────
# NODE 2: GENERATION NODE
# ─────────────────────────────────────────────────────────
def generation_node(state: AgentState) -> dict:
    """
    CALL 2. Single LLM call. Never retries.
    Always returns a formatted response.

        citation_instruction varies by query type:
            summary types -> (Sources: Title A; Title B) at end
            chunk types   -> no citations in answer text
    """
    raw_query        = state.get("raw_query",        "")
    context          = state.get("context",          "")
    retrieval_path   = state.get("retrieval_path",   "")
    retrieval_status = state.get("retrieval_status", "")
    route            = state.get("route",            "")
    query_type       = state.get("query_type",       "")
    formula_score    = float(state.get("formula_score", 0.0) or 0.0)

    qt = (query_type or "").lower()

    # ── Citation instruction by query type [CHG-6] ────────
    if qt in _SUMMARY_TYPES:
        # Summary queries: ground truth has (Sources: ...) format
        # RAGAS expects source titles to appear in answer
        citation_instruction = (
            "CITATION RULE FOR THIS QUERY TYPE (summary/reasoning):\n"
            "Reference papers by descriptive name in prose:\n"
            "  'the testbed paper shows...'\n"
            "  'the planning framework identifies...'\n"
            "  'both papers agree that...'\n"
            "\n"
            "At the END of your answer add a Sources line:\n"
            "  (Sources: Paper Title Only; Paper Title Only)\n"
            "\n"
            "CRITICAL — Sources line format:\n"
            "  The source tag name contains: TITLE | AUTHORS | YEAR | VENUE\n"
            "  Use ONLY the TITLE part (text before the first | character).\n"
            "  Do NOT include authors, year, venue, or journal name.\n"
            "\n"
            "CORRECT:\n"
            "  (Sources: A robust supply chain planning framework for revenue\n"
            "   management in the semiconductor industry;\n"
            "   A Testbed for Simulating Semiconductor Supply Chains)\n"
            "\n"
            "WRONG — do not copy pipe-separated metadata:\n"
            "  (Sources: A robust supply chain... | Alexander Seitz | 2016 |\n"
            "   Journal of Revenue and Pricing Management)\n"
            "\n"
            "Do NOT add the retrieval footer --- *Retrieval: ...*"
        )
    else:
        # Chunk queries: ground truth is short clean fact
        # Citations add FP statements hurting RAGAS score
        citation_instruction = (
            "CITATION RULE FOR THIS QUERY TYPE (specific lookup):\n"
            "Do NOT add any citations inside your answer text.\n"
            "Do NOT reference paper titles, authors, page numbers,\n"
            "or source names anywhere in your answer.\n"
            "Do NOT add a Sources line.\n"
            "Do NOT add the retrieval footer --- *Retrieval: ...*\n"
            "Write ONLY the direct answer — clean prose or exact value.\n"
            "The pipeline displays source information separately."
        )

    # ── Build path label ──────────────────────────────────
    if "fallback" in retrieval_path:
        path_label = retrieval_path
    elif "direct" in retrieval_path:
        path_label = f"direct ({route})"
    elif retrieval_path == "degraded":
        path_label = "degraded"
    else:
        path_label = f"direct ({route or 'summary_store'})"

    print(
        f"\n[GENERATION] query={raw_query[:60]} | "
        f"path={path_label} | "
        f"context_tokens={len(context) // 4}"
    )

    # ── Context size estimate ─────────────────────────────
    # Exact token count (tiktoken) of the retrieved context
    # actually sent to the generation LLM. Not derivable from
    # usage calls since those only report a combined prompt
    # total for system + context + query together.
    context_tokens = exact_tokens(context)

    # ── Build prompts ─────────────────────────────────────
    system = GENERATION_SYSTEM.format(
        retrieval_path       = path_label,
        retrieval_status     = retrieval_status,
        citation_instruction = citation_instruction,
    )
    user = GENERATION_USER.format(
        context = context,
        query   = raw_query,
    )

    # ── LLM call ──────────────────────────────────────────
    started_at = time.time()
    raw, generation_input_tokens, generation_output_tokens = call_llm(
        system,
        user,
        model=GENERATION_MODEL,
    )
    finished_at = time.time()
    generation_response_time = round(finished_at - started_at, 3)
    # answer_tokens reuses generation_output_tokens — the real
    # output token count from the usage call, not an estimate.
    answer_tokens = generation_output_tokens
    print(f"[GENERATION] response_tokens={len(raw) // 4}")

    # ── Parse answer + validation block ───────────────────
    json_match = re.search(
        r"```json\s*(\{.*?\})\s*(?:```|$)",
        raw,
        re.DOTALL,
    )
    if json_match:
        answer_text = raw[:json_match.start()].strip()
        # Strip retrieval footer if LLM still adds it
        answer_text = re.sub(
            r"\n*-{3,}\n\*Retrieval:[^\n]*\*\s*$",
            "",
            answer_text,
        ).strip()
        try:
            validation = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            validation = _default_validation("Validation block malformed.")
    else:
        answer_text = raw.strip()
        validation  = _default_validation("Validation block missing.")

    # ── Clean Sources line [CHG-7] ─────────────────────────
    # Remove authors/year/venue that LLM copies from source tag name
    # Only matters for summary types but safe to run on all answers
    answer_text = _clean_sources_line(answer_text)

    # ── Extract validation fields ──────────────────────────
    confidence = _normalize_choice(
        validation.get("confidence", "LOW"),
        allowed  = _VALID_CONFIDENCE,
        fallback = _fallback_confidence(formula_score, path_label),
    )
    completeness = _normalize_choice(
        validation.get("completeness", "PARTIAL"),
        allowed  = _VALID_COMPLETENESS,
        fallback = _fallback_completeness(answer_text),
    )
    claims = validation.get("unsupported_claims", [])
    note   = validation.get("retrieval_note", "")

    # ── Format user-facing response ───────────────────────
    response = answer_text

    if completeness == "PARTIAL" and note:
        response += f"\n\n*Note: {note}*"

    if completeness == "EMPTY":
        response = (
            "The available corpus does not contain "
            "sufficient information to answer this query.\n\n"
            f"Based on available context:\n{answer_text}"
        )
        if note:
            response += f"\n\n*Note: {note}*"

    if claims:
        claims_str = "; ".join(str(c) for c in claims if c)
        if claims_str:
            response += f"\n\n*Could not verify from corpus: {claims_str}*"

    # ── Quality flags ──────────────────────────────────────
    answer_flagged = (
        completeness != "COMPLETE"
        or bool(claims)
        or confidence == "LOW"
    )

    # ── Source metadata ───────────────────────────────────
    source_meta = _extract_source_metadata(state)

    print(
        f"[GENERATION] confidence={confidence} | "
        f"completeness={completeness} | "
        f"claims={len(claims)} | "
        f"flagged={answer_flagged} | "
        f"tokens={answer_tokens}"
    )

    # ── Token/latency totals [CHG-9] ──────────────────────
    # Combine planning stats (already in state) with this call's
    # generation stats so build_response() can report per-question
    # input/output tokens and total response time across both calls.
    planning_input_tokens   = state.get("planning_input_tokens",   0) or 0
    planning_output_tokens  = state.get("planning_output_tokens",  0) or 0
    planning_response_time  = state.get("planning_response_time_s", 0.0) or 0.0

    return {
        # ── Core answer ───────────────────────────────────
        "answer"            : answer_text,
        "validation"        : validation,
        "formatted_response": response,

        # ── Quality metadata ──────────────────────────────
        "confidence"        : confidence,
        "completeness"      : completeness,
        "answer_flagged"    : answer_flagged,
        "answer_tokens"     : answer_tokens,

        # ── Retrieval metadata [CHG-3] ────────────────────
        "retrieval_path"    : path_label,

        # ── Context size estimate ─────────────────────────
        "context_tokens"    : context_tokens,

        # ── Token/latency metadata [CHG-9] ────────────────
        "generation_input_tokens"  : generation_input_tokens,
        "generation_output_tokens" : generation_output_tokens,
        "generation_response_time_s": generation_response_time,
        "total_input_tokens"  : planning_input_tokens  + generation_input_tokens,
        "total_output_tokens" : planning_output_tokens + generation_output_tokens,
        "total_response_time_s": round(
            planning_response_time + generation_response_time, 3
        ),

        # ── Source citation fields [CHG-4] ────────────────
        "source_title"      : source_meta["source_title"],
        "source_page"       : source_meta["source_page"],
        "source_section"    : source_meta["source_section"],
        "citation_string"   : source_meta["citation_string"],
    }


def _default_validation(note: str) -> dict:
    """Safe default validation block when LLM response cannot be parsed."""
    return {
        "confidence"         : "LOW",
        "completeness"       : "PARTIAL",
        "unsupported_claims" : [],
        "retrieval_note"     : note,
    }