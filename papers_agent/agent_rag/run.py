"""
run.py
========================================================
Entry point and CLI. No Gradio. Pure Python for testing
and debugging. Imports everything from other modules.
CLI usage:
    python -m agent_rag.run --query "What methods does Paper9 use?"
    python -m agent_rag.run --interactive
"""
import argparse
import os
import sys
from pathlib import Path

from .retrieval import set_embedder, LiteLLMEmbedder
from .graph     import build_graph
from .state     import AgentState


# ─────────────────────────────────────────────────────────
# PAYLOAD HELPER
# ─────────────────────────────────────────────────────────
def _payload_to_context_text(payload: dict) -> str:
    """Convert one retrieved payload into a concise text context for evaluation."""
    title   = payload.get("title", "")
    page    = payload.get("page", "")
    heading = payload.get("heading_path", "")
    body    = payload.get("chunk_text", "") or payload.get("summary_text", "")

    provenance = " | ".join(
        p for p in [
            str(title).strip(),
            f"p.{page}" if page not in (None, "") else "",
            str(heading).strip(),
        ]
        if p
    )
    if provenance and body:
        return f"[{provenance}]\n{body}"
    return body or ""


# ─────────────────────────────────────────────────────────
# RESPONSE BUILDER
# ─────────────────────────────────────────────────────────
def build_response(state: AgentState) -> dict:
    """Extracts final response from completed graph state."""
    payloads = state.get("retrieved_payloads", []) or []
    retrieved_contexts = [
        ctx for ctx in (_payload_to_context_text(p) for p in payloads) if ctx
    ]

    return {
        # ── Core answer ───────────────────────────────────
        "answer"             : state.get("formatted_response", ""),

        # ── Quality flags ─────────────────────────────────
        "quality_flags": {
            "confidence" : state.get("confidence",    "LOW"),
            "completeness": state.get("completeness", "PARTIAL"),
            "flagged"    : state.get("answer_flagged", False),
        },

        # ── Retrieval metadata ────────────────────────────
        "retrieval_path"     : state.get("retrieval_path",  ""),
        "formula_score"      : state.get("formula_score",   0.0),
        "query_type"         : state.get("query_type",       ""),
        "context_tokens"     : state.get("context_tokens",   0),

        "retrieved_contexts" : retrieved_contexts,
        "retrieved_context"  : retrieved_contexts,   # backward compat
        "retrieval_scores"   : state.get("retrieval_scores", []),
        "source_ids"         : state.get("source_ids",       []),

        # ── Source citation fields [CHG-1] ────────────────
        # Set by generation_node from retrieved_payloads.
        # Replaces inline citations that were embedded in answer text.
        # Used by print_response() and saved to pipeline_results.json.
        "source_title"       : state.get("source_title",    ""),
        "source_page"        : state.get("source_page",     None),
        "source_section"     : state.get("source_section",  ""),
        "citation_string"    : state.get("citation_string", ""),

        # ── Legacy keys kept for backward compat ──────────
        # Old scripts that read paper_id / paper_page still work.
        "paper_id"           : state.get("source_title",    ""),
        "paper_page"         : state.get("source_page",     ""),
    }


# ─────────────────────────────────────────────────────────
# RUN QUERY
# ─────────────────────────────────────────────────────────
def run_query(query: str, graph) -> dict:
    """Builds initial state and calls graph.invoke(). Returns build_response."""
    initial_state = {
        # ── Query inputs ──────────────────────────────────
        "raw_query"          : query,
        "query_type"         : "",
        "route"              : "",
        "metadata_filters"   : {},
        "is_degraded"        : False,

        # ── Retrieval outputs ─────────────────────────────
        "retrieved_payloads" : [],
        "formula_score"      : 0.0,
        "retrieval_path"     : "",
        "context"            : "",
        "context_tokens"     : 0,
        "retrieval_status"   : "",
        "retrieval_scores"   : [],
        "source_ids"         : [],

        # ── Generation outputs ────────────────────────────
        "answer"             : "",
        "validation"         : {},
        "formatted_response" : "",
        "confidence"         : "",
        "completeness"       : "",
        "answer_flagged"     : False,

        # ── Citation fields [CHG-2] ───────────────────────
        # Initialized here so KeyError never occurs if
        # generation_node doesn't set them due to an early failure path.
        "source_title"       : "",
        "source_page"        : None,
        "source_section"     : "",
        "citation_string"    : "",
    }
    final_state = graph.invoke(initial_state)
    return build_response(final_state)


# ─────────────────────────────────────────────────────────
# PRINT RESPONSE
# ─────────────────────────────────────────────────────────

def print_response(response: dict) -> None:
    """Prints structured response for debugging."""
    print("\n" + "═" * 60)

    print("ANSWER:")
    print(response["answer"])

    flags = response["quality_flags"]
    print(f"\n{'─' * 60}")
    print(f"  Query type     : {response['query_type']}")
    print(f"  Retrieval path : {response['retrieval_path']}")
    print(f"  Formula score  : {response['formula_score']:.4f}")
    print(f"  Confidence     : {flags['confidence']}")
    print(f"  Completeness   : {flags['completeness']}")
    print(f"  Flagged        : {flags['flagged']}")
    print(f"  Context tokens : {response.get('context_tokens', 0)}")

    # ── Source citation block ─────────────────────────────
    citation = response.get("citation_string", "")
    source   = response.get("source_title",    "")

    if citation:
        print(f"  Citation       : {citation}")

    if source:
        # Multi-source: split by " | " separator
        if " | " in source and response.get("query_type") in (
            "multi_hop", "comparative"
        ):
            titles = source.split(" | ")
            print(f"  Sources        :")
            for i, t in enumerate(titles, 1):
                print(f"    [{i}] {t}")         # ← full title, no truncation
        else:
            print(f"  Source         : {source}") # ← full title, no truncation

    # Page and section only for single-source queries
    if response.get("source_page") is not None and " | " not in source:
        print(f"  Page           : {response['source_page']}")
    if response.get("source_section") and " | " not in source:
        print(f"  Section        : {response['source_section']}") # ← no truncation

    print("═" * 60)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG agent for academic papers."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query string. Run single query and exit.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive mode (loop until quit/exit/q).",
    )
    args = parser.parse_args()

    # Check environment variables
    missing = [v for v in ("API_KEY", "CERT_PATH") if not os.getenv(v)]
    if missing:
        print(
            f"[ERROR] Missing environment variables: {', '.join(missing)}\n"
            "  Windows: $env:API_KEY='...' ; $env:CERT_PATH='...'\n"
            "  Linux:   export API_KEY=...   CERT_PATH=...",
            file=sys.stderr,
        )
        sys.exit(1)

    # Startup sequence
    print("Loading embedder...")
    embedder = LiteLLMEmbedder()
    set_embedder(embedder)
    print("Building graph...")
    graph = build_graph()
    print("Ready.\n")

    def _repl():
        """Shared REPL loop."""
        print("RAG Agent — type 'quit', 'exit', or 'q' to stop.\n")
        while True:
            try:
                query = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break
            if not query:
                continue
            response = run_query(query, graph)
            print_response(response)

    if args.query:
        response = run_query(args.query, graph)
        print_response(response)
    else:
        # Interactive mode (--interactive flag or no args)
        _repl()


if __name__ == "__main__":
    main()