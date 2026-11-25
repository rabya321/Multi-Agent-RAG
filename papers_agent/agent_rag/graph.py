"""
graph.py
========================================================
LangGraph wiring only. No LLM calls. No Qdrant calls.
No prompt strings. Import everything from other modules.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import planning_node, generation_node
from .retrieval import run_retrieval


def build_graph():
    """
    Builds and compiles the RAG agent graph.
    set_embedder() must be called before this function.
    Takes no arguments — embedder set via retrieval.set_embedder().
    """
    workflow = StateGraph(AgentState)

    # ── Retrieval node wrapper (defined inside build_graph) ──
    def retrieval_node(state: AgentState) -> dict:
        if state.get("is_degraded"):
            return {
                "context": "",
                "retrieval_status": (
                    "DEGRADED: Both stores failed. "
                    "Proceeding with empty context."
                ),
                "retrieved_payloads": [],
                "retrieval_scores": [],
                "source_ids": [],
                "formula_score": 0.0,
                "retrieval_path": "degraded",
                "is_degraded"       : True,
            }

        return run_retrieval(
            query=state["raw_query"],
            route=state.get("route", "summary_store"),
            metadata_filters=state.get("metadata_filters", {}),
            query_type=state.get("query_type", "superficial"),
        )

    # ── Register nodes ────────────────────────────────────
    workflow.add_node("planning", planning_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)

    # ── Entry point ───────────────────────────────────────
    workflow.set_entry_point("planning")

    # ── Fixed edge after planning ─────────────────────────
    workflow.add_edge("planning", "retrieval")

    # ── Fixed edge after retrieval ────────────────────────
    # No conditional routing after retrieval.
    # Degraded path handled by retrieval_status string
    # passed to generation node.
    workflow.add_edge("retrieval", "generation")

    # ── Fixed edge after generation ───────────────────────
    workflow.add_edge("generation", END)

    return workflow.compile()
