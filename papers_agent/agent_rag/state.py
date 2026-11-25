"""
state.py
========================================================
AgentState TypedDict definition.
All nodes return partial dicts. LangGraph merges them.
Uses total=False so partial updates work without KeyError.
"""

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    raw_query: str
    query_type: str
    route: str
    metadata_filters: dict
    clarifying_question: str
    is_degraded: bool
    retrieved_payloads: list
    formula_score: float
    retrieval_path: str
    context: str
    retrieval_status: str
    answer: str
    validation: dict
    formatted_response: str
    confidence: str
    completeness: str
    answer_flagged: bool
    answer_tokens : int
    context_tokens: int
    retrieval_scores: list
    source_ids: list
    source_title: str
    source_page: object
    source_section: str
    citation_string: str

    # ── Token/latency metadata [CHG-9] ────────────────────
    planning_input_tokens: int
    planning_output_tokens: int
    planning_response_time_s: float
    generation_input_tokens: int
    generation_output_tokens: int
    generation_response_time_s: float
    total_input_tokens: int
    total_output_tokens: int
    total_response_time_s: float
