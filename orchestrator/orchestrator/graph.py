"""Stateful orchestrator graph.

classify -> (Send fan-out) wiki_agent / papers_agent -> synthesize
         -> ask_clarification
         -> reject

Interactive test (from orchestrator/):
    uv run python -m orchestrator.graph [--scope-mode topics|full]
"""
import argparse
import operator
import sys
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite, Send

from orchestrator.adapters import ADAPTERS, AgentResult, close as close_adapters
from orchestrator.config import make_llm
from orchestrator.router import classify_query

AnswerType = Literal["grounded", "partial", "general_knowledge", "clarify", "out_of_scope"]


class OrchestratorState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages] #persisted through checkpoint, appended each turn
    decision: dict                                   # RouteDecision, replaced each turn, no reducer
    results: Annotated[list[AgentResult], operator.add] # .add combines results from multiple agents, cleared each turn through Overwrite in classify
    answer_type: AnswerType


class AgentTask(TypedDict):
    query: str
    rewritten: bool


# ---------------------------------------------------------------------------
# Prompts for synthesis and fallback 
#  ---------------------------------------------------------------------------

COMBINE_PROMPT = """You combine answers from two knowledge sources into one answer to the user's question.
- WIKI is Infineon internal supply-chain training material (how Infineon works in practice).
- PAPERS are academic research papers (what research proposed and found).
Rules:
- Use only the content of the source answers below. Do not add facts.
- Say which statements come from WIKI and which from PAPERS.
- If the sources disagree, state the disagreement instead of resolving it.
- Plain prose, concise. Do not add a source list; it is appended separately."""

FALLBACK_PROMPT = """The internal knowledge sources contained no usable evidence for this question.
Answer briefly from general knowledge.
- Interpret the question in the context of semiconductor supply chain and manufacturing.
- State uncertainty where it exists.
- Do not invent Infineon-specific facts (processes, systems, roles, numbers). If the question
  asks for them, say that they were not found in the knowledge sources.
- Plain prose, concise. Do not add a disclaimer; one is added separately."""


def _source_line(r: AgentResult) -> str:
    line = f"Source: {r['source'].upper()} · evidence: {r['sufficiency']}"
    if r["citations"]:
        line += f"\n  Citations: {'; '.join(r['citations'])}"
    if r["rewritten"]:
        line += f"\n  Question sent: {r['query']}"
    return line


def _fallback_label(results: list[AgentResult]) -> str:
    sources = " or ".join(r["source"].upper() for r in results)
    return f"Not found in {sources}. General knowledge answer, not from the knowledge bases:"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def make_classify(scope_mode: str):
    def classify(state: OrchestratorState) -> dict:
        messages = state["messages"]
        decision = classify_query(str(messages[-1].content), messages[:-1], scope_mode)
        return {
            "decision": decision.model_dump(),
            "results": Overwrite([]),   # clear last turn's agent results (§4.3)
        }
    return classify


def route_after_classify(state: OrchestratorState):
    decision = state["decision"]
    if decision["route"] == "clarify":
        return "ask_clarification"
    if decision["route"] == "out_of_scope":
        return "reject"
    question = str(state["messages"][-1].content)
    return [
        Send(f"{t['source']}_agent", AgentTask(query=t["query"] or question, rewritten=bool(t["query"])))
        for t in decision["targets"]
    ]


def wiki_agent(task: AgentTask) -> dict:
    return {"results": [ADAPTERS["wiki"](task["query"], task["rewritten"])]}


def papers_agent(task: AgentTask) -> dict:
    return {"results": [ADAPTERS["papers"](task["query"], task["rewritten"])]}


def _answer_message(body: str, footer: str) -> AIMessage:
    # The user sees body + source line; the router's history reads only the body.
    return AIMessage(f"{body}\n\n{footer}", additional_kwargs={"history_text": body})


def synthesize(state: OrchestratorState) -> dict:
    results = sorted(state["results"], key=lambda r: r["source"])   # stable order: papers, wiki
    usable = [r for r in results if r["answer"]]   # empty answer = agent found nothing
    question = str(state["messages"][-1].content)

    if not usable:
        standalone = next((r["query"] for r in results if r["rewritten"]), question)
        reply = make_llm().invoke([SystemMessage(FALLBACK_PROMPT), HumanMessage(standalone)])
        text = f"{_fallback_label(results)}\n\n{reply.content}"
        return {"messages": [AIMessage(text)], "answer_type": "general_knowledge"}

    answer_type: AnswerType = "grounded" if any(r["sufficiency"] == "sufficient" for r in usable) else "partial"
    footer = "\n".join(_source_line(r) for r in usable)

    if len(usable) == 1:
        return {"messages": [_answer_message(usable[0]["answer"], footer)], "answer_type": answer_type}

    blocks = "\n\n".join(
        f"[{r['source'].upper()}] (evidence: {r['sufficiency']}; question sent: {r['query']})\n{r['answer']}"
        for r in usable
    )
    reply = make_llm().invoke([
        SystemMessage(COMBINE_PROMPT),
        HumanMessage(f"User question:\n{question}\n\nSource answers:\n{blocks}"),
    ])
    return {"messages": [_answer_message(str(reply.content), footer)], "answer_type": answer_type}


def ask_clarification(state: OrchestratorState) -> dict:
    return {
        "messages": [AIMessage(state["decision"]["clarifying_question"])],
        "answer_type": "clarify",
    }


def reject(state: OrchestratorState) -> dict:
    text = (
        "I can only answer questions covered by WIKI (Infineon supply-chain training material) "
        f"or PAPERS (academic research papers). {state['decision']['rejection_reason']}"
    )
    return {"messages": [AIMessage(text)], "answer_type": "out_of_scope"}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_orchestrator(scope_mode: str = "topics", checkpointer=None):
    builder = StateGraph(OrchestratorState)
    builder.add_node("classify", make_classify(scope_mode))
    builder.add_node("wiki_agent", wiki_agent)
    builder.add_node("papers_agent", papers_agent)
    builder.add_node("synthesize", synthesize)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("reject", reject)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", route_after_classify,
        ["wiki_agent", "papers_agent", "ask_clarification", "reject"],
    )
    builder.add_edge("wiki_agent", "synthesize")
    builder.add_edge("papers_agent", "synthesize")
    builder.add_edge("synthesize", END)
    builder.add_edge("ask_clarification", END)
    builder.add_edge("reject", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


# ---------------------------------------------------------------------------
# Interactive loop for manual testing
# ---------------------------------------------------------------------------

def _print_turn(state: dict) -> None:
    d = state["decision"]
    targets = ", ".join(
        f"{t['source']}" + (f" (rewritten: {t['query']})" if t["query"] else "") for t in d["targets"]
    )
    print(f"\n[route] {d['route']}" + (f" -> {targets}" if targets else ""))
    print(f"[reasoning] {d['reasoning']}")
    for r in state.get("results", []):
        print(f"[{r['source']}] sufficiency={r['sufficiency']} score={r['agent_score']}"
              + (f" error={r['error']}" if r["error"] else ""))
    print(f"[answer_type] {state['answer_type']}\n")
    print(state["messages"][-1].content)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chat with the orchestrator.")
    parser.add_argument("--scope-mode", choices=["topics", "full"], default="topics")
    args = parser.parse_args()

    graph = build_orchestrator(args.scope_mode)
    thread_id = f"conv-{uuid.uuid4().hex[:8]}"
    print(f"Thread {thread_id}. Type /new for a new conversation, /exit to quit.")
    try:
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question == "/exit":
                break
            if question == "/new":
                thread_id = f"conv-{uuid.uuid4().hex[:8]}"
                print(f"New thread {thread_id}.")
                continue
            final_state = graph.invoke(
                {"messages": [HumanMessage(question)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            _print_turn(final_state)
    finally:
        close_adapters()   # close Qdrant while Python is still running (avoids the shutdown traceback)
