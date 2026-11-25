"""LLM query router: one structured-output call per question.
"""
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestrator.config import make_llm

SCOPE_DIR = Path(__file__).parent   # scope_topics.txt / scope_full.txt, made by scope.py

Source = Literal["wiki", "papers"]

HISTORY_TURNS = 3          # one turn = one user message + one assistant reply
# Earlier assistant replies are cut to this length. Measured on saved agent runs: the longest
# single-source answer is 1,497 characters (wiki max 1,267, papers max 1,497), so 1,500 keeps all of them.
HISTORY_CHAR_LIMIT = 1500 # applied to each assistant message, not the whole history


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class Target(BaseModel):
    source: Source
    query: str | None = Field(
        default=None,
        description="Leave empty to send the user's question unchanged. "
                    "Fill only when a rewrite is needed (rules Q1-Q4).",
    )


class RouteDecision(BaseModel):
    reasoning: str = Field(description="1-3 sentences: what is asked and why this route and these sources.")
    route: Literal["retrieve", "clarify", "out_of_scope"]
    targets: list[Target] = Field(default_factory=list, description="Only for route=retrieve: one or two sources.")
    clarifying_question: str | None = Field(default=None, description="Only for route=clarify.")
    rejection_reason: str | None = Field(default=None, description="Only for route=out_of_scope.")


# ---------------------------------------------------------------------------
# Prompt 
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the query router of a question-answering system with two knowledge sources, WIKI and PAPERS.
You do not answer questions. You decide where each question should be sent.

<scope>
{scope}
</scope>

Choose exactly one route:
- retrieve: the question is in domain. Choose one or both sources.
- clarify: the question has no clear meaning, even after reading the conversation. Ask one short question.
- out_of_scope: the question is outside the domain boundary, or needs live, future or personal information.

Choosing the route
R1 Choose retrieve whenever the question is in domain, even if no listed topic mentions it.
R2 Choose clarify only if no reasonable reading of the question exists. If one reading is clearly
   most likely, choose retrieve with that reading.
R3 Choose out_of_scope only if the question is clearly outside the domain boundary. If unsure, choose retrieve.
R4 If one part of the question is in domain and another is not, retrieve the in-domain part and
   name the dropped part in your reasoning.
R5 Questions about one specific record (a named customer's open orders, an order's delivery status,
   today's stock of a product) need live data that neither source holds: choose out_of_scope,
   even if the topic is in domain. Questions about how such data is handled in general are retrieve.

Choosing sources
S1 Decide by the kind of answer needed, using each source's "Answers questions like" and
   "Does NOT contain" lines: how something is done at Infineon in practice -> WIKI;
   what research proposed, tested or found -> PAPERS.
S2 A term that appears in a topic list is a hint, not proof. Some terms appear in both lists.
S3 Choose both sources when the question asks for both practice and research, when both could
   answer it, or when you are unsure. Otherwise choose one source.
S4 Do not prefer a source because its list is longer or because it is shown first.

Rewriting the question (optional)
Q1 By default leave `query` empty: the user's question is sent unchanged.
Q2 Write a `query` only if (a) the question cannot be understood without the conversation
   (for example "it", "this", "that paper", or a reply to your clarifying question), or
   (b) only part of the question belongs to this source.
Q3 When you write a query, keep the user's meaning and terms, including acronyms.
   Do not add facts, assumptions or extra scope.
Q4 If both sources need the same rewritten question, use the same text for both.

Using the conversation
H1 Use the conversation only to understand what the current question refers to. Route the current question.
H2 If the current message answers your earlier clarifying question, combine it with the original question.
H3 If the current question starts a new topic, ignore the earlier conversation.

Output
O1 Write reasoning first: 1-3 sentences on what is asked and why this route and these sources.
O2 Fill only the fields of the chosen route."""


def _format_history(history: list[BaseMessage]) -> str:
    recent = history[-2 * HISTORY_TURNS:]
    if not recent:
        return "Conversation so far: (none)"
    lines = ["Conversation so far (oldest first):"]
    for msg in recent:
        text = str(msg.content).strip()
        if isinstance(msg, AIMessage):
            # Answer text only, without the source line (set by graph.synthesize).
            text = str(msg.additional_kwargs.get("history_text", text)).strip()
            if len(text) > HISTORY_CHAR_LIMIT:
                text = text[:HISTORY_CHAR_LIMIT] + " ..."
            lines.append(f"Assistant: {text}")
        else:
            lines.append(f"User: {text}")
    return "\n".join(lines)


def build_messages(question: str, history: list[BaseMessage], scope_mode: str = "topics") -> list[BaseMessage]:
    human = f"{_format_history(history)}\n\nCurrent question:\n{question}" #history+question
    # replace() instead of format(): the scope text may contain braces.
    scope = (SCOPE_DIR / f"scope_{scope_mode}.txt").read_text(encoding="utf-8")
    system = SYSTEM_PROMPT.replace("{scope}", scope)
    return [SystemMessage(system), HumanMessage(human)]


def classify_query(
    question: str,
    history: list[BaseMessage] | None = None,
    scope_mode: str = "topics",
) -> RouteDecision:
    """Route one question. Returns the router LLM's decision unchanged."""
    llm = make_llm().with_structured_output(RouteDecision)
    return llm.invoke(build_messages(question, history or [], scope_mode))
