"""Agent adapters.

Each adapter calls its agent unchanged and normalises the output into one AgentResult.
Normalisation happens here, right after the agent call; the agents know nothing about it.
The agents are imported lazily, because importing them builds their models/clients.
"""
import contextlib
import io
from typing import Literal, TypedDict

from orchestrator.config import REPO_ROOT

WIKI_INDEX_PATH = str(REPO_ROOT / "wiki_agent" / "wiki" / "global_index.yaml")

Sufficiency = Literal["sufficient", "partial", "insufficient"]


class AgentResult(TypedDict):
    source: Literal["wiki", "papers"]
    query: str                     # exactly what the agent received
    rewritten: bool                # True if the router rewrote the question
    answer: str                    # "" when the agent found no answer -> fallback
    citations: list[str]
    sufficiency: Sufficiency       # the agent's own evidence label, shown to the user
    agent_score: float | None      # wiki: confidence_score 0-100; papers: formula_score 0-1
    error: str | None


def _error_result(source: str, query: str, rewritten: bool, error: str) -> AgentResult:
    return AgentResult(source=source, query=query, rewritten=rewritten, answer="", citations=[],
                       sufficiency="insufficient", agent_score=None, error=error)


def _no_answer(answer: str) -> bool:
    # Both agent prompts say: reply exactly "No answer found." when nothing relevant was found.
    # Only a line that is exactly that phrase counts, so a partial answer that mentions it
    # in a sentence is kept. The papers agent puts it on its own line after a preamble.
    lines = [line.strip().lower().rstrip(".") for line in answer.splitlines()]
    return not answer.strip() or "no answer found" in lines


# ---------------------------------------------------------------------------
# Wiki agent
# ---------------------------------------------------------------------------

def _wiki_sufficiency(score: int) -> Sufficiency:
    if score < 45:
        return "insufficient"
    # 70 = the "mostly complete" band in the wiki agent's own prompt (STEP 5).
    return "sufficient" if score >= 70 else "partial"


def ask_wiki(query: str, rewritten: bool = False) -> AgentResult:
    try:
        from retrieval_agent.search_agent import run_query
        raw = run_query(query, index_path=WIKI_INDEX_PATH)
    except Exception as e:
        raw = {"error": f"{type(e).__name__}: {e}"}
    if "error" in raw:   # the wiki agent also reports its own failures this way
        return _error_result("wiki", query, rewritten, str(raw["error"]))

    answer = raw.get("answer", "") or ""
    if _no_answer(answer):
        answer = ""
    score = int(raw.get("confidence_score", 0) or 0)
    return AgentResult(
        source="wiki", query=query, rewritten=rewritten, answer=answer,
        citations=list(raw.get("answered_modules") or []),
        sufficiency=_wiki_sufficiency(score) if answer else "insufficient",
        agent_score=score,
        error=None,
    )


# ---------------------------------------------------------------------------
# Papers agent
# ---------------------------------------------------------------------------

_papers_graph = None

# The papers agent's own completeness label, renamed to the shared labels.
_PAPERS_SUFFICIENCY: dict[str, Sufficiency] = {
    "COMPLETE": "sufficient",
    "PARTIAL": "partial",
    "EMPTY": "insufficient",
}


def _get_papers_graph():
    # The papers graph (planning -> retrieval -> generation) is built on the first question
    # and reused for all later questions, so the wiring is not redone for every query.
    # build_graph() itself does not open Qdrant. The Qdrant connection is opened on the first
    # search by agent_rag.retrieval.get_qdrant_client() and then reused by every query, because
    # the local Qdrant store allows only one open connection at a time. close() below closes it.
    global _papers_graph
    if _papers_graph is None:
        from agent_rag.graph import build_graph
        from agent_rag.retrieval import LiteLLMEmbedder, set_embedder
        set_embedder(LiteLLMEmbedder())
        _papers_graph = build_graph()
    return _papers_graph


def ask_papers(query: str, rewritten: bool = False) -> AgentResult:
    try:
        from agent_rag.run import run_query
        # The papers agent prints debug lines ([ROUTE], [GENERATION]) while it runs.
        # redirect_stdout sends every print inside this block into io.StringIO(), an in-memory
        # text buffer that is never shown, so those lines do not clutter the chat or eval output.
        # After the block ends, print works normally again.
        with contextlib.redirect_stdout(io.StringIO()):
            raw = run_query(query, _get_papers_graph())
    except Exception as e:
        return _error_result("papers", query, rewritten, f"{type(e).__name__}: {e}")

    completeness = str((raw.get("quality_flags") or {}).get("completeness", "")).upper()
    answer = raw.get("answer", "") or ""
    # Fallback only when the agent itself says it found no answer; otherwise its answer is shown.
    if raw.get("clarifying_question") or _no_answer(answer):
        answer = ""
    # A missing completeness value: the papers agent's own default is PARTIAL.
    sufficiency = _PAPERS_SUFFICIENCY.get(completeness, "partial") if answer else "insufficient"

    citation = (raw.get("citation_string") or "").strip()
    return AgentResult(
        source="papers", query=query, rewritten=rewritten, answer=answer,
        citations=[citation] if citation else [],
        sufficiency=sufficiency,
        # formula_score: best reranked retrieval score; the papers agent's own threshold is 0.45.
        agent_score=round(float(raw.get("formula_score") or 0.0), 4),
        error=None,
    )


def close() -> None:
    """Close the papers agent's local Qdrant client before Python shuts down."""
    if _papers_graph is None:
        return
    from agent_rag import retrieval
    if retrieval._qdrant_client is not None:
        retrieval._qdrant_client.close()
        retrieval._qdrant_client = None


ADAPTERS = {"wiki": ask_wiki, "papers": ask_papers}
