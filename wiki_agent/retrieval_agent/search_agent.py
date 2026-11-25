import argparse
import json
import re
import time
from pathlib import Path
from pydantic import BaseModel, Field
import tiktoken
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelRetryMiddleware
from config import QUERY_MODEL, API_KEY, BASE_URL, CERT_PATH, WIKI_MAX_TOKENS, ANSWER_LENGTH_INSTRUCTION
from wiki_tools import read_file
from retrieval_agent.prompt import SYSTEM_PROMPT
import httpx

DEFAULT_INDEX = "wiki/index.md"
MAX_TURNS = 10
MAX_TOTAL_PAGES = 8
MAX_TOOL_CALLS = 5
MAX_PAGE_CONTENT_TOKENS = 5000

# --------------------------
# STRUCTURED OUTPUT SCHEMA
# --------------------------

class WikiAnswer(BaseModel):
    """Structured answer produced by the retrieval agent."""
    question_type: str | None = Field(default=None, description="Internal classification only. Never repeat inside answer.")
    classification_reasoning: str | None = Field(default=None, description="Internal reasoning only. Never repeat inside answer.")
    selection_log: str | None = Field(default=None, description="Internal retrieval note only. Never repeat inside answer.")
    validated_chain: str | None = Field(default=None, description="Internal chain note only. Never repeat inside answer.")
    supporting_concepts: list[str] = Field(description="Internal: slugs of read pages that support claims in the final answer, excluding unused pages.")
    answered_modules: list[str] = Field(description="User-facing: exact source module IDs supporting the final answer; select only sources of supporting_concepts, not all retrieved sources.")
    answer: str = Field(description=ANSWER_LENGTH_INSTRUCTION)
    confidence_score: int = Field(ge=0, le=100, description="User-facing, filled last after the final answer: evidence sufficiency from 0 to 100, not a probability of correctness.")


# -------------------------
# TOOL
# -------------------------

pages_read: list[str] = []
# path -> full page content exactly as returned to the model
pages_content: dict[str, str] = {}
tool_calls_made = 0
page_content_tokens = 0
retrieval_budget_exhausted = False

_READ_ERROR_PREFIXES = (
    "File path is required",
    "Path escapes project root",
    "File does not exist",
    "Failed to read file",
)


def _build_token_encoder():
    """Get tokenizer aligned to model where possible, with stable fallback."""
    try:
        return tiktoken.encoding_for_model(QUERY_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


TOKEN_ENCODER = _build_token_encoder()


def _token_count(text: str) -> int:
    """Count tokens using tiktoken for retrieval budgeting."""
    if not text:
        return 0
    return len(TOKEN_ENCODER.encode(text))


@tool
def read_wiki(paths: list[str]) -> str:
    """Read one or more wiki pages and return their content.

    Args:
        paths: List of relative paths to concept pages in the wiki.
               Example: ["wiki/concepts/concept_a.md", "wiki/concepts/concept_b.md"]

    Returns:
        str: Concatenated content of the specified wiki pages.
    """
    global tool_calls_made, page_content_tokens, retrieval_budget_exhausted

    tool_calls_made += 1
    if tool_calls_made > MAX_TOOL_CALLS:
        retrieval_budget_exhausted = True
        return (
            f"Tool-call budget exhausted. Maximum read_wiki calls per query is "
            f"{MAX_TOOL_CALLS}."
        )

    if retrieval_budget_exhausted:
        return (
            "Retrieval budget exhausted. Answer using already retrieved pages "
            "and report unsupported parts explicitly."
        )

    remaining = MAX_TOTAL_PAGES - len(pages_read)
    if remaining <= 0:
        retrieval_budget_exhausted = True
        return f"Page budget exhausted. Maximum total pages per query is {MAX_TOTAL_PAGES}."
    results = []
    for path in paths[:remaining]:
        content = read_file(path)
        header = f"###{path}"
        error_read = content.startswith(_READ_ERROR_PREFIXES)

        if not error_read:
            candidate_tokens = _token_count(content)
            if page_content_tokens + candidate_tokens > MAX_PAGE_CONTENT_TOKENS:
                retrieval_budget_exhausted = True
                results.append(
                    f"{header}\n"
                    "Retrieval content-token budget exhausted before reading this page. "
                    f"Limit: {MAX_PAGE_CONTENT_TOKENS} tokens."
                )
                break
            page_content_tokens += candidate_tokens

        results.append(f"{header}\n{content}")
        if path not in pages_read:
            pages_read.append(path)
        if not error_read:
            pages_content[path] = content

    if page_content_tokens >= MAX_PAGE_CONTENT_TOKENS:
        retrieval_budget_exhausted = True

    return "\n\n---\n\n".join(results) if results else "No content found for the specified paths."


# -------------------------
# MODEL
# -------------------------

def _build_agent():
    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    model = ChatOpenAI(
        model=QUERY_MODEL,
        temperature=0.0,
        max_tokens=WIKI_MAX_TOKENS,
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
    )
    return create_agent(
        model=model,
        tools=[read_wiki],
        response_format=WikiAnswer,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="read_wiki",
                run_limit=MAX_TOOL_CALLS,
                exit_behavior="continue",
            ),
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        ],
    )


AGENT = _build_agent()


_ANSWER_SCAFFOLD_RE = re.compile(
    r"^(?:question[ _]type|classification[ _]reasoning|validated[ _]chain|selection[ _]log)\s*:|^[A-Za-z0-9_\-/]+:\s+selected via",
    re.IGNORECASE,
)


def _sanitize_answer(answer: str) -> str:
    """Drop labeled diagnostic blocks, including embedded/multiline leakage.

    Resume prose only at a blank line or explicit answer marker. If no prose
    can be separated safely, return empty rather than return the leaked text.
    """
    if not answer:
        return answer
    kept: list[str] = []
    skipping = False
    for line in answer.splitlines():
        stripped = line.strip().lstrip("#* ").replace("**", "")
        marker = re.match(r"^(?:final[ _]answer|answer)\s*:\s*(.*)$", stripped, re.IGNORECASE)
        if marker:
            skipping = False
            if marker.group(1):
                kept.append(marker.group(1))
            continue
        if _ANSWER_SCAFFOLD_RE.match(stripped):
            skipping = True
            continue
        if not stripped:
            skipping = False
        if skipping:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _finalize_provenance(answer: dict, contents: dict[str, str]) -> None:
    """Keep retrieved provenance distinct from model-selected supporting sources."""
    sources_by_concept: dict[str, set[str]] = {}
    for path, content in contents.items():
        parts = content.split("---", 2)
        if not content.startswith("---") or len(parts) < 3:
            continue
        frontmatter = yaml.safe_load(parts[1])
        if not isinstance(frontmatter, dict):
            continue
        sources_by_concept[Path(path).stem] = {
            re.sub(r"\s*\(primary source\)$", "", str(source)).strip()
            for source in frontmatter.get("sources") or []
        }
    supporting = set(answer.get("supporting_concepts") or []) & sources_by_concept.keys()
    answer["supporting_concepts"] = sorted(supporting)
    retrieved = set().union(*sources_by_concept.values()) if sources_by_concept else set()
    supported = set().union(*(sources_by_concept[slug] for slug in supporting)) if supporting else set()
    selected = {
        re.sub(r"\s*\(primary source\)$", "", str(module)).strip()
        for module in answer.get("answered_modules") or []
    }
    answer["retrieved_modules"] = sorted(retrieved)
    answer["answered_modules"] = sorted(selected & supported)


# -------------------------
# HELPERS
# -------------------------

def usage_stats(final_state: dict) -> dict:
    """Sum input/output tokens and count model calls and read_wiki tool calls."""
    stats = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0, "tool_calls": 0}
    for msg in final_state.get("messages", []):
        if not isinstance(msg, AIMessage):
            continue
        stats["model_calls"] += 1
        stats["tool_calls"] += sum(1 for call in msg.tool_calls if call["name"] == "read_wiki")
        usage = msg.usage_metadata or {}
        stats["input_tokens"] += usage.get("input_tokens", 0) or 0
        stats["output_tokens"] += usage.get("output_tokens", 0) or 0
    return stats


def load_questions(path: str) -> list[dict]:
    """Load questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError(
        f"Unrecognized format in {path}. "
        f"Expected a list of questions or a dict with a 'questions' key."
    )


# -------------------------
# CORE AGENT
# -------------------------

def run_query(
        query: str,
        index_path: str = DEFAULT_INDEX,
) -> dict:
    global pages_read, pages_content, tool_calls_made, page_content_tokens, retrieval_budget_exhausted
    pages_read = []
    pages_content = {}
    tool_calls_made = 0
    page_content_tokens = 0
    retrieval_budget_exhausted = False

    index_content = read_file(index_path)
    if index_content.startswith("File does not exist"):
        return {
            "error": f"Index file not found: {index_path}",
            "run_status": "error",
            "total_tokens": 0,
            "latency_ms": 0,
        }

    start_time = time.monotonic()

    try:
        final_state = AGENT.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"GLOBAL INDEX:\n{index_content}"
                            f"\n\nQUESTION:{query}"
                        )
                    )
                ]
            },
            config={"recursion_limit": MAX_TURNS},
        )
    except Exception as e:
        return {
            "error": f"Agent invocation failed: {type(e).__name__}: {str(e)}",
            "run_status": "error",
            "total_tokens": 0,
            "latency_ms": int((time.monotonic() - start_time) * 1000),
        }

    latency_ms = int((time.monotonic() - start_time) * 1000)
    stats = usage_stats(final_state)
    tokens = stats["input_tokens"] + stats["output_tokens"]

    structured = final_state.get("structured_response")
    if structured is not None:
        answer = structured.model_dump()
        answer["answer"] = _sanitize_answer(answer.get("answer", ""))
        answer["pages_read"] = list(pages_read)
        answer["retrieved_contexts"] = list(pages_content.values())
        answer["retrieved_concepts"] = [Path(path).stem for path in pages_read]
        _finalize_provenance(answer, pages_content)

        has_support = bool(answer.get("supporting_concepts"))
        has_answer = bool(answer.get("answer", "").strip())
        if not has_support or not has_answer:
            answer["answer"] = "No answer found."
            answer["confidence_score"] = 0
            answer["run_status"] = "no_answer"
        else:
            answer["run_status"] = "ok"

        answer["retrieval_budget"] = {
            "max_tool_calls": MAX_TOOL_CALLS,
            "tool_calls_made": tool_calls_made,
            "max_page_content_tokens": MAX_PAGE_CONTENT_TOKENS,
            "page_content_tokens_used": page_content_tokens,
            "max_pages": MAX_TOTAL_PAGES,
            "pages_read_count": len(pages_read),
            "budget_exhausted": retrieval_budget_exhausted,
        }
        answer["total_tokens"] = tokens
        answer.update(stats)
        answer["latency_ms"] = latency_ms
        return answer

    final_text = ""
    for msg in reversed(final_state.get("messages", [])):
        content = getattr(msg, "content", "")
        if content and isinstance(content, str):
            final_text = content
            break

    return {
        "error": "Model did not produce a valid structured answer.",
        "run_status": "error",
        "raw_response": final_text[:500],
        "total_tokens": tokens,
        **stats,
        "latency_ms": latency_ms,
    }


# -------------------------
# BATCH RUNNER
# -------------------------

def run_batch(
    questions_paths: list[str],
    index_path: str = DEFAULT_INDEX,
    limit: int | None = None,
    question_type: str | None = None,
    output_path: str | None = None,
) -> None:
    """Run the agent on one or more question sets.

    Writes one JSON record per question to a single JSONL file.
    Flushes after every question so partial runs are not lost.
    Each record contains agent output + ground truth fields
    so the file is self-contained for RAGAS evaluation.

    Args:
        questions_paths: paths to standard and/or cross-track questions JSON;
                         all questions are combined into one results file
        index_path:     path to concept index
        limit:          only run first N questions
        question_type:  filter to this question type only
        output_path:    where to write JSONL output
    """
    questions = []
    for path in questions_paths:
        questions.extend(load_questions(path))

    ids = [q.get("id") for q in questions]
    duplicates = {i for i in ids if i is not None and ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids across files: {sorted(duplicates)}")

    if question_type:
        questions = [
            q for q in questions
            if q.get("type") == question_type
        ]
        if not questions:
            print(f"No questions of type '{question_type}' found.")
            return

    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be >= 0")
        questions = questions[:limit]

    if output_path is None:
        timestamp = int(time.time())
        stem = (
            Path(questions_paths[0]).stem
            if len(questions_paths) == 1 else "combined"
        )
        type_suffix = f"_{question_type}" if question_type else ""
        out_path = Path(f"wiki_query_{stem}{type_suffix}_{timestamp}.jsonl")
    else:
        out_path = Path(output_path)

    print(f"Running {len(questions)} questions.")
    print(f"Output: {out_path}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            q_id   = q.get("id", f"Q{i}")
            q_type = q.get("type", "unknown")

            print(
                f"[{i}/{len(questions)}] {q_id} ({q_type})...",
                end=" ",
                flush=True,
            )

            result = run_query(q["question"], index_path)

            gt = q.get("ground_truth", {})
            record = {
                "id":       q_id,
                "type":     q_type,
                "evaluation_mode": q.get("evaluation_mode"),
                "track":    q.get("track"),
                "question": q["question"],
                "required_modules": q.get("required_modules", []),
                "required_concepts": q.get("required_concepts", []),
                "gold_chain": q.get("chain"),
                **result,
                "ground_truth": {"answer": gt.get("answer", "")},
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if result.get("error"):
                print(f"ERROR: {result['error'][:60]}")
            else:
                print(
                    f"tokens={result.get('total_tokens', 0)} "
                    f"latency={result.get('latency_ms', 0)}ms"
                )

    print(f"\nDone. {len(questions)} questions written to {out_path}")


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wiki retrieval agent for Infineon Supply Chain Academy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single query:
        uv run python retrieval_agent/search_agent.py --query "Why do Robust packages need two electroplating steps?"
  Standard questions:
        uv run python retrieval_agent/search_agent.py --eval evaluation/output/standard_questions.json
  Cross-track questions:
        uv run python retrieval_agent/search_agent.py --eval evaluation/output/cross_track_questions.json
  Standard + cross-track into one results file:
        uv run python retrieval_agent/search_agent.py --eval evaluation/output/standard_questions.json evaluation/output/cross_track_questions.json --output results/agent_all.jsonl
  First 10 questions only:
        uv run python retrieval_agent/search_agent.py --eval evaluation/output/standard_questions.json --limit 10
  Multi-hop questions only:
        uv run python retrieval_agent/search_agent.py --eval evaluation/output/standard_questions.json --type multi_hop
  Save to file:
        uv run python retrieval_agent/search_agent.py --query "..." --output result.json
        uv run python retrieval_agent/search_agent.py --eval ... --output results.jsonl
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--query", "-q",
        type=str,
        help="Single question to answer.",
    )
    mode.add_argument(
        "--eval",
        type=str,
        nargs="+",
        metavar="QUESTIONS_PATH",
        help=(
            "One or more evaluation JSON files (standard and/or "
            "cross-track). All are combined into one results file."
        ),
    )

    parser.add_argument(
        "--index", "-i",
        type=str,
        default=DEFAULT_INDEX,
        help=f"Path to concept index. Default: {DEFAULT_INDEX}",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=(
            "Output file. JSON for --query, JSONL for --eval. "
            "Auto-generated with timestamp if not set."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="(--query only) Print internal diagnostics as well as the user-facing fields.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(--eval only) Only run first N questions.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["multi_hop", "comparative_synthesis", "index_navigation"],
        help="(--eval only) Filter to this question type only.",
    )

    args = parser.parse_args()

    if args.query:
        result = run_query(args.query, args.index)
        visible = result if args.debug or result.get("error") else {
            key: result[key] for key in ("answer", "answered_modules", "confidence_score")
        }
        print(json.dumps(visible, indent=2, ensure_ascii=False))
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nWritten to: {args.output}")

    elif args.eval:
        run_batch(
            questions_paths=args.eval,
            index_path=args.index,
            limit=args.limit,
            question_type=args.type,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()