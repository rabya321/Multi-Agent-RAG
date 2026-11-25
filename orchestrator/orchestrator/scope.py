"""Scope descriptions of the wiki and papers knowledge bases, used in the router prompt.

Two modes, each saved as a plain-text file that is pasted into the router prompt as is:
  topics: source cards + LLM-generated topic map per source + domain boundary  -> scope_topics.txt
  full:   source cards + every concept/paper title with keywords (baseline) + domain boundary -> scope_full.txt

Generate a file (topics makes one LLM call per source; full makes none):
  uv run python -m orchestrator.scope --mode topics
  uv run python -m orchestrator.scope --mode full
The router only reads the saved file. Re-run this script when the wiki or the papers change.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage

from orchestrator.config import REPO_ROOT, make_llm

WIKI_INDEX = REPO_ROOT / "wiki_agent" / "wiki" / "global_index.yaml"
PAPERS_STEP2_DIR = REPO_ROOT / "papers_agent" / "output_preprocessing" / "step2_summarize_metadata"
PAPERS_META_DIR = PAPERS_STEP2_DIR / "metadata"
PAPERS_SUMMARY_DIR = PAPERS_STEP2_DIR / "summaries"
SCOPE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# SOURCE CARDS; Hand-written parts (do not grow with the number of documents)
# Both cards use the same fields and the same number of items per field, and describe
# the kind of question each source answers rather than trigger words, so that neither
# source is favoured by wording.
# ---------------------------------------------------------------------------

WIKI_CARD = """WIKI: Infineon internal supply-chain training material
Nature: concept pages rewritten from Infineon e-learning slide decks; describes how Infineon works in practice.
Answers questions like: how an internal process works and what its steps are; who is responsible for what; which internal system or tool is used and how; which rules, settings and parameters Infineon applies; what an Infineon term or acronym means.
Does NOT contain: research studies, experiments, research results, comparisons of algorithms, paper authors."""

PAPERS_CARD = """PAPERS: academic research papers
Nature: peer-reviewed conference and journal papers; describes what research proposed and found.
Answers questions like: what problem a study addressed; which method, model or algorithm it proposed; how its experiments or simulations were set up; what results it reported and how it compared with other approaches; who wrote which paper.
Does NOT contain: internal procedures, instructions for internal systems, organisational roles and responsibilities, company rules and settings."""

DOMAIN_BOUNDARY = """DOMAIN BOUNDARY
In domain: supply chain management, operations, logistics, planning, semiconductor manufacturing, and every topic listed under WIKI or PAPERS, including topics beyond supply chain (e.g. semantic web and ontologies, quantum optimization).
In-domain questions are retrievable even if no listed topic mentions them explicitly.
Out of scope: topics unrelated to the above, and questions that need live, future or personal information (current prices, market forecasts, personal data)."""

# ---------------------------------------------------------------------------
# Topic-map generation prompts 
# ---------------------------------------------------------------------------

TOPIC_MAP_PROMPT = """You are building a compact topic map of a knowledge base. A query router will read it to decide whether a user 
question can be answered from this knowledge base.

Knowledge base: {name}
Entries ({n}):
{entries}

Write the topic map:
- Group the entries into topic areas by what they are about, not by course, module or file names.
- One line per topic area, in the form: "- <topic area>: <comma-separated key terms>"
- Every entry must be covered by some topic area.
- Keep acronyms, named methods and distinctive terms a user might type.
- Output only the lines, nothing else."""


# ---------------------------------------------------------------------------
# Loading entries (no LLM)
# ---------------------------------------------------------------------------

def load_wiki_entries() -> list[dict]:
    with WIKI_INDEX.open(encoding="utf-8") as f:
        docs = [d for d in yaml.safe_load_all(f) if isinstance(d, dict) and d.get("id")]
    return [
        {
            "title": d["title"],
            "area": " / ".join(p for p in (d.get("track"), d.get("subtrack")) if p),
            "description": d.get("description", ""),
            "keywords": d.get("keywords") or [],
        }
        for d in docs
    ]


def load_paper_entries() -> list[dict]:
    # The one-sentence research problem plays the same role as the wiki concept description,
    # so both knowledge bases give the topic-map generator the same kind of input.
    entries = []
    for path in sorted(PAPERS_META_DIR.glob("step2_metadata_*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        summary_path = PAPERS_SUMMARY_DIR / f"step2_summary_{meta['doc_id']}.json"
        description = ""
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            description = summary.get("research_problem", {}).get("core_problem", "")
        entries.append(
            {
                "title": meta["title"],
                "area": meta.get("domain", ""),
                "description": description,
                "keywords": meta.get("all_keywords") or [],
            }
        )
    return entries

# from dict to one line per entry, with title, area, description(if) and keywords. The LLM will group them into topic areas.
def _format_entries(entries: list[dict], with_description: bool) -> str:
    lines = []
    for e in entries:
        line = f"- {e['title']} [{e['area']}]"
        if with_description and e["description"]:
            line += f": {e['description']}"
        if e["keywords"]:
            line += f" (keywords: {', '.join(e['keywords'])})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generating and loading the scope text
# ---------------------------------------------------------------------------

def generate_topic_map(llm, name: str, entries: list[dict]) -> str:
    prompt = TOPIC_MAP_PROMPT.format(
        name=name,
        n=len(entries),
        entries=_format_entries(entries, with_description=True),
    )
    reply = llm.invoke([HumanMessage(prompt)])
    text = "\n".join(
        line.strip() for line in reply.content.splitlines() if line.strip().startswith("- ")
    )
    if not text:
        raise RuntimeError(f"{name}: no topic lines in reply:\n{reply.content}")
    return text


def build_scope(mode: str) -> str:
    if mode == "topics":
        llm = make_llm(max_tokens=4000)
        wiki_list = generate_topic_map(llm, "WIKI", load_wiki_entries())
        papers_list = generate_topic_map(llm, "PAPERS", load_paper_entries())
        label = "Topics"
    else:
        wiki_list = _format_entries(load_wiki_entries(), with_description=False)
        papers_list = _format_entries(load_paper_entries(), with_description=False)
        label = "Contents"
    return (
        f"{WIKI_CARD}\n{label}:\n{wiki_list}\n\n"
        f"{PAPERS_CARD}\n{label}:\n{papers_list}\n\n"
        f"{DOMAIN_BOUNDARY}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate and save the router scope text.")
    parser.add_argument("--mode", choices=["topics", "full"], required=True)
    args = parser.parse_args()

    out_path = SCOPE_DIR / f"scope_{args.mode}.txt"
    out_path.write_text(build_scope(args.mode), encoding="utf-8")
    print(f"[saved] {out_path}")
