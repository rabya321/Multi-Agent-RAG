"""
Step 2: Reference Stripping + Summary Generation +
        Metadata Extraction
========================================================
Reads Step 1 parsed output, strips references from
full_markdown, sends cleaned document to LLM in one
call, receives structured summary and metadata,
saves to separate output files.


Output files:
    output/step2_summarize_metadata/summaries/step2_summary_{doc_id}.json
    output/step2_summarize_metadata/metadata/step2_metadata_{doc_id}.json
    output/step2_summarize_metadata/embed_text/step2_embed_{doc_id}.txt

USAGE:
    python Ingestion/step2_summarize_metadata.py output/step1_parsed/           # folder (recommended)
    python Ingestion/step2_summarize_metadata.py output/step1_doc_xyz.json      # single file
    python Ingestion/step2_summarize_metadata.py output/step1_parsed/ --overwrite
    python Ingestion/step2_summarize_metadata.py --input output/step1_doc_xyz.json
    python Ingestion/step2_summarize_metadata.py --input_dir output/step1_parsed/
"""

import os
import re
import json
import csv
import argparse
import time
import tiktoken
from pathlib import Path
from typing import List, Optional, Tuple
import litellm
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

API_KEY   = os.getenv("API_KEY")
BASE_URL  = os.getenv("BASE_URL")
CERT_PATH = os.getenv("CERT_PATH")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.2")

if CERT_PATH:
    litellm.ssl_verify = CERT_PATH

TOKENIZER = tiktoken.get_encoding("cl100k_base")

DOMAIN_LABELS = [
    "supply chain optimization",
    "inventory management",
    "demand forecasting",
    "semiconductor manufacturing",
    "deep reinforcement learning",
    "machine learning",
    "operations research",
    "simulation modeling",
    "process optimization",
    "other",
]

METHODOLOGY_LABELS = [
    "empirical",
    "simulation",
    "theoretical",
    "survey",
    "case_study",
    "mixed",
    "other",
]


# ─────────────────────────────────────────────────────────
# BUILD LLM CLIENT
# ─────────────────────────────────────────────────────────

def build_litellm_config() -> dict:
    """
    Validates LiteLLM gateway config used for completions.
    """
    if not API_KEY:
        raise ValueError(
            "API_KEY environment variable not set.\n"
            "Windows: $env:API_KEY='your_key'"
        )
    return {"api_key": API_KEY, "base_url": BASE_URL}


# ─────────────────────────────────────────────────────────
# REFERENCE STRIPPING
# ─────────────────────────────────────────────────────────

REFERENCES_PATTERNS = [
    re.compile(r'^references?$',               re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+references?$',     re.IGNORECASE),
    re.compile(r'^bibliography$',              re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+bibliography$',    re.IGNORECASE),
    re.compile(r'^works\s+cited$',             re.IGNORECASE),
    re.compile(r'^author\s+biograph',          re.IGNORECASE),
    re.compile(r'^acknowledgem',               re.IGNORECASE),
    re.compile(r'^\d+[\.\s]+acknowledgem',     re.IGNORECASE),
]


def is_references_heading(text: str) -> bool:
    t = text.strip()
    return any(p.match(t) for p in REFERENCES_PATTERNS)


def strip_references_from_markdown(
    full_markdown:      str,
    elements:           list,
    full_text_fallback: str = ""
) -> Tuple[str, bool]:
    """
    Strips references section from full_markdown.
    Finds references heading in elements, truncates there.
    Returns (cleaned_text, used_fallback).
    Falls back to full_text_fallback if heading not found
    in markdown. Logs a warning if fallback was used.
    """
    ref_heading_text = None
    for e in elements:
        if e.get("element_type") != "header":
            continue
        content = e.get("content", "").strip()
        if is_references_heading(content):
            ref_heading_text = content
            break

    if ref_heading_text is None:
        print(f"  [STRIP] No references heading detected")
        return full_markdown, False

    patterns_to_try = [
        f"## {ref_heading_text}",
        f"# {ref_heading_text}",
        f"### {ref_heading_text}",
        ref_heading_text,
    ]

    for pattern in patterns_to_try:
        idx = full_markdown.find(pattern)
        if idx != -1:
            cleaned = full_markdown[:idx].strip()
            saved   = len(full_markdown) - len(cleaned)
            print(
                f"  [STRIP] Removed at: "
                f"'{pattern[:50]}' "
                f"({saved} chars removed)"
            )
            return cleaned, False

    lower_md  = full_markdown.lower()
    lower_ref = ref_heading_text.lower()
    idx       = lower_md.find(lower_ref)
    if idx != -1:
        cleaned = full_markdown[:idx].strip()
        saved   = len(full_markdown) - len(cleaned)
        print(
            f"  [STRIP] Removed (case-insensitive, "
            f"{saved} chars removed)"
        )
        return cleaned, False

    print(f"  [STRIP] Warning: heading not found in markdown")
    if full_text_fallback:
        print(f"  [STRIP] Used full_text fallback")
        return full_text_fallback, True
    return full_markdown, False


def count_tokens_exact(text: str) -> int:
    """Returns exact token count using cl100k_base encoding."""
    if not text or not text.strip():
        return 0
    return len(TOKENIZER.encode(text))


# ─────────────────────────────────────────────────────────
# LLM PROMPT
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an academic paper analyzer
for a research corpus in semiconductor supply chain
and operations research.

Your task is to generate a structured summary and
extract metadata from the provided paper text.

CRITICAL INSTRUCTIONS FOR SUMMARY:
- Target less than 2500 tokens for the summary
- Remove less relevant detail if the summary exceeds the target.
- Use the paper's exact terminology throughout
- Do not paraphrase technical terms into generic language
- Include specific numbers, percentages, and metrics
  from the results section
- Include specific method names, algorithm names,
  and variable names as they appear in the paper
- The summary must be detailed enough that specific
  technical queries can be answered directly from it
  without loading the full document

CRITICAL INSTRUCTIONS FOR METADATA:
- authors: extract from the FRONT MATTER section provided
           Use "Firstname Lastname" format
           e.g. "Fazail Ahmad" not "Ahmad Fazail"
           The front matter contains author names either
           as plain text or in a table structure
           Do NOT invent or guess author names
           Only use names explicitly present in front matter

- year: look for it in the proceedings/conference name
        e.g. "Proceedings of the 2022 Winter Simulation
        Conference" means year=2022

- venue: extract the full conference or journal name
     Look in front matter text

Return ONLY valid JSON. No explanation. No markdown fences.

Schema:
{   
    "summary": {

    "research_problem": {
      "core_problem": "<specific problem in 1-2 sentences>",
      "why_existing_methods_fail": "<what gap or limitation
                                    motivates this work>",
      "problem_class": "<optimization | forecasting |
                        scheduling | planning | classification
                        | simulation | other>"
    },

    "methodology": {
      "primary_method": "<exact method/algorithm name>",
      "key_design_choices": "<WHY this method was chosen,
                              what properties it exploits,
                              2-3 sentences>",
      "technical_components": "<specific modules, layers,
                                parameters, formulations>",
      "assumptions": "<key assumptions the method relies on,
                       what conditions must hold>"
    },

    "positioning": {
      "baselines_compared": ["<method name 1>",
                             "<method name 2>"],
      "how_it_differs": "<what makes this approach distinct
                          from baselines and prior work,
                          2-3 sentences>",
      "claimed_advantages": "<specific claimed benefits
                              over alternatives>",
      "applicable_conditions": "<when/where this method
                                 works best, problem
                                 characteristics it suits>"
    },

    "theoretical_framework": "<formal model, theory,
                               or mathematical basis
                               if present, else empty>",

    "key_findings": {
      "primary_result": "<main quantitative result
                          with exact numbers>",
      "comparison_vs_baselines": "<performance vs each
                                   baseline with numbers>",
      "conditions_of_success": "<under what experimental
                                 conditions results hold>",
      "surprising_or_notable": "<any unexpected finding
                                 or nuance worth noting>"
    },

    "limitations": {
      "stated_limitations": "<what authors admit>",
      "scope_constraints": "<what problem types or
                             conditions this does NOT
                             handle well>",
      "future_work": "<what authors suggest next>"
    },

    "practical_recommendations": "<actionable insights
                                   for practitioners,
                                   1-2 sentences>"
  },
  "metadata": {
    "title": "<exact paper title>",
    "authors": ["Firstname Lastname", ...],
    "year": <integer year from proceedings name or null>,
    "venue": "<full conference or journal name from front matter or hints>",
    "domain": "<exactly one of: supply chain optimization | inventory management | demand forecasting | semiconductor manufacturing | deep reinforcement learning | machine learning | operations research | simulation modeling | process optimization | other>",
        "methodology_type": "<exactly one of: empirical | simulation | theoretical | survey | case_study | mixed | other>",
    "keywords_explicit": ["<only if paper has explicit Keywords section, else empty list>"],
    "keywords_inferred": ["<5-8 specific terms from paper vocabulary, return empty list if explicit keywords found>"]
  }
}
"""


def build_user_message(
    cleaned_markdown: str,
    front_matter:     str,
) -> str:
    """
    Builds user message for summary + metadata extraction.
    Authors always come from front_matter (Docling output).
    """
    return f"""FRONT MATTER (title, authors, affiliations):
---
{front_matter[:2000]}
---

IMPORTANT FOR AUTHORS:
Extract author names from the FRONT MATTER above.
Author names appear after the title, either as plain
text or in a two-column layout.
Use "Firstname Lastname" format (e.g. "Fazail Ahmad").

FULL PAPER TEXT (references removed):
---
{cleaned_markdown}
---

Generate the structured summary and metadata JSON."""


# ─────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────

def call_llm(
    client:           dict,
    cleaned_markdown: str,
    front_matter:     str,
) -> dict:
    """
    Single LLM call returning summary + metadata.
    Authors extracted from front_matter (Docling content).
    """
    user_msg     = build_user_message(
        cleaned_markdown, front_matter
    )
    input_tokens = count_tokens_exact(
        SYSTEM_PROMPT + user_msg
    )
    print(
        f"  [LLM] Model: {LLM_MODEL} | "
        f"~{input_tokens} input tokens"
    )

    messages   = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    last_error = ""
    delays     = [0, 2, 4]

    for attempt in range(1, 4):
        if attempt > 1:
            delay = delays[attempt - 1]
            print(
                f"  [LLM] Attempt {attempt}/3 "
                f"after {delay}s..."
            )
            time.sleep(delay)
        try:
            response = litellm.completion(
                model    = LLM_MODEL,
                messages = messages,
                api_base = client["base_url"],
                api_key  = client["api_key"],
                temperature = 0,
                timeout  = 60,
            )
            raw = (response.choices[0].message.content or "").strip()

            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",          "", raw)

            result = json.loads(raw)
            print(f"  [LLM] Response parsed successfully")
            return result

        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed: {e}"
            print(f"  [LLM] {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"  [LLM] Call failed: {e}")

    return {"_error": last_error}


# ─────────────────────────────────────────────────────────
# POST VALIDATION
# ─────────────────────────────────────────────────────────


class ResearchProblemSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    core_problem: str = ""
    why_existing_methods_fail: str = ""
    problem_class: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _to_clean_str(cls, value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return ""
        return str(value).strip()


class MethodologySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_method: str = ""
    key_design_choices: str = ""
    technical_components: str = ""
    assumptions: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _to_clean_str(cls, value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return ""
        return str(value).strip()


class PositioningSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    baselines_compared: list[str] = Field(default_factory=list)
    how_it_differs: str = ""
    claimed_advantages: str = ""
    applicable_conditions: str = ""

    @field_validator("baselines_compared", mode="before")
    @classmethod
    def _normalize_baselines(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(v).strip() for v in value if str(v).strip()]

    @field_validator(
        "how_it_differs",
        "claimed_advantages",
        "applicable_conditions",
        mode="before",
    )
    @classmethod
    def _to_clean_str(cls, value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return ""
        return str(value).strip()


class KeyFindingsSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_result: str = ""
    comparison_vs_baselines: str = ""
    conditions_of_success: str = ""
    surprising_or_notable: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _to_clean_str(cls, value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return ""
        return str(value).strip()


class LimitationsSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stated_limitations: str = ""
    scope_constraints: str = ""
    future_work: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _to_clean_str(cls, value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return ""
        return str(value).strip()


class SummarySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    research_problem: ResearchProblemSchema = Field(
        default_factory=ResearchProblemSchema
    )
    methodology: MethodologySchema = Field(
        default_factory=MethodologySchema
    )
    positioning: PositioningSchema = Field(
        default_factory=PositioningSchema
    )
    theoretical_framework: str = ""
    key_findings: KeyFindingsSchema = Field(
        default_factory=KeyFindingsSchema
    )
    limitations: LimitationsSchema = Field(
        default_factory=LimitationsSchema
    )
    practical_recommendations: str = ""

    @field_validator(
        "research_problem",
        "methodology",
        "positioning",
        "key_findings",
        "limitations",
        mode="before",
    )
    @classmethod
    def _coerce_nested_dict(cls, value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return {"core_problem": value.strip()}
        return {}

    @field_validator(
        "theoretical_framework",
        "practical_recommendations",
        mode="before",
    )
    @classmethod
    def _coerce_text_field(cls, value):
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(str(v) for v in value.values() if v).strip()
        if isinstance(value, list):
            return " ".join(str(v) for v in value if v).strip()
        return str(value).strip()


class MetadataSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = Field(default=None, ge=1950, le=2030)
    venue: str = ""
    domain: str = "other"
    methodology_type: str = "other"
    keywords_explicit: list[str] = Field(default_factory=list)
    keywords_inferred: list[str] = Field(default_factory=list)

    @field_validator("title", "venue", mode="before")
    @classmethod
    def _normalize_text(cls, value):
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("authors", mode="before")
    @classmethod
    def _normalize_authors(cls, value):
        if not isinstance(value, list):
            return []
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        cleaned = [a for a in cleaned if len(a.split()) <= 6]
        return cleaned[:20]

    @field_validator("year", mode="before")
    @classmethod
    def _normalize_year(cls, value):
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize_domain(cls, value):
        domain = str(value or "").strip().lower()
        if domain in DOMAIN_LABELS:
            return domain
        return next(
            (label for label in DOMAIN_LABELS if domain in label or label in domain),
            "other",
        )

    @field_validator("methodology_type", mode="before")
    @classmethod
    def _normalize_methodology(cls, value):
        method = str(value or "").strip().lower()
        if method in METHODOLOGY_LABELS:
            return method
        return next(
            (
                label
                for label in METHODOLOGY_LABELS
                if method in label or label in method
            ),
            "other",
        )

    @field_validator("keywords_explicit", "keywords_inferred", mode="before")
    @classmethod
    def _normalize_keywords(cls, value):
        if not isinstance(value, list):
            return []
        return [str(v).strip().lower() for v in value if str(v).strip()]


def validate_summary(raw_summary: dict, notes: list) -> dict:
    """Validate LLM summary output with Pydantic schema."""
    try:
        return SummarySchema.model_validate(raw_summary or {}).model_dump()
    except ValidationError as e:
        notes.append(f"summary: schema validation failed: {e.errors()}")
        return SummarySchema().model_dump()


def validate_metadata(raw_meta: dict, notes: list) -> dict:
    """Validate LLM metadata output with Pydantic schema."""
    try:
        data = MetadataSchema.model_validate(raw_meta or {}).model_dump()
        data["keywords_inferred"] = data["keywords_inferred"][:10]
        return data
    except ValidationError as e:
        notes.append(f"metadata: schema validation failed: {e.errors()}")
        return MetadataSchema().model_dump()


# ─────────────────────────────────────────────────────────
# EMBED TEXT BUILDER
#
# EMBEDDING STRATEGY
# ──────────────────
# embed_text (fed to the embedding model):
#   Flat labelled prose of every summary sub-field
#   + Keywords from metadata.
#   This is what creates the semantic vector.
#   Does NOT include title or structured metadata —
#   those live in the vector_payload.
#
# vector_payload (loaded into LLM context at retrieval):
#   doc_id, title, full metadata fields, and the flat
#   summary prose (same as embed_text).
#   This dict is what the LLM sees when answering a query.
#
# context_tokens = count_tokens_exact(json(vector_payload))
#   Token budget consumed per paper at LLM inference time.
# ─────────────────────────────────────────────────────────

def build_embed_text(
    summary:      dict,
    all_keywords: list,
) -> str:
    """
    Builds the flat text string fed to the embedding model.
    Iterates sub-fields of nested summary dicts with
    human-readable labels. Skips empty values silently.
    Returns a flat prose string joined with double newlines.
    """
    parts = []

    def add(label: str, value) -> None:
        """Add a labelled part if value is non-empty."""
        if isinstance(value, list):
            v = ", ".join(str(i) for i in value if str(i).strip())
        else:
            v = str(value).strip() if value else ""
        if v:
            parts.append(f"{label}: {v}")

    rp = summary.get("research_problem", {})
    add("Research Problem",        rp.get("core_problem", ""))
    add("Gap in Existing Methods", rp.get("why_existing_methods_fail", ""))
    add("Problem Class",           rp.get("problem_class", ""))

    meth = summary.get("methodology", {})
    add("Primary Method",       meth.get("primary_method", ""))
    add("Design Rationale",     meth.get("key_design_choices", ""))
    add("Method Assumptions",   meth.get("assumptions", ""))
    add("Technical Components", meth.get("technical_components", ""))

    pos = summary.get("positioning", {})
    add("How It Differs From Prior Work", pos.get("how_it_differs", ""))
    add("Applicable Conditions",          pos.get("applicable_conditions", ""))
    add("Claimed Advantages",             pos.get("claimed_advantages", ""))
    add("Baselines Compared",             pos.get("baselines_compared", []))

    kf = summary.get("key_findings", {})
    add("Primary Result",          kf.get("primary_result", ""))
    add("Comparison vs Baselines", kf.get("comparison_vs_baselines", ""))
    add("Conditions of Success",   kf.get("conditions_of_success", ""))
    add("Notable Findings",        kf.get("surprising_or_notable", ""))

    lim = summary.get("limitations", {})
    add("Stated Limitations", lim.get("stated_limitations", ""))
    add("Scope Constraints",  lim.get("scope_constraints", ""))
    add("Future Work",        lim.get("future_work", ""))

    add("Theoretical Framework",     summary.get("theoretical_framework", ""))
    add("Practical Recommendations", summary.get("practical_recommendations", ""))

    add("Keywords", all_keywords)

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────
# PROCESS ONE DOCUMENT
# ─────────────────────────────────────────────────────────

def process_document(
    step1_data: dict,
    client:     dict
) -> tuple:
    """
    Strips references, calls LLM, validates output.
    """
    doc_id        = step1_data["doc_id"]
    filename      = step1_data["filename"]
    full_markdown = step1_data.get("full_markdown", "")
    front_matter  = step1_data.get("front_matter",  "")
    elements      = step1_data.get("elements",      [])
    full_text     = step1_data.get("full_text",     "")
    full_doc_toks = step1_data.get("full_text_tokens", 0)

    summary_notes  = []
    metadata_notes = []

    # Strip references again for LLM input.
    # Step 1 stripping is used for stripped_tokens reporting,
    # but Step 1 still persists full_markdown/full_text.
    # Step 2 therefore applies stripping on the actual
    # markdown passed into summarization.
    cleaned_markdown, _ = \
        strip_references_from_markdown(
            full_markdown, elements, full_text
        )
    cleaned_tokens = count_tokens_exact(cleaned_markdown)
    print(
        f"  [CLEAN] {full_doc_toks} → "
        f"{cleaned_tokens} tokens after stripping"
    )

    # Single LLM call
    raw = call_llm(
        client, cleaned_markdown, front_matter
    )

    if "_error" in raw:
        summary_notes.append(f"LLM failed: {raw['_error']}")
        error_out = {
            "doc_id"  : doc_id,
            "filename": filename,
            "_error"  : raw["_error"],
            "notes"   : summary_notes,
        }
        return error_out, error_out, ""

    # Validate and split
    validated_summary  = validate_summary(
        raw.get("summary",  {}), summary_notes
    )
    validated_metadata = validate_metadata(
        raw.get("metadata", {}),
        metadata_notes,
    )

    all_keywords = list(dict.fromkeys(
        validated_metadata["keywords_explicit"] +
        validated_metadata["keywords_inferred"]
    ))
    validated_metadata["all_keywords"] = all_keywords

    # Build flat embed text: summary prose + keywords
    # (fed to the embedding model for vector creation)
    embed_text   = build_embed_text(
        validated_summary, all_keywords
    )

    # Build vector_payload: what gets loaded into LLM
    # context at inference time (title + metadata + summary)
    vector_payload = {
        "doc_id"          : doc_id,
        "title"           : validated_metadata["title"],
        "authors"         : validated_metadata["authors"],
        "year"            : validated_metadata["year"],
        "venue"           : validated_metadata["venue"],
        "domain"          : validated_metadata["domain"],
        "all_keywords"    : all_keywords,
        "methodology_type": validated_metadata["methodology_type"],
        "summary_text"    : embed_text,
    }
    context_tokens = count_tokens_exact(
        json.dumps(vector_payload, ensure_ascii=False)
    )

    # Print summary diagnostics
    rp   = validated_summary.get("research_problem", {})
    meth = validated_summary.get("methodology",       {})
    pos  = validated_summary.get("positioning",       {})
    kf   = validated_summary.get("key_findings",      {})
    cp   = rp.get("core_problem",    "")
    pm   = meth.get("primary_method", "")
    bl   = pos.get("baselines_compared", [])
    pr   = kf.get("primary_result",   "")

    print(
        f"  [SUMMARY] research_problem.core_problem: "
        f"{count_tokens_exact(cp)} tokens"
    )
    print(f"  [SUMMARY] methodology.primary_method: {pm}")
    if bl:
        print(
            f"  [SUMMARY] positioning.baselines: "
            f"{', '.join(bl)}"
        )
    print(
        f"  [SUMMARY] key_findings.primary_result: "
        f"{count_tokens_exact(pr)} tokens"
    )
    print(
        f"  [SUMMARY] context_tokens: {context_tokens} "
        f"(target: 1200-1800)"
    )

    # Extract flat metadata fields from nested summary
    

    summary_out = {
        "step"                     : "step2_summary",
        "doc_id"                   : doc_id,
        "filename"                 : filename,
        "research_problem"         : validated_summary[
            "research_problem"
        ],
        "methodology"              : validated_summary[
            "methodology"
        ],
        "positioning"              : validated_summary[
            "positioning"
        ],
        "theoretical_framework"    : validated_summary[
            "theoretical_framework"
        ],
        "key_findings"             : validated_summary[
            "key_findings"
        ],
        "limitations"              : validated_summary[
            "limitations"
        ],
        "practical_recommendations": validated_summary[
            "practical_recommendations"
        ],
        "embed_text"               : embed_text,
        "vector_payload"           : vector_payload,
        "context_tokens"           : context_tokens,
        "full_doc_tokens"          : full_doc_toks,
        "notes"                    : summary_notes,
    }

    metadata_out = {
        "step"              : "step2_metadata",
        "doc_id"            : doc_id,
        "filename"          : filename,
        "title"             : validated_metadata["title"],
        "authors"           : validated_metadata["authors"],
        "year"              : validated_metadata["year"],
        "venue"             : validated_metadata["venue"],
        "domain"            : validated_metadata["domain"],
        "methodology_type"  : validated_metadata[
            "methodology_type"
        ],
        "keywords_explicit" : validated_metadata[
            "keywords_explicit"
        ],
        "keywords_inferred" : validated_metadata[
            "keywords_inferred"
        ],
        "all_keywords"      : all_keywords,
        "notes"             : metadata_notes,

    }

    return summary_out, metadata_out, embed_text


# ─────────────────────────────────────────────────────────
# OUTPUT PATH HELPERS
# ─────────────────────────────────────────────────────────

def get_summary_path(output_dir: str, doc_id: str) -> str:
    return os.path.join(
        output_dir,
        "step2_summarize_metadata",
        "summaries",
        f"step2_summary_{doc_id}.json"
    )


def get_metadata_path(output_dir: str, doc_id: str) -> str:
    return os.path.join(
        output_dir,
        "step2_summarize_metadata",
        "metadata",
        f"step2_metadata_{doc_id}.json"
    )


def get_embed_text_path(output_dir: str, doc_id: str) -> str:
    return os.path.join(
        output_dir,
        "step2_summarize_metadata",
        "embed_text",
        f"step2_embed_{doc_id}.txt"
    )


# ─────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────

def save_outputs(
    summary:    dict,
    metadata:   dict,
    embed_text: str,
    output_dir: str
) -> tuple:
    doc_id         = summary["doc_id"]
    summary_path   = get_summary_path(output_dir, doc_id)
    metadata_path  = get_metadata_path(output_dir, doc_id)
    embed_text_path = get_embed_text_path(output_dir, doc_id)

    os.makedirs(
        os.path.dirname(summary_path),   exist_ok=True
    )
    os.makedirs(
        os.path.dirname(metadata_path),  exist_ok=True
    )
    os.makedirs(
        os.path.dirname(embed_text_path), exist_ok=True
    )

    # JSON files: structured, for inspection and debugging
    with open(
        summary_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            summary, f, indent=2, ensure_ascii=False
        )

    with open(
        metadata_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            metadata, f, indent=2, ensure_ascii=False
        )

    # Flat text file: fed directly to the embedding model
    with open(
        embed_text_path, "w", encoding="utf-8"
    ) as f:
        f.write(embed_text)

    print(f"  [SAVE] Summary   : {summary_path}")
    print(f"  [SAVE] Metadata  : {metadata_path}")
    print(f"  [SAVE] Embed text: {embed_text_path}")
    return summary_path, metadata_path, embed_text_path


# ─────────────────────────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────────────────────────

def preview(summary: dict, metadata: dict):
    rp   = summary.get("research_problem", {})
    meth = summary.get("methodology",       {})
    pos  = summary.get("positioning",       {})
    kf   = summary.get("key_findings",      {})

    print(f"\n{'─'*55}")
    print(f"  SUMMARY PREVIEW")
    print(f"{'─'*55}")
    print(
        f"  core_problem    : "
        f"{rp.get('core_problem', '')[:90]}"
    )
    print(
        f"  primary_method  : "
        f"{meth.get('primary_method', '')[:60]}"
    )
    print(
        f"  how_it_differs  : "
        f"{pos.get('how_it_differs', '')[:90]}"
    )
    print(
        f"  baselines       : "
        f"{pos.get('baselines_compared', [])}"
    )
    print(
        f"  primary_result  : "
        f"{kf.get('primary_result', '')[:90]}"
    )
    print(
        f"  context_tokens  : "
        f"{summary.get('context_tokens', 0)}"
    )
    print(f"\n{'─'*55}")
    print(f"  METADATA PREVIEW")
    print(f"{'─'*55}")
    print(
        f"  title         : "
        f"{metadata.get('title','')[:70]}"
    )
    print(f"  authors       : {metadata.get('authors', [])}")
    print(f"  year          : {metadata.get('year')}")
    print(
        f"  venue         : "
        f"{metadata.get('venue','')[:60]}"
    )
    print(f"  domain        : {metadata.get('domain')}")
    print(
        f"  methodology   : "
        f"{metadata.get('methodology_type', '')}"
    )
    print(
        f"  keywords      : "
        f"{metadata.get('all_keywords', [])[:5]}"
    )
    if summary.get("notes"):
        print(f"\n  Summary Notes:")
        for n in summary["notes"]:
            print(f"    ⚠  {n}")
    if metadata.get("notes"):
        print(f"\n  Metadata Notes:")
        for n in metadata["notes"]:
            print(f"    ⚠  {n}")


# ─────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────

def already_processed(
    output_dir: str,
    doc_id:     str
) -> bool:
    return (
        os.path.exists(
            get_summary_path(output_dir, doc_id)
        ) and
        os.path.exists(
            get_metadata_path(output_dir, doc_id)
        ) and
        os.path.exists(
            get_embed_text_path(output_dir, doc_id)
        )
    )


def run_batch(
    input_files: List[Path],
    output_dir:  str,
    overwrite:   bool,
    client:      dict
):
    total             = len(input_files)
    done = skipped = failed = 0
    token_rows: list = []  # for CSV

    print(f"\n{'='*55}")
    print(f"STEP 2: SUMMARIZATION + METADATA EXTRACTION")
    print(f"{'='*55}")
    print(f"  Documents    : {total}")
    print(f"  Model        : {LLM_MODEL}")
    print(f"  Endpoint     : {BASE_URL}")
    print(
        f"  SSL verify   : "
        f"{CERT_PATH if CERT_PATH else 'default TLS'}"
    )
    print(
        f"  Summaries    : "
        f"{os.path.abspath(os.path.join(output_dir, 'step2_summarize_metadata', 'summaries'))}"
    )
    print(
        f"  Metadata     : "
        f"{os.path.abspath(os.path.join(output_dir, 'step2_summarize_metadata', 'metadata'))}"
    )
    print(
        f"  Embed text   : "
        f"{os.path.abspath(os.path.join(output_dir, 'step2_summarize_metadata', 'embed_text'))}"
    )
    print(f"  Overwrite    : {overwrite}")
    print(f"  Authors from : Docling front_matter only")
    print(f"  Venue/Year   : LLM only")

    for idx, path in enumerate(input_files, 1):
        print(f"\n[{idx}/{total}] {path.name}")

        try:
            with open(
                path, "r", encoding="utf-8"
            ) as f:
                step1_data = json.load(f)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            failed += 1
            continue

        doc_id = step1_data.get("doc_id", "unknown")

        if not overwrite and already_processed(
            output_dir, doc_id
        ):
            print(
                f"  SKIPPED (outputs exist). "
                f"Use --overwrite to redo."
            )
            skipped += 1
            continue

        try:
            summary, metadata, embed_text = process_document(
                step1_data, client
            )

            if "_error" in summary:
                print(f"  FAILED: {summary['_error']}")
                failed += 1
                continue

            preview(summary, metadata)
            save_outputs(
                summary, metadata, embed_text, output_dir
            )
            token_rows.append({
                "doc_id"        : summary["doc_id"],
                "filename"      : summary["filename"],
                "title"         : metadata.get("title", ""),
                "context_tokens": summary.get("context_tokens", 0),
            })
            done += 1

            if idx < total:
                time.sleep(1)

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"BATCH COMPLETE")
    print(f"  Processed : {done}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Total     : {total}")
    print(f"{'='*55}")

    # Write token stats CSV
    if token_rows:
        csv_path = os.path.join(
            output_dir,
            "step2_summarize_metadata",
            "vectorpayload_token_stats.csv"
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(token_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(token_rows)
        print(
            f"  Token stats: {os.path.abspath(csv_path)}"
        )


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: Reference Stripping + "
            "Summary + Metadata Extraction"
        )
    )

    # Accept a positional path (file or directory) OR
    # the legacy --input / --input_dir flags.
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Path to a single step1_*.json file "
            "OR a directory containing step1_*.json files"
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--input",
        help="Path to single step1_*.json file"
    )
    group.add_argument(
        "--input_dir",
        help="Directory containing step1_*.json files"
    )

    parser.add_argument(
        "--output_dir", default="output",
        help="Where to save outputs (default: output)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process even if outputs exist"
    )

    args  = parser.parse_args()
    start = time.time()

    # Resolve the input source: positional > --input > --input_dir
    raw_path = args.path or args.input or args.input_dir
    if not raw_path:
        parser.error(
            "Provide a file or directory path as a "
            "positional argument, or use --input / --input_dir"
        )

    p = Path(raw_path)

    if p.is_file():
        input_files = [p]
    elif p.is_dir():
        input_files = sorted(p.rglob("step1_*.json"))
        if not input_files:
            print(
                f"No step1_*.json files found in: {p}"
            )
            return
    else:
        print(f"ERROR: path does not exist: {p}")
        return

    try:
        client = build_litellm_config()
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    print(f"\n  LLM endpoint : {BASE_URL}")
    print(f"  LLM model    : {LLM_MODEL}")
    print(
        f"  SSL verify   : "
        f"{CERT_PATH if CERT_PATH else 'default TLS'}"
    )

    run_batch(
        input_files,
        args.output_dir,
        args.overwrite,
        client
    )

    print(f"\n  Total time : {time.time() - start:.1f}s")
    print(
        f"  Outputs    : "
        f"{os.path.abspath(args.output_dir)}"
    )
    print(f"  Next step  : step3_chunk.py")


if __name__ == "__main__":
    main()