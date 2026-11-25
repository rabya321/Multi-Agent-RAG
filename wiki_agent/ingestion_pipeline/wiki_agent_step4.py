import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from pydantic import BaseModel, Field, ValidationError

from config import (
    WIKI_MODEL,
    WIKI_DIR,
    INDEX_PATH,
    LOG_PATH,
    DEBUG_DIR,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    TOOL_CALL_HARD_LIMIT,
    WIKI_MAX_TOKENS,
    LOG_TOKEN_USAGE,
    API_KEY,
    BASE_URL,
    CERT_PATH,
)
from models import (
    Assembly,
    AssemblyBlock,
    WikiEnvelope,
    WikiFile,
    IndexUpdate,
    TokenUsage,
)
from ingestion_pipeline.wiki_prompts import WIKI_SYSTEM_PROMPT


INDEX_PROMPT_CHAR_LIMIT = 50000
API_TIMEOUT_SECONDS = 120
MAX_EXTRA_ROUNDS_AFTER_TOOL_LIMIT = 2

if CERT_PATH:
    litellm.ssl_verify = CERT_PATH


# ---------------------------------------------------------------------------
# Embedded log utility (replaces utils.append_log)
# ---------------------------------------------------------------------------

def _append_log(entry: str) -> None:
    """
    Append a timestamped entry to wiki/log.md.
    Creates log.md if it does not exist.
    Format: YYYY-MM-DD HH:MM:SS | {entry}
    """
    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {entry}\n"
        )


# ---------------------------------------------------------------------------
# Pydantic validation models
# ---------------------------------------------------------------------------

class WikiFileModel(BaseModel):
    filename: str
    content: str


class IndexUpdateModel(BaseModel):
    """
    Simplified IndexUpdate — errors and synthesis triggers removed.
    All fields have defaults so partial model output does not
    cause validation errors.
    """
    modules: list[dict] = Field(default_factory=list)
    concepts_added: list[str] = Field(default_factory=list)
    concepts_updated: list[str] = Field(default_factory=list)


class WikiEnvelopeModel(BaseModel):
    """
    Simplified WikiEnvelope — error_entries removed entirely.
    All fields have defaults for robustness against partial output.
    """
    module_page: WikiFileModel
    concept_pages: list[WikiFileModel] = Field(default_factory=list)
    index_updates: IndexUpdateModel = Field(default_factory=IndexUpdateModel)
    log_entries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool handler — read_file (renamed from read_wiki_file)
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    """
    Read any wiki file by path.
    Tries path as given first, then relative to wiki/.
    Returns file contents or error message if not found.
    """
    try:
        requested = Path(path)
        base = Path(WIKI_DIR).resolve()

        candidate = requested if requested.is_absolute() else (base / requested)
        resolved = candidate.resolve()

        if not str(resolved).startswith(str(base)):
            return f"Access denied outside wiki directory: {path}"

        if resolved.exists() and resolved.is_file():
            return resolved.read_text(encoding="utf-8")

        if resolved.exists() and resolved.is_dir():
            return f"Path is a directory, not a file: {path}"

        return f"File does not exist: {path}"
    except Exception as e:
        return f"Error reading file {path}: {e}"


def _read_index() -> str:
    """
    Read current wiki/index.md contents.
    Empty string if not yet created — first module ingestion.
    Read fresh at call time — never cached.
    """
    index_path = Path(INDEX_PATH)
    if index_path.exists():
        text = index_path.read_text(encoding="utf-8")
        if len(text) > INDEX_PROMPT_CHAR_LIMIT:
            return (
                text[:INDEX_PROMPT_CHAR_LIMIT]
                + "\n\n[TRUNCATED: wiki index too large for full prompt context]"
            )
        return text
    return ""


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _build_slide_text_part(block: AssemblyBlock) -> dict:
    """
    Build text content part for NAVIGATION and TEXT_ONLY slides.
    speaker_notes is primary source.
    body_text only included if notes are empty.
    """
    lines = [
        f"[SLIDE {block.slide_number} | CATEGORY: {block.category}]",
        f"TITLE: {block.title}",
    ]
    if block.speaker_notes:
        lines.append(f"NOTES: {block.speaker_notes}")
    if block.body_text:
        lines.append(f"BODY: {block.body_text}")
    lines.append("---")
    return {"type": "text", "text": "\n".join(lines) + "\n"}


def _build_image_text_part(block: AssemblyBlock) -> dict:
    """
    Build text annotation part for IMAGE slides.
    Image path normalised to forward slashes.
    """
    image_path = block.image_path.replace("\\", "/")
    lines = [
        f"[SLIDE {block.slide_number} | CATEGORY: IMAGE | IMAGE PATH: {image_path}]",
        f"TITLE: {block.title}",
    ]
    if block.speaker_notes:
        lines.append(f"NOTES: {block.speaker_notes}")
    if block.body_text:
        lines.append(f"BODY: {block.body_text}")
    lines.append("---")
    return {"type": "text", "text": "\n".join(lines) + "\n"}


def _build_image_part(image_path: str) -> dict:
    """
    Build base64 image content part.
    Presence of image_url activates model vision capability.
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_b64}",
        },
    }


def _build_content_parts(assembly: Assembly, wiki_state: str) -> list[dict]:
    """
    Build the full multimodal message content parts list.
    [0] module context block (text)
    [1..N] one or two parts per slide in slide_number order
    """
    meta = assembly.module_metadata
    context_text = (
        "MODULE CONTEXT\n"
        "==============\n"
        f"ID: {meta.id}\n"
        f"Title: {meta.title}\n"
        f"Track: {meta.track}\n"
        f"Subtrack: {meta.subtrack}\n"
        f"Contact: {meta.contact}\n"
        f"Date: {meta.date}\n"
        "\n"
        "Current wiki state:\n"
        f"{wiki_state}\n"
        "==============\n"
    )
    parts: list[dict] = [{"type": "text", "text": context_text}]

    for block in assembly.blocks:
        if (block.category == "IMAGE"
                and block.image_path
                and Path(block.image_path).exists()):
            parts.append(_build_image_text_part(block))
            parts.append(_build_image_part(block.image_path))
        else:
            if block.category == "IMAGE" and block.image_path:
                _append_log(
                    f"warning | missing image file for slide {block.slide_number}: "
                    f"{block.image_path}"
                )
            parts.append(_build_slide_text_part(block))

    return parts


def _strip_image_parts(messages: list) -> None:
    """
    Remove base64 image parts from user messages in place.
    Images needed only for first reasoning pass.
    Draft content preserved in history via model_dump().
    Prevents re-uploading images on every tool call iteration.
    """
    for msg in messages:
        if (msg.get("role") == "user"
                and isinstance(msg.get("content"), list)):
            msg["content"] = [
                part for part in msg["content"]
                if not (
                    isinstance(part, dict)
                    and part.get("type") == "image_url"
                )
            ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip markdown code fences or surrounding prose."""
    if not raw:
        return ""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return text.strip()


def _save_debug_response(raw_response: Any, module_id: str) -> None:
    """Save raw response to debug file on parse failure."""
    debug_dir = Path(DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{module_id}_raw_response.txt"
    safe_text = "" if raw_response is None else str(raw_response)
    debug_path.write_text(safe_text, encoding="utf-8")


def _serialise_assistant_message(message: Any) -> dict:
    """Serialise only chat fields expected by most gateways."""
    tool_calls = []
    for tc in (getattr(message, "tool_calls", None) or []):
        tool_calls.append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        })

    payload: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", "") or "",
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _should_strip_images(message: Any) -> bool:
    """
    Strip images only after the model emits substantive text content.
    Tool-call-only turns must not cause early stripping.
    """
    content = getattr(message, "content", None)
    return isinstance(content, str) and bool(content.strip())


def _is_retryable_error(err: Exception) -> bool:
    """Retry transient network/gateway failures only."""
    status = getattr(err, "status_code", None)
    if isinstance(status, int):
        if status in (400, 401, 403, 404, 422):
            return False
        if 500 <= status < 600:
            return True

    text = str(err).lower()
    non_retryable_markers = [
        "invalid_request_error",
        "authentication",
        "unauthorized",
        "forbidden",
        "bad request",
        "schema",
        "validation",
    ]
    if any(marker in text for marker in non_retryable_markers):
        return False

    retryable_markers = [
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "rate limit",
        "service unavailable",
    ]
    return any(marker in text for marker in retryable_markers)


def _parse_envelope(
    raw_json: str,
    module_id: str,
    finish_reason: str = ""
) -> WikiEnvelope:
    """
    Parse and validate raw JSON string into WikiEnvelope.
    finish_reason=stop   → complete output
    finish_reason=length → truncated — increase WIKI_MAX_TOKENS
    All fields have pydantic defaults — no required field check needed.
    """
    if finish_reason == "length":
        _save_debug_response(raw_json, module_id)
        raise ValueError(
            f"Model response for module {module_id} was truncated "
            f"(finish_reason=length) — increase WIKI_MAX_TOKENS in "
            f"config.py (currently {WIKI_MAX_TOKENS}). "
            f"Raw response saved to {DEBUG_DIR}/{module_id}_raw_response.txt"
        )

    try:
        data = json.loads(_extract_json(raw_json))
    except json.JSONDecodeError as e:
        _save_debug_response(raw_json, module_id)
        raise ValueError(
            f"Model response was not valid JSON for module {module_id}. "
            f"Raw response saved to {DEBUG_DIR}/{module_id}_raw_response.txt"
        ) from e

    try:
        validated = WikiEnvelopeModel.model_validate(data)
    except ValidationError as e:
        _save_debug_response(raw_json, module_id)
        raise ValueError(
            f"Model response failed schema validation for module {module_id}. "
            f"Raw response saved to {DEBUG_DIR}/{module_id}_raw_response.txt"
        ) from e

    return WikiEnvelope(
        module_page=WikiFile(
            filename=validated.module_page.filename,
            content=validated.module_page.content,
        ),
        concept_pages=[
            WikiFile(filename=p.filename, content=p.content)
            for p in validated.concept_pages
        ],
        index_updates=IndexUpdate(
            modules=validated.index_updates.modules,
            concepts_added=validated.index_updates.concepts_added,
            concepts_updated=validated.index_updates.concepts_updated,
        ),
        log_entries=validated.log_entries,
    )


# ---------------------------------------------------------------------------
# API call kwargs builder
# ---------------------------------------------------------------------------

def _build_completion_kwargs(messages: list, tools: list) -> dict:
    """Assemble litellm.completion kwargs with auth and gateway routing."""
    kwargs = {
        "model":      WIKI_MODEL,
        "messages":   messages,
        "max_tokens": WIKI_MAX_TOKENS,
        "timeout":    API_TIMEOUT_SECONDS,
    }
    if tools:
        kwargs["tools"] = tools
    if API_KEY:
        kwargs["api_key"]  = API_KEY
    if BASE_URL:
        kwargs["api_base"] = BASE_URL
    return kwargs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_wiki(assembly: Assembly) -> tuple[WikiEnvelope, TokenUsage]:
    """
    Call the LLM to produce wiki pages from an Assembly.
    Returns (WikiEnvelope, TokenUsage).

    Vision: activated by image_url content parts — no flag needed.
    Tool:   read_file for concept extension and relationship verification.
    Images: stripped after first response — draft preserved in history.
    Limit:  TOOL_CALL_HARD_LIMIT total tool calls.

    finish_reason=stop   → complete output
    finish_reason=length → truncated — increase WIKI_MAX_TOKENS
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read any wiki file by path. TWO USES ONLY:\n"
                    "1. Before extending an existing concept page — "
                    "read current content to produce merged output. "
                    "Call read_file('wiki/concepts/{slug}.md') for "
                    "every slug confirmed in the Concepts table.\n"
                    "2. After completing wiki output — read up to 3 "
                    "candidate pages to verify relationships from "
                    "actual content not slug names.\n"
                    "Never call speculatively for slugs not in index."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "File path to read. "
                                "e.g. wiki/concepts/semantic_web_stack.md "
                                "or wiki/modules/00_05_02__rdf.md"
                            ),
                        }
                    },
                    "required": ["path"],
                },
            },
        }
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    final_finish_reason = ""
    final_tool_call_count = 0
    # Snapshot index once for this ingestion run; retries reuse it.
    wiki_state_snapshot = _read_index()

    for attempt in range(MAX_RETRIES + 1):
        content_parts = _build_content_parts(assembly, wiki_state_snapshot)
        messages = [
            {"role": "system", "content": WIKI_SYSTEM_PROMPT},
            {"role": "user",   "content": content_parts},
        ]

        tool_call_count = 0
        raw_json = ""
        finish_reason = ""
        images_stripped = False
        disable_tools = False
        extra_rounds_after_limit = 0

        try:
            while True:
                kwargs = _build_completion_kwargs(
                    messages,
                    [] if disable_tools else tools,
                )
                if disable_tools:
                    kwargs["tool_choice"] = "none"

                response = litellm.completion(**kwargs)
                message       = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                usage = getattr(response, "usage", None)
                total_input_tokens += getattr(usage, "prompt_tokens", 0) if usage else 0
                total_output_tokens += getattr(usage, "completion_tokens", 0) if usage else 0
                total_tokens += getattr(usage, "total_tokens", 0) if usage else 0

                if not images_stripped and _should_strip_images(message):
                    _strip_image_parts(messages)
                    images_stripped = True

                if message.tool_calls:
                    messages.append(_serialise_assistant_message(message))

                    for tool_call in message.tool_calls:
                        tool_call_count += 1

                        if tool_call_count > TOOL_CALL_HARD_LIMIT:
                            disable_tools = True
                            extra_rounds_after_limit += 1
                            messages.append({
                                "role":        "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    "Tool call limit reached. "
                                    "Produce final JSON output now."
                                ),
                            })
                            if extra_rounds_after_limit > MAX_EXTRA_ROUNDS_AFTER_TOOL_LIMIT:
                                raise RuntimeError(
                                    "Model kept requesting tools after tool-call limit. "
                                    "Aborting to prevent infinite loop."
                                )
                            continue

                        try:
                            if tool_call.function.name == "read_file":
                                args = json.loads(tool_call.function.arguments or "{}")
                                if not isinstance(args, dict) or "path" not in args:
                                    result = "Tool error: missing required argument 'path'."
                                else:
                                    result = _read_file(str(args["path"]))
                            else:
                                result = f"Unknown tool: {tool_call.function.name}"
                        except Exception as e:
                            result = f"Tool error in {tool_call.function.name}: {e}"

                        messages.append({
                            "role":        "tool",
                            "tool_call_id": tool_call.id,
                            "content":     result,
                        })

                else:
                    raw_json = message.content or ""
                    break

            envelope = _parse_envelope(raw_json, assembly.module_id, finish_reason)
            final_finish_reason = finish_reason
            final_tool_call_count = tool_call_count

            token_usage = TokenUsage(
                module_id=assembly.module_id,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_tokens,
                finish_reason=final_finish_reason,
                tool_calls_made=final_tool_call_count,
            )

            if LOG_TOKEN_USAGE:
                _append_log(
                    f"{assembly.module_id} — "
                    f"input: {total_input_tokens} tokens, "
                    f"output: {total_output_tokens} tokens, "
                    f"finish: {final_finish_reason}, "
                    f"tool_calls: {final_tool_call_count}"
                )

            return envelope, token_usage

        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable_error(e):
                backoff = RETRY_DELAY_SECONDS * (2 ** attempt)
                time.sleep(backoff)
                continue
            raise RuntimeError(
                f"LLM API call failed for module {assembly.module_id} "
                f"after {MAX_RETRIES + 1} attempts: {e}"
            ) from e

    raise RuntimeError(
        f"LLM API call failed for module {assembly.module_id} after retries."
    )