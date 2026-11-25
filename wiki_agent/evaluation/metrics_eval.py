import json
import sys
import time
import types
from pathlib import Path
import argparse
import pandas as pd
import tiktoken
from datetime import datetime, timezone

# ragas==0.4.3 unconditionally imports `ChatVertexAI` from
# `langchain_community.chat_models.vertexai`, a module removed in
# langchain-community>=0.3. We don't use Vertex AI, so inject a stub
# module satisfying the import instead of patching the installed
# package (which would be lost on every reinstall/upgrade).
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerCorrectness, ContextRecall
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import EVAL_MODEL, API_KEY, BASE_URL, CERT_PATH
import httpx

METRIC_COLS = [
    "faithfulness",
    "answer_correctness",
    "context_recall",
]
# Cost/size descriptor (not a quality metric).
COST_COLS = ["context_tokens"]


def _build_token_encoder():
    """Use model-matched tokenizer when available, with stable fallback."""
    try:
        return tiktoken.encoding_for_model(EVAL_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


TOKEN_ENCODER = _build_token_encoder()


def _token_count(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_ENCODER.encode(text))
# -------------------------
# RAGAS LLM
# -------------------------

def build_ragas_llm():
    """
    RAGAS requires its own LLM instance wrapped in LangchainLLMWrapper.
    The wrapper adapts LangChain's ChatOpenAI to RAGAS's internal
    BaseRagasLLM interface. Needed because we use a custom base_url
    and CERT_PATH that LangChain's ChatOpenAI handles correctly.

    ragas.evaluate() runs everything through asyncio, and ChatOpenAI
    builds separate sync/async OpenAI clients. Only setting http_client
    (sync) leaves the async client with default SSL verification, which
    fails against our custom-CA endpoint. Both must be set explicitly.
    """
    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    http_async_client = httpx.AsyncClient(verify=CERT_PATH) if CERT_PATH else None
    llm = ChatOpenAI(
        model=EVAL_MODEL,
        temperature=0.0,
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
        http_async_client=http_async_client,
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings():
    """
    Embeddings required by AnswerCorrectness.
    Uses the same embedding model as the naive RAG baseline.
    """
    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    http_async_client = httpx.AsyncClient(verify=CERT_PATH) if CERT_PATH else None
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
        http_async_client=http_async_client,
    )
    return LangchainEmbeddingsWrapper(embeddings)


# -------------------------
# LOAD RESULTS
# -------------------------

def load_results(path: str) -> list[dict]:
    """Load agent output records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# -------------------------
# BUILD RAGAS DATASET
# -------------------------

def build_ragas_dataset(records: list[dict]) -> tuple[EvaluationDataset, list[dict]]:
    """
    Convert agent output records into a RAGAS-compatible dataset.

    RAGAS requires four fields per sample:
      user_input         — the query
      response           — the generated answer prose
      retrieved_contexts — list of retrieved text chunks
      reference          — the reference answer prose

    Mapping:
      user_input         = record["question"]
      response           = record["answer"]
      retrieved_contexts = record["retrieved_contexts"]  (full text the
                           generator saw: wiki pages read by the agent,
                           or top-k chunks for naive RAG).
      reference          = record["ground_truth"]["answer"]

    Records with errors or missing required fields are skipped.
    Returns the dataset and metadata list for reassembly after eval.
    """
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []
    meta          = []
    skipped       = 0

    for record in records:
        if record.get("error"):
            skipped += 1
            continue

        question     = record.get("question", "")
        answer       = record.get("answer", "")
        retrieved    = record.get("retrieved_contexts") or [""]
        gt_answer    = record.get("ground_truth", {}).get("answer", "")

        if not question or not answer or not gt_answer:
            skipped += 1
            continue

        questions.append(question)
        answers.append(answer)
        contexts.append(list(retrieved))
        ground_truths.append(gt_answer)
        row_meta = {
            "id":            record.get("id"),
            "type":          record.get("type"),
            "evaluation_mode": record.get("evaluation_mode"),
            "track":         record.get("track"),
            "context_tokens": sum(_token_count(c) for c in contexts[-1]),
        }
        meta.append(row_meta)

    if skipped:
        print(f"Skipped {skipped} records with errors or missing fields.")

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=g,
        )
        for q, a, c, g in zip(questions, answers, contexts, ground_truths)
    ]
    dataset = EvaluationDataset(samples=samples)
    return dataset, meta


# -------------------------
# CONFIGURE METRICS
# -------------------------

def build_metrics(
    ragas_llm,
    ragas_embeddings,
) -> list:
    """
    Configure RAGAS metrics.

        Answer Correctness:
            RAGAS metric combining claim agreement and semantic alignment.

    Context Recall:
      No change. GT answer sentences attributed to retrieved_contexts.

    Faithfulness:
      No change. Answer claims checked against retrieved_contexts.
    """
    return [
        Faithfulness(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextRecall(llm=ragas_llm),
    ]


# -------------------------
# PRINT SUMMARY
# -------------------------

def _print_rows(group: pd.DataFrame) -> None:
    """Print mean/min/max for quality metrics and cost columns."""
    for col in METRIC_COLS:
        if col in group.columns:
            print(
                f"  {col:<25} "
                f"{group[col].mean():>7.3f} "
                f"{group[col].min():>7.3f} "
                f"{group[col].max():>7.3f}"
            )
    for col in COST_COLS:
        if col in group.columns and group[col].notna().any():
            print(
                f"  {col:<25} "
                f"{group[col].mean():>7.0f} "
                f"{group[col].min():>7.0f} "
                f"{group[col].max():>7.0f}"
            )


def print_summary(df: pd.DataFrame) -> None:
    """Print overall and per-question-type metric summaries."""

    # overall
    print("\n" + "=" * 50)
    print(f"  OVERALL  (n={len(df)})")
    print("=" * 50)
    print(f"  {'METRIC':<25} {'MEAN':>7} {'MIN':>7} {'MAX':>7}")
    print("-" * 50)
    _print_rows(df)

    if "evaluation_mode" in df.columns and df["evaluation_mode"].notna().any():
        print("\n" + "=" * 50)
        print("  BY EVALUATION MODE")
        print("=" * 50)
        for mode in ["within_track", "cross_track"]:
            group = df[df["evaluation_mode"] == mode]
            if group.empty:
                continue
            print(f"\n  {mode}  (n={len(group)})")
            print(f"  {'METRIC':<25} {'MEAN':>7} {'MIN':>7} {'MAX':>7}")
            print("  " + "-" * 46)
            _print_rows(group)

    if "type" not in df.columns:
        print("\n" + "=" * 50)
        return

    type_order = ["multi_hop", "comparative_synthesis", "index_navigation"]
    print("\n" + "=" * 50)
    print("  BY QUESTION TYPE")
    print("=" * 50)

    remaining = [t for t in df["type"].unique() if t not in type_order]
    for q_type in type_order + remaining:
        group = df[df["type"] == q_type]
        if group.empty:
            continue
        print(f"\n  {q_type}  (n={len(group)})")
        print(f"  {'METRIC':<25} {'MEAN':>7} {'MIN':>7} {'MAX':>7}")
        print("  " + "-" * 46)
        _print_rows(group)

    print("\n" + "=" * 50)


def _summary_table(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    cols = [c for c in METRIC_COLS + COST_COLS if c in df.columns]
    if group_col is None:
        return pd.DataFrame([{"group": "overall", "n": len(df), **{c: df[c].mean() for c in cols}}])

    rows = []
    for group_value, group_df in df.groupby(group_col, dropna=False):
        rows.append({
            "group": group_value,
            "n": len(group_df),
            **{c: group_df[c].mean() for c in cols},
        })
    return pd.DataFrame(rows)


def _markdown_table(table: pd.DataFrame) -> str:
    """Render summary tables without pandas' optional tabulate dependency."""
    def cell(value) -> str:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [list(table.columns), ["---"] * len(table.columns)]
    rows.extend(list(row) for row in table.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)


def _write_summary_tables(df: pd.DataFrame, out_path: Path) -> None:
    overall = _summary_table(df)
    by_type = _summary_table(df, "type") if "type" in df.columns else pd.DataFrame()
    by_mode = _summary_table(df, "evaluation_mode") if "evaluation_mode" in df.columns else pd.DataFrame()

    summary_path = out_path.with_name(out_path.stem + "_summary.md")
    sections = ["# Evaluation Summary\n"]
    sections.append("## Overall\n")
    sections.append(_markdown_table(overall))
    if not by_type.empty:
        sections.append("\n\n## By Question Type\n")
        sections.append(_markdown_table(by_type))
    if not by_mode.empty:
        sections.append("\n\n## By Evaluation Mode\n")
        sections.append(_markdown_table(by_mode))
    summary_path.write_text("".join(sections), encoding="utf-8")
    print(f"Summary tables written to {summary_path}")


# -------------------------
# EVALUATE
# -------------------------

def run_evaluation(
    results_path: str,
    output_path: str | None = None,
) -> None:
    """
    Run RAGAS evaluation on agent output records.

    Args:
        results_path: path to JSONL file from run_batch
        output_path:  where to write evaluation results JSON
    """
    print(f"Loading results from {results_path}...")
    records = load_results(results_path)
    print(f"Loaded {len(records)} records.")

    dataset, meta = build_ragas_dataset(records)
    print(f"Built dataset with {len(dataset)} evaluable samples.")

    if len(dataset) == 0:
        print("No evaluable samples found. Check your results file.")
        return

    print("Building RAGAS LLM...")
    ragas_llm        = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()
    metrics          = build_metrics(ragas_llm, ragas_embeddings)

    print("Running RAGAS evaluation...")
    # Align concurrency/retry/timeout settings with papers_agent evaluation.
    run_config = RunConfig(
        max_workers=4,
        max_retries=3,
        timeout=300,
    )
    eval_started_at = datetime.now(timezone.utc)
    eval_start_monotonic = time.perf_counter()
    results    = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )
    eval_finished_at = datetime.now(timezone.utc)
    latency_ms = int((time.perf_counter() - eval_start_monotonic) * 1000)

    df = results.to_pandas()

    # attach metadata
    for key in ["id", "type", "evaluation_mode", "track"] + COST_COLS:
        df[key] = [m[key] for m in meta]
    df["eval_started_at_utc"] = eval_started_at.isoformat()
    df["eval_finished_at_utc"] = eval_finished_at.isoformat()
    df["eval_latency_ms"] = latency_ms

    # reorder columns
    front_cols = ["id", "type", "evaluation_mode", "track", "question"]
    lead_cols  = front_cols + METRIC_COLS + COST_COLS
    other_cols = [c for c in df.columns if c not in lead_cols]
    df = df[[c for c in lead_cols + other_cols if c in df.columns]]

    if output_path is None:
        stem     = Path(results_path).stem
        out_path = Path(
            f"ragas_eval_{stem}_{int(datetime.now(timezone.utc).timestamp())}.json"
        )
    else:
        out_path = Path(output_path)

    df.to_json(out_path, orient="records", indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Evaluation started at (UTC): {eval_started_at.isoformat()}")
    print(f"Evaluation finished at (UTC): {eval_finished_at.isoformat()}")
    print(f"Evaluation latency: {latency_ms / 1000:.1f}s")

    _write_summary_tables(df, out_path)

    print_summary(df)


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for wiki retrieval agent outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Evaluate combined results:
        uv run python -m results.metrics_eval --results all_results.jsonl

  Evaluate standard results only:
        uv run python -m results.metrics_eval --results standard_results.jsonl

  Save to specific file:
        uv run python -m results.metrics_eval --results all_results.jsonl --output eval_report.json
        """,
    )
    parser.add_argument(
        "--results", "-r",
        type=str,
        required=True,
        help="Path to JSONL file produced by run_batch.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=(
            "Output JSON file for evaluation results. "
            "Auto-generated with timestamp if not set."
        ),
    )

    args = parser.parse_args()
    run_evaluation(
        results_path=args.results,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()