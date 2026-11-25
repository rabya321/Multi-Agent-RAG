"""
run_testset.py
========================================================
Runs the existing agent on all test questions
and saves generated answers for later RAGAS evaluation.
Also saves formula score analytics CSV/Excel.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# ─────────────────────────────────────────────────────────
# IMPORT YOUR EXISTING AGENT
# ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rag.run       import LiteLLMEmbedder, run_query
from agent_rag.retrieval import set_embedder, CONFIDENCE_THRESHOLD
from agent_rag.graph     import build_graph

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
SLEEP_BETWEEN = 0.5   # seconds between questions


# ─────────────────────────────────────────────────────────
# ENVIRONMENT VALIDATION
# ─────────────────────────────────────────────────────────
def validate_env():
    missing = [v for v in ("API_KEY", "CERT_PATH") if not os.getenv(v)]
    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}\n"
            "Windows: $env:API_KEY='...' ; $env:CERT_PATH='...'"
        )
    cert = Path(os.getenv("CERT_PATH", ""))
    if not cert.exists():
        raise FileNotFoundError(f"Certificate not found: {cert}")


# ─────────────────────────────────────────────────────────
# SAVE AND LOAD RESULTS
# ─────────────────────────────────────────────────────────
def load_existing_results(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        existing = {
            r["question_id"]: r
            for r in data.get("results", [])
            if r.get("question_id")
        }
        print(f"  [RESUME] {len(existing)} existing results found.")
        return existing
    except Exception as e:
        print(f"  [RESUME] Could not load: {e}")
        return {}


def save_results(results: list, output_path: Path, total_qs: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at"   : datetime.now().isoformat(),
                "total_questions": total_qs,
                "total_generated": len(results),
                "results"        : results,
            },
            f, indent=2, ensure_ascii=False,
        )


# ─────────────────────────────────────────────────────────
# BUILD RESULT RECORD
# ─────────────────────────────────────────────────────────
def _build_result(q: dict, response: dict, idx: int) -> dict:
    q_id   = q.get("question_id", f"Q{idx:04d}")
    q_type = q.get("question_type", "")
    flags  = response.get("quality_flags", {})

    return {
        # ── From test set ──────────────────────────────────
        "question_id"        : q_id,
        "question"           : q.get("question",        ""),
        "question_type"      : q_type,
        "expected_answer"    : q.get("expected_answer",  ""),
        "expected_winner"    : q.get("expected_winner",  ""),
        "source_titles"      : q.get("source_titles",   []),

        # ── From agent response ───────────────────────────
        "generated_answer"   : response.get("answer",           ""),
        "retrieval_path"     : response.get("retrieval_path",   ""),
        "formula_score"      : response.get("formula_score",    0.0),
        "query_type_pred"    : response.get("query_type",       ""),
        "context_tokens"     : response.get("context_tokens",    0),

        "answer_flagged"     : flags.get("flagged",          False),
        "confidence"         : flags.get("confidence",       "LOW"),
        "completeness"       : flags.get("completeness",  "PARTIAL"),

        # ── Source citation fields ─────────────────────────
        "source_title"       : response.get("source_title",     ""),
        "source_page"        : response.get("source_page",    None),
        "source_section"     : response.get("source_section",   ""),
        "citation_string"    : response.get("citation_string",  ""),

        # ── Retrieval details ─────────────────────────────
        "retrieved_contexts" : response.get(
                                   "retrieved_contexts",
                                   response.get("retrieved_context", [])
                               ),
        "retrieval_scores"   : response.get("retrieval_scores", []),
        "source_ids"         : response.get("source_ids",       []),

        # ── Timestamp ─────────────────────────────────────
        "generated_at"       : datetime.now().isoformat(),
    }


def _build_error_result(q: dict, error: Exception, idx: int) -> dict:
    q_id   = q.get("question_id", f"Q{idx:04d}")
    q_type = q.get("question_type", "")
    return {
        "question_id"        : q_id,
        "question"           : q.get("question",       ""),
        "question_type"      : q_type,
        "expected_answer"    : q.get("expected_answer", ""),
        "expected_winner"    : q.get("expected_winner", ""),
        "source_titles"      : q.get("source_titles",  []),
        "generated_answer"   : f"[ERROR] {error}",
        "retrieval_path"     : "error",
        "formula_score"      : 0.0,
        "query_type_pred"    : q_type,
        "context_tokens"     : 0,

        "answer_flagged"     : True,
        "confidence"         : "LOW",
        "completeness"       : "EMPTY",
        "source_title"       : "",
        "source_page"        : None,
        "source_section"     : "",
        "citation_string"    : "",
        "retrieved_contexts" : [],
        "retrieval_scores"   : [],
        "source_ids"         : [],
        "generated_at"       : datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────
# ANALYTICS CSV / EXCEL
# ─────────────────────────────────────────────────────────
def save_analytics(results: list, output_dir: Path) -> None:
    """
    Saves two analytics files:
    1. formula_score_by_query_type.csv  — aggregated per type
    2. per_question_summary.csv         — one row per question
    3. pipeline_analytics.xlsx          — both as Excel sheets

    Columns in formula_score_by_query_type:
      question_type, count,
      formula_score_mean, formula_score_std,
      formula_score_min,  formula_score_max,
      formula_score_median,
      above_threshold_count, above_threshold_pct,
      direct_path_count,  fallback_count, degraded_count,
      flagged_count,      flagged_pct,
      completeness_empty_count, completeness_empty_pct,
      classification_correct_count, classification_accuracy
    """
    if not results:
        return

    # ── Sheet 1: Per-question summary ─────────────────────
    per_q_rows = []
    for r in results:
        path = str(r.get("retrieval_path", "")).lower()
        per_q_rows.append({
            "question_id"          : r.get("question_id",       ""),
            "question_type"        : r.get("question_type",     ""),
            "query_type_pred"      : r.get("query_type_pred",   ""),
            "classification_correct": (
                r.get("question_type", "") ==
                r.get("query_type_pred", "")
            ),
            "formula_score"        : r.get("formula_score",     0.0),
            "above_threshold"      : r.get("formula_score", 0.0)
                                      >= CONFIDENCE_THRESHOLD,
            "retrieval_path"       : r.get("retrieval_path",    ""),
            "is_direct"            : "direct"   in path,
            "is_fallback"          : "fallback" in path,
            "is_degraded"          : "degraded" in path,
            "confidence"           : r.get("confidence",        ""),
            "completeness"         : r.get("completeness",      ""),
            "answer_flagged"       : r.get("answer_flagged",  False),
            "question"             : r.get("question",          "")[:80],
        })

    per_q_df = pd.DataFrame(per_q_rows)

    # ── Sheet 2: Aggregated by query type ─────────────────
    type_rows = []
    query_types = sorted(per_q_df["question_type"].unique())

    for qt in query_types:
        subset = per_q_df[per_q_df["question_type"] == qt]
        scores = subset["formula_score"]
        n      = len(subset)

        type_rows.append({
            "question_type"              : qt,
            "count"                      : n,

            # Formula score stats
            "formula_score_mean"         : round(scores.mean(),   4),
            "formula_score_std"          : round(scores.std(),    4) if n > 1 else 0.0,
            "formula_score_min"          : round(scores.min(),    4),
            "formula_score_max"          : round(scores.max(),    4),
            "formula_score_median"       : round(scores.median(), 4),

            # Threshold analysis
            "above_threshold_count"      : int(subset["above_threshold"].sum()),
            "above_threshold_pct"        : round(
                subset["above_threshold"].mean() * 100, 1
            ),

            # Retrieval path breakdown
            "direct_count"               : int(subset["is_direct"].sum()),
            "fallback_count"             : int(subset["is_fallback"].sum()),
            "degraded_count"             : int(subset["is_degraded"].sum()),

            # Answer quality
            "flagged_count"              : int(subset["answer_flagged"].sum()),
            "flagged_pct"                : round(
                subset["answer_flagged"].mean() * 100, 1
            ),
            "completeness_empty_count"   : int(
                (subset["completeness"] == "EMPTY").sum()
            ),
            "completeness_empty_pct"     : round(
                (subset["completeness"] == "EMPTY").mean() * 100, 1
            ),

            # Classification accuracy
            "classification_correct"     : int(
                subset["classification_correct"].sum()
            ),
            "classification_accuracy_pct": round(
                subset["classification_correct"].mean() * 100, 1
            ),
        })

    # Add TOTAL row
    scores_all = per_q_df["formula_score"]
    type_rows.append({
        "question_type"              : "TOTAL",
        "count"                      : len(per_q_df),
        "formula_score_mean"         : round(scores_all.mean(),   4),
        "formula_score_std"          : round(scores_all.std(),    4),
        "formula_score_min"          : round(scores_all.min(),    4),
        "formula_score_max"          : round(scores_all.max(),    4),
        "formula_score_median"       : round(scores_all.median(), 4),
        "above_threshold_count"      : int(per_q_df["above_threshold"].sum()),
        "above_threshold_pct"        : round(
            per_q_df["above_threshold"].mean() * 100, 1
        ),
        "direct_count"               : int(per_q_df["is_direct"].sum()),
        "fallback_count"             : int(per_q_df["is_fallback"].sum()),
        "degraded_count"             : int(per_q_df["is_degraded"].sum()),
        "flagged_count"              : int(per_q_df["answer_flagged"].sum()),
        "flagged_pct"                : round(
            per_q_df["answer_flagged"].mean() * 100, 1
        ),
        "completeness_empty_count"   : int(
            (per_q_df["completeness"] == "EMPTY").sum()
        ),
        "completeness_empty_pct"     : round(
            (per_q_df["completeness"] == "EMPTY").mean() * 100, 1
        ),
        "classification_correct"     : int(
            per_q_df["classification_correct"].sum()
        ),
        "classification_accuracy_pct": round(
            per_q_df["classification_correct"].mean() * 100, 1
        ),
    })

    type_df = pd.DataFrame(type_rows)

    # ── Save CSV ───────────────────────────────────────────
    type_csv_path  = output_dir / "formula_score_by_query_type.csv"
    per_q_csv_path = output_dir / "per_question_summary.csv"

    type_df.to_csv(type_csv_path,  index=False, encoding="utf-8")
    per_q_df.to_csv(per_q_csv_path, index=False, encoding="utf-8")

    print(f"  [ANALYTICS] Formula score by type : {type_csv_path}")
    print(f"  [ANALYTICS] Per-question summary  : {per_q_csv_path}")

    # ── Save Excel (both sheets) ───────────────────────────
    xlsx_path = output_dir / "pipeline_analytics.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            type_df.to_excel(
                writer,
                sheet_name="Formula Score by Type",
                index=False,
            )
            per_q_df.to_excel(
                writer,
                sheet_name="Per Question Summary",
                index=False,
            )
        print(f"  [ANALYTICS] Excel report          : {xlsx_path}")
    except ImportError:
        print(
            "  [ANALYTICS] openpyxl not installed — "
            "Excel skipped. CSVs saved. "
            "Install with: pip install openpyxl"
        )


# ─────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────
def run_on_testset(
    questions:   list,
    output_path: Path,
    graph,
    overwrite:   bool,
):
    existing    = {} if overwrite else load_existing_results(output_path)
    total       = len(questions)
    done        = 0
    errors      = 0
    all_results = list(existing.values())
    start_time  = time.time()

    print(f"\n  Processing {total} questions...")
    print(f"  Resume mode  : {not overwrite}")
    print(f"  Already done : {len(existing)}")

    for idx, q in enumerate(questions, 1):
        q_id   = q.get("question_id", f"Q{idx:04d}")
        q_type = q.get("question_type", "")

        if q_id in existing:
            continue

        print(
            f"\n  [{idx:>4}/{total}] "
            f"{q_type:<30} "
            f"{q.get('question','')[:60]}"
        )

        try:
            response = run_query(q.get("question", ""), graph)
            result   = _build_result(q, response, idx)
        except Exception as e:
            print(f"    [ERROR] {e}")
            errors += 1
            result  = _build_error_result(q, e, idx)

        all_results.append(result)
        done += 1

        print(
            f"    path={result['retrieval_path']} | "
            f"score={result['formula_score']:.3f} | "
            f"ctx_tokens={result['context_tokens']} | "
            f"flagged={result['answer_flagged']}"
        )

        if done <= 5 and q_type != "out_of_scope":
            print(f"    preview: {result['generated_answer'][:150]}")

        if done % 10 == 0:
            save_results(all_results, output_path, total)
            elapsed   = time.time() - start_time
            per_q     = elapsed / done
            remaining = (total - len(all_results)) * per_q
            print(
                f"\n  [CHECKPOINT] {done} new saved. "
                f"~{remaining/60:.1f} min remaining.\n"
            )

        if idx < total:
            time.sleep(SLEEP_BETWEEN)

    save_results(all_results, output_path, total)
    return done, errors, all_results


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run existing agent_rag pipeline on test set. "
            "Saves answers + analytics for RAGAS evaluation."
        )
    )
    parser.add_argument("--test_set",   required=True)
    parser.add_argument("--output_dir", default="Evaluation")
    parser.add_argument("--sample",     type=int, default=None)
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    validate_env()

    with open(args.test_set, "r", encoding="utf-8") as f:
        questions = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        questions   = questions[:args.sample]
        output_path = out_dir / "pipeline_results_sample.json"
        print(f"\n[SAMPLE MODE] {args.sample} questions.")
    else:
        output_path = out_dir / "pipeline_results.json"

    print(f"\n{'='*60}")
    print(f"RUN TESTSET ON AGENT")
    print(f"{'='*60}")
    print(f"  Test set    : {args.test_set}")
    print(f"  Questions   : {len(questions)}")
    print(f"  Output      : {output_path}")
    print(f"  Overwrite   : {args.overwrite}")

    print("\n  Loading embedder...")
    embedder = LiteLLMEmbedder()
    set_embedder(embedder)
    print("  Building graph...")
    graph = build_graph()
    print("  Ready.\n")

    start_time = time.time()
    done, errors, all_results = run_on_testset(
        questions   = questions,
        output_path = output_path,
        graph       = graph,
        overwrite   = args.overwrite,
    )
    elapsed = time.time() - start_time

    # ── Save analytics ─────────────────────────────────────
    print(f"\n  Saving analytics...")
    save_analytics(all_results, out_dir)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"  Generated : {done}")
    print(f"  Errors    : {errors}")
    print(f"  Time      : {elapsed/60:.1f} minutes")
    print(f"  Saved to  : {output_path}")
    print(f"\n  Analytics files:")
    print(f"    {out_dir}/formula_score_by_query_type.csv")
    print(f"    {out_dir}/per_question_summary.csv")
    print(f"    {out_dir}/pipeline_analytics.xlsx")
    print(f"\n  Next step:")
    print(f"    python ragas_evaluate.py")
    print(f"      --input  {output_path}")
    print(f"      --output {args.output_dir}/evaluation_results")


if __name__ == "__main__":
    main()