import sys
import types
import os
import warnings
import json
import time
import asyncio
import logging
import argparse
from typing import Optional, List, Tuple
from datetime import datetime

import pandas as pd
import litellm
from litellm import embedding as litellm_embedding

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


def _stub_missing_langchain_module(module_path: str, class_names: list[str]) -> None:
    if module_path in sys.modules:
        return
    fake_module = types.ModuleType(module_path)
    for class_name in class_names:
        setattr(fake_module, class_name, type(class_name, (), {}))
    sys.modules[module_path] = fake_module


_stub_missing_langchain_module(
    "langchain_community.chat_models.vertexai",
    ["ChatVertexAI"],
)

from ragas import evaluate
from ragas.metrics import answer_correctness, context_recall, faithfulness
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from langchain_core.outputs import LLMResult, Generation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
for noisy_logger in ["litellm", "LiteLLM", "httpx", "httpcore", "openai", "urllib3"]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
CERT_PATH = os.getenv("CERT_PATH")
EVAL_MODEL = "openai/gpt-5-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

DEFAULT_INPUT_JSON = "Evaluation/pipeline_results.json"
DEFAULT_OUTPUT_DIR = "Evaluation/evaluation_results"

TEST_RUN = False
TEST_RUN_SIZE = 5

FIELD_MAP = {
    "answer": "generated_answer",
    "ground_truth": "expected_answer",
    "contexts": "retrieved_contexts",
}


def call_llm(system: str, user: str, model: str = EVAL_MODEL, max_retries: int = 2) -> str:
    litellm.ssl_verify = CERT_PATH
    for attempt in range(max_retries + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                api_base=BASE_URL,
                api_key=API_KEY,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                log.warning(
                    "LLM call failed (attempt %s), retrying in %ss: %s",
                    attempt + 1,
                    wait,
                    exc,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"LLM call failed after {max_retries + 1} attempts: {exc}"
            ) from exc


class LiteLLMWrapper(BaseRagasLLM):
    def __init__(self, run_config: RunConfig = None):
        super().__init__()
        self.run_config = run_config or RunConfig()

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def _extract_system_and_user(self, prompt) -> Tuple[str, str]:
        system_msg = ""
        user_msg = ""
        if hasattr(prompt, "to_messages"):
            for msg in prompt.to_messages():
                msg_type = getattr(msg, "type", None)
                if msg_type == "system":
                    system_msg = msg.content
                elif msg_type in ("human", "user"):
                    user_msg = msg.content
        if not user_msg:
            user_msg = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        return system_msg, user_msg

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        system_msg, user_msg = self._extract_system_and_user(prompt)
        text = call_llm(
            system=system_msg or "You are a helpful evaluation assistant.",
            user=user_msg,
        )
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text,
            prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )


class LiteLLMEmbeddings(BaseRagasEmbeddings):
    def __init__(self, run_config: RunConfig = None):
        super().__init__()
        self.run_config = run_config or RunConfig()

    def embed_query(self, text: str) -> list[float]:
        litellm.ssl_verify = CERT_PATH
        response = litellm_embedding(
            model=EMBEDDING_MODEL,
            input=[text],
            api_base=BASE_URL,
            api_key=API_KEY,
        )
        return response.data[0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        litellm.ssl_verify = CERT_PATH
        response = litellm_embedding(
            model=EMBEDDING_MODEL,
            input=texts,
            api_base=BASE_URL,
            api_key=API_KEY,
        )
        return [item["embedding"] for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def load_and_validate(filepath: str) -> list[dict]:
    log.info("Loading data from: %s", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        data = raw.get("results", [])
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError(f"Unexpected JSON structure: {type(raw)}")

    for item in data:
        for script_name, json_name in FIELD_MAP.items():
            if json_name in item and script_name not in item:
                item[script_name] = item[json_name]

    required_fields = ["question", "answer", "ground_truth"]
    errors = []
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                errors.append(f"Row {i}: missing required field '{field}'")
    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))

    log.info("Loaded %d questions successfully.", len(data))
    return data


def build_ragas_dataset(data: list[dict]) -> EvaluationDataset:
    # context_recall uses the reference answer and retrieved_contexts.
    # No separate reference-context extraction step is needed.
    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            reference=item["ground_truth"],
            retrieved_contexts=item.get("retrieved_contexts", []),
        )
        for item in data
    ]
    log.info("Built RAGAS dataset with %d samples.", len(samples))
    return EvaluationDataset(samples=samples)


def build_per_question_csv(data: list[dict]) -> pd.DataFrame:
    rows = []
    for item in data:
        rows.append(
            {
                "question_id": item.get("question_id", ""),
                "question": item.get("question", ""),
                "question_type": item.get("question_type", "N/A"),
                "generated_answer": item.get("answer", ""),
                "expected_answer": item.get("ground_truth", ""),
                "answer_correctness": item.get("ragas_answer_correctness", None),
                "context_recall": item.get("ragas_context_recall", None),
                "faithfulness": item.get("ragas_faithfulness", None),
                "retrieval_path": item.get("retrieval_path", "N/A"),
                "confidence": item.get("confidence", "N/A"),
                "completeness": item.get("completeness", "N/A"),
                "answer_tokens": item.get("answer_tokens", None),
                "context_tokens": item.get("context_tokens", None),
            }
        )
    return pd.DataFrame(rows)


def build_query_type_breakdown(data: list[dict]) -> pd.DataFrame:
    rows = []
    question_types = sorted(set(item.get("question_type", "unknown") for item in data))

    for qtype in question_types:
        subset = [d for d in data if d.get("question_type", "unknown") == qtype]

        def _mean(field: str):
            vals = [d[field] for d in subset if d.get(field) is not None]
            return round(pd.Series(vals).mean(), 4) if vals else None

        rows.append(
            {
                "question_type": qtype,
                "n_questions": len(subset),
                "answer_correctness_mean": _mean("ragas_answer_correctness"),
                "context_recall_mean": _mean("ragas_context_recall"),
                "faithfulness_mean": _mean("ragas_faithfulness"),
            }
        )

    def _overall(field: str):
        vals = [d[field] for d in data if d.get(field) is not None]
        return round(pd.Series(vals).mean(), 4) if vals else None

    rows.append(
        {
            "question_type": "TOTAL",
            "n_questions": len(data),
            "answer_correctness_mean": _overall("ragas_answer_correctness"),
            "context_recall_mean": _overall("ragas_context_recall"),
            "faithfulness_mean": _overall("ragas_faithfulness"),
        }
    )

    return pd.DataFrame(rows)


def build_summary_csv(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["answer_correctness", "context_recall", "faithfulness"]:
        if metric in results_df.columns:
            scores = results_df[metric].dropna()
            if len(scores) > 0:
                rows.append(
                    {
                        "metric": metric,
                        "score_mean": round(float(scores.mean()), 4),
                        "n_questions": len(scores),
                    }
                )
    return pd.DataFrame(rows)


def save_excel(
    per_question_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    query_type_df: pd.DataFrame,
    output_dir: str,
    run_tag: str,
) -> str:
    xlsx_path = os.path.join(output_dir, f"evaluation_{run_tag}.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            query_type_df.to_excel(writer, sheet_name="By Query Type", index=False)
            per_question_df.to_excel(writer, sheet_name="Per Question", index=False)
        log.info("Excel saved: %s", xlsx_path)
        return xlsx_path
    except ImportError:
        log.warning("openpyxl not installed - Excel skipped. Install: uv add openpyxl")
        return ""


def run_evaluation(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    run_prefix = f"TEST{TEST_RUN_SIZE}_" if TEST_RUN else ""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{run_prefix}{run_timestamp}"

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION")
    print("=" * 60)
    print(f"  Input file  : {input_path}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Eval model  : {EVAL_MODEL}")
    print(f"  Embed model : {EMBEDDING_MODEL}")

    data = load_and_validate(input_path)

    if TEST_RUN:
        total_available = len(data)
        data = data[:TEST_RUN_SIZE]
        print(f"  TEST MODE: Using {len(data)} of {total_available} questions")
    else:
        print(f"  Running on all {len(data)} questions")

    ragas_dataset = build_ragas_dataset(data)

    eval_run_config = RunConfig(timeout=300, max_workers=4, max_retries=3)
    ragas_llm = LiteLLMWrapper(run_config=eval_run_config)
    ragas_embeddings = LiteLLMEmbeddings(run_config=eval_run_config)

    ac_metric = answer_correctness
    cr_metric = context_recall
    fa_metric = faithfulness

    for metric in [ac_metric, cr_metric, fa_metric]:
        metric.llm = ragas_llm
    ac_metric.embeddings = ragas_embeddings

    print("\n  Running metrics: answer_correctness + context_recall + faithfulness")
    ragas_results = evaluate(
        dataset=ragas_dataset,
        metrics=[ac_metric, cr_metric, fa_metric],
        run_config=eval_run_config,
    )

    results_df = ragas_results.to_pandas()

    for i, item in enumerate(data):
        item["ragas_answer_correctness"] = float(
            results_df.iloc[i].get("answer_correctness", float("nan"))
        )
        item["ragas_context_recall"] = float(
            results_df.iloc[i].get("context_recall", float("nan"))
        ) if "context_recall" in results_df.columns else None
        item["ragas_faithfulness"] = float(
            results_df.iloc[i].get("faithfulness", float("nan"))
        ) if "faithfulness" in results_df.columns else None

    per_question_df = build_per_question_csv(data)
    summary_df = build_summary_csv(results_df)
    query_type_df = build_query_type_breakdown(data)

    per_q_csv = os.path.join(output_dir, f"per_question_metrics_{run_tag}.csv")
    per_question_df.to_csv(per_q_csv, index=False, encoding="utf-8")

    summary_csv = os.path.join(output_dir, f"evaluation_summary_{run_tag}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    query_type_csv = os.path.join(output_dir, f"query_type_breakdown_{run_tag}.csv")
    query_type_df.to_csv(query_type_csv, index=False, encoding="utf-8")

    xlsx_path = save_excel(per_question_df, summary_df, query_type_df, output_dir, run_tag)

    json_path = os.path.join(output_dir, f"per_question_full_{run_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    def _fmt(value, decimals=4) -> str:
        if value is None:
            return "N/A"
        try:
            if pd.isna(value):
                return "N/A"
        except (TypeError, ValueError):
            pass
        return f"{value:.{decimals}f}"

    ac_scores = results_df["answer_correctness"].dropna()
    cr_scores = results_df["context_recall"].dropna() if "context_recall" in results_df.columns else pd.Series([], dtype=float)
    fa_scores = results_df["faithfulness"].dropna() if "faithfulness" in results_df.columns else pd.Series([], dtype=float)

    print("\n" + "=" * 60)
    print("           EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Total Questions         : {len(data)}")
    print(f"  Answer Correctness Mean : {_fmt(ac_scores.mean())}")
    if len(cr_scores) > 0:
        print(f"  Context Recall Mean     : {_fmt(cr_scores.mean())}")
    if len(fa_scores) > 0:
        print(f"  Faithfulness Mean       : {_fmt(fa_scores.mean())}")
    print("-" * 60)
    print("  OUTPUT FILES")
    print(f"  Per-question CSV    : {per_q_csv}")
    print(f"  Summary CSV         : {summary_csv}")
    print(f"  Query Type CSV      : {query_type_csv}")
    if xlsx_path:
        print(f"  Excel               : {xlsx_path}")
    print(f"  Full JSON backup    : {json_path}")
    print("=" * 60)

    return {
        "per_question_df": per_question_df,
        "summary_df": summary_df,
        "query_type_df": query_type_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation script for results JSON files."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_JSON,
        help="Path to results JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation outputs.",
    )
    args = parser.parse_args()

    run_evaluation(
        input_path=args.input,
        output_dir=args.output,
    )
