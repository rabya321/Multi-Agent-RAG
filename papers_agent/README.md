# Papers Agent

This project contains a complete paper-question answering workflow with:

- Ingestion pipeline (PDF -> structured artifacts -> vectors)
- Agentic retrieval pipeline (`agent_rag`)
- Naive retrieval baseline (`naive_rag`)
- Evaluation and statistical comparison (`Evaluation`)

## Repository workflow

1. Run ingestion Step 1 to Step 4.
2. Run the agent pipeline on a test set.
3. Run the naive baseline on the same test set.
4. Evaluate both outputs with RAGAS.
5. Run Wilcoxon significance tests on the resulting per-question scores.

## Main folders

- `Ingestion/`
	- `step1_parse.py`
	- `step2_summarize_metadata.py`
	- `step3_chunk.py`
	- `step4_embed_index.py`
- `agent_rag/`
	- `run.py` (single query / interactive)
	- `run_testset.py` (batch run)
- `naive_rag/`
	- `retriever.py` (chunk-only baseline)
- `Evaluation/`
	- `ragas_evaluate.py`
	- `wilcoxon_test.py`

## Environment variables

Set these before running ingestion/retrieval/evaluation:

- `API_KEY`
- `CERT_PATH`
- `BASE_URL`

## Python and dependencies


- Recommended Python: 3.11 
- Recommended package manager: `uv`

Create and activate a virtual environment (Windows PowerShell):

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
```

Install project dependencies:

```powershell
uv pip install docling qdrant-client litellm langgraph langchain-core tiktoken pandas numpy scipy ragas datasets openpyxl
```

`openpyxl` is used only for Excel export (`.xlsx`) in analytics outputs (for example from testset and evaluation scripts). CSV and JSON outputs still work without it.


## Ingestion pipeline

Run from the `papers_agent` root.

### Step 1: Parse PDFs

Single file:

```powershell
python Ingestion/step1_parse.py --file path/to/paper.pdf --output_dir output
```

Whole folder:

```powershell
python Ingestion/step1_parse.py --folder path/to/pdfs --output_dir output
```

### Step 2: Summaries and metadata

```powershell
python Ingestion/step2_summarize_metadata.py output/step1_parsed --output_dir output
```

### Step 3: Chunking

Folder mode:

```powershell
python Ingestion/step3_chunk.py --pdf_folder path/to/pdfs --input_dir output --output_dir output
```

Single file mode:

```powershell
python Ingestion/step3_chunk.py --pdf_file path/to/paper.pdf --input_dir output --output_dir output
```

### Step 4: Embedding and indexing

```powershell
python Ingestion/step4_embed_index.py --output_dir output
```

## Agent pipeline (`agent_rag`)

### Interactive or single-query run

Interactive:

```powershell
python -m agent_rag.run --interactive
```

Single query:

```powershell
python -m agent_rag.run --query "What methods does Paper9 use?"
```

### Testset run

```powershell
python -m agent_rag.run_testset --test_set Evaluation/your_testset.json --output_dir Evaluation --overwrite
```

Main output: `Evaluation/pipeline_results.json`

## Naive baseline (`naive_rag`)

This baseline routes every query directly to `chunk_store` with no planning and no reranking.

```powershell
python naive_rag/retriever.py --test_set Evaluation/your_testset.json --output_dir Evaluation
```

Main output: `Evaluation/naive_rag_results.json`

## Evaluation

### RAGAS metrics

Runs:

- `answer_correctness`
- `context_recall`
- `faithfulness`

```powershell
python Evaluation/ragas_evaluate.py --input Evaluation/pipeline_results.json --output Evaluation/evaluation_results_agent
```

For naive baseline:

```powershell
python Evaluation/ragas_evaluate.py --input Evaluation/naive_rag_results.json --output Evaluation/evaluation_results_naive
```

### Wilcoxon significance test

Edit file paths in `Evaluation/wilcoxon_test.py` first (`AGENT_FILE`, `BASELINES`), then run:

```powershell
python Evaluation/wilcoxon_test.py
```


