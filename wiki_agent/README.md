# SCA Wiki Project

1. Wiki ingestion + retrieval agent
2. Naive RAG baseline
3. Statistical comparison and metric-based evaluation

## Dependencies

- Python 3.11+
- LibreOffice (for PPTX to PDF conversion in ingestion)
- Core Python packages:

```bash
pip install python-pptx pydantic pypdfium2 litellm python-dotenv \
            langchain langchain-openai langchain-core langchain-text-splitters \
            langchain-qdrant qdrant-client pandas scipy ragas tiktoken pyyaml httpx
```

## Ingestion Pipeline

Purpose: convert modules into structured wiki pages and index entries.

Steps:
1. Parse PPTX slides and notes (deterministic metadata extraction from slide 1)
2. Categorize slides as NAVIGATION, TEXT_ONLY, or IMAGE
3. Assemble ordered module context
4. Generate wiki content with an LLM (module page + concept pages + index updates)
5. Persist pages, index updates, and ingestion logs

Run:

```bash
uv run python -m ingestion_pipeline.main \
  --module 00_05_02_Resource_Description_Framework \
  --track General
```

## Retrieval Agent

Purpose: answer user questions from wiki concept pages using bounded tool-calling.

Key design:
1. search path with index and graph traversal via related concepts
2. `read_wiki` tool with fixed retrieval limits
3. Structured output containing answer, supporting concepts, modules, and confidence

Run single query:

```bash
uv run python -m retrieval_agent.search_agent \
  --query "How does demand planning affect inventory management?" \
  --index wiki/index.md
```

## Naive Baseline

Purpose: baseline retrieval without graph-style tool planning.

Current setup:
1. Uses compiled slide images as source
2. Converts images to text with Claude Sonnet 5.5
3. Chunks text and stores embeddings in local Qdrant
4. Answers via top-k similarity retrieval

Run:

```bash
uv run python -m naive_rag \
  --questions evaluation/output/standard_questions.json \
  --compiled-slides raw_sources/general/images \
  --store qdrant_store
```

## Evaluation

Two evaluation layers are used:
1. RAGAS metrics: answer_correctness, context_recall, faithfulness
2. Paired Wilcoxon signed-rank tests between systems

Run metrics:

```bash
uv run python -m evaluation.metrics_eval \
  --results results/agent_all.jsonl
```

Run Wilcoxon tests:

```bash
uv run python -m evaluation.wilcoxon_test
```

## Outputs

- Ingestion: `wiki/modules`, `wiki/concepts`, `wiki/index.md`, `wiki/log.md`
- Retrieval/naive runs: JSONL result files
- Evaluation: JSON/CSV/Markdown summaries and Wilcoxon reports

