# Agentic RAG Project

This repository contains two retrieval agents and one orchestrator that routes each question to the right source.

- Wiki Agent: answers from internal e-learning wiki content.
- Papers Agent: answers from indexed research papers.
- Orchestrator: decides route (retrieve, clarify, out_of_scope), calls one or both agents, and evaluates routing quality.

## Project layout

- wiki_agent: wiki ingestion, retrieval agent, naive baseline, evaluation.
- papers_agent: paper ingestion, agentic RAG, naive baseline, evaluation.
- orchestrator: router + orchestration graph + routing dataset/evaluation scripts.

## Requirements

- Python 3.11+
- Recommended: uv
- Access to your LLM endpoint

Environment variables used across components:

- API_KEY
- BASE_URL (or LITELLM_BASE_URL for orchestrator)
- CERT_PATH (optional if TLS cert pinning is required)

## Quick setup

```powershell
cd c:\Users\alir\Desktop\csc-clm-agent
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install python-dotenv httpx pydantic langgraph langchain-core langchain-openai litellm qdrant-client tiktoken pandas scipy ragas datasets python-pptx pypdfium2 pyyaml
```

## Minimal run flow

1. Build data/indexes for each agent:
- Run the wiki ingestion pipeline in wiki_agent.
- Run the papers ingestion pipeline in papers_agent.

2. Run orchestrator interactively:

```powershell
cd orchestrator
uv run python -m orchestrator.graph --scope-mode topics
```

3. Build routing dev/test sets:

```powershell
cd orchestrator\evaluation
python build_routing_testset.py --write
```

4. Evaluate router:

```powershell
cd orchestrator
uv run python evaluation/eval_router.py
uv run python evaluation/eval_router.py --split test --final
```

## Detailed docs

- See wiki_agent/README.md for wiki pipeline, baseline, and metrics.
- See papers_agent/README.md for papers pipeline, baseline, and metrics.
