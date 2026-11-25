import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ORCHESTRATOR_DIR.parent

load_dotenv(ORCHESTRATOR_DIR / ".env")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.icp.infineon.com")
CERT_PATH = os.getenv("CERT_PATH")

# Same model for the topic map and the router, so the map's wording matches how the router reads it.
ROUTER_MODEL = "gpt-5.2"


def make_llm(max_tokens: int = 2000) -> ChatOpenAI:
    http_client = httpx.Client(verify=CERT_PATH) if CERT_PATH else None
    return ChatOpenAI(
        model=ROUTER_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client,
    )
