import os
import platform
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------

PPTS_DIR   = "raw_sources/general/ppts"
IMAGES_DIR = "raw_sources/general/images"

WIKI_DIR      = "wiki"
MODULES_DIR   = "wiki/modules"
CONCEPTS_DIR  = "wiki/concepts"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

INDEX_PATH         = "wiki/index.md"
LOG_PATH           = "wiki/log.md"
INGESTION_LOG_PATH = "wiki/ingestion_log.csv"
DEBUG_DIR          = "ingestion_pipeline/debug"


# ---------------------------------------------------------------------------
# LiteLLM configuration
# ---------------------------------------------------------------------------

API_KEY   = os.getenv("API_KEY")
BASE_URL  = os.getenv("LITELLM_BASE_URL", "https://litellm.icp.infineon.com")
CERT_PATH = os.getenv("CERT_PATH")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

WIKI_MODEL  = "claudesonnet5.5"
QUERY_MODEL = "gpt-5.2"  
EVAL_MODEL  = "claudesonnet4.6"

# Shared answer-format instruction for every compared system, so answer
# length does not confound claim-level F1 in answer correctness.
ANSWER_LENGTH_INSTRUCTION = (
    "Plain prose, no headers or bullets, at most 180 words. "
    "Use fewer words when sufficient; there is no minimum length. "
    "Answer every part of the question with only necessary, supported claims. "
    "State the requested relationship directly, without a generic closing insight."
)
# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------
# Verify actual slide aspect ratio before using:
#   prs.slide_width / prs.slide_height
#   16:9 = 1.7778 → IMAGE_HEIGHT = 882
#   4:3  = 1.3333 → IMAGE_HEIGHT = 1176
# Standard tier long-edge limit = 1568px
# Final render height is computed per module from actual slide aspect ratio.
# IMAGE_HEIGHT is the fallback reference when module-specific height is unavailable.

IMAGE_WIDTH  = 1568
IMAGE_HEIGHT = 1176

# LibreOffice path for PPTX → PDF conversion.
# Rendering pipeline is strict: LibreOffice + pypdfium2.
# If LibreOffice is unavailable, IMAGE slides are downgraded to TEXT_ONLY.
if platform.system() == "Windows":
    LIBREOFFICE_PATH = r"C:\Program Files\LibreOffice\program\soffice.exe"
elif platform.system() == "Darwin":
    LIBREOFFICE_PATH = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
else:
    LIBREOFFICE_PATH = "soffice"
LIBREOFFICE_PATH = os.getenv("LIBREOFFICE_PATH", LIBREOFFICE_PATH)

# ---------------------------------------------------------------------------
# LLM call settings
# ---------------------------------------------------------------------------

MAX_RETRIES          = 1
RETRY_DELAY_SECONDS  = 30

# Max output tokens — increase if finish_reason=length
# Run probe_gateway.py to confirm gateway ceiling
WIKI_MAX_TOKENS = 32000

# Tool calls per ingestion call
# read_file only — used for concept page extension
# Hard limit across all uses
TOOL_CALL_HARD_LIMIT = 10

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_TOKEN_USAGE = True
# True  → log input/output token counts to wiki/log.md
# False → disable token logging