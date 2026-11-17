"""
Central configuration for SmartPaper AI.

All model settings, limits, storage paths, and API-key handling live
here, so no module hardcodes a model string or a path.

API keys
--------
The deployed app is "bring your own keys": each visitor supplies their
own OpenRouter / Google / Groq keys in the UI (all three are required),
and those keys travel through the pipeline as an explicit ApiKeys
object. Nothing in this
module copies a key into os.environ, because an environment variable
is process-wide and would leak one visitor's key into another
visitor's requests.

Keys found in the environment (a local .env, or secrets set on the
host) are only a *fallback*: keys_from_env() returns them so that
local runs and test_pipeline.py work without typing keys into a UI.
On a public deployment, leave those three variables unset unless you
intend to pay for every visitor's usage.

IMPORTANT: the original notebook had a live OpenRouter key hardcoded
in plaintext (OPEN_ROUTER_API_KEY = "sk-or-v1-..."). That key should
be rotated/revoked on the OpenRouter dashboard before this project is
pushed anywhere public, since it was exposed in cleartext.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    """Read an integer setting from the environment, ignoring bad values."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Model settings ---
# Three providers, each doing the job its free tier can actually carry:
#
#   Groq        table + text summaries (~80 requests per uploaded paper)
#   Google      figure summaries, and answers that need to see an image
#   OpenRouter  the agent's text reasoning, on a free model
#
# Free OpenRouter models are limited to 50 requests per day per key
# (1,000 once the account has bought 10 credits). That covers
# questions, at 2-3 requests each, but not the summaries of even one
# paper - which is why summarising is on Groq (limited per minute,
# not per day) rather than on OpenRouter as in the original notebook.
#
# Providers also retire models. Groq used to serve
# llama-3.3-70b-versatile, which every Groq role here was built on;
# its API now answers 404 "model does not exist" for it, so the Groq
# roles use openai/gpt-oss-120b, the closest model in size that Groq
# currently serves. Each provider's model can be changed with an
# environment variable, so the next retirement is a settings change
# on the host, not a code change. models/llm_clients.check_keys()
# reports a configured model that the provider no longer lists.
_GROQ_MODEL = os.getenv("SMARTPAPER_GROQ_MODEL") or "openai/gpt-oss-120b"
_GEMINI_MODEL = os.getenv("SMARTPAPER_GEMINI_MODEL") or "gemini-2.5-flash"
_OPENROUTER_MODEL = os.getenv("SMARTPAPER_OPENROUTER_MODEL") or "nvidia/nemotron-3-ultra-550b-a55b:free"

TABLE_SUMMARY_MODEL = _GROQ_MODEL                 # via ChatGroq
TABLE_SUMMARY_PROVIDER = "groq"
TABLE_SUMMARY_TEMPERATURE = 0

IMAGE_SUMMARY_MODEL = _GEMINI_MODEL               # via ChatGoogleGenerativeAI
IMAGE_SUMMARY_TEMPERATURE = 0

TEXT_SUMMARY_MODEL = _GROQ_MODEL                  # via ChatGroq
TEXT_SUMMARY_TEMPERATURE = 0.5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Agent reasoning LLM (contextualizer/planner/reasoner nodes) - OpenRouter
AGENT_LLM_MODEL = _OPENROUTER_MODEL               # via ChatOpenRouter
AGENT_LLM_TEMPERATURE = 0
# Answers the agent's calls when OpenRouter fails - in practice, when
# the key's 50 free requests for the day are used up.
AGENT_FALLBACK_MODEL = _GROQ_MODEL                # via ChatGroq

# Vision-capable LLM for figure-query reasoning in the agent
AGENT_VISION_MODEL = _GEMINI_MODEL

# --- LLM call behaviour ---
LLM_TIMEOUT_SECONDS = _env_int("SMARTPAPER_LLM_TIMEOUT", 120)
LLM_MAX_RETRIES = _env_int("SMARTPAPER_LLM_MAX_RETRIES", 4)
# The agent's OpenRouter call gets a much shorter leash than the rest.
# Free models are served best-effort: in a real run one request simply
# hung, and with the general timeout the visitor waited over three
# minutes before the Groq fallback answered. The model normally
# replies in a few seconds, so past this it is quicker to fall back.
AGENT_TIMEOUT_SECONDS = _env_int("SMARTPAPER_AGENT_TIMEOUT", 45)

# --- Retrieval ---
# How many records a question retrieves. The pipeline originally got
# Chroma's default of 4 (retrieve()'s old top_k argument was accepted
# and ignored), which with section-sized chunks left answers short of
# context - e.g. the table listing a dataset's features was ranked 5th
# for "what input features does the dataset contain?".
RETRIEVAL_TOP_K = _env_int("SMARTPAPER_TOP_K", 6)

# The paper's opening - title, authors, abstract - is given to the
# model with every question, up to this many characters. Similarity
# search is bad at "what is this paper about?" and "who wrote it?":
# on a real paper it returned pages of the reference list instead of
# the title page.
FRONT_MATTER_MAX_CHARS = 3500

# When the critic rejects an answer, the retry searches this many
# times wider (RETRIEVAL_TOP_K x 2, then x 3). Repeating the same
# search with the same context rarely produces a different answer.
RETRY_WIDEN_MAX = 3

# Summaries that fail (in practice: Groq's per-minute rate limit) get
# one more attempt, one at a time, after this pause.
SUMMARY_RETRY_PAUSE_SECONDS = _env_int("SMARTPAPER_SUMMARY_RETRY_PAUSE", 20)

# For a question classified as being about a table (or a figure), this
# many of the retrieved results are guaranteed to be tables (figures),
# provided the paper has that many.
MODALITY_TOP_K = 2

# How many previous chat messages the agent sees when rewriting a
# follow-up question into a standalone one.
HISTORY_MESSAGES = 6

# --- Parsing: text chunks ---
# Body text is grouped by section into chunks of about TARGET
# characters (never more than MAX); sections shorter than
# COMBINE_UNDER are merged with their neighbour instead of standing
# alone. Sized so that RETRIEVAL_TOP_K chunks give the model a few
# thousand characters of coherent context.
CHUNK_TARGET_CHARS = 1500
CHUNK_MAX_CHARS = 2000
CHUNK_COMBINE_UNDER_CHARS = 500
# A line shorter than this that recurs on several pages is a running
# header/footer or a page number, and is dropped.
BOILERPLATE_MAX_CHARS = 150

# --- Parsing: figures ---
# An image crop is treated as decoration, not a figure, if it is
# smaller than this on its short side ...
IMAGE_MIN_SIDE_PX = 100
# ... or it has no caption and: the same crop size shows up on this
# many pages (a logo in the page header) ...
IMAGE_REPEAT_PAGES = 3
# ... or it has fewer than this many characters of text inside it
# (an author portrait, an ornament) ...
IMAGE_MIN_OCR_CHARS = 20
# ... or it is a wide strip (a masthead banner).
IMAGE_MAX_UNCAPTIONED_ASPECT = 3.0
# A caption belongs to the figure/table nearest to it on the page, and
# only if the gap between them is under this fraction of the page
# height. On the paper these rules were tuned on, a figure's own
# caption was 1-2% of the page height away and the next candidate
# never under 2% (usually far more).
CAPTION_MAX_GAP_FRACTION = 0.06

# How many figures per paper may be described by the vision model at
# upload, to enrich the search index. Default 0: figures are indexed by
# their caption and the text printed inside them, which costs no API
# calls. Gemini's free tier allows about 20 requests PER DAY on
# gemini-2.5-flash - one figure-heavy paper used to burn all of them
# during upload, leaving none for answering figure and table
# questions. Raise this only for keys on a paid tier.
IMAGE_SUMMARY_BUDGET = _env_int("SMARTPAPER_VISION_SUMMARY_BUDGET", 0)

# --- Limits ---
MAX_UPLOAD_MB = _env_int("SMARTPAPER_MAX_UPLOAD_MB", 20)
MAX_PDF_PAGES = _env_int("SMARTPAPER_MAX_PDF_PAGES", 50)
MAX_QUESTION_CHARS = 2000
# Oldest papers (by last use) are deleted once the library exceeds this.
MAX_STORED_PAPERS = _env_int("SMARTPAPER_MAX_STORED_PAPERS", 50)

# --- Storage paths ---
# Everything the app writes lives under one directory, so a host only
# has to make one path writable (or mount one volume). The default is
# ./data next to this file, which works on Windows and Linux alike;
# set SMARTPAPER_DATA_DIR to move it (for example to a host's
# persistent-storage path).
DATA_DIR = Path(os.getenv("SMARTPAPER_DATA_DIR") or (_REPO_ROOT / "data")).resolve()
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma")
CHROMA_COLLECTION_NAME = "mm_rag"
PAPERS_DIR = DATA_DIR / "papers"            # manifest + raw content per paper
UPLOAD_DIR = DATA_DIR / "uploads"           # PDFs, only while being ingested
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted"   # figure crops, only while being ingested

# Bump when the on-disk format of a stored paper changes; papers saved
# with a different version are treated as not ingested and rebuilt.
STORE_SCHEMA_VERSION = 3


def ensure_data_dirs() -> None:
    """Create the storage directories if they do not exist yet."""
    for path in (DATA_DIR, Path(CHROMA_PERSIST_DIR), PAPERS_DIR, UPLOAD_DIR, EXTRACTED_IMAGES_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --- Fine-tuned model (Day 4) ---
FINETUNED_MODEL_REPO = os.getenv(
    "FINETUNED_MODEL_REPO", "Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER"
)


# --- API keys ---
def _clean(value) -> str:
    return (value or "").strip()


@dataclass(frozen=True)
class ApiKeys:
    """
    One visitor's set of provider keys. All three are required:
    openrouter answers questions, groq summarises uploads, google
    reads figures.
    """

    groq: str = ""
    google: str = ""
    openrouter: str = ""

    def __post_init__(self):
        object.__setattr__(self, "groq", _clean(self.groq))
        object.__setattr__(self, "google", _clean(self.google))
        object.__setattr__(self, "openrouter", _clean(self.openrouter))

    def missing(self) -> list[str]:
        """Names of the required keys that are not set."""
        missing = []
        if not self.openrouter:
            missing.append("OPENROUTER_API_KEY")
        if not self.google:
            missing.append("GOOGLE_API_KEY")
        if not self.groq:
            missing.append("GROQ_API_KEY")
        return missing

    def is_complete(self) -> bool:
        return not self.missing()

    def secrets(self) -> list[str]:
        """The non-empty key strings, for scrubbing them out of error text."""
        return [k for k in (self.groq, self.google, self.openrouter) if k]

    def __repr__(self) -> str:
        # Never let a key reach a log line or a traceback via repr().
        def mask(key: str) -> str:
            return "set" if key else "missing"

        return (
            f"ApiKeys(groq={mask(self.groq)}, google={mask(self.google)}, "
            f"openrouter={mask(self.openrouter)})"
        )


def keys_from_env() -> ApiKeys:
    """Keys from the environment / .env - the fallback for local runs."""
    return ApiKeys(
        groq=os.getenv("GROQ_API_KEY", ""),
        google=os.getenv("GOOGLE_API_KEY", ""),
        openrouter=os.getenv("OPENROUTER_API_KEY", ""),
    )


def validate_config() -> ApiKeys:
    """
    For command-line entry points (test_pipeline.py): return the keys
    from the environment, or raise a clear error if a required one is
    missing, instead of failing deep in a chain.

    The Streamlit app does not call this - it collects keys from the
    visitor and reports missing ones in the UI.
    """
    keys = keys_from_env()
    missing = keys.missing()
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Set them in a .env file at the project root."
        )
    return keys
