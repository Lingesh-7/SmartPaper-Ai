"""
Central configuration for SmartPaper AI.

All API keys and model settings are loaded from environment variables
(via a .env file with python-dotenv) rather than hardcoded, so this
file is safe to commit to GitHub.

IMPORTANT: the original notebook had a live OpenRouter key hardcoded
in plaintext (OPEN_ROUTER_API_KEY = "sk-or-v1-..."). That key should
be rotated/revoked on the OpenRouter dashboard before this project is
pushed anywhere public, since it was exposed in cleartext.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (set these in a .env file, never commit the .env itself) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# Propagate into os.environ for libraries that read directly from there
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# --- Model settings ---
TABLE_SUMMARY_MODEL = "llama-3.3-70b-versatile"   # via ChatGroq
TABLE_SUMMARY_PROVIDER = "groq"
TABLE_SUMMARY_TEMPERATURE = 0

IMAGE_SUMMARY_MODEL = "gemini-2.5-flash"          # via ChatGoogleGenerativeAI
IMAGE_SUMMARY_TEMPERATURE = 0

TEXT_SUMMARY_MODEL = "meta-llama/llama-3.3-70b-instruct"  # via ChatOpenRouter
TEXT_SUMMARY_TEMPERATURE = 0.5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Agent reasoning LLM (planner/reasoner/critic nodes) - Groq, same as table summaries
AGENT_LLM_MODEL = "llama-3.3-70b-versatile"
AGENT_LLM_TEMPERATURE = 0

# Vision-capable LLM for figure-query reasoning in the agent
AGENT_VISION_MODEL = "gemini-2.5-flash"

# --- Storage paths ---
CHROMA_PERSIST_DIR = "/tmp/chroma_db"
CHROMA_COLLECTION_NAME = "mm_rag"
EXTRACTED_IMAGES_DIR = "./extracted_data"
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# --- Fine-tuned model (Day 4) ---
FINETUNED_MODEL_REPO = os.getenv(
    "FINETUNED_MODEL_REPO", "your-hf-username/smartpaper-phi3-qasper"
)

# --- Validation (call this at app startup) ---
def validate_config():
    """Raise a clear error if required keys are missing, instead of failing deep in a chain."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Set them in a .env file at the project root."
        )
