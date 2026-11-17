"""
LLM client construction for SmartPaper AI.

Why this module exists: the original notebook reused one mutable `llm`
variable, reassigned three times (table summaries -> text summaries ->
agent reasoning). The first modular version replaced that with one
explicitly named client per job, each created once at module import
from the keys in .env.

Import-time creation cannot survive "bring your own keys": every
visitor has different keys, so a client built at import would bill
whoever's key happened to be in the environment. The explicit naming
is kept, the import-time creation is not. build_clients() makes one
LLMClients bundle per set of keys, and each field still has exactly
one job, so there is never any ambiguity about which model answers
which call. Callers build the bundle once (per browser session, or
once per script run) and pass it down explicitly.

Which provider does which job, and why:

    Groq        table + text summaries   ~80 requests per uploaded paper
    Google      figure summaries, and answers that need to see an image
    OpenRouter  the agent's text reasoning (rewrite / classify / answer)

The agent runs on a *free* OpenRouter model. OpenRouter limits free
models to 50 requests per day per key (1,000 once the account has
bought 10 credits), which is plenty for questions at 2-3 requests
each but cannot cover the ~80 summaries of a single upload. That is
why summarising stays on Groq, whose free tier is limited per minute
rather than per day. When the daily OpenRouter allowance does run
out, the agent falls back to the Groq model instead of failing - see
_agent_llm().
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter

import config

logger = logging.getLogger("smartpaper.llm")


@dataclass(frozen=True)
class LLMClients:
    """One explicitly named client per job, all bound to one visitor's keys."""

    table_summary_llm: Runnable     # Groq - ingestion
    text_summary_llm: Runnable      # Groq - ingestion
    image_summary_llm: Runnable     # Gemini vision - ingestion
    agent_llm: Runnable             # OpenRouter (free), falling back to Groq - contextualizer / planner / reasoner
    agent_vision_llm: Runnable      # Gemini vision - reasoner when a figure or table image is in play


def _note_fallback(value):
    logger.warning(
        "OpenRouter request failed; answering with the Groq fallback model %s instead",
        config.AGENT_FALLBACK_MODEL,
    )
    return value


def _agent_llm(keys: config.ApiKeys, common: dict) -> Runnable:
    """
    The agent's text model: the free OpenRouter model, with Groq as a
    fallback.

    The fallback exists because of the 50-requests-per-day limit on
    free OpenRouter models. Without it, a visitor's 20th question or
    so would simply fail until the next day. A daily-limit 429 cannot
    be cured by retrying, so the OpenRouter client does not retry: one
    failure, or config.AGENT_TIMEOUT_SECONDS without a reply, and the
    fallback takes over.
    """
    primary = ChatOpenRouter(
        model=config.AGENT_LLM_MODEL,
        temperature=config.AGENT_LLM_TEMPERATURE,
        api_key=keys.openrouter,
        # ChatOpenRouter takes its timeout in MILLISECONDS (the Groq and
        # Google clients take seconds). Passing seconds here gives a
        # timeout of a few dozen ms: every request fails, and the SDK's
        # retry loop makes that look like a multi-minute hang.
        timeout=config.AGENT_TIMEOUT_SECONDS * 1000,
        max_retries=0,
    )
    fallback = ChatGroq(
        model=config.AGENT_FALLBACK_MODEL,
        temperature=config.AGENT_LLM_TEMPERATURE,
        api_key=keys.groq,
        **common,
    )
    return primary.with_fallbacks([RunnableLambda(_note_fallback) | fallback])


def build_clients(keys: config.ApiKeys) -> LLMClients:
    """Build the full set of clients for one set of keys."""
    missing = keys.missing()
    if missing:
        raise ValueError(f"Missing required API keys: {', '.join(missing)}")

    common = {
        "timeout": config.LLM_TIMEOUT_SECONDS,
        "max_retries": config.LLM_MAX_RETRIES,
    }

    table_summary_llm = ChatGroq(
        model=config.TABLE_SUMMARY_MODEL,
        temperature=config.TABLE_SUMMARY_TEMPERATURE,
        api_key=keys.groq,
        **common,
    )

    text_summary_llm = ChatGroq(
        model=config.TEXT_SUMMARY_MODEL,
        temperature=config.TEXT_SUMMARY_TEMPERATURE,
        api_key=keys.groq,
        **common,
    )

    image_summary_llm = ChatGoogleGenerativeAI(
        model=config.IMAGE_SUMMARY_MODEL,
        temperature=config.IMAGE_SUMMARY_TEMPERATURE,
        google_api_key=keys.google,
        **common,
    )

    agent_vision_llm = ChatGoogleGenerativeAI(
        model=config.AGENT_VISION_MODEL,
        temperature=0,
        google_api_key=keys.google,
        **common,
    )

    return LLMClients(
        table_summary_llm=table_summary_llm,
        text_summary_llm=text_summary_llm,
        image_summary_llm=image_summary_llm,
        agent_llm=_agent_llm(keys, common),
        agent_vision_llm=agent_vision_llm,
    )


def message_text(message) -> str:
    """
    Return a chat response as a plain string.

    Groq and OpenRouter return `content` as a str. Gemini can return a
    list of content blocks instead (for example when the response
    carries more than plain text), and code that calls .strip() or
    .lower() on that list crashes. This flattens both shapes.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


# --- Key checks -------------------------------------------------------------
# A wrong key would otherwise only surface minutes into an ingestion,
# after the PDF has already been parsed. Each provider has a cheap
# authenticated endpoint that costs no tokens (and, for OpenRouter,
# none of the 50 daily requests), so the UI checks keys when they are
# entered.
#
# The same calls also tell us whether the models in config.py still
# exist. That check is here because it has already happened once:
# Groq retired the model this app was built on, and the only symptom
# was a 404 several minutes into an upload.

@dataclass(frozen=True)
class KeyCheck:
    """
    ok is True (usable), False (not usable - the message says whether
    the key or the configured model is the problem) or None (could not
    be checked, e.g. no network).
    """

    ok: bool | None
    message: str


_PROVIDERS = {
    # provider: (display name, authenticated URL, auth header builder)
    "groq": (
        "Groq",
        "https://api.groq.com/openai/v1/models",
        lambda key: {"Authorization": f"Bearer {key}"},
    ),
    "google": (
        "Google",
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        lambda key: {"x-goog-api-key": key},
    ),
    "openrouter": (
        "OpenRouter",
        "https://openrouter.ai/api/v1/key",
        lambda key: {"Authorization": f"Bearer {key}"},
    ),
}

# OpenRouter's key endpoint does not list models; its model list is a
# separate, public endpoint.
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _required_models(provider: str) -> set[str]:
    if provider == "groq":
        return {config.TABLE_SUMMARY_MODEL, config.TEXT_SUMMARY_MODEL, config.AGENT_FALLBACK_MODEL}
    if provider == "google":
        return {config.IMAGE_SUMMARY_MODEL, config.AGENT_VISION_MODEL}
    if provider == "openrouter":
        return {config.AGENT_LLM_MODEL}
    return set()


def _listed_models(provider: str, response, timeout: float) -> set[str] | None:
    """Model ids the provider currently offers, or None if they cannot be read."""
    try:
        if provider == "groq":
            return {m["id"] for m in response.json()["data"]}
        if provider == "google":
            return {m["name"].removeprefix("models/") for m in response.json()["models"]}
        if provider == "openrouter":
            listing = httpx.get(_OPENROUTER_MODELS_URL, timeout=timeout)
            if listing.status_code != 200:
                return None
            return {m["id"] for m in listing.json()["data"]}
    except Exception:  # noqa: BLE001 - an odd payload must not block a valid key
        return None
    return None


def _check(provider: str, key: str, timeout: float) -> KeyCheck:
    name, url, headers = _PROVIDERS[provider]
    try:
        response = httpx.get(url, headers=headers(key), timeout=timeout)
    except httpx.HTTPError as exc:
        # A network problem says nothing about the key - don't block on it.
        logger.warning("Could not reach %s to check the key: %s", name, type(exc).__name__)
        return KeyCheck(None, f"Could not reach {name} to check the key.")

    if response.status_code == 200:
        listed = _listed_models(provider, response, timeout)
        if listed:
            missing = sorted(_required_models(provider) - listed)
            if missing:
                logger.error("%s no longer lists configured model(s): %s", name, missing)
                return KeyCheck(
                    False,
                    f"Your {name} key is fine, but {name} no longer offers the model "
                    f"this app is set up to use ({', '.join(missing)}). "
                    "The app's owner needs to update its model settings.",
                )
        return KeyCheck(True, f"{name} key accepted.")
    if response.status_code in (400, 401, 403):
        return KeyCheck(False, f"{name} rejected this key. Check that it is copied in full.")
    if response.status_code == 429:
        return KeyCheck(None, f"{name} is rate-limiting this key right now.")
    return KeyCheck(None, f"{name} returned HTTP {response.status_code} while checking the key.")


def check_keys(keys: config.ApiKeys, timeout: float = 10.0) -> dict[str, KeyCheck]:
    """
    Check each supplied key against its provider, and that the
    provider still offers the configured models. Returns a dict with
    an entry for each of "groq", "google", "openrouter" whose key was
    supplied.
    """
    supplied = {
        provider: key
        for provider, key in (("groq", keys.groq), ("google", keys.google), ("openrouter", keys.openrouter))
        if key
    }
    if not supplied:
        return {}

    with ThreadPoolExecutor(max_workers=len(supplied)) as pool:
        futures = {
            provider: pool.submit(_check, provider, key, timeout)
            for provider, key in supplied.items()
        }
        return {provider: future.result() for provider, future in futures.items()}
