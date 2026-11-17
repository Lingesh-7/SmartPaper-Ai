"""
The application layer of SmartPaper AI: everything a front end needs,
with no front-end code in it.

    ingest_pdf()        bytes of a PDF  -> an ingested, queryable paper
    answer_question()   question        -> answer + sources
    summarize_paper()   paper           -> overview (cached per paper)

Why this module exists: in the Flask version these steps lived inside
route handlers in app.py, together with a module-level dict of parsed
papers. That made them impossible to reuse from another front end or
to test without an HTTP client. The Streamlit app and test_pipeline.py
both call the functions below, so there is exactly one implementation
of "what happens when a PDF is uploaded".

Things this layer owns that no lower module does:

- Paper identity. A paper's id is a hash of the PDF's bytes, so
  uploading a file that was already ingested (by anyone) reuses the
  stored result instead of spending minutes and API quota again.
- Limits. File size, page count, and one ingestion at a time (the
  layout model is memory-hungry; two at once can exhaust a small
  host).
- Cleanup. The uploaded PDF and the extracted figure crops are deleted
  as soon as ingestion ends; only the derived content is kept.
  Least-recently-used papers are evicted past config.MAX_STORED_PAPERS.
- Error wording. Provider exceptions are turned into messages a
  visitor can act on, with their API keys scrubbed out.
"""

import hashlib
import logging
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

import config
from agents.graph import run_agent
from embeddings import store
from embeddings.embedder import IngestionError, delete_paper, ingest_paper, load_retriever
from models.llm_clients import LLMClients, message_text
from parsers.pdf_parser import PdfError, count_pages, parse_pdf

logger = logging.getLogger("smartpaper.pipeline")

config.ensure_data_dirs()


class PaperError(Exception):
    """Something went wrong that the visitor can understand. The message is user-facing."""


@dataclass
class PaperInfo:
    paper_id: str
    filename: str
    pages: int
    text_chunks: int
    tables: int
    images: int
    reused: bool = False                 # True if an earlier ingestion was reused
    warnings: list = field(default_factory=list)


# --- Retriever cache --------------------------------------------------------
# Retrievers are rebuilt from disk on demand, so this is only a cache:
# losing an entry costs a reload, never a paper.
_RETRIEVER_CACHE_SIZE = 8
_retrievers: "OrderedDict[str, object]" = OrderedDict()
_retrievers_lock = threading.Lock()

# One ingestion at a time, process-wide.
_ingest_lock = threading.Lock()
_INGEST_WAIT_SLICE = 2.0     # seconds between "still queued" progress messages
_INGEST_WAIT_MAX = 15 * 60   # give up waiting for the lock after this long

_STALE_TEMP_SECONDS = 6 * 60 * 60


def _cache_put(paper_id: str, retriever) -> None:
    with _retrievers_lock:
        _retrievers[paper_id] = retriever
        _retrievers.move_to_end(paper_id)
        while len(_retrievers) > _RETRIEVER_CACHE_SIZE:
            _retrievers.popitem(last=False)


def _cache_get(paper_id: str):
    with _retrievers_lock:
        retriever = _retrievers.get(paper_id)
        if retriever is not None:
            _retrievers.move_to_end(paper_id)
        return retriever


def _cache_drop(paper_id: str) -> None:
    with _retrievers_lock:
        _retrievers.pop(paper_id, None)


def _info_from_manifest(manifest: dict, reused: bool = False) -> PaperInfo:
    return PaperInfo(
        paper_id=manifest["paper_id"],
        filename=manifest.get("filename", "paper.pdf"),
        pages=manifest.get("pages", 0),
        text_chunks=manifest.get("text_chunks", 0),
        tables=manifest.get("tables", 0),
        images=manifest.get("images", 0),
        reused=reused,
        warnings=list(manifest.get("warnings", [])),
    )


# --- Error wording ----------------------------------------------------------
def scrub(text: str, secrets: list[str]) -> str:
    """Remove API keys from a piece of text before it is shown or logged."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[key hidden]")
    return text


def describe_error(exc: BaseException, keys: config.ApiKeys | None = None) -> str:
    """
    Turn an exception into one sentence a visitor can act on.
    PaperError / PdfError / IngestionError messages are already
    user-facing; provider errors are classified by their text.
    """
    secrets = keys.secrets() if keys is not None else []
    raw = scrub(str(exc), secrets)
    if isinstance(exc, (PaperError, PdfError)):
        return raw
    lowered = raw.lower()

    def provider() -> str:
        if "groq" in lowered:
            return "Groq"
        if "openrouter" in lowered:
            return "OpenRouter"
        if "google" in lowered or "gemini" in lowered or "generativelanguage" in lowered:
            return "Google"
        return "A provider"

    if any(s in lowered for s in ("model_not_found", "does not exist or you do not have access",
                                  "is not a valid model", "no endpoints found")):
        return (
            f"{provider()} no longer offers the model this app is set up to use. "
            "The app's owner needs to update its model settings."
        )
    if any(s in lowered for s in ("401", "invalid api key", "invalid_api_key", "api_key_invalid",
                                  "api key not valid", "unauthorized", "permission_denied",
                                  "no auth credentials", "user not found")):
        return f"{provider()} rejected the API key. Check the key in the sidebar and try again."
    if any(s in lowered for s in ("429", "rate limit", "rate_limit", "resource_exhausted", "quota")):
        return (
            f"{provider()} is rate-limiting your key (too many requests, or the daily quota is used up). "
            "Wait a minute and try again."
        )
    if any(s in lowered for s in ("timeout", "timed out", "connection error", "connecterror")):
        return "A model provider could not be reached. Check your connection and try again."

    if isinstance(exc, IngestionError):
        return raw[:400]
    return f"Something went wrong ({type(exc).__name__}). {raw[:300]}".strip()


# --- Housekeeping -----------------------------------------------------------
def cleanup_temp_files() -> None:
    """Delete upload / extraction leftovers from ingestions that died midway."""
    cutoff = time.time() - _STALE_TEMP_SECONDS
    for directory in (config.UPLOAD_DIR, config.EXTRACTED_IMAGES_DIR):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove stale temp entry %s: %s", entry.name, exc)


def cleanup_stale_papers() -> None:
    """
    Delete stored papers this version can no longer read: directories
    without a valid manifest (an ingestion that died midway) or written
    by an older store format. They are invisible to the library - never
    reused, never evicted - so without this they would only pile up.
    """
    if not config.PAPERS_DIR.is_dir():
        return
    cutoff = time.time() - _STALE_TEMP_SECONDS
    for entry in config.PAPERS_DIR.iterdir():
        if not entry.is_dir():
            continue
        try:
            readable = store.load_manifest(entry.name) is not None
            # Leave recent ones alone: an ingestion may be writing it right now.
            if readable or entry.stat().st_mtime > cutoff:
                continue
            logger.info("Removing unreadable stored paper %s", entry.name)
            _cache_drop(entry.name)
            delete_paper(entry.name)
        except (OSError, ValueError) as exc:
            logger.warning("Could not clean up stored paper %s: %s", entry.name, exc)


def _evict_old_papers(keep: str) -> None:
    """Delete least-recently-used papers beyond config.MAX_STORED_PAPERS."""
    manifests = store.list_manifests()
    excess = len(manifests) - config.MAX_STORED_PAPERS
    if excess <= 0:
        return
    manifests.sort(key=lambda m: m.get("last_used", 0))
    for manifest in manifests:
        if excess <= 0:
            break
        paper_id = manifest["paper_id"]
        if paper_id == keep:
            continue
        logger.info("Evicting paper %s (library is full)", paper_id)
        _cache_drop(paper_id)
        delete_paper(paper_id)
        excess -= 1


# --- Lookup -----------------------------------------------------------------
def paper_id_for(data: bytes) -> str:
    """A paper's id: the first 32 hex characters of the SHA-256 of its bytes."""
    return hashlib.sha256(data).hexdigest()[:32]


def get_paper_info(paper_id: str) -> PaperInfo | None:
    """Details of a stored paper, or None if it is not (or no longer) stored."""
    try:
        manifest = store.load_manifest(paper_id)
    except ValueError:
        return None
    return _info_from_manifest(manifest) if manifest else None


def get_retriever(paper_id: str):
    """
    Return a paper's retriever, loading it from disk if this process
    has not used it yet. Raises PaperError if the paper is not stored
    (it was never uploaded, or it has been evicted).
    """
    retriever = _cache_get(paper_id)
    if retriever is not None:
        return retriever

    try:
        retriever = load_retriever(paper_id)
    except ValueError:
        retriever = None
    if retriever is None:
        raise PaperError("This paper is no longer available on the server. Please upload it again.")

    _cache_put(paper_id, retriever)
    store.touch(paper_id)
    return retriever


# --- Ingestion --------------------------------------------------------------
def _report(progress, message: str, fraction: float) -> None:
    if progress is not None:
        progress(message, max(0.0, min(1.0, fraction)))


def _reuse_if_stored(paper_id: str, filename: str) -> PaperInfo | None:
    """Return the stored paper if a complete, consistent copy exists."""
    manifest = store.load_manifest(paper_id)
    if manifest is None:
        return None
    retriever = _cache_get(paper_id) or load_retriever(paper_id)
    if retriever is None:
        return None
    _cache_put(paper_id, retriever)
    store.touch(paper_id)
    info = _info_from_manifest(manifest, reused=True)
    info.filename = filename or info.filename
    return info


def ingest_pdf(
    data: bytes,
    filename: str,
    clients: LLMClients,
    progress=None,
    force: bool = False,
) -> PaperInfo:
    """
    Ingest a PDF given as bytes and return its PaperInfo.

    Args:
        data: the PDF file's bytes
        filename: the original file name, for display only
        clients: LLM clients bound to the current visitor's keys
        progress: optional callable(message: str, fraction: float)
        force: re-ingest even if this exact file is already stored

    Raises:
        PaperError / PdfError / IngestionError with user-facing
        messages for expected failures. Anything else is unexpected;
        pass it to describe_error() before showing it.
    """
    if not data:
        raise PaperError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise PaperError(f"This PDF is larger than the {config.MAX_UPLOAD_MB} MB limit.")
    if b"%PDF-" not in data[:1024]:
        raise PaperError("This file is not a PDF.")

    paper_id = paper_id_for(data)
    filename = (filename or "paper.pdf").strip()[:200]

    if not force:
        reused = _reuse_if_stored(paper_id, filename)
        if reused is not None:
            _report(progress, "This paper was already processed - loaded from the library", 1.0)
            return reused

    # Unique per attempt, so two visitors uploading the same file at
    # the same moment never write to the same temp paths.
    attempt = f"{paper_id}-{uuid.uuid4().hex[:8]}"
    pdf_path = config.UPLOAD_DIR / f"{attempt}.pdf"
    extract_dir = config.EXTRACTED_IMAGES_DIR / attempt

    config.ensure_data_dirs()
    have_lock = False
    try:
        pdf_path.write_bytes(data)

        _report(progress, "Checking the PDF", 0.01)
        pages = count_pages(str(pdf_path))
        if pages == 0:
            raise PaperError("This PDF has no pages.")
        if pages > config.MAX_PDF_PAGES:
            raise PaperError(
                f"This PDF has {pages} pages; the limit is {config.MAX_PDF_PAGES}."
            )

        waited = 0.0
        while not _ingest_lock.acquire(timeout=_INGEST_WAIT_SLICE):
            waited += _INGEST_WAIT_SLICE
            if waited >= _INGEST_WAIT_MAX:
                raise PaperError("The server is busy processing other papers. Please try again in a few minutes.")
            _report(progress, "Another paper is being processed - yours is next in line", 0.02)
        have_lock = True

        # Someone may have finished ingesting this same file while we waited.
        if not force:
            reused = _reuse_if_stored(paper_id, filename)
            if reused is not None:
                _report(progress, "This paper was already processed - loaded from the library", 1.0)
                return reused

        # Clear anything already stored under this id: a forced
        # re-ingest, or a copy this version cannot read (older format).
        _cache_drop(paper_id)
        delete_paper(paper_id)

        _report(progress, f"Reading the layout of {pages} pages (the slowest step)", 0.05)
        extract_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        parsed_data = parse_pdf(str(pdf_path), str(extract_dir))
        parse_seconds = time.monotonic() - started

        if not (parsed_data["text_chunks"] or parsed_data["tables"] or parsed_data["images"]):
            raise PaperError(
                "No text, tables or figures could be extracted from this PDF. "
                "It may be a scanned document without readable text."
            )

        # Parsing is ~40% of the bar; summarising and indexing fill the rest.
        retriever = ingest_paper(
            parsed_data,
            paper_id,
            clients,
            meta={
                "filename": filename,
                "pages": pages,
                "size_bytes": len(data),
            },
            progress=lambda message, fraction: _report(progress, message, 0.4 + 0.6 * fraction),
        )
        _cache_put(paper_id, retriever)
        logger.info(
            "Ingested %s: %d pages, parse %.0fs, summarise+index %.0fs",
            paper_id, pages, parse_seconds, time.monotonic() - started - parse_seconds,
        )

        manifest = store.load_manifest(paper_id)
        if manifest is None:
            raise PaperError("The paper was processed but could not be saved. Please try again.")

        _evict_old_papers(keep=paper_id)
        return _info_from_manifest(manifest)

    finally:
        if have_lock:
            _ingest_lock.release()
        pdf_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


# --- Question answering -----------------------------------------------------
def answer_question(
    question: str,
    paper_id: str,
    clients: LLMClients,
    chat_history: list | None = None,
    on_step=None,
) -> dict:
    """
    Answer a question about a stored paper.

    Returns the dict from agents.graph.run_agent: final_answer,
    retrieval_type, retry_count, sources, standalone_question.
    """
    question = (question or "").strip()
    if not question:
        raise PaperError("Type a question first.")
    if len(question) > config.MAX_QUESTION_CHARS:
        raise PaperError(f"That question is too long (limit: {config.MAX_QUESTION_CHARS} characters).")

    retriever = get_retriever(paper_id)
    result = run_agent(question, retriever, clients, chat_history=chat_history, on_step=on_step)

    if not result["final_answer"].strip():
        raise PaperError("The model returned an empty answer. Please try rephrasing the question.")
    return result


# --- Summary ----------------------------------------------------------------
_SUMMARY_CHUNKS = 5


def summarize_paper(paper_id: str, clients: LLMClients) -> str:
    """
    A short overview of the paper, generated from its opening passages.
    Generated once per paper and cached in the paper's manifest.
    """
    manifest = store.load_manifest(paper_id)
    if manifest is None:
        raise PaperError("This paper is no longer available on the server. Please upload it again.")
    if manifest.get("summary"):
        return manifest["summary"]

    records = store.load_records(paper_id) or {}
    # docstore.json preserves insertion order: text records first, in
    # reading order - so the first few are the abstract/introduction.
    opening = [
        r["content"] for r in records.values()
        if r.get("type") == "text" and r.get("section", "body") == "body"
    ][:_SUMMARY_CHUNKS]
    if not opening:
        raise PaperError("This paper has no extracted text to summarise.")

    prompt = (
        "Based on the following text from the start of a research paper, provide:\n"
        "1. A 3-sentence overview\n"
        "2. 5 key contributions as bullet points\n"
        "3. 3 suggested follow-up questions a reader might ask\n\n"
        f"Text:\n{chr(10).join(opening)}"
    )
    summary = message_text(clients.agent_llm.invoke([HumanMessage(content=prompt)])).strip()
    if not summary:
        raise PaperError("The model returned an empty summary. Please try again.")

    store.update_manifest(paper_id, summary=summary)
    return summary


# --- Background ingestion ---------------------------------------------------
class IngestJob:
    """
    Runs ingest_pdf() in a worker thread and exposes its progress.

    Why a thread: ingestion takes minutes, and an interactive front end
    cannot be trusted to keep one request alive that long. In
    Streamlit, any click or a page refresh while a script is running
    stops that script at its next UI call - which would abort an
    ingestion that is four minutes in. A job keeps going regardless:
    the UI only polls it. If the visitor's session disappears, the
    paper still lands in the library, and re-uploading the same file
    picks it up instantly.

    Read message / fraction / done / info / error from any thread.
    """

    def __init__(
        self,
        data: bytes,
        filename: str,
        clients: LLMClients,
        keys: config.ApiKeys | None = None,
        force: bool = False,
    ):
        self.filename = filename
        self.message = "Starting"
        self.fraction = 0.0
        self.info: PaperInfo | None = None
        self.error: str | None = None
        self._started = time.monotonic()
        self._finished: float | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            args=(data, filename, clients, keys, force),
            name="smartpaper-ingest",
            daemon=True,
        )
        self._thread.start()

    def _progress(self, message: str, fraction: float) -> None:
        with self._lock:
            self.message = message
            # never let the bar move backwards (a queued job reports 2% repeatedly)
            self.fraction = max(self.fraction, fraction)

    def _run(self, data, filename, clients, keys, force) -> None:
        try:
            info = ingest_pdf(data, filename, clients, progress=self._progress, force=force)
            with self._lock:
                self.info = info
        except Exception as exc:  # noqa: BLE001 - every failure must reach the UI as a message
            message = describe_error(exc, keys)
            logger.error("Ingestion of %r failed (%s): %s", filename, type(exc).__name__, message)
            logger.debug("Ingestion traceback", exc_info=True)
            with self._lock:
                self.error = message
        finally:
            with self._lock:
                self._finished = time.monotonic()

    @property
    def done(self) -> bool:
        with self._lock:
            return self._finished is not None

    @property
    def elapsed(self) -> float:
        with self._lock:
            end = self._finished if self._finished is not None else time.monotonic()
        return end - self._started

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the job finishes (or timeout). Returns True if it finished."""
        self._thread.join(timeout)
        return self.done
