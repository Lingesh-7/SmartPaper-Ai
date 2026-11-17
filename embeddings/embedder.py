"""
Embedding module for SmartPaper AI.

Summarizes tables and text (Groq), builds an index text for each
figure from its caption and the text printed in it, then builds a
Chroma-backed MultiVectorRetriever that indexes those texts but
returns raw content (text/table strings, base64 images) at query
time.

NOTE on the original notebook: it reused a single mutable `llm`
variable across table summarization, text summarization, and the
agent's planner/reasoner, reassigning it three separate times
(ChatGroq -> ChatOpenRouter -> ChatGroq again). That works in a
notebook because cells run top-to-bottom in a fixed order, but it's
fragile and confusing in a module. Here each summarization step uses
its own explicitly named client from the LLMClients bundle
(models/llm_clients.py), so there's no ambiguity about which model
answers which call. The bundle is passed in rather than created at
import, because each visitor brings their own keys.

Other deliberate differences from the first modular version:

- The docstore holds typed records ({type, content, page_number, ...})
  instead of bare strings, and is written to disk (embeddings/store.py).
  That is what makes page citations possible and what lets a paper
  survive a restart.
- A summary that fails (rate limit, timeout) no longer aborts the
  whole ingestion. The item's own text is embedded in place of the
  missing summary - a weaker retrieval key than a summary, but far
  better than throwing away minutes of parsing. If *every* call to a
  provider fails, that is a bad key or an outage, and ingestion stops
  with an error instead of persisting a hollow paper.
- The embedding model is loaded on first use, not at import, so the
  app can start and show its UI before the model is in memory.
- Text summaries are produced by Groq, not OpenRouter as in the
  notebook. The agent now runs on a free OpenRouter model, and free
  OpenRouter models are capped at 50 requests per day per key - less
  than the summaries of one paper. See models/llm_clients.py.
- Chroma comes from langchain-chroma. The langchain-community class
  the notebook used is deprecated and that package is being sunset.
"""

import base64
import logging
import os
import threading
import time
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.stores import InMemoryStore

import config
from embeddings import store
from models.llm_clients import LLMClients, message_text

logger = logging.getLogger("smartpaper.embedder")


class IngestionError(Exception):
    """Ingestion could not produce a usable paper. The message is user-facing."""


# --- Embedding model (shared across text, table, image summary vectors) ---
_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    """Load the sentence-transformers model once per process, on first use."""
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info("Loading embedding model %s", config.EMBEDDING_MODEL)
            _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        return _embeddings


def _collection_name(paper_id: str) -> str:
    return f"{config.CHROMA_COLLECTION_NAME}_{store.validate_paper_id(paper_id)}"


def _vectorstore(paper_id: str) -> Chroma:
    return Chroma(
        collection_name=_collection_name(paper_id),
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def _report(progress, message: str, fraction: float) -> None:
    if progress is not None:
        progress(message, max(0.0, min(1.0, fraction)))


# --- Summarization helpers ---
_FALLBACK_SUMMARY_CHARS = 1500
_BATCH_GROUP = 10   # progress is reported after each group of this many items


def _fallback_summary(item: str) -> str:
    """What gets embedded when an item's summary could not be generated."""
    return item[:_FALLBACK_SUMMARY_CHARS]


def _summarize_batch(chain, items, max_concurrency, label, provider, on_progress=None):
    """
    Summarize items in groups, tolerating individual failures.

    Returns (summaries, failed_count); summaries is parallel to items,
    with failed items replaced by _fallback_summary(item). Raises
    IngestionError if the first group, or every item, fails - that
    pattern means a bad key or a provider outage, not a flaky call.

    Items that fail in the first pass get one more attempt at the end,
    one at a time, after a pause. Nearly all such failures are Groq's
    per-minute rate limit, which the client's own quick retries (a few
    seconds apart) cannot outlast but a short wait can.
    """
    summaries: list[str] = []
    failed = 0
    first_error = None
    failed_positions: list[int] = []

    for start in range(0, len(items), _BATCH_GROUP):
        group = items[start:start + _BATCH_GROUP]
        results = chain.batch(
            group,
            config={"max_concurrency": max_concurrency},
            return_exceptions=True,
        )
        for item, result in zip(group, results):
            if isinstance(result, Exception):
                failed += 1
                first_error = first_error or result
                logger.warning("%s summary failed: %s", label, type(result).__name__)
                failed_positions.append(len(summaries))
                summaries.append(_fallback_summary(item))
            elif not str(result).strip():
                failed += 1
                failed_positions.append(len(summaries))
                summaries.append(_fallback_summary(item))
            else:
                summaries.append(str(result))

        if failed == len(summaries) and first_error is not None:
            # Nothing has succeeded so far. Stop now rather than sending
            # the remaining items into the same wall.
            raise IngestionError(
                f"Every {label} summary request to {provider} failed. "
                f"First error: {first_error}"
            ) from first_error

        if on_progress is not None:
            on_progress(len(summaries), len(items))

    if failed_positions:
        logger.info(
            "Retrying %d failed %s summaries after %ds",
            len(failed_positions), label, config.SUMMARY_RETRY_PAUSE_SECONDS,
        )
        time.sleep(config.SUMMARY_RETRY_PAUSE_SECONDS)
        for position in failed_positions:
            try:
                result = str(chain.invoke(items[position]))
            except Exception as exc:  # noqa: BLE001 - keep the fallback summary
                logger.warning("%s summary failed again: %s", label, type(exc).__name__)
                continue
            if result.strip():
                summaries[position] = result
                failed -= 1

    return summaries, failed


# --- Table summarization (Groq) ---
_table_prompt = ChatPromptTemplate.from_template(
    """
You are an assistant tasked with summarizing tables for retrieval.
These summaries will be embedded and used to retrieve the raw table elements.
Give a concise summary of the table that is well optimized for retrieval
Table:
{table}
"""
)


def summarize_tables(tables: list[str], clients: LLMClients, on_progress=None):
    """
    Summarize a list of table strings for retrieval embedding.
    Returns (summaries, failed_count).
    """
    if not tables:
        return [], 0
    chain = (
        {"table": RunnableLambda(lambda x: x)}
        | _table_prompt
        | clients.table_summary_llm
        | StrOutputParser()
    )
    # Concurrency 2, like the text summaries: at the notebook's 12, two
    # of a paper's six tables lost their summaries to Groq's rate limit.
    return _summarize_batch(chain, tables, 2, "table", "Groq", on_progress)


# --- Figure indexing ---
# A figure is found through text, like everything else: the multi-vector
# pattern embeds a piece of text and returns the figure itself. The
# notebook got that text by asking a vision model to describe every
# image. That cannot be the default any more: Gemini's free tier allows
# about 20 requests per day on gemini-2.5-flash, and a real 20-page
# paper produced 37 image crops - the upload used the whole day's
# allowance, 33 of the 37 descriptions failed, the figures that failed
# were dropped as unsearchable, and nothing was left for answering.
#
# So the index text of a figure is now built from what the parser
# already has, at no cost: the author's caption ("FIGURE 9. ROC curves
# of the evaluated models.") and the text the layout model read inside
# the figure (titles, axis labels, legend entries). A caption is the
# author's own description of the figure - a better search key than a
# generic one. A vision description is added on top only for as many
# figures as config.IMAGE_SUMMARY_BUDGET allows (default 0), and a
# figure is never dropped because that call failed.
_IMAGE_SUMMARY_PROMPT = (
    "You are an assistant tasked with summarizing images for retrieval. "
    "Give a concise summary optimized for retrieval."
)

_MAX_OCR_INDEX_CHARS = 600

# After this many vision failures in a row, stop asking for the rest of
# the paper: the quota is gone or the key is bad, and each further
# attempt only adds waiting time.
_VISION_GIVE_UP_AFTER = 2


def encode_image(image_path: str) -> str:
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _summarize_image(img_base64: str, clients: LLMClients, prompt: str = _IMAGE_SUMMARY_PROMPT) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img_base64}",
            },
        ]
    )
    response = clients.image_summary_llm.invoke([message])
    return message_text(response)


def figure_index_text(image: dict) -> str:
    """The text a figure is indexed by: its caption and the text printed in it."""
    parts = []
    caption = (image.get("caption") or "").strip()
    if caption:
        parts.append(caption)
    ocr_text = (image.get("ocr_text") or "").strip()
    if ocr_text:
        parts.append(f"Text in the figure: {ocr_text[:_MAX_OCR_INDEX_CHARS]}")
    if not parts:
        parts.append(f"Figure on page {image.get('page_number')}")
    return "\n".join(parts)


def index_figures(images: list[dict], clients: LLMClients, on_progress=None):
    """
    Build the index text for each figure (as returned by
    parsers.pdf_parser.parse_pdf: {path, page_number, caption, label,
    ocr_text}).

    Returns:
        (kept, index_texts, vision_failed) where kept is a list of
        {"base64", "page_number", "caption", "label", "ocr_text"}
        dicts parallel to index_texts, and vision_failed counts the
        figures whose optional vision description could not be
        obtained (they are kept, indexed by caption).
    """
    kept = []
    index_texts = []
    vision_failed = 0
    failures_in_a_row = 0
    budget = max(0, config.IMAGE_SUMMARY_BUDGET)

    candidates = [
        image for image in images
        if os.path.isfile(image["path"])
        and image["path"].lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for index, image in enumerate(candidates, start=1):
        try:
            base64_image = encode_image(image["path"])
        except OSError as exc:
            logger.warning("Could not read figure %s: %s", os.path.basename(image["path"]), exc)
            continue

        index_text = figure_index_text(image)

        if budget > 0 and failures_in_a_row < _VISION_GIVE_UP_AFTER:
            budget -= 1
            try:
                description = _summarize_image(base64_image, clients).strip()
                if not description:
                    raise ValueError("empty description")
                index_text = f"{index_text}\n{description}"
                failures_in_a_row = 0
            except Exception as exc:  # noqa: BLE001 - the description is an extra, never a requirement
                vision_failed += 1
                failures_in_a_row += 1
                logger.warning(
                    "Figure description failed (%s): %s",
                    os.path.basename(image["path"]), type(exc).__name__,
                )

        kept.append({
            "base64": base64_image,
            "page_number": image.get("page_number"),
            "caption": image.get("caption") or "",
            "label": image.get("label") or "",
            "ocr_text": image.get("ocr_text") or "",
        })
        index_texts.append(index_text)

        if on_progress is not None:
            on_progress(index, len(candidates))

    return kept, index_texts, vision_failed


# --- Text summarization (Groq) ---
# The notebook used ChatOpenRouter here; this is a ChatGroq client
# (see the module docstring for why) - the rest of the pipeline
# doesn't care which provider produced the summary text. Groq's free
# tier has tight per-minute token limits: a real run of a 76-passage
# paper at concurrency 3 drew more 429 responses than successes, and
# 5 passages ran out of retries. It therefore runs at concurrency 2.
_text_prompt = ChatPromptTemplate.from_template(
    """You are an assistant tasked with summarizing text for retrieval.
These summaries will be embedded and used to retrieve the raw text elements.
Give a concise summary of the text that is well optimized for retrieval
Text:
{text}"""
)


def summarize_texts(texts: list[str], clients: LLMClients, on_progress=None):
    """
    Summarize a list of text chunks for retrieval embedding.
    Returns (summaries, failed_count).
    """
    if not texts:
        return [], 0
    chain = (
        {"text": RunnableLambda(lambda x: x)}
        | _text_prompt
        | clients.text_summary_llm
        | StrOutputParser()
    )
    return _summarize_batch(chain, texts, 2, "text", "Groq", on_progress)


# --- Multi-vector retriever construction ---
def build_retriever(paper_id: str, records: dict) -> MultiVectorRetriever:
    """
    Build a paper's retriever from its Chroma collection (summary
    vectors) and its records (raw content, keyed by doc_id).

    The MultiVector pattern is unchanged: summaries are what gets
    searched, records are what gets returned. search_kwargs carries
    config.RETRIEVAL_TOP_K, which the original construction never
    passed (so Chroma's default k silently applied).
    """
    docstore = InMemoryStore()
    docstore.mset(list(records.items()))
    return MultiVectorRetriever(
        vectorstore=_vectorstore(paper_id),
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": config.RETRIEVAL_TOP_K},
        # Read by embeddings.retriever.front_matter().
        metadata={"front_matter": _front_matter(records)},
    )


def _front_matter(records: dict) -> list:
    """
    The paper's opening text records - title, authors, abstract: the
    first body-text records on the first page, up to
    config.FRONT_MATTER_MAX_CHARS. Records are stored in reading
    order, so these are simply the first ones.
    """
    opening = []
    total = 0
    first_page = None
    for record in records.values():
        if record.get("type") != "text" or record.get("section", "body") != "body":
            continue
        if first_page is None:
            first_page = record.get("page_number")
        if record.get("page_number") != first_page:
            break
        total += len(record["content"])
        if opening and total > config.FRONT_MATTER_MAX_CHARS:
            break
        opening.append(record)
    return opening


def _table_snapshot(image_path) -> str | None:
    """Base64 of a table's cropped snapshot, for display in the UI only."""
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        return encode_image(image_path)
    except OSError:
        return None


def ingest_paper(
    parsed_data: dict,
    paper_id: str,
    clients: LLMClients,
    meta: dict | None = None,
    progress=None,
) -> MultiVectorRetriever:
    """
    Take the output of parsers.pdf_parser.parse_pdf() and build a
    paper-scoped multi-vector retriever: summarizes text/tables/images,
    embeds the summaries, and stores raw content for retrieval - in
    memory for this process and on disk for later ones.

    Args:
        parsed_data: dict with text_chunks, images, tables (from parse_pdf)
        paper_id: unique identifier for this paper, used as the Chroma
                  collection name so multiple papers don't collide
        clients: the LLM clients bound to the current visitor's keys
        meta: extra fields to record in the paper's manifest
              (file name, hash, page count, ...)
        progress: optional callable(message: str, fraction: float)

    Returns:
        a MultiVectorRetriever scoped to this paper. The paper's
        manifest (counts, warnings) is available via
        embeddings.store.load_manifest(paper_id).
    """
    text_chunks = parsed_data["text_chunks"]
    tables = parsed_data["tables"]
    images = parsed_data["images"]

    text_strings = [chunk["text"] for chunk in text_chunks]
    table_strings = [table["text"] for table in tables]

    logger.info(
        "[ingest_paper:%s] %d text chunks, %d tables, %d images",
        paper_id, len(text_strings), len(table_strings), len(images),
    )

    # Fractions of the progress bar given to each stage.
    _report(progress, f"Summarising {len(text_strings)} text passages", 0.0)
    text_summaries, text_failed = summarize_texts(
        text_strings, clients,
        lambda done, total: _report(
            progress, f"Summarising text passages ({done}/{total})", 0.5 * done / total
        ),
    )

    _report(progress, f"Summarising {len(table_strings)} tables", 0.5)
    table_summaries, table_failed = summarize_tables(
        table_strings, clients,
        lambda done, total: _report(
            progress, f"Summarising tables ({done}/{total})", 0.5 + 0.1 * done / total
        ),
    )

    _report(progress, f"Indexing {len(images)} figures", 0.6)
    kept_images, image_summaries, image_failed = index_figures(
        images, clients,
        lambda done, total: _report(
            progress, f"Indexing figures ({done}/{total})", 0.6 + 0.3 * done / total
        ),
    )

    # Pair every summary with the record it should retrieve.
    records: dict = {}
    summary_docs: list[Document] = []

    def add(summary: str, record: dict) -> None:
        doc_id = str(uuid.uuid4())
        records[doc_id] = record
        # "section" is set on every vector (not only reference chunks)
        # so that a "$ne references" filter never depends on how the
        # vector store treats a missing field.
        metadata = {
            "doc_id": doc_id,
            "type": record["type"],
            "section": record.get("section", "body"),
        }
        if record.get("page_number") is not None:
            metadata["page_number"] = int(record["page_number"])
        if record.get("label"):
            # Lets "explain Figure 9" fetch Figure 9 itself (retriever.py).
            metadata["label"] = record["label"]
        summary_docs.append(Document(page_content=summary, metadata=metadata))

    for chunk, summary in zip(text_chunks, text_summaries):
        add(summary, {
            "type": "text",
            "content": chunk["text"],
            "page_number": chunk.get("page_number"),
            "page_end": chunk.get("page_end") or chunk.get("page_number"),
            "section": chunk.get("section", "body"),
        })
    for table, summary in zip(tables, table_summaries):
        add(summary, {
            "type": "table",
            "content": table["text"],
            "page_number": table.get("page_number"),
            "image": _table_snapshot(table.get("image_path")),
            "caption": table.get("caption") or "",
            "label": table.get("label") or "",
        })
    for image, summary in zip(kept_images, image_summaries):
        add(summary, {
            "type": "image",
            "content": image["base64"],
            "page_number": image.get("page_number"),
            "caption": image["caption"],
            "label": image["label"],
            "ocr_text": image["ocr_text"],
        })

    if not summary_docs:
        raise IngestionError(
            "No text, tables or figures could be extracted from this PDF. "
            "It may be a scanned document without readable text."
        )

    _report(progress, "Indexing the paper", 0.9)

    warnings = []
    if text_failed:
        warnings.append(
            f"{text_failed} of {len(text_strings)} text passages could not be summarised "
            "and were indexed by their raw text instead."
        )
    if table_failed:
        warnings.append(
            f"{table_failed} of {len(table_strings)} tables could not be summarised "
            "and were indexed by their raw text instead."
        )
    if image_failed:
        warnings.append(
            f"{image_failed} of {len(kept_images)} figures could not be described by the "
            "vision model; they are indexed by their captions instead."
        )

    manifest = {
        **(meta or {}),
        "text_chunks": len(text_strings),
        "tables": len(table_strings),
        "images": len(kept_images),
        "images_skipped": parsed_data.get("images_skipped", 0),
        "warnings": warnings,
    }

    # Start from an empty collection: a previous, interrupted ingestion
    # of the same paper may have left vectors behind.
    _drop_collection(paper_id)
    try:
        _vectorstore(paper_id).add_documents(summary_docs)
        store.save_paper(paper_id, manifest, records)
    except Exception:
        # Never leave vectors without a manifest, or a manifest without vectors.
        _drop_collection(paper_id)
        store.delete_paper_files(paper_id)
        raise

    retriever = build_retriever(paper_id, records)

    _report(progress, "Paper ready", 1.0)
    logger.info(
        "[ingest_paper:%s] Done. %d text, %d tables, %d images indexed.",
        paper_id, len(text_strings), len(table_strings), len(kept_images),
    )
    return retriever


def load_retriever(paper_id: str) -> MultiVectorRetriever | None:
    """
    Rebuild the retriever of a previously ingested paper from disk.
    Returns None if the paper is not stored, or if its vectors and its
    records no longer match (in which case it must be re-ingested).
    """
    if store.load_manifest(paper_id) is None:
        return None
    records = store.load_records(paper_id)
    if not records:
        return None

    try:
        vector_count = _vectorstore(paper_id)._collection.count()
    except Exception as exc:  # noqa: BLE001 - any Chroma failure means "not usable"
        logger.warning("Could not open collection for %s: %s", paper_id, exc)
        return None

    if vector_count != len(records):
        logger.warning(
            "Paper %s is inconsistent on disk (%d vectors, %d records); it will be re-ingested.",
            paper_id, vector_count, len(records),
        )
        return None

    return build_retriever(paper_id, records)


def _drop_collection(paper_id: str) -> None:
    try:
        _vectorstore(paper_id).delete_collection()
    except Exception as exc:  # noqa: BLE001 - nothing to delete is fine
        logger.debug("delete_collection(%s): %s", paper_id, exc)


def delete_paper(paper_id: str) -> None:
    """Remove a paper's Chroma collection and its stored content."""
    _drop_collection(paper_id)
    store.delete_paper_files(paper_id)
    logger.info("[delete_paper] Deleted %s", paper_id)
