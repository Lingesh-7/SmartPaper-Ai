"""
Retrieval module for SmartPaper AI.

Wraps a paper's MultiVectorRetriever with a clean retrieve() function
that returns typed context dicts (content, type, page_number, ...),
and provides the image helper the agent's reasoner node needs to build
multimodal prompts.

What changed from the first modular version, and why:

- The docstore now holds typed records (see embeddings/store.py), so
  retrieve() reads each result's type and page number directly. The
  old helpers that guessed "is this string an image?" by decoding
  base64 header bytes (looks_like_base64, is_image_data,
  split_image_text_types) are gone, along with the rule that any
  string starting with "table on page" must be a table.
- page_number is real. It used to be hardcoded to None because the
  docstore stored bare strings.
- top_k is honoured. It used to be accepted and ignored.
- A question about a table or a figure is guaranteed to get one.
  Plain similarity search ranks every summary together, and a paper
  has dozens of text passages for each table, so "what are the key
  results in the tables?" used to come back with four text passages
  and no table at all. With prefer_type set, the best-matching
  records of that type are fetched with a metadata filter and placed
  first; ordinary results fill the remaining slots.
- A question that names a figure or table gets that one. "Explain
  Figure 9" is matched on the label the parser read from the caption,
  not left to embedding similarity - which has no idea that "9" is
  the important word.
- The reference list is searched only when the question is about
  references or citations. Its entries are full of the paper's own
  keywords and would otherwise crowd out the paper's actual text.
- front_matter() exposes the paper's title/authors/abstract records so
  the agent can give them to the model with every question.
- exclude_type keeps figures out of the context of a question that is
  not about one. A figure in the context means a vision request, and
  those are the scarcest thing a free-tier visitor has.
- Images are no longer forced to 1300x600. The notebook's
  resize_base64_image() stretched every figure to that exact size,
  which distorts any figure that is not roughly 2:1 (a tall
  architecture diagram becomes a squashed one) before the vision
  model sees it. prepare_image_for_llm() only ever shrinks, and keeps
  the aspect ratio.
"""

import base64
import io
import logging
import re

from PIL import Image as PILImage
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

import config

logger = logging.getLogger("smartpaper.retriever")


def prepare_image_for_llm(base64_string: str, max_side: int = 1300) -> str:
    """
    Return a base64 JPEG no larger than max_side on its longest edge.

    Images already within the limit are returned untouched. Anything
    that cannot be decoded is also returned untouched, so a single odd
    image degrades to "sent as-is" rather than an exception.
    """
    try:
        img = PILImage.open(io.BytesIO(base64.b64decode(base64_string)))
        if max(img.size) <= max_side and img.format == "JPEG":
            return base64_string
        img.thumbnail((max_side, max_side), PILImage.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not resize image for the vision model: %s", exc)
        return base64_string


# "figure 9", "Fig. 3", "fig 12", "Table 2", "table II" - but only
# roman numerals written in capitals, so that "table in" or "figure i
# mentioned" is not read as Table 1 / Figure 1.
_LABEL_RE = re.compile(
    r"\b(fig(?:ure)?s?\.?|tables?)\s*"
    r"((?:\d{1,3}|[IVXLC]{1,6})(?:\s*(?:,\s*and|,|and|&)\s*\d{1,3})*)\b"
)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(text: str) -> int:
    total, previous = 0, 0
    for char in reversed(text):
        value = _ROMAN[char]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total


def labels_in_query(query: str) -> list[str]:
    """
    The figures and tables a question names, as labels in the form the
    parser stores them: "Explain Figure 9 and Table II" ->
    ["Figure 9", "Table 2"]; "compare tables 4 and 5" ->
    ["Table 4", "Table 5"].
    """
    labels = []
    for match in re.finditer(_LABEL_RE.pattern, query, flags=re.IGNORECASE):
        kind = "Figure" if match.group(1).lower().startswith("fig") else "Table"
        for number in re.findall(r"\d{1,3}|[IVXLCivxlc]{1,6}", match.group(2)):
            if number.isdigit():
                value = int(number)
            elif number.isupper():
                value = _roman_to_int(number)
            else:
                continue      # "and", or a lower-case word that only looks roman
            label = f"{kind} {value}"
            if label not in labels:
                labels.append(label)
    return labels[:config.RETRIEVAL_TOP_K]


_ABOUT_REFERENCES_RE = re.compile(
    r"\b(references?|cit(?:e|es|ed|ing|ation|ations)|bibliograph\w*|related work|prior work)\b",
    re.IGNORECASE,
)


def asks_about_references(query: str) -> bool:
    """True if the question is about what the paper cites."""
    return _ABOUT_REFERENCES_RE.search(query) is not None


def front_matter(retriever: MultiVectorRetriever) -> list:
    """
    The paper's opening records (title, authors, abstract) as result
    dicts marked "pinned", or [] if the retriever carries none.
    """
    records = (getattr(retriever, "metadata", None) or {}).get("front_matter") or []
    results = []
    for record in records:
        result = _as_result(record)
        if result:
            result["pinned"] = True
            results.append(result)
    return results


def _as_result(record) -> dict | None:
    if not isinstance(record, dict) or "content" not in record:
        logger.warning("Skipping unexpected docstore value of type %s", type(record).__name__)
        return None
    return {
        "content": record["content"],
        "type": record.get("type", "text"),
        "page_number": record.get("page_number"),
        "page_end": record.get("page_end") or record.get("page_number"),
        "image": record.get("image"),
        "caption": record.get("caption") or "",
        "label": record.get("label") or "",
        "ocr_text": record.get("ocr_text") or "",
        "section": record.get("section") or "body",
        "pinned": False,
    }


def _filtered_records(retriever: MultiVectorRetriever, query: str, where: dict, k: int) -> list:
    """The k records matching a metadata filter whose index text best matches the query."""
    summaries = retriever.vectorstore.similarity_search(query, k=k, filter=where)
    doc_ids = []
    for summary in summaries:
        doc_id = summary.metadata.get(retriever.id_key)
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    records = [record for record in retriever.docstore.mget(doc_ids) if record is not None]
    return [r for r in (_as_result(record) for record in records) if r]


def _identity(result: dict) -> tuple:
    """What makes two results the same record (the docstore id is not carried along)."""
    return (result["type"], result["page_number"], result["label"], result["caption"], result["content"])


def retrieve(
    retriever: MultiVectorRetriever,
    query: str,
    top_k: int | None = None,
    prefer_type: str | None = None,
    exclude_type: str | None = None,
) -> list:
    """
    Query a paper's retriever and return a list of typed context dicts.

    Args:
        retriever: the MultiVectorRetriever for a specific paper
                   (from embedder.ingest_paper / embedder.load_retriever)
        query: the search query
        top_k: number of results to return. None uses the retriever's
               own setting (config.RETRIEVAL_TOP_K).
        prefer_type: "table" or "image" to guarantee that the paper's
                     best-matching records of that type come first
                     (up to config.MODALITY_TOP_K of them). If the
                     paper has no record of that type, the result is
                     the ordinary search.
        exclude_type: a record type to leave out of the ordinary
                      search (a figure or table the query names is
                      still returned).

    Returns:
        list of at most top_k dicts, in this order: records the query
        names by label, then preferred-type records, then the rest.
            {"content": <text, table string, or base64 image>,
             "type": "text" | "table" | "image",
             "page_number": <int or None>, "page_end": <int or None>,
             "image": <base64 snapshot of a table, or None>,
             "caption": <str>, "label": <"Figure 5" / "Table 2" / "">,
             "ocr_text": <text read inside a figure>,
             "section": "body" | "references", "pinned": False}
        Reference-list chunks are left out unless the query is about
        references or citations.
    """
    k = top_k if top_k is not None else retriever.search_kwargs.get("k", config.RETRIEVAL_TOP_K)

    conditions = []
    if exclude_type:
        conditions.append({"type": {"$ne": exclude_type}})
    if not asks_about_references(query):
        conditions.append({"section": {"$ne": "references"}})

    search_kwargs = {**retriever.search_kwargs, "k": k}
    if len(conditions) == 1:
        search_kwargs["filter"] = conditions[0]
    elif conditions:
        search_kwargs["filter"] = {"$and": conditions}
    if search_kwargs != retriever.search_kwargs:
        # Copy rather than mutate: the retriever is shared between
        # every session that has this paper open.
        retriever = retriever.model_copy(update={"search_kwargs": search_kwargs})

    general = [r for r in (_as_result(record) for record in retriever.invoke(query)) if r]

    first = []
    for label in labels_in_query(query):
        first.extend(_filtered_records(retriever, query, {"label": label}, 1))
    if prefer_type is not None:
        first.extend(
            _filtered_records(retriever, query, {"type": prefer_type}, min(config.MODALITY_TOP_K, k))
        )

    results, seen = [], set()
    for result in first + general:
        key = _identity(result)
        if key not in seen:
            seen.add(key)
            results.append(result)
    return results[:k]
