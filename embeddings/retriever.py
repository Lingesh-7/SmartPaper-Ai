"""
Retrieval module for SmartPaper AI.

Wraps a paper's MultiVectorRetriever with a clean retrieve() function
that returns typed context dicts (content, type, page_number), and
provides the image/text splitting helpers the agent's reasoner node
needs to build multimodal prompts.
"""

import base64
import io
import re

from PIL import Image as PILImage
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever


def looks_like_base64(sb: str) -> bool:
    """Check if a string looks like base64."""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data: str) -> bool:
    """Check if base64 data is an image by inspecting its header bytes."""
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig in image_signatures:
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string: str, size: tuple = (1300, 600)) -> str:
    """Resize a base64-encoded image, returning a re-encoded base64 string."""
    img_data = base64.b64decode(base64_string)
    img = PILImage.open(io.BytesIO(img_data))
    resized_img = img.resize(size, PILImage.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs: list) -> dict:
    """
    Split a list of retrieved docs (raw strings or Documents) into
    base64 images vs. plain text/table strings.

    Returns:
        {"images": [...], "texts": [...]}
    """
    b64_images = []
    texts = []

    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content

        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)

    return {"images": b64_images, "texts": texts}


def retrieve(retriever: MultiVectorRetriever, query: str, top_k: int = 3) -> list:
    """
    Query a paper's retriever and return a list of typed context dicts.

    Args:
        retriever: the MultiVectorRetriever for a specific paper
                   (from embedder.ingest_paper)
        query: the search query
        top_k: number of results to retrieve (note: the underlying
               MultiVectorRetriever does not currently expose top_k
               directly in this pipeline's construction - this param
               is accepted for API compatibility with the Day 5 spec
               and reserved for when retriever search_kwargs are wired
               through; currently the retriever's own default k applies)

    Returns:
        list of {"content": ..., "type": "text"|"table"|"image", "page_number": ...}

    NOTE: page_number is not populated here. The original notebook's
    add_documents() stores raw content (plain strings / base64 images)
    in the docstore with NO metadata at all - page numbers from
    parse_pdf()'s text_chunks are dropped before they ever reach the
    vectorstore or docstore. To get real page citations working, the
    embedder's _add_documents() would need to store (content, page_number)
    tuples or a dict instead of bare strings, and this function would
    unpack that. That's flagged as a known gap rather than silently
    faked here.
    """
    docs = retriever.invoke(query)
    split = split_image_text_types(docs)

    results = []
    for text in split["texts"]:
        content_type = "table" if text.strip().lower().startswith("table on page") else "text"
        results.append({
            "content": text,
            "type": content_type,
            "page_number": None,
        })
    for image in split["images"]:
        results.append({
            "content": image,
            "type": "image",
            "page_number": None,
        })

    return results