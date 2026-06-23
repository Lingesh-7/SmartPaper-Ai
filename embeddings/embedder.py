"""
Embedding module for SmartPaper AI.

Summarizes tables (Groq), images (Gemini), and text (OpenRouter Llama),
then builds a Chroma-backed MultiVectorRetriever that indexes summaries
but returns raw content (text/table strings, base64 images) at query
time.

NOTE on the original notebook: it reused a single mutable `llm`
variable across table summarization, text summarization, and the
agent's planner/reasoner, reassigning it three separate times
(ChatGroq -> ChatOpenRouter -> ChatGroq again). That works in a
notebook because cells run top-to-bottom in a fixed order, but it's
fragile and confusing in a module. Here each summarization step gets
its own explicitly named client so there's no ambiguity about which
model answers which call.
"""

import base64
import os
import uuid

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

import config

config.validate_config()

# --- Embedding model (shared across text, table, image summary vectors) ---
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


# --- Table summarization (Groq) ---
_table_summary_llm = ChatGroq(
    model=config.TABLE_SUMMARY_MODEL,
    temperature=config.TABLE_SUMMARY_TEMPERATURE,
)

_table_prompt = ChatPromptTemplate.from_template(
    """
You are an assistant tasked with summarizing tables for retrieval.
These summaries will be embedded and used to retrieve the raw table elements.
Give a concise summary of the table that is well optimized for retrieval
Table:
{table}
"""
)

_table_summarize_chain = (
    {"table": RunnableLambda(lambda x: x)}
    | _table_prompt
    | _table_summary_llm
    | StrOutputParser()
)


def summarize_tables(tables: list[str]) -> list[str]:
    """Summarize a list of table strings for retrieval embedding."""
    if not tables:
        return []
    return _table_summarize_chain.batch(tables, config={"max_concurrency": 12})


# --- Image summarization (Gemini) ---
_image_summary_llm = ChatGoogleGenerativeAI(
    model=config.IMAGE_SUMMARY_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=config.IMAGE_SUMMARY_TEMPERATURE,
)

_IMAGE_SUMMARY_PROMPT = (
    "You are an assistant tasked with summarizing images for retrieval. "
    "Give a concise summary optimized for retrieval."
)


def encode_image(image_path: str) -> str:
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _summarize_image(img_base64: str, prompt: str = _IMAGE_SUMMARY_PROMPT) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img_base64}",
            },
        ]
    )
    response = _image_summary_llm.invoke([message])
    return response.content


def generate_img_summaries(image_paths: list[str]) -> tuple[list[str], list[str]]:
    """
    Generate base64 encodings + retrieval summaries for a list of image
    file paths (as returned by parsers.pdf_parser.parse_pdf).

    Returns:
        (img_base64_list, image_summaries) - parallel lists.
    """
    img_base64_list = []
    image_summaries = []

    for file_path in image_paths:
        if not os.path.isfile(file_path):
            continue
        if not file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            print(f"Processing: {file_path}")
            base64_image = encode_image(file_path)
            summary = _summarize_image(base64_image)
            img_base64_list.append(base64_image)
            image_summaries.append(summary)
        except Exception as e:
            print(f"Skip: {file_path} ({e})")

    return img_base64_list, image_summaries


# --- Text summarization (OpenRouter Llama-3.3-70b) ---
# Uses ChatOpenRouter, matching the validated notebook. If
# langchain_openrouter is unavailable in your environment, swap this
# for ChatGroq with the same model family - the rest of the pipeline
# doesn't care which provider produced the summary text.
from langchain_openrouter import ChatOpenRouter  # noqa: E402

_text_summary_llm = ChatOpenRouter(
    model=config.TEXT_SUMMARY_MODEL,
    temperature=config.TEXT_SUMMARY_TEMPERATURE,
)

_text_prompt = ChatPromptTemplate.from_template(
    """You are an assistant tasked with summarizing text for retrieval.
These summaries will be embedded and used to retrieve the raw text elements.
Give a concise summary of the text that is well optimized for retrieval
Text:
{text}"""
)

_text_summarize_chain = (
    {"text": RunnableLambda(lambda x: x)}
    | _text_prompt
    | _text_summary_llm
    | StrOutputParser()
)


def summarize_texts(texts: list[str]) -> list[str]:
    """Summarize a list of NarrativeText strings for retrieval embedding."""
    if not texts:
        return []
    return _text_summarize_chain.batch(texts, config={"max_concurrency": 5})


# --- Multi-vector retriever construction ---
def _add_documents(retriever: MultiVectorRetriever, doc_summaries: list[str], doc_content: list):
    """Embed summaries into the vectorstore, store raw content in the docstore."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_content]
    summary_docs = [
        Document(page_content=s, metadata={"doc_id": doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_content)))


def create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    images,
) -> MultiVectorRetriever:
    """
    Create a retriever that indexes summaries but returns raw content
    (text strings, table strings, or base64 images) on retrieval.
    """
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
    )

    if text_summaries:
        _add_documents(retriever, text_summaries, texts)
    if table_summaries:
        _add_documents(retriever, table_summaries, tables)
    if image_summaries:
        _add_documents(retriever, image_summaries, images)

    return retriever


def ingest_paper(parsed_data: dict, paper_id: str) -> MultiVectorRetriever:
    """
    Take the output of parsers.pdf_parser.parse_pdf() and build a fresh,
    paper-scoped multi-vector retriever: summarizes text/tables/images,
    embeds the summaries, and stores raw content for retrieval.

    Args:
        parsed_data: dict with text_chunks, images, tables (from parse_pdf)
        paper_id: unique identifier for this paper, used as the Chroma
                  collection name so multiple papers don't collide

    Returns:
        a compiled MultiVectorRetriever scoped to this paper
    """
    text_strings = [chunk["text"] for chunk in parsed_data["text_chunks"]]
    table_strings = parsed_data["tables"]
    image_paths = parsed_data["images"]

    print(f"[ingest_paper:{paper_id}] Summarizing {len(text_strings)} text chunks...")
    text_summaries = summarize_texts(text_strings)

    print(f"[ingest_paper:{paper_id}] Summarizing {len(table_strings)} tables...")
    table_summaries = summarize_tables(table_strings)

    print(f"[ingest_paper:{paper_id}] Summarizing {len(image_paths)} images...")
    img_base64_list, image_summaries = generate_img_summaries(image_paths)

    collection_name = f"{config.CHROMA_COLLECTION_NAME}_{paper_id}"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    retriever = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        text_strings,
        table_summaries,
        table_strings,
        image_summaries,
        img_base64_list,
    )

    print(f"[ingest_paper:{paper_id}] Done. "
          f"{len(text_strings)} text, {len(table_strings)} tables, "
          f"{len(img_base64_list)} images indexed.")

    return retriever


def delete_paper(paper_id: str):
    """Remove a paper's Chroma collection for cleanup between uploads."""
    collection_name = f"{config.CHROMA_COLLECTION_NAME}_{paper_id}"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    vectorstore.delete_collection()
    print(f"[delete_paper] Deleted collection: {collection_name}")