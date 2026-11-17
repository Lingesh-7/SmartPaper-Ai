"""
LangGraph agent nodes for SmartPaper AI.

Five nodes: contextualizer (rewrites a follow-up into a standalone
question), planner (classifies query type), retriever (fetches context
via embeddings.retriever), reasoner (generates an answer, routing to a
vision model when images are present), and critic (accepts the answer
or triggers a retry, up to 2 retries).

This mirrors the validated notebook logic from the LangGraph Agent
section, with these structural changes:

- The original notebook reused a single module-level `llm` variable
  that got reassigned three separate times across the notebook (table
  summaries -> text summaries -> agent reasoning), which only works
  because notebook cells run in a fixed order. Here the agent's text
  LLM and vision LLM are separate, explicitly named clients
  (clients.agent_llm, clients.agent_vision_llm).
- Those clients are passed into the nodes rather than created at
  module import. Every visitor brings their own API keys, so a client
  created at import would answer everyone's questions on one person's
  key. agents/graph.py binds the clients by closure, the same way it
  binds the paper's retriever.
- state["context"] is the list of retrieved records, each with its
  type and page number, rather than {"texts": [...], "images": [...]}.
  The reasoner labels every piece of context with its page so the
  model can cite it, and the records are returned to the UI as the
  answer's sources.
- The paper's title, authors and abstract are added to the context of
  every question, and a retry searches wider than the attempt before
  it. Both come from watching real questions fail: "who wrote this
  paper?" retrieved the reference list, and retrying a rejected
  answer with the identical context produced the identical answer.
- The contextualizer is new. Without it every question is answered in
  isolation, so "explain that further" retrieves nothing useful.
- The planner's classification now steers retrieval and reasoning. It
  used to be informational only, so a question about a table could be
  answered from four text passages with no table among them. Now a
  table_query is guaranteed the paper's best-matching tables and a
  figure_query its best-matching figures (embeddings/retriever.py),
  the reasoner is told to answer from them, and for a table_query it
  is also shown the table's image: unstructured's text dump of a
  table loses the row/column structure that the numbers depend on.
- The vision model is used ONLY for figure and table questions, and a
  failed vision call falls back to the text model. Gemini's free tier
  allows about 20 requests per day. Before, any figure that happened
  to be retrieved - even for "what datasets were used?" - turned the
  answer into a vision request, and once the allowance was gone every
  such question failed. Now a text question never retrieves a figure,
  and when the vision model is unavailable the question is answered
  from captions, the text read inside the figures, and the tables'
  text, with the answer marked as such (state["vision_fallback"]).
"""

import logging

from langchain_core.messages import HumanMessage

import config
from embeddings.retriever import front_matter, prepare_image_for_llm, retrieve
from models.llm_clients import LLMClients, message_text

logger = logging.getLogger("smartpaper.agent")

VALID_RETRIEVAL_TYPES = {"text_query", "figure_query", "table_query"}

REFUSAL_PHRASES = [
    # The reasoner's prompt says "If the information is insufficient,
    # say so clearly" - so this is the wording a refusal usually takes.
    "insufficient information", "insufficient context",
    "not enough", "cannot answer", "i don't know", "no information",
    "does not provide", "does not mention", "does not contain",
    "doesn't provide", "doesn't mention", "doesn't contain",
    "not provided in", "not available in the", "no mention of",
]

# Citations are written "(page 4)", never "(p. 4)": critic_node counts
# sentences by splitting on ".", and an abbreviation with a period
# would make every cited answer look long enough.
_CITATION_RULE = (
    "Each piece of context is labelled with the page it comes from. "
    "When you use one, cite it in the form (page N). "
    "Only cite page numbers that appear in the labels."
)

_HISTORY_CHARS_PER_MESSAGE = 600

# Which kind of record a classified question must be answered from.
_PREFERRED_RECORD_TYPE = {"table_query": "table", "figure_query": "image"}

_FOCUS_RULE = {
    "table_query": (
        "The question is about a table. Base your answer on the table(s) provided, "
        "reading values from the table image when one is given. "
        "Refer to a table by its number when it has one. "
    ),
    "figure_query": (
        "The question is about a figure. Base your answer on the figure(s) provided. "
        "Refer to a figure by its number when it has one. "
    ),
}


def contextualizer_node(state: dict, clients: LLMClients) -> dict:
    """
    Rewrite a follow-up question into a standalone one using the chat
    history. With no history (the first question) this is a no-op and
    makes no LLM call. Any failure falls back to the question as asked.
    """
    question = state["question"]
    state["original_question"] = question
    state["standalone_question"] = question

    history = (state.get("chat_history") or [])[-config.HISTORY_MESSAGES:]
    if not history:
        return state

    transcript = "\n".join(
        f"{'User' if turn.get('role') == 'user' else 'Assistant'}: "
        f"{str(turn.get('content', ''))[:_HISTORY_CHARS_PER_MESSAGE]}"
        for turn in history
    )
    prompt = (
        "Below is a conversation about a research paper, followed by a new question.\n"
        "Rewrite the new question so that it can be understood without the conversation, "
        "replacing words like 'it', 'that' or 'this' with what they refer to.\n"
        "If the new question is already self-contained, return it unchanged.\n"
        "Do not answer the question. Respond with ONLY the rewritten question.\n\n"
        f"Conversation:\n{transcript}\n\n"
        f"New question: {question}\n\n"
        "Rewritten question:"
    )

    try:
        response = clients.agent_llm.invoke([HumanMessage(content=prompt)])
        rewritten = message_text(response).strip().strip('"').strip()
    except Exception as exc:  # noqa: BLE001 - a rewrite is an optimisation, never fatal
        logger.warning("[Contextualizer] Failed, using the question as asked: %s", type(exc).__name__)
        return state

    if rewritten and len(rewritten) <= config.MAX_QUESTION_CHARS:
        state["question"] = rewritten
        state["standalone_question"] = rewritten
        logger.info("[Contextualizer] Rewrote follow-up question")
    return state


def planner_node(state: dict, clients: LLMClients) -> dict:
    """Classify the question into text_query, figure_query, or table_query."""
    question = state["question"]

    classification_prompt = f"""Classify the following question into exactly one of these categories:
- text_query: asking about concepts, explanations, methods, or general content
- figure_query: asking about a figure, diagram, chart, or visualization
- table_query: asking about numbers, results, scores, or comparisons in a table

Question: {question}

Respond with ONLY one word: text_query, figure_query, or table_query"""

    response = clients.agent_llm.invoke([HumanMessage(content=classification_prompt)])
    retrieval_type = message_text(response).strip().lower()

    if retrieval_type not in VALID_RETRIEVAL_TYPES:
        retrieval_type = "text_query"

    state["retrieval_type"] = retrieval_type
    logger.info("[Planner] Classified as: %s", retrieval_type)
    return state


def retriever_node(state: dict, paper_retriever) -> dict:
    """
    Fetch context for the question from the given paper's retriever.

    paper_retriever is the MultiVectorRetriever returned by
    embeddings.embedder.ingest_paper() for the paper currently in
    session - it is passed explicitly rather than read from global
    state, since the app may be serving multiple papers/users
    concurrently and a single global retriever (as in the original
    notebook) would not be safe for that.
    """
    question = state["question"]
    retrieval_type = state.get("retrieval_type")
    # A retry after the critic rejected an answer searches wider. The
    # same search would return the same context and, mostly, the same
    # answer.
    widen = min(state.get("retry_count", 0) + 1, config.RETRY_WIDEN_MAX)
    records = retrieve(
        paper_retriever,
        question,
        top_k=config.RETRIEVAL_TOP_K * widen,
        prefer_type=_PREFERRED_RECORD_TYPE.get(retrieval_type),
        # Only a question about a figure gets figures: each one in the
        # context costs a vision request.
        exclude_type=None if retrieval_type == "figure_query" else "image",
    )

    # The paper's title, authors and abstract ride along with every
    # question (see config.FRONT_MATTER_MAX_CHARS for why).
    found = {record["content"] for record in records}
    records = records + [r for r in front_matter(paper_retriever) if r["content"] not in found]

    state["context"] = records
    n_images = sum(1 for r in records if r["type"] == "image")
    logger.info(
        "[Retriever] Got %d text/table chunks and %d images",
        len(records) - n_images, n_images,
    )
    return state


def _page_label(record: dict) -> str:
    page = record.get("page_number")
    if page is None:
        return "page unknown"
    page_end = record.get("page_end")
    if page_end is not None and page_end != page:
        return f"pages {page}-{page_end}"
    return f"page {page}"


def _record_name(record: dict, fallback: str) -> str:
    """ "Figure 5" / "Table 2" when the paper numbers it, else the generic word."""
    return record.get("label") or fallback


def _format_text_context(records: list) -> str:
    """
    Join the records into one context string, each labelled with its
    page. A figure contributes what is known about it in words - its
    caption and the text printed inside it - so that it is part of the
    context even when no model can look at the image.
    """
    blocks = []
    for record in records:
        if record["type"] == "image":
            lines = [f"[{_record_name(record, 'Figure')}, {_page_label(record)}]"]
            if record.get("caption"):
                lines.append(record["caption"])
            if record.get("ocr_text"):
                lines.append(f"Text printed in the figure: {record['ocr_text']}")
            if len(lines) > 1:
                blocks.append("\n".join(lines))
            continue
        if record["type"] == "table":
            kind = _record_name(record, "Table")
        elif record.get("pinned"):
            kind = "Title and abstract"
        else:
            kind = "Text"
        blocks.append(f"[{kind}, {_page_label(record)}]\n{record['content']}")
    return "\n\n".join(blocks)


def _text_answer(question: str, formatted_texts: str, focus_rule: str, clients: LLMClients):
    prompt = (
        "You are a research paper assistant. "
        "Answer ONLY using the provided context. "
        "Provide specific details from the context. "
        f"{focus_rule}"
        f"{_CITATION_RULE} "
        "If the information is insufficient, say so clearly.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{formatted_texts}"
    )
    return clients.agent_llm.invoke([HumanMessage(content=prompt)])


def reasoner_node(state: dict, clients: LLMClients) -> dict:
    """
    Generate a draft answer. A question about a figure or a table is
    shown the retrieved figures / the retrieved tables' snapshots by
    the vision LLM; everything else, and any question for which the
    vision call fails, is answered by the text LLM.
    """
    question = state["question"]
    records = state["context"]
    retrieval_type = state.get("retrieval_type")
    formatted_texts = _format_text_context(records)
    focus_rule = _FOCUS_RULE.get(retrieval_type, "")

    # (kind, record, base64) in the order the model should see them.
    visuals = []
    if retrieval_type == "figure_query":
        visuals = [("Figure", r, r["content"]) for r in records if r["type"] == "image"]
    elif retrieval_type == "table_query":
        visuals = [("Table", r, r["image"]) for r in records if r["type"] == "table" and r.get("image")]

    response = None
    if visuals:
        content = []
        counters = {"Figure": 0, "Table": 0}
        for kind, record, image_base64 in visuals:
            counters[kind] += 1
            name = _record_name(record, f"{kind} {counters[kind]}")
            caption = f" {record['caption']}" if record.get("caption") else ""
            content.append({
                "type": "text",
                "text": f"{name} ({_page_label(record)}):{caption}",
            })
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{prepare_image_for_llm(image_base64)}",
            })
        content.append({
            "type": "text",
            "text": (
                "You are a research paper assistant. "
                "Answer ONLY using the provided image(s) and context. "
                "Provide specific details from figures, charts, tables, "
                "or diagrams when available. "
                f"{focus_rule}"
                f"{_CITATION_RULE} "
                "If the information is insufficient, say so clearly.\n\n"
                f"Question:\n{question}\n\n"
                f"Additional Context:\n{formatted_texts}"
            ),
        })
        try:
            response = clients.agent_vision_llm.invoke([HumanMessage(content=content)])
            state["vision_fallback"] = False
        except Exception as exc:  # noqa: BLE001 - answer from text rather than not at all
            logger.warning(
                "[Reasoner] Vision model unavailable (%s); answering from text instead",
                type(exc).__name__,
            )
            state["vision_fallback"] = True

    if response is None:
        response = _text_answer(question, formatted_texts, focus_rule, clients)

    state["draft_answer"] = message_text(response)
    logger.info("[Reasoner] Draft answer generated")
    return state


def critic_node(state: dict) -> dict:
    """Accept the draft answer, or trigger a retry (max 2) if it looks weak/refusal-like."""
    draft = state["draft_answer"]
    retry_count = state.get("retry_count", 0)

    is_long_enough = len(draft.strip().split(".")) >= 2
    is_not_refusal = not any(phrase in draft.lower() for phrase in REFUSAL_PHRASES)

    if (is_long_enough and is_not_refusal) or retry_count >= 2:
        state["final_answer"] = draft
        logger.info("[Critic] Accepted answer (retry_count=%d)", retry_count)
    else:
        state["retry_count"] = retry_count + 1
        state["question"] = state["question"] + " Please be more specific and detailed."
        logger.info("[Critic] Rejected, retrying (retry_count=%d)", state["retry_count"])

    return state


def should_continue(state: dict) -> str:
    """Conditional edge: route to END once final_answer is set, else back to retriever."""
    if state.get("final_answer"):
        return "end"
    return "retry"
