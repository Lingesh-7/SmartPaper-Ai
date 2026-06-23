"""
LangGraph agent nodes for SmartPaper AI.

Four nodes: planner (classifies query type), retriever (fetches
context via embeddings.retriever), reasoner (generates an answer,
routing to a vision model when images are present), and critic
(accepts the answer or triggers a retry, up to 2 retries).

This mirrors the validated notebook logic from the LangGraph Agent
section, with one structural change: the original notebook reused a
single module-level `llm` variable that got reassigned three separate
times across the notebook (table summaries -> text summaries -> agent
reasoning), which only works because notebook cells run in a fixed
order. Here, the agent's text LLM and vision LLM are each given their
own explicitly named client, created once at module import time
(rather than on every reasoner_node call, which the notebook version
did unnecessarily).
"""

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from embeddings.retriever import retrieve

# --- LLM clients, created once at import time ---
agent_llm = ChatGroq(
    model=config.AGENT_LLM_MODEL,
    temperature=config.AGENT_LLM_TEMPERATURE,
)

agent_vision_llm = ChatGoogleGenerativeAI(
    model=config.AGENT_VISION_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0,
)

VALID_RETRIEVAL_TYPES = {"text_query", "figure_query", "table_query"}

REFUSAL_PHRASES = [
    "not enough", "cannot answer", "i don't know", "no information",
    "does not provide", "does not mention", "does not contain",
    "doesn't provide", "doesn't mention", "doesn't contain",
    "not provided in", "not available in the", "no mention of",
]


def planner_node(state: dict) -> dict:
    """Classify the question into text_query, figure_query, or table_query."""
    question = state["question"]

    classification_prompt = f"""Classify the following question into exactly one of these categories:
- text_query: asking about concepts, explanations, methods, or general content
- figure_query: asking about a figure, diagram, chart, or visualization
- table_query: asking about numbers, results, scores, or comparisons in a table

Question: {question}

Respond with ONLY one word: text_query, figure_query, or table_query"""

    response = agent_llm.invoke([HumanMessage(content=classification_prompt)])
    retrieval_type = response.content.strip().lower()

    if retrieval_type not in VALID_RETRIEVAL_TYPES:
        retrieval_type = "text_query"

    state["retrieval_type"] = retrieval_type
    print(f"[Planner] Classified as: {retrieval_type}")
    return state


def retriever_node(state: dict, paper_retriever) -> dict:
    """
    Fetch context for the question from the given paper's retriever.

    paper_retriever is the MultiVectorRetriever returned by
    embeddings.embedder.ingest_paper() for the paper currently in
    session - it is passed explicitly rather than read from global
    state, since a Flask app may be serving multiple papers/users
    concurrently and a single global retriever (as in the original
    notebook) would not be safe for that.
    """
    question = state["question"]
    context_list = retrieve(paper_retriever, question)

    texts = [c["content"] for c in context_list if c["type"] in ("text", "table")]
    images = [c["content"] for c in context_list if c["type"] == "image"]

    state["context"] = {"texts": texts, "images": images}
    print(f"[Retriever] Got {len(texts)} text/table chunks and {len(images)} images")
    return state


def reasoner_node(state: dict) -> dict:
    """Generate a draft answer, routing to the vision LLM when images are present."""
    question = state["question"]
    context = state["context"]
    formatted_texts = "\n".join(context["texts"])

    if context["images"]:
        content = []
        for image in context["images"]:
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            })
        content.append({
            "type": "text",
            "text": (
                "You are a research paper assistant. "
                "Answer ONLY using the provided image(s) and context. "
                "Provide specific details from figures, charts, tables, "
                "or diagrams when available. "
                "If the information is insufficient, say so clearly.\n\n"
                f"Question:\n{question}\n\n"
                f"Additional Context:\n{formatted_texts}"
            ),
        })
        response = agent_vision_llm.invoke([HumanMessage(content=content)])
    else:
        prompt = (
            "You are a research paper assistant. "
            "Answer ONLY using the provided context. "
            "Provide specific details from the context. "
            "If the information is insufficient, say so clearly.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{formatted_texts}"
        )
        response = agent_llm.invoke([HumanMessage(content=prompt)])

    state["draft_answer"] = response.content
    print("[Reasoner] Draft answer generated")
    return state


def critic_node(state: dict) -> dict:
    """Accept the draft answer, or trigger a retry (max 2) if it looks weak/refusal-like."""
    draft = state["draft_answer"]
    retry_count = state.get("retry_count", 0)

    is_long_enough = len(draft.strip().split(".")) >= 2
    is_not_refusal = not any(phrase in draft.lower() for phrase in REFUSAL_PHRASES)

    if (is_long_enough and is_not_refusal) or retry_count >= 2:
        state["final_answer"] = draft
        print(f"[Critic] Accepted answer (retry_count={retry_count})")
    else:
        state["retry_count"] = retry_count + 1
        state["question"] = state["question"] + " Please be more specific and detailed."
        print(f"[Critic] Rejected, retrying (retry_count={state['retry_count']})")

    return state


def should_continue(state: dict) -> str:
    """Conditional edge: route to END once final_answer is set, else back to retriever."""
    if state.get("final_answer"):
        return "end"
    return "retry"