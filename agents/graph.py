"""
LangGraph pipeline setup for SmartPaper AI.

create_graph(paper_retriever, clients) builds and compiles the
StateGraph for a specific paper's retriever and a specific visitor's
LLM clients. run_agent(...) compiles a graph bound to both and runs
it, returning the final answer plus supporting metadata.

NOTE on design: the original notebook's retriever_node closed over a
single global retriever_multi_vector_img variable. Since this module
needs to support multiple papers (each with its own retriever from
embeddings.embedder.ingest_paper), retriever_node takes the retriever
as an explicit argument. To keep the LangGraph node signature
compatible with StateGraph.add_node (which expects a single-argument
function: state -> state), this module binds the paper's retriever
via a closure when building the graph, rather than changing
LangGraph's calling convention. The visitor's LLM clients are bound
the same way, for the same reason: they differ per request and must
not be module-level state.

This module no longer keeps a paper_id -> retriever registry.
run_agent() takes the retriever itself; looking a paper up (from
memory, or from disk after a restart) is pipeline.py's job.
"""

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.nodes import (
    contextualizer_node,
    planner_node,
    retriever_node,
    reasoner_node,
    critic_node,
    should_continue,
)

logger = logging.getLogger("smartpaper.agent")


class AgentState(TypedDict, total=False):
    question: str              # the question the nodes work on (may be rewritten)
    original_question: str     # exactly what the user typed
    standalone_question: str   # after the contextualizer, before any critic retry suffix
    chat_history: list         # [{"role": "user"|"assistant", "content": str}, ...]
    retrieval_type: str
    context: list              # retrieved records (see embeddings.retriever.retrieve)
    vision_fallback: bool      # True if the vision model failed and text answered instead
    draft_answer: str
    final_answer: str
    retry_count: int


def create_graph(paper_retriever, clients):
    """
    Build and compile the LangGraph StateGraph, with retriever_node
    bound to the given paper's retriever and the LLM-calling nodes
    bound to the given clients, via closures.
    """

    def _bound_contextualizer_node(state: AgentState) -> AgentState:
        return contextualizer_node(state, clients)

    def _bound_planner_node(state: AgentState) -> AgentState:
        return planner_node(state, clients)

    def _bound_retriever_node(state: AgentState) -> AgentState:
        return retriever_node(state, paper_retriever)

    def _bound_reasoner_node(state: AgentState) -> AgentState:
        return reasoner_node(state, clients)

    graph = StateGraph(AgentState)

    graph.add_node("contextualizer", _bound_contextualizer_node)
    graph.add_node("planner", _bound_planner_node)
    graph.add_node("retriever", _bound_retriever_node)
    graph.add_node("reasoner", _bound_reasoner_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("contextualizer")
    graph.add_edge("contextualizer", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner", "critic")

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "retry": "retriever",
        },
    )

    return graph.compile()


def run_agent(
    question: str,
    paper_retriever,
    clients,
    chat_history: list | None = None,
    on_step=None,
) -> dict:
    """
    Run the full agent pipeline for a question against a specific
    paper's retriever.

    Args:
        question: the user's question, as typed
        paper_retriever: the paper's MultiVectorRetriever
        clients: models.llm_clients.LLMClients for the current visitor
        chat_history: earlier turns of this conversation, oldest first,
                      as [{"role", "content"}] - used only to make a
                      follow-up question standalone
        on_step: optional callable(node_name: str, state: dict), called
                 after each node finishes, for live progress in the UI

    Returns:
        dict with keys: final_answer, retrieval_type, retry_count,
        sources (the retrieved records the answer was based on),
        standalone_question, vision_fallback (True if a figure/table
        question had to be answered without the vision model)
    """
    app = create_graph(paper_retriever, clients)

    state: dict = {
        "question": question,
        "chat_history": list(chat_history or []),
        "retry_count": 0,
    }

    # stream_mode="updates" yields {node_name: state_returned_by_node}
    # after every node, which is what lets the UI show each step as it
    # completes instead of one long wait.
    for chunk in app.stream(state, stream_mode="updates"):
        for node_name, update in chunk.items():
            if isinstance(update, dict):
                state.update(update)
            if on_step is not None:
                try:
                    on_step(node_name, state)
                except Exception:  # noqa: BLE001 - a display problem must not lose the answer
                    logger.exception("on_step callback failed for node %s", node_name)

    return {
        "final_answer": state.get("final_answer", ""),
        "retrieval_type": state.get("retrieval_type", ""),
        "retry_count": state.get("retry_count", 0),
        "sources": state.get("context", []),
        "standalone_question": state.get("standalone_question", question),
        "vision_fallback": bool(state.get("vision_fallback")),
    }
