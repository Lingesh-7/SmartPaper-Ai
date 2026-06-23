"""
LangGraph pipeline setup for SmartPaper AI.

create_graph(paper_retriever) builds and compiles the StateGraph for
a specific paper's retriever. run_agent(question, paper_id) looks up
the right retriever for that paper, compiles a graph bound to it, and
invokes it, returning the final answer plus supporting metadata.

NOTE on design: the original notebook's retriever_node closed over a
single global retriever_multi_vector_img variable. Since this module
needs to support multiple papers (each with its own retriever from
embeddings.embedder.ingest_paper), retriever_node now takes the
retriever as an explicit argument. To keep the LangGraph node
signature compatible with StateGraph.add_node (which expects a
single-argument function: state -> state), this module binds the
paper's retriever via a closure when building the graph, rather than
changing LangGraph's calling convention.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.nodes import (
    planner_node,
    retriever_node,
    reasoner_node,
    critic_node,
    should_continue,
)


class AgentState(TypedDict):
    question: str
    retrieval_type: str
    context: dict
    draft_answer: str
    final_answer: str
    retry_count: int


# Cache of paper_id -> retriever, populated by ingest_paper() calls
# from the Flask /upload route (or directly, for testing). Keeping
# this as a simple module-level dict is fine for a single-process
# Flask dev server; if this app were ever run with multiple worker
# processes, this cache would need to move to something shared
# (e.g. re-querying Chroma by collection name on each request).
_paper_retrievers: dict = {}


def register_retriever(paper_id: str, retriever):
    """Register a paper's retriever so run_agent() can find it later."""
    _paper_retrievers[paper_id] = retriever


def create_graph(paper_retriever):
    """
    Build and compile the LangGraph StateGraph, with retriever_node
    bound to the given paper's retriever via closure.
    """

    def _bound_retriever_node(state: AgentState) -> AgentState:
        return retriever_node(state, paper_retriever)

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", _bound_retriever_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")
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


def run_agent(question: str, paper_id: str) -> dict:
    """
    Run the full agent pipeline for a question against a specific
    paper. The paper must already have been ingested (its retriever
    registered via register_retriever, which embeddings.embedder
    .ingest_paper's caller is expected to do - see app.py's /upload
    route).

    Returns:
        dict with keys: final_answer, retrieval_type, retry_count
    """
    if paper_id not in _paper_retrievers:
        raise ValueError(
            f"No retriever registered for paper_id='{paper_id}'. "
            f"Call register_retriever(paper_id, retriever) after "
            f"ingest_paper() before running the agent."
        )

    paper_retriever = _paper_retrievers[paper_id]
    app = create_graph(paper_retriever)

    result = app.invoke({
        "question": question,
        "retry_count": 0,
    })

    return {
        "final_answer": result.get("final_answer", ""),
        "retrieval_type": result.get("retrieval_type", ""),
        "retry_count": result.get("retry_count", 0),
    }