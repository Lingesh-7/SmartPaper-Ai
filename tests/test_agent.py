"""Tests for the agent nodes, the graph, and image preparation."""

import base64
import io

from PIL import Image

from agents import nodes
from agents.graph import run_agent
from embeddings.retriever import prepare_image_for_llm
from helpers import FakeLLM, jpeg_base64, make_clients

GOOD_ANSWER = "The model uses attention (page 1). It was trained on WMT 2014 (page 7)."


class FakeRetriever:
    """Returns fixed records regardless of the query, and remembers the queries."""

    def __init__(self, records):
        self.records = records
        self.queries = []
        self.search_kwargs = {"k": 4}

    def invoke(self, query):
        self.queries.append(query)
        return self.records

    def model_copy(self, update=None):
        # retrieve() copies the retriever to change its search settings
        self.search_kwargs_used = (update or {}).get("search_kwargs")
        return self


TEXT_RECORDS = [
    {"type": "text", "content": "The Transformer relies on attention.", "page_number": 1},
    {"type": "table", "content": "Table on page 8: BLEU 28.4", "page_number": 8, "image": None},
]
IMAGE_RECORD = {
    "type": "image", "content": jpeg_base64(), "page_number": 3,
    "caption": "FIGURE 1. The Transformer architecture.", "label": "Figure 1",
    "ocr_text": "Encoder Decoder",
}


def scripted_agent_llm(answers, rewrite="What dataset was the Transformer trained on?"):
    """An agent LLM that classifies, rewrites follow-ups, and answers from a script."""
    remaining = list(answers)

    def reply(prompt):
        if prompt.startswith("Classify the following question"):
            return "text_query"
        if "Rewritten question:" in prompt:
            return rewrite
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return FakeLLM(reply)


# --- critic ---
def test_critic_accepts_a_substantive_answer():
    state = nodes.critic_node({"draft_answer": GOOD_ANSWER, "question": "q", "retry_count": 0})

    assert state["final_answer"] == GOOD_ANSWER
    assert nodes.should_continue(state) == "end"


def test_critic_rejects_a_refusal_and_asks_for_a_retry():
    state = nodes.critic_node({
        "draft_answer": "The context does not mention the dataset. Sorry.",
        "question": "q",
        "retry_count": 0,
    })

    assert "final_answer" not in state
    assert state["retry_count"] == 1
    assert state["question"].endswith("Please be more specific and detailed.")
    assert nodes.should_continue(state) == "retry"


def test_critic_accepts_anything_after_two_retries():
    state = nodes.critic_node({"draft_answer": "I don't know", "question": "q", "retry_count": 2})

    assert state["final_answer"] == "I don't know"


def test_a_page_citation_alone_does_not_make_an_answer_long_enough():
    """The citation format has no period, so it cannot fake a second sentence."""
    state = nodes.critic_node({"draft_answer": "Attention (page 4)", "question": "q", "retry_count": 0})

    assert "final_answer" not in state


# --- planner ---
def test_planner_falls_back_to_text_query_on_unexpected_output():
    clients = make_clients(agent_llm=FakeLLM("I think this is about figures"))

    assert nodes.planner_node({"question": "q"}, clients)["retrieval_type"] == "text_query"


def test_planner_accepts_a_valid_category():
    clients = make_clients(agent_llm=FakeLLM("  Figure_Query\n"))

    assert nodes.planner_node({"question": "q"}, clients)["retrieval_type"] == "figure_query"


# --- contextualizer ---
def test_first_question_is_not_rewritten_and_costs_no_llm_call():
    llm = FakeLLM("should not be called")
    state = nodes.contextualizer_node({"question": "What is attention?"}, make_clients(agent_llm=llm))

    assert state["question"] == "What is attention?"
    assert state["standalone_question"] == "What is attention?"
    assert llm.prompts == []


def test_follow_up_is_rewritten_using_the_history():
    llm = FakeLLM('"What dataset was the Transformer trained on?"')
    history = [
        {"role": "user", "content": "What is the Transformer?"},
        {"role": "assistant", "content": "A model based on attention."},
    ]
    state = nodes.contextualizer_node(
        {"question": "What was it trained on?", "chat_history": history},
        make_clients(agent_llm=llm),
    )

    assert state["question"] == "What dataset was the Transformer trained on?"
    assert state["original_question"] == "What was it trained on?"
    assert "User: What is the Transformer?" in llm.prompts[0]
    assert "Assistant: A model based on attention." in llm.prompts[0]


def test_a_failing_rewrite_falls_back_to_the_question_as_asked():
    def boom(prompt):
        raise RuntimeError("provider down")

    state = nodes.contextualizer_node(
        {"question": "What was it trained on?", "chat_history": [{"role": "user", "content": "hi"}]},
        make_clients(agent_llm=FakeLLM(boom)),
    )

    assert state["question"] == "What was it trained on?"


# --- reasoner ---
def test_reasoner_labels_context_with_pages_and_uses_the_text_model():
    text_llm, vision_llm = FakeLLM(GOOD_ANSWER), FakeLLM("vision")
    clients = make_clients(agent_llm=text_llm, agent_vision_llm=vision_llm)

    state = nodes.reasoner_node({"question": "What is it?", "context": TEXT_RECORDS}, clients)

    assert state["draft_answer"] == GOOD_ANSWER
    assert "[Text, page 1]" in text_llm.prompts[0]
    assert "[Table, page 8]" in text_llm.prompts[0]
    assert "(page N)" in text_llm.prompts[0]
    assert vision_llm.prompts == []


def test_reasoner_shows_the_figure_to_the_vision_model_for_a_figure_question():
    text_llm, vision_llm = FakeLLM("text"), FakeLLM("The figure shows a stack. It has six layers.")
    clients = make_clients(agent_llm=text_llm, agent_vision_llm=vision_llm)

    state = nodes.reasoner_node(
        {"question": "What does the figure show?", "retrieval_type": "figure_query",
         "context": TEXT_RECORDS + [IMAGE_RECORD]},
        clients,
    )

    assert state["draft_answer"].startswith("The figure shows")
    assert state["vision_fallback"] is False
    # the figure is introduced by its own number and caption, then shown
    assert "Figure 1 (page 3): FIGURE 1. The Transformer architecture." in vision_llm.prompts[0]
    assert "<image>" in vision_llm.prompts[0]
    assert "[Text, page 1]" in vision_llm.prompts[0]
    assert text_llm.prompts == []


def test_a_text_question_never_spends_a_vision_request():
    """A free Google key allows ~20 vision requests a day; text questions must not use them."""
    text_llm, vision_llm = FakeLLM(GOOD_ANSWER), FakeLLM("vision")
    clients = make_clients(agent_llm=text_llm, agent_vision_llm=vision_llm)

    nodes.reasoner_node(
        {"question": "What datasets were used?", "retrieval_type": "text_query",
         "context": TEXT_RECORDS + [IMAGE_RECORD]},
        clients,
    )

    assert vision_llm.prompts == []
    # the figure still contributes what is known about it in words
    assert "[Figure 1, page 3]" in text_llm.prompts[0]
    assert "FIGURE 1. The Transformer architecture." in text_llm.prompts[0]
    assert "Text printed in the figure: Encoder Decoder" in text_llm.prompts[0]


def test_figure_question_is_answered_from_captions_when_the_vision_model_is_unavailable():
    """What happens once the day's Gemini allowance is used up."""
    def quota_exhausted(prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: requests per day limit exceeded")

    text_llm = FakeLLM("Figure 1 shows the Transformer architecture (page 3). It has an encoder and a decoder.")
    clients = make_clients(agent_llm=text_llm, agent_vision_llm=FakeLLM(quota_exhausted))

    state = nodes.reasoner_node(
        {"question": "What does Figure 1 show?", "retrieval_type": "figure_query",
         "context": [IMAGE_RECORD, TEXT_RECORDS[0]]},
        clients,
    )

    assert state["vision_fallback"] is True
    assert state["draft_answer"].startswith("Figure 1 shows")
    assert "FIGURE 1. The Transformer architecture." in text_llm.prompts[0]


def test_a_chunk_spanning_two_pages_is_labelled_with_both():
    llm = FakeLLM(GOOD_ANSWER)
    records = [{"type": "text", "content": "Spans a page break.", "page_number": 4, "page_end": 5}]

    nodes.reasoner_node({"question": "q", "context": records}, make_clients(agent_llm=llm))

    assert "[Text, pages 4-5]" in llm.prompts[0]


def test_a_numbered_table_is_named_by_its_number_in_the_context():
    llm = FakeLLM(GOOD_ANSWER)
    records = [{"type": "table", "content": "Table on page 8: TABLE 2. BLEU.", "page_number": 8, "label": "Table 2"}]

    nodes.reasoner_node({"question": "q", "context": records}, make_clients(agent_llm=llm))

    assert "[Table 2, page 8]" in llm.prompts[0]


def test_reasoner_handles_a_record_without_a_page_number():
    llm = FakeLLM(GOOD_ANSWER)
    records = [{"type": "text", "content": "No page here.", "page_number": None}]

    nodes.reasoner_node({"question": "q", "context": records}, make_clients(agent_llm=llm))

    assert "[Text, page unknown]" in llm.prompts[0]


# --- whole graph ---
def test_run_agent_returns_answer_sources_and_reports_each_step():
    retriever = FakeRetriever(TEXT_RECORDS)
    steps = []

    result = run_agent(
        "What is the Transformer?",
        retriever,
        make_clients(agent_llm=scripted_agent_llm([GOOD_ANSWER])),
        on_step=lambda node, state: steps.append(node),
    )

    assert result["final_answer"] == GOOD_ANSWER
    assert result["retrieval_type"] == "text_query"
    assert result["retry_count"] == 0
    assert [s["page_number"] for s in result["sources"]] == [1, 8]
    assert steps == ["contextualizer", "planner", "retriever", "reasoner", "critic"]


def test_run_agent_retries_after_a_refusal_then_accepts():
    retriever = FakeRetriever(TEXT_RECORDS)
    llm = scripted_agent_llm(["The context does not mention it. Sorry.", GOOD_ANSWER])

    result = run_agent("What dataset?", retriever, make_clients(agent_llm=llm))

    assert result["final_answer"] == GOOD_ANSWER
    assert result["retry_count"] == 1
    assert len(retriever.queries) == 2
    assert retriever.queries[1].endswith("Please be more specific and detailed.")


def test_run_agent_gives_up_after_two_retries():
    retriever = FakeRetriever(TEXT_RECORDS)
    llm = scripted_agent_llm(["I don't know. Sorry."])

    result = run_agent("What dataset?", retriever, make_clients(agent_llm=llm))

    assert result["final_answer"] == "I don't know. Sorry."
    assert result["retry_count"] == 2
    assert len(retriever.queries) == 3


def test_run_agent_searches_with_the_rewritten_follow_up():
    retriever = FakeRetriever(TEXT_RECORDS)
    history = [
        {"role": "user", "content": "What is the Transformer?"},
        {"role": "assistant", "content": "A model based on attention."},
    ]

    result = run_agent(
        "What was it trained on?",
        retriever,
        make_clients(agent_llm=scripted_agent_llm([GOOD_ANSWER])),
        chat_history=history,
    )

    assert retriever.queries == ["What dataset was the Transformer trained on?"]
    assert result["standalone_question"] == "What dataset was the Transformer trained on?"


def test_a_failing_step_callback_does_not_lose_the_answer():
    def broken(node, state):
        raise RuntimeError("display failed")

    result = run_agent(
        "q?", FakeRetriever(TEXT_RECORDS),
        make_clients(agent_llm=scripted_agent_llm([GOOD_ANSWER])), on_step=broken,
    )

    assert result["final_answer"] == GOOD_ANSWER


# --- images ---
def _size(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def test_small_images_are_sent_to_the_vision_model_untouched():
    small = jpeg_base64(400, 300)

    assert prepare_image_for_llm(small) == small


def test_large_images_are_shrunk_without_distortion():
    tall = jpeg_base64(1000, 2600)   # a tall architecture diagram

    width, height = _size(prepare_image_for_llm(tall))

    assert height == 1300
    assert width == 500              # 1000:2600 preserved, not forced to 1300x600


def test_undecodable_image_data_is_passed_through():
    assert prepare_image_for_llm("not-an-image") == "not-an-image"


# --- table / figure questions are answered from tables / figures ---
TABLE_WITH_SNAPSHOT = {
    "type": "table", "content": "Table on page 8: BLEU 28.4", "page_number": 8,
    "image": jpeg_base64(color=(10, 10, 10)),
    "caption": "TABLE 2. BLEU scores.", "label": "Table 2",
}


def test_table_question_shows_the_table_image_to_the_vision_model():
    text_llm, vision_llm = FakeLLM("text"), FakeLLM("The big model scores 28.4 BLEU (page 8). It is the best.")
    clients = make_clients(agent_llm=text_llm, agent_vision_llm=vision_llm)

    state = nodes.reasoner_node(
        {"question": "What BLEU does Table 2 report?", "retrieval_type": "table_query",
         "context": [TABLE_WITH_SNAPSHOT, TEXT_RECORDS[0]]},
        clients,
    )

    assert state["draft_answer"].startswith("The big model scores 28.4")
    prompt = vision_llm.prompts[0]
    assert "Table 2 (page 8): TABLE 2. BLEU scores." in prompt and "<image>" in prompt
    assert "The question is about a table" in prompt
    assert "[Table 2, page 8]" in prompt, "the table's text stays in the context too"
    assert text_llm.prompts == []


def test_a_table_met_in_passing_is_not_sent_as_an_image():
    """Only a question *about* a table spends a vision call on the table's snapshot."""
    text_llm, vision_llm = FakeLLM(GOOD_ANSWER), FakeLLM("vision")
    clients = make_clients(agent_llm=text_llm, agent_vision_llm=vision_llm)

    nodes.reasoner_node(
        {"question": "What is attention?", "retrieval_type": "text_query",
         "context": [TABLE_WITH_SNAPSHOT, TEXT_RECORDS[0]]},
        clients,
    )

    assert vision_llm.prompts == []
    assert "The question is about a table" not in text_llm.prompts[0]


def test_figure_question_tells_the_model_to_answer_from_the_figure():
    vision_llm = FakeLLM("The figure shows a stack. It has six layers.")

    nodes.reasoner_node(
        {"question": "What does Figure 1 show?", "retrieval_type": "figure_query",
         "context": [IMAGE_RECORD, TEXT_RECORDS[0]]},
        make_clients(agent_vision_llm=vision_llm),
    )

    assert "The question is about a figure" in vision_llm.prompts[0]
    assert "Figure 1 (page 3):" in vision_llm.prompts[0]


def test_retriever_node_asks_for_the_record_type_the_question_is_about(monkeypatch):
    asked = []

    def fake_retrieve(retriever, question, top_k=None, prefer_type=None, exclude_type=None):
        asked.append((prefer_type, exclude_type))
        return []

    monkeypatch.setattr(nodes, "retrieve", fake_retrieve)
    for retrieval_type in ("table_query", "figure_query", "text_query"):
        nodes.retriever_node({"question": "q", "retrieval_type": retrieval_type}, object())

    # only a figure question is allowed to pull figures into the context
    assert asked == [("table", "image"), ("image", None), (None, "image")]


def test_questions_naming_figures_and_tables_are_recognised():
    from embeddings.retriever import labels_in_query

    assert labels_in_query("Explain Figure 9 and Table II") == ["Figure 9", "Table 2"]
    assert labels_in_query("what does fig. 3 show?") == ["Figure 3"]
    assert labels_in_query("compare tables 4 and 5") == ["Table 4", "Table 5"]
    assert labels_in_query("Summarize Table 2.") == ["Table 2"]
    # ordinary words after "table"/"figure" are not numbers
    assert labels_in_query("which table in the paper shows it") == []
    assert labels_in_query("the figure i mentioned") == []
