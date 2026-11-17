"""
Tests for the Streamlit front end, driven headlessly with AppTest.

The pipeline underneath is real (Chroma, the paper store); LLMs,
parse_pdf and the key check are faked. These tests exercise each
state of the page: no keys, keys rejected, upload, processing,
upload failed, chat, a failing answer.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
import pipeline
from helpers import FakeLLM, make_clients, make_pdf_bytes
from models import llm_clients
from models.llm_clients import KeyCheck

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
KEYS = config.ApiKeys(groq="gsk_test", google="AIza_test", openrouter="sk-or-test")
GOOD_ANSWER = "The model uses attention (page 1). It was trained on WMT 2014 (page 7)."


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120)


def answering_clients(answer=GOOD_ANSWER):
    def reply(prompt):
        if prompt.startswith("Classify the following question"):
            return "text_query"
        if "Rewritten question:" in prompt:
            return "What dataset was the Transformer trained on?"
        if callable(answer):
            return answer(prompt)
        return answer

    return make_clients(agent_llm=FakeLLM(reply), agent_vision_llm=FakeLLM(reply))


def app_with_paper(fake_parse, clients=None) -> AppTest:
    """An app session that already has keys and an ingested paper."""
    clients = clients or answering_clients()
    info = pipeline.ingest_pdf(make_pdf_bytes(), "attention.pdf", clients)
    at = new_app()
    at.session_state["api_keys"] = KEYS
    at.session_state["clients"] = clients
    at.session_state["paper"] = info
    return at.run()


def all_text(at: AppTest) -> str:
    """Everything a visitor could read on the page."""
    parts = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "error", "warning", "info"):
        parts.extend(str(element.value) for element in getattr(at, kind))
    return "\n".join(parts)


# --- no keys ---
def test_first_visit_asks_for_keys_and_nothing_else():
    at = new_app().run()

    assert not at.exception
    assert "First, add your API keys" in all_text(at)
    assert len(at.text_input) == 3
    assert len(at.chat_input) == 0, "no chat before there are keys and a paper"


def test_saving_incomplete_keys_names_the_missing_ones():
    at = new_app().run()
    at.text_input[0].set_value("sk-or-something")
    at.button[0].click().run()

    assert not at.exception
    assert "Still needed: Google AI Studio API key, Groq API key." in all_text(at)
    assert at.session_state["api_keys"] is None


def test_a_rejected_key_is_reported_and_not_saved(monkeypatch):
    monkeypatch.setattr(llm_clients, "check_keys", lambda keys: {
        "groq": KeyCheck(True, "Groq key accepted."),
        "google": KeyCheck(False, "Google rejected this key. Check that it is copied in full."),
        "openrouter": KeyCheck(True, "OpenRouter key accepted."),
    })
    at = new_app().run()
    for box in at.text_input:
        box.set_value("some-key")
    at.button[0].click().run()

    assert "Google rejected this key" in all_text(at)
    assert at.session_state["api_keys"] is None


def test_accepted_keys_lead_to_the_upload_step(monkeypatch):
    monkeypatch.setattr(llm_clients, "check_keys", lambda keys: {
        "groq": KeyCheck(True, "ok"), "google": KeyCheck(True, "ok"),
        "openrouter": KeyCheck(None, "Could not reach OpenRouter to check the key."),
    })
    monkeypatch.setattr(llm_clients, "build_clients", lambda keys: make_clients())
    at = new_app().run()
    for box, value in zip(at.text_input, ("sk-or-mine", "AIza-mine", "gsk-mine")):
        box.set_value(value)
    at.button[0].click().run()

    assert not at.exception
    assert at.session_state["api_keys"].openrouter == "sk-or-mine"
    assert at.session_state["api_keys"].groq == "gsk-mine"
    assert "Upload a paper" in all_text(at)
    # a key that could not be checked is accepted, with a visible note
    assert "Could not reach OpenRouter" in all_text(at)
    assert "saved for this session" in all_text(at)


def test_keys_never_appear_on_the_page(fake_parse):
    at = app_with_paper(fake_parse)

    page = all_text(at)
    for secret in KEYS.secrets():
        assert secret not in page


# --- upload and processing ---
class FakeJob:
    def __init__(self, done=False, info=None, error=None):
        self.filename = "attention.pdf"
        self.message = "Summarising text passages (20/76)"
        self.fraction = 0.5
        self.elapsed = 75.0
        self.done = done
        self.info = info
        self.error = error


def test_a_running_job_shows_progress_and_hides_the_chat():
    at = new_app()
    at.session_state["api_keys"] = KEYS
    at.session_state["clients"] = make_clients()
    at.session_state["job"] = FakeJob()
    at.run()

    assert not at.exception
    assert "Reading your paper" in all_text(at)
    assert "1:15 elapsed" in all_text(at)
    assert len(at.chat_input) == 0


def test_a_finished_job_opens_the_chat(fake_parse):
    clients = answering_clients()
    info = pipeline.ingest_pdf(make_pdf_bytes(), "attention.pdf", clients)
    at = new_app()
    at.session_state["api_keys"] = KEYS
    at.session_state["clients"] = clients
    at.session_state["job"] = FakeJob(done=True, info=info)
    at.run()

    assert not at.exception
    assert at.session_state["job"] is None
    assert at.session_state["paper"].paper_id == info.paper_id
    assert len(at.chat_input) == 1
    assert "3 passages" in all_text(at)


def test_a_failed_job_returns_to_the_upload_step_with_the_reason():
    at = new_app()
    at.session_state["api_keys"] = KEYS
    at.session_state["clients"] = make_clients()
    at.session_state["job"] = FakeJob(done=True, error="This PDF has 99 pages; the limit is 50.")
    at.run()

    assert not at.exception
    assert at.session_state["paper"] is None
    assert "This PDF has 99 pages; the limit is 50." in all_text(at)
    assert "Upload a paper" in all_text(at)


def test_a_real_background_job_ingests_and_reports(fake_parse):
    job = pipeline.IngestJob(make_pdf_bytes(), "attention.pdf", make_clients(), KEYS)

    assert job.wait(60)
    assert job.error is None
    assert job.info.text_chunks == 3
    assert job.fraction == 1.0


def test_a_background_job_turns_a_failure_into_a_message_without_the_key(fake_parse):
    def dead(prompt):
        raise RuntimeError(f"401 invalid api key {KEYS.groq} (groq)")

    job = pipeline.IngestJob(
        make_pdf_bytes(), "attention.pdf", make_clients(text_summary_llm=FakeLLM(dead)), KEYS
    )

    assert job.wait(60)
    assert job.info is None
    assert "Groq rejected the API key" in job.error
    assert KEYS.groq not in job.error


# --- chat ---
def test_paper_view_offers_suggestions_and_a_chat_box(fake_parse):
    at = app_with_paper(fake_parse)

    assert not at.exception
    assert len(at.chat_input) == 1
    assert "attention.pdf" in all_text(at).replace("\\", "")
    assert "2 pages · 3 passages · 1 tables · 1 figures" in all_text(at)


def test_asking_a_question_shows_the_answer_with_its_sources(fake_parse):
    at = app_with_paper(fake_parse)
    at.chat_input[0].set_value("What is the Transformer?").run()

    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == GOOD_ANSWER
    assert len(messages[1]["sources"]) == config.RETRIEVAL_TOP_K
    assert GOOD_ANSWER in all_text(at)
    # AppTest lists an st.expander among its status/expander blocks
    labels = [block.label for block in list(at.expander) + list(at.status)]
    assert any(label.startswith("Sources (4)") and "pages" in label for label in labels)


def test_a_follow_up_is_sent_with_the_conversation_so_far(fake_parse):
    clients = answering_clients()
    at = app_with_paper(fake_parse, clients)
    at.chat_input[0].set_value("What is the Transformer?").run()
    at.chat_input[0].set_value("What was it trained on?").run()

    assert not at.exception
    messages = at.session_state["messages"]
    assert len(messages) == 4
    assert messages[3]["understood_as"] == "What dataset was the Transformer trained on?"
    rewrite_prompts = [p for p in clients.agent_llm.prompts if "Rewritten question:" in p]
    assert "User: What is the Transformer?" in rewrite_prompts[-1]


def test_a_failing_answer_becomes_one_error_message_and_does_not_loop(fake_parse):
    calls = {"n": 0}

    def failing(prompt):
        calls["n"] += 1
        raise RuntimeError("Error code: 429 - groq rate limit exceeded")

    clients = make_clients(agent_llm=FakeLLM(failing))
    info = pipeline.ingest_pdf(make_pdf_bytes(), "attention.pdf", make_clients())
    at = new_app()
    at.session_state["api_keys"] = KEYS
    at.session_state["clients"] = clients
    at.session_state["paper"] = info
    at.run()
    at.chat_input[0].set_value("What is the Transformer?").run()

    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["error"] is True
    assert "rate-limiting your key" in messages[1]["content"]
    assert calls["n"] == 1, "a failed answer must not be retried automatically"

    # the next question does not carry the error text as conversation history
    at.session_state["clients"] = answering_clients()
    at.chat_input[0].set_value("And the dataset?").run()
    assert at.session_state["messages"][-1]["content"] == GOOD_ANSWER
    history_prompt = [p for p in at.session_state["clients"].agent_llm.prompts if "Rewritten question:" in p][0]
    assert "rate-limiting" not in history_prompt


def test_a_question_about_an_evicted_paper_sends_the_visitor_back_to_upload(fake_parse):
    at = app_with_paper(fake_parse)
    paper_id = at.session_state["paper"].paper_id
    pipeline._retrievers.clear()
    pipeline.delete_paper(paper_id)

    at.chat_input[0].set_value("What is the Transformer?").run()

    assert not at.exception
    assert at.session_state["paper"] is None
    assert "no longer on the server" in all_text(at)
    assert "Upload a paper" in all_text(at)


def test_clear_chat_and_use_another_paper(fake_parse):
    at = app_with_paper(fake_parse)
    at.chat_input[0].set_value("What is the Transformer?").run()
    assert len(at.session_state["messages"]) == 2

    next(b for b in at.button if b.label == "Clear chat").click().run()
    assert at.session_state["messages"] == []

    next(b for b in at.button if b.label == "Use another paper").click().run()
    assert at.session_state["paper"] is None
    assert "Upload a paper" in all_text(at)


def test_server_keys_are_used_when_present(monkeypatch):
    """Local runs: keys in .env skip the key form."""
    monkeypatch.setattr(config, "keys_from_env", lambda: KEYS)
    monkeypatch.setattr(llm_clients, "build_clients", lambda keys: make_clients())
    at = new_app().run()

    assert not at.exception
    assert at.session_state["keys_from_server"] is True
    assert "Upload a paper" in all_text(at)
    assert "set on the server" in all_text(at)


# --- figure / table questions ---
def test_an_answer_given_without_the_vision_model_says_so(fake_parse):
    """Once a free Google key's ~20 daily requests are gone, the visitor is told."""
    def quota_exhausted(prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: requests per day limit exceeded")

    def agent(prompt):
        if prompt.startswith("Classify the following question"):
            return "figure_query"
        return "Figure 1 is a diagram of the model (page 3). It shows an encoder and a decoder."

    clients = make_clients(agent_llm=FakeLLM(agent), agent_vision_llm=FakeLLM(quota_exhausted))
    at = app_with_paper(fake_parse, clients)
    at.chat_input[0].set_value("What does Figure 1 show?").run()

    assert not at.exception
    message = at.session_state["messages"][-1]
    assert message.get("error") is not True, "a vision failure must not fail the question"
    assert message["vision_fallback"] is True
    assert message["sources"][0]["label"] == "Figure 1"
    assert "vision model could not be reached" in all_text(at)


def test_sources_show_the_figure_number_caption_and_page_range(fake_parse):
    def agent(prompt):
        if prompt.startswith("Classify the following question"):
            return "figure_query"
        return GOOD_ANSWER

    clients = make_clients(agent_llm=FakeLLM(agent), agent_vision_llm=FakeLLM(GOOD_ANSWER))
    at = app_with_paper(fake_parse, clients)
    at.chat_input[0].set_value("What does Figure 1 show?").run()

    assert not at.exception
    page = all_text(at)
    assert ":blue-badge[Figure 1]" in page
    assert "FIGURE 1. Diagram number 1 of the model." in page      # the caption, shown under the image
    assert "vision model could not be reached" not in page
    assert at.session_state["messages"][-1]["vision_fallback"] is False
