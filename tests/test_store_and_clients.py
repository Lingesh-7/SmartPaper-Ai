"""Tests for the on-disk paper store, API-key handling, and LLM client helpers."""

import httpx
import pytest
from langchain_core.messages import AIMessage

import config
from embeddings import store
from models import llm_clients
from models.llm_clients import build_clients, check_keys, message_text

PAPER_ID = "a" * 32


def _records():
    return {
        "doc-1": {"type": "text", "content": "First passage.", "page_number": 1},
        "doc-2": {"type": "table", "content": "Table on page 4: a b", "page_number": 4, "image": None},
    }


# --- store ---
def test_save_and_load_round_trip():
    manifest = store.save_paper(PAPER_ID, {"filename": "x.pdf", "pages": 3}, _records())

    assert manifest["paper_id"] == PAPER_ID
    assert store.load_manifest(PAPER_ID)["filename"] == "x.pdf"
    assert store.load_records(PAPER_ID) == _records()
    # insertion order is what summarize_paper relies on
    assert list(store.load_records(PAPER_ID)) == ["doc-1", "doc-2"]


def test_missing_paper_reads_as_not_ingested():
    assert store.load_manifest(PAPER_ID) is None
    assert store.load_records(PAPER_ID) is None


def test_docstore_without_manifest_is_not_ingested():
    """An interrupted ingestion leaves a docstore but no manifest."""
    store.save_paper(PAPER_ID, {}, _records())
    (store.paper_dir(PAPER_ID) / "manifest.json").unlink()

    assert store.load_manifest(PAPER_ID) is None
    assert store.list_manifests() == []


def test_corrupt_manifest_is_not_ingested():
    store.save_paper(PAPER_ID, {}, _records())
    (store.paper_dir(PAPER_ID) / "manifest.json").write_text("{not json", encoding="utf-8")

    assert store.load_manifest(PAPER_ID) is None


def test_other_schema_version_is_not_ingested(monkeypatch):
    store.save_paper(PAPER_ID, {}, _records())
    monkeypatch.setattr(config, "STORE_SCHEMA_VERSION", config.STORE_SCHEMA_VERSION + 1)

    assert store.load_manifest(PAPER_ID) is None


def test_other_embedding_model_is_not_ingested(monkeypatch):
    store.save_paper(PAPER_ID, {}, _records())
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "some/other-model")

    assert store.load_manifest(PAPER_ID) is None


def test_update_manifest_and_delete():
    store.save_paper(PAPER_ID, {"filename": "x.pdf"}, _records())

    assert store.update_manifest(PAPER_ID, summary="An overview.")["summary"] == "An overview."
    assert store.load_manifest(PAPER_ID)["filename"] == "x.pdf"

    store.delete_paper_files(PAPER_ID)
    assert store.load_manifest(PAPER_ID) is None
    assert store.update_manifest(PAPER_ID, summary="x") is None
    store.delete_paper_files(PAPER_ID)   # deleting twice is not an error


@pytest.mark.parametrize("bad_id", ["", "ab", "../etc", "a/b", "a\\b", "a.b", "x" * 65, None])
def test_paper_ids_that_could_escape_the_data_dir_are_rejected(bad_id):
    with pytest.raises(ValueError):
        store.paper_dir(bad_id)


# --- ApiKeys ---
def test_api_keys_are_stripped_and_never_shown_by_repr():
    keys = config.ApiKeys(groq="  gsk_secret  ", google="AIza_secret")

    assert keys.groq == "gsk_secret"
    assert "secret" not in repr(keys)
    assert "secret" not in str(keys)
    assert keys.secrets() == ["gsk_secret", "AIza_secret"]


def test_all_three_keys_are_required():
    assert config.ApiKeys(groq="g", google="a", openrouter="o").is_complete()
    assert config.ApiKeys(groq="g", google="a").missing() == ["OPENROUTER_API_KEY"]
    assert config.ApiKeys().missing() == ["OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]


def test_validate_config_raises_without_env_keys():
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        config.validate_config()


# --- clients ---
def test_build_clients_refuses_an_incomplete_key_set():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_clients(config.ApiKeys(groq="g", google="a"))


def test_each_job_runs_on_its_intended_provider_and_model():
    clients = build_clients(config.ApiKeys(groq="g", google="a", openrouter="o"))

    # summaries: Groq
    assert clients.text_summary_llm.model_name == config.TEXT_SUMMARY_MODEL
    assert clients.table_summary_llm.model_name == config.TABLE_SUMMARY_MODEL
    # the agent: OpenRouter first, Groq only as the fallback
    assert clients.agent_llm.runnable.model_name == config.AGENT_LLM_MODEL
    assert config.AGENT_LLM_MODEL.endswith(":free")
    assert len(clients.agent_llm.fallbacks) == 1


def test_two_visitors_get_clients_bound_to_their_own_keys():
    first = build_clients(config.ApiKeys(groq="groq-one", google="a", openrouter="or-one"))
    second = build_clients(config.ApiKeys(groq="groq-two", google="a", openrouter="or-two"))

    assert first.text_summary_llm.groq_api_key.get_secret_value() == "groq-one"
    assert second.text_summary_llm.groq_api_key.get_secret_value() == "groq-two"
    assert first.agent_llm.runnable.openrouter_api_key.get_secret_value() == "or-one"
    assert second.agent_llm.runnable.openrouter_api_key.get_secret_value() == "or-two"


def test_agent_falls_back_to_groq_when_openrouter_fails(monkeypatch):
    """What a visitor hits after their 50 free OpenRouter requests for the day."""
    from langchain_core.messages import AIMessage as Message
    from langchain_core.runnables import RunnableLambda

    def openrouter_down(*args, **kwargs):
        raise RuntimeError("429 Rate limit exceeded: free-models-per-day")

    monkeypatch.setattr(llm_clients, "ChatOpenRouter", lambda **kwargs: RunnableLambda(openrouter_down))
    monkeypatch.setattr(
        llm_clients, "ChatGroq",
        lambda **kwargs: RunnableLambda(lambda value: Message(content="answered by groq")),
    )

    clients = build_clients(config.ApiKeys(groq="g", google="a", openrouter="o"))

    assert message_text(clients.agent_llm.invoke("a question")) == "answered by groq"


def test_message_text_flattens_every_content_shape():
    assert message_text(AIMessage(content="plain")) == "plain"
    blocks = [{"type": "text", "text": "Hello "}, {"type": "thinking", "thinking": "x"}, "world"]
    assert message_text(AIMessage(content=blocks)) == "Hello world"
    assert message_text("already a string") == "already a string"


# --- key checks ---
class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def test_check_keys_reports_accepted_rejected_and_unreachable(monkeypatch):
    def fake_get(url, headers, timeout):
        if "groq" in url:
            return _Response(200)
        if "googleapis" in url:
            return _Response(400)
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(llm_clients.httpx, "get", fake_get)
    result = check_keys(config.ApiKeys(groq="g", google="a", openrouter="o"))

    assert result["groq"].ok is True
    assert result["google"].ok is False
    assert result["openrouter"].ok is None    # a network error is not a bad key


def test_check_keys_reports_a_model_the_provider_no_longer_offers(monkeypatch):
    """What happened when Groq retired llama-3.3-70b-versatile."""
    def fake_get(url, headers, timeout):
        if "groq" in url:
            return _Response(200, {"data": [{"id": "some-other-model"}]})
        return _Response(200, {"models": [{"name": f"models/{config.AGENT_VISION_MODEL}"}]})

    monkeypatch.setattr(llm_clients.httpx, "get", fake_get)
    result = check_keys(config.ApiKeys(groq="g", google="a"))

    assert result["groq"].ok is False
    assert config.TEXT_SUMMARY_MODEL in result["groq"].message
    assert "key is fine" in result["groq"].message
    assert result["google"].ok is True


def test_check_keys_reports_a_retired_openrouter_model(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/key"):
            return _Response(200, {"data": {}})
        return _Response(200, {"data": [{"id": "some/other-model:free"}]})   # the public model list

    monkeypatch.setattr(llm_clients.httpx, "get", fake_get)
    result = check_keys(config.ApiKeys(openrouter="o"))

    assert result["openrouter"].ok is False
    assert config.AGENT_LLM_MODEL in result["openrouter"].message


def test_check_keys_skips_keys_that_were_not_supplied(monkeypatch):
    monkeypatch.setattr(llm_clients.httpx, "get", lambda url, headers, timeout: _Response(200))

    assert set(check_keys(config.ApiKeys(groq="g", google="a"))) == {"groq", "google"}
    assert check_keys(config.ApiKeys()) == {}
