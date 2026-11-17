"""
Tests for pipeline.py: ingestion, the saved paper library, limits,
failure handling, and question answering against a stored paper.

parse_pdf and every LLM are faked (see conftest.py); Chroma and the
on-disk store are real.
"""

import io

import pytest

import config
import pipeline
from embeddings import embedder, store
from embeddings.embedder import IngestionError
from embeddings.retriever import retrieve
from parsers.pdf_parser import PdfError
from pipeline import PaperError
from helpers import FakeLLM, make_clients, make_pdf_bytes

GOOD_ANSWER = "The model uses attention (page 1). It was trained on WMT 2014 (page 7)."


def answering_clients(answer=GOOD_ANSWER, **overrides):
    def reply(prompt):
        if prompt.startswith("Classify the following question"):
            return "text_query"
        if "Rewritten question:" in prompt:
            return "What dataset was the Transformer trained on?"
        return answer

    # With 5 records and top_k=4 a figure is usually retrieved, which
    # routes the answer to the vision model - so both models answer alike.
    overrides.setdefault("agent_vision_llm", FakeLLM(answer))
    return make_clients(agent_llm=FakeLLM(reply), **overrides)


def _llm_calls(clients) -> int:
    return sum(
        len(llm.prompts)
        for llm in (clients.text_summary_llm, clients.table_summary_llm,
                    clients.image_summary_llm, clients.agent_llm, clients.agent_vision_llm)
    )


def _temp_dirs_are_empty() -> bool:
    return not any(config.UPLOAD_DIR.iterdir()) and not any(config.EXTRACTED_IMAGES_DIR.iterdir())


# --- the happy path ---
def test_ingest_stores_typed_records_with_page_numbers(fake_parse):
    progress = []
    info = pipeline.ingest_pdf(
        make_pdf_bytes(pages=2), "attention.pdf", make_clients(),
        progress=lambda message, fraction: progress.append((message, fraction)),
    )

    assert (info.text_chunks, info.tables, info.images, info.pages) == (3, 1, 1, 2)
    assert info.filename == "attention.pdf"
    assert info.reused is False
    assert info.warnings == []

    records = list(store.load_records(info.paper_id).values())
    assert [r["type"] for r in records] == ["text", "text", "text", "table", "image"]
    assert [r["page_number"] for r in records] == [1, 7, 6, 8, 3]
    assert records[1]["page_end"] == 8    # a chunk that runs over a page break
    assert records[3]["content"].startswith("Table on page 8:")
    assert records[3]["image"]            # the table's snapshot, for display
    assert (records[3]["label"], records[3]["caption"]) == ("Table 2", "TABLE 2. BLEU scores.")
    assert records[4]["content"]          # the figure itself, as base64
    assert records[4]["label"] == "Figure 1"
    assert records[4]["caption"].startswith("FIGURE 1.")
    assert records[4]["ocr_text"] == "Encoder Decoder Multi-Head Attention"
    assert store.load_manifest(info.paper_id)["images_skipped"] == 4

    fractions = [fraction for _, fraction in progress]
    assert fractions == sorted(fractions), "progress must never go backwards"
    assert fractions[-1] == 1.0


def test_uploaded_pdf_and_figure_crops_are_deleted_after_ingestion(fake_parse):
    pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())

    assert _temp_dirs_are_empty()


def test_retrieval_returns_typed_records_and_honours_top_k(fake_parse):
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    default = retrieve(retriever, "attention")
    assert len(default) == config.RETRIEVAL_TOP_K
    expected_fields = {"content", "type", "page_number", "page_end", "image", "caption", "label", "ocr_text",
                       "section", "pinned"}
    assert all(set(r) == expected_fields for r in default)

    assert len(retrieve(retriever, "attention", top_k=2)) == 2
    assert len(retrieve(retriever, "attention", top_k=5)) == 5
    # asking for a different k must not change the shared retriever
    assert retriever.search_kwargs == {"k": config.RETRIEVAL_TOP_K}


def test_answer_question_returns_answer_with_page_tagged_sources(fake_parse):
    clients = answering_clients()
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    result = pipeline.answer_question("What is the Transformer?", info.paper_id, clients)

    assert result["final_answer"] == GOOD_ANSWER
    assert result["sources"]
    assert all(source["page_number"] is not None for source in result["sources"])


# --- the saved library ---
def test_uploading_the_same_file_again_reuses_the_stored_paper(fake_parse):
    data = make_pdf_bytes()
    first = pipeline.ingest_pdf(data, "a.pdf", make_clients())

    second_clients = make_clients()
    second = pipeline.ingest_pdf(data, "renamed.pdf", second_clients)

    assert second.paper_id == first.paper_id
    assert second.reused is True
    assert second.filename == "renamed.pdf"
    assert fake_parse["calls"] == 1, "the PDF must not be parsed twice"
    assert _llm_calls(second_clients) == 0, "a reused paper must cost no API calls"


def test_a_paper_survives_a_restart(fake_parse):
    clients = answering_clients()
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    pipeline._retrievers.clear()     # what a restart does to in-memory state

    assert pipeline.get_paper_info(info.paper_id).text_chunks == 3
    result = pipeline.answer_question("What is the Transformer?", info.paper_id, clients)
    assert result["final_answer"] == GOOD_ANSWER
    assert len(result["sources"]) == config.RETRIEVAL_TOP_K


def test_force_reingests_a_stored_paper(fake_parse):
    data = make_pdf_bytes()
    pipeline.ingest_pdf(data, "a.pdf", make_clients())

    info = pipeline.ingest_pdf(data, "a.pdf", make_clients(), force=True)

    assert info.reused is False
    assert fake_parse["calls"] == 2
    assert len(store.load_records(info.paper_id)) == 5, "old vectors/records must be replaced, not doubled"
    assert len(retrieve(pipeline.get_retriever(info.paper_id), "attention", top_k=10)) == 5


def test_two_papers_never_share_an_extraction_directory_or_results(fake_parse):
    first = pipeline.ingest_pdf(make_pdf_bytes(marker="one"), "one.pdf", make_clients())

    fake_parse["result"] = {
        "text_chunks": [{"text": "Second paper only.", "page_number": 2, "category": "NarrativeText"}],
        "tables": [],
        "images": [],
    }
    second = pipeline.ingest_pdf(make_pdf_bytes(marker="two"), "two.pdf", make_clients())

    assert first.paper_id != second.paper_id
    assert len(set(fake_parse["extract_dirs"])) == 2
    contents = [r["content"] for r in retrieve(pipeline.get_retriever(second.paper_id), "anything", top_k=10)]
    assert contents == ["Second paper only."]


def test_a_paper_whose_vectors_are_gone_is_reingested_not_reused(fake_parse):
    data = make_pdf_bytes()
    info = pipeline.ingest_pdf(data, "a.pdf", make_clients())
    pipeline._retrievers.clear()
    embedder._drop_collection(info.paper_id)      # e.g. the Chroma directory was wiped

    with pytest.raises(PaperError, match="upload it again"):
        pipeline.get_retriever(info.paper_id)

    again = pipeline.ingest_pdf(data, "a.pdf", make_clients())
    assert again.reused is False
    assert fake_parse["calls"] == 2


def test_least_recently_used_papers_are_evicted(fake_parse, monkeypatch):
    monkeypatch.setattr(config, "MAX_STORED_PAPERS", 2)
    ids = []
    for index in range(3):
        info = pipeline.ingest_pdf(make_pdf_bytes(marker=f"paper-{index}"), f"{index}.pdf", make_clients())
        ids.append(info.paper_id)
        store.update_manifest(info.paper_id, last_used=1000 + index)

    assert pipeline.get_paper_info(ids[0]) is None, "the oldest paper should have been evicted"
    assert pipeline.get_paper_info(ids[1]) is not None
    assert pipeline.get_paper_info(ids[2]) is not None
    with pytest.raises(PaperError):
        pipeline.get_retriever(ids[0])


# --- limits and bad input ---
@pytest.mark.parametrize("data, message", [
    (b"", "empty"),
    (b"just some text, not a pdf", "not a PDF"),
])
def test_non_pdf_uploads_are_rejected_before_parsing(fake_parse, data, message):
    with pytest.raises(PaperError, match=message):
        pipeline.ingest_pdf(data, "x.pdf", make_clients())

    assert fake_parse["calls"] == 0


def test_oversized_upload_is_rejected(fake_parse, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)

    with pytest.raises(PaperError, match="larger than"):
        pipeline.ingest_pdf(make_pdf_bytes(), "x.pdf", make_clients())


def test_pdf_with_too_many_pages_is_rejected_before_parsing(fake_parse, monkeypatch):
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 3)

    with pytest.raises(PaperError, match="4 pages; the limit is 3"):
        pipeline.ingest_pdf(make_pdf_bytes(pages=4), "x.pdf", make_clients())

    assert fake_parse["calls"] == 0
    assert _temp_dirs_are_empty()


def test_damaged_pdf_is_rejected_with_a_readable_message(fake_parse):
    with pytest.raises(PdfError, match="could not be read"):
        pipeline.ingest_pdf(b"%PDF-1.7 this is not really a pdf", "x.pdf", make_clients())

    assert _temp_dirs_are_empty()


def test_password_protected_pdf_is_rejected(fake_parse):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(PdfError, match="password-protected"):
        pipeline.ingest_pdf(buffer.getvalue(), "locked.pdf", make_clients())

    assert fake_parse["calls"] == 0


def test_pdf_with_nothing_extractable_is_rejected_and_not_stored(fake_parse):
    fake_parse["result"] = {"text_chunks": [], "tables": [], "images": []}
    data = make_pdf_bytes()

    with pytest.raises(PaperError, match="scanned"):
        pipeline.ingest_pdf(data, "scan.pdf", make_clients())

    assert pipeline.get_paper_info(pipeline.paper_id_for(data)) is None
    assert _temp_dirs_are_empty()


# --- provider failures ---
def test_one_failed_summary_does_not_abort_the_ingestion(fake_parse):
    def flaky(prompt):
        if "WMT 2014" in prompt:
            raise RuntimeError("429 rate limit")
        return "Summary of a passage."

    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients(text_summary_llm=FakeLLM(flaky)))

    assert info.text_chunks == 3
    assert len(info.warnings) == 1 and "1 of 3 text passages" in info.warnings[0]
    # the warning is stored, so the next visitor who gets this paper sees it too
    assert pipeline.get_paper_info(info.paper_id).warnings == info.warnings


def test_a_provider_failing_every_request_aborts_and_stores_nothing(fake_parse):
    def dead(prompt):
        raise RuntimeError("Error code: 401 - invalid api key")

    data = make_pdf_bytes()
    with pytest.raises(IngestionError, match="request to Groq failed"):
        pipeline.ingest_pdf(data, "a.pdf", make_clients(text_summary_llm=FakeLLM(dead)))

    paper_id = pipeline.paper_id_for(data)
    assert pipeline.get_paper_info(paper_id) is None, "a failed ingestion must not be reusable"
    assert embedder.load_retriever(paper_id) is None
    assert _temp_dirs_are_empty()

    # and the same file ingests cleanly afterwards
    assert pipeline.ingest_pdf(data, "a.pdf", make_clients()).reused is False


def test_uploading_a_paper_spends_no_vision_requests(fake_parse):
    """
    A free Google key allows ~20 vision requests a day. Describing every
    figure at upload used all of them on one paper; figures are now
    indexed by caption and the text printed in them.
    """
    fake_parse["figures"] = 5
    vision = FakeLLM("should not be called")

    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients(image_summary_llm=vision))

    assert info.images == 5
    assert vision.prompts == []
    assert info.warnings == []


def test_a_figure_is_found_by_its_caption_without_any_vision_model(fake_parse):
    fake_parse["figures"] = 3
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    # the index text of a figure is exactly its caption + the text printed in it
    hits = retriever.vectorstore.similarity_search("anything", k=10, filter={"type": "image"})
    assert sorted(hit.page_content for hit in hits) == [
        f"FIGURE {n}. Diagram number {n} of the model.\nText in the figure: Encoder Decoder Multi-Head Attention"
        for n in (1, 2, 3)
    ]


def test_vision_descriptions_are_an_optional_extra_within_a_budget(fake_parse, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_SUMMARY_BUDGET", 2)
    fake_parse["figures"] = 4
    vision = FakeLLM("A diagram with boxes and arrows.")

    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients(image_summary_llm=vision))

    assert info.images == 4
    assert len(vision.prompts) == 2, "only as many vision requests as the budget allows"
    hits = pipeline.get_retriever(info.paper_id).vectorstore.similarity_search(
        "x", k=10, filter={"type": "image"}
    )
    assert sum("A diagram with boxes and arrows." in hit.page_content for hit in hits) == 2


def test_a_failing_vision_model_never_costs_a_figure(fake_parse, monkeypatch):
    """Before: 33 of 37 figures were dropped as 'not searchable' when Gemini's quota ran out."""
    monkeypatch.setattr(config, "IMAGE_SUMMARY_BUDGET", 10)
    fake_parse["figures"] = 6

    def quota_exhausted(prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    vision = FakeLLM(quota_exhausted)
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients(image_summary_llm=vision))

    assert info.images == 6, "every figure is kept, indexed by its caption"
    assert len(vision.prompts) == 2, "after two failures in a row the rest are not attempted"
    assert any("2 of 6 figures could not be described" in w for w in info.warnings)
    stored_types = [r["type"] for r in store.load_records(info.paper_id).values()]
    assert stored_types.count("image") == 6


def test_ingestion_lock_is_released_after_a_failure(fake_parse):
    def dead(prompt):
        raise RuntimeError("boom")

    with pytest.raises(IngestionError):
        pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients(text_summary_llm=FakeLLM(dead)))

    assert pipeline._ingest_lock.acquire(blocking=False), "a failed ingestion must not block the next one"
    pipeline._ingest_lock.release()


# --- questions ---
def test_question_validation(fake_parse):
    clients = answering_clients()
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    with pytest.raises(PaperError, match="Type a question"):
        pipeline.answer_question("   ", info.paper_id, clients)
    with pytest.raises(PaperError, match="too long"):
        pipeline.answer_question("x" * (config.MAX_QUESTION_CHARS + 1), info.paper_id, clients)


def test_question_about_an_unknown_paper_is_a_readable_error():
    with pytest.raises(PaperError, match="upload it again"):
        pipeline.answer_question("What?", "f" * 32, answering_clients())
    with pytest.raises(PaperError, match="upload it again"):
        pipeline.answer_question("What?", "../../etc/passwd", answering_clients())


def test_empty_model_answer_is_reported_not_shown(fake_parse):
    clients = answering_clients(answer="   ")
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    with pytest.raises(PaperError, match="empty answer"):
        pipeline.answer_question("What?", info.paper_id, clients)


# --- summary ---
def test_summary_is_generated_once_and_then_cached(fake_parse):
    clients = make_clients(agent_llm=FakeLLM("An overview of the paper."))
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    assert pipeline.summarize_paper(info.paper_id, clients) == "An overview of the paper."
    assert "The Transformer relies entirely on attention." in clients.agent_llm.prompts[0]

    other_visitor = make_clients(agent_llm=FakeLLM("should not be called"))
    assert pipeline.summarize_paper(info.paper_id, other_visitor) == "An overview of the paper."
    assert other_visitor.agent_llm.prompts == []


def test_summary_of_a_paper_without_text_is_a_readable_error(fake_parse):
    fake_parse["result"] = {
        "text_chunks": [],
        "tables": [{"text": "Table on page 1: a b", "page_number": 1, "image_path": None}],
        "images": [],
    }
    clients = make_clients()
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    with pytest.raises(PaperError, match="no extracted text"):
        pipeline.summarize_paper(info.paper_id, clients)


# --- error wording ---
def test_describe_error_hides_api_keys():
    keys = config.ApiKeys(groq="gsk_live_SECRET", google="AIza_SECRET")
    error = RuntimeError("Request failed for key gsk_live_SECRET with AIza_SECRET")

    message = pipeline.describe_error(error, keys)

    assert "SECRET" not in message


@pytest.mark.parametrize("raw, expected", [
    ("Error code: 401 - groq: Invalid API Key", "Groq rejected the API key"),
    ("openrouter: User not found.", "OpenRouter rejected the API key"),
    ("429 RESOURCE_EXHAUSTED: gemini quota exceeded", "Google is rate-limiting"),
    ("Error code: 404 - groq model_not_found", "Groq no longer offers the model"),
    ("Request timed out.", "could not be reached"),
])
def test_describe_error_turns_provider_errors_into_actions(raw, expected):
    assert expected in pipeline.describe_error(RuntimeError(raw))


def test_describe_error_passes_user_facing_messages_through():
    assert pipeline.describe_error(PaperError("This PDF has 99 pages; the limit is 50.")) == \
        "This PDF has 99 pages; the limit is 50."


# --- table / figure questions are guaranteed a table / figure ---
def _paper_with_many_passages(fake_parse):
    """30 text passages, 2 tables, 1 figure: plain search would mostly return text."""
    import base64
    from pathlib import Path

    from helpers import jpeg_base64

    def build(extract_dir):
        figure = Path(extract_dir) / "figure-3-1.jpg"
        figure.write_bytes(base64.b64decode(jpeg_base64()))
        return {
            "text_chunks": [
                {"text": f"Passage number {i} about attention.", "page_number": i % 9 + 1, "category": "NarrativeText"}
                for i in range(30)
            ],
            "tables": [
                {"text": "Table on page 8: BLEU scores 27.3 28.4", "page_number": 8, "image_path": None},
                {"text": "Table on page 9: variations of the model", "page_number": 9, "image_path": None},
            ],
            "images": [{"path": str(figure), "page_number": 3}],
        }

    return build


def test_a_table_question_is_guaranteed_tables_first(fake_parse, monkeypatch):
    build = _paper_with_many_passages(fake_parse)
    monkeypatch.setattr(pipeline, "parse_pdf", lambda path, extract_dir: build(extract_dir))
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    results = retrieve(retriever, "What are the key results?", prefer_type="table")

    assert len(results) == config.RETRIEVAL_TOP_K
    assert [r["type"] for r in results[:2]] == ["table", "table"]
    assert len({r["content"] for r in results}) == len(results), "no record may appear twice"


def test_a_figure_question_is_guaranteed_the_figure_first(fake_parse, monkeypatch):
    build = _paper_with_many_passages(fake_parse)
    monkeypatch.setattr(pipeline, "parse_pdf", lambda path, extract_dir: build(extract_dir))
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    results = retrieve(retriever, "What does the diagram show?", prefer_type="image")

    assert results[0]["type"] == "image"            # the paper has only one figure
    assert len(results) == config.RETRIEVAL_TOP_K


def test_a_question_naming_a_figure_or_table_gets_exactly_that_one(fake_parse):
    fake_parse["figures"] = 6
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    for number in (2, 5):
        results = retrieve(retriever, f"Explain Figure {number}", prefer_type="image")
        assert results[0]["label"] == f"Figure {number}"

    results = retrieve(retriever, "What does Table 2 report?", prefer_type="table", exclude_type="image")
    assert results[0]["label"] == "Table 2"

    # a figure the paper does not have changes nothing
    assert len(retrieve(retriever, "Explain Figure 99")) == config.RETRIEVAL_TOP_K


def test_figures_are_kept_out_of_a_text_questions_context(fake_parse):
    fake_parse["figures"] = 6
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    with_figures = retrieve(retriever, "attention", top_k=10)
    without = retrieve(retriever, "attention", top_k=10, exclude_type="image")

    assert any(r["type"] == "image" for r in with_figures)
    assert [r["type"] for r in without].count("image") == 0
    assert len(without) == 4                                  # 3 passages + 1 table
    assert "filter" not in retriever.search_kwargs, "the shared retriever must not be modified"


def test_a_paper_stored_in_an_older_format_is_reprocessed(fake_parse, monkeypatch):
    data = make_pdf_bytes()
    current = config.STORE_SCHEMA_VERSION
    monkeypatch.setattr(config, "STORE_SCHEMA_VERSION", current - 1)
    old = pipeline.ingest_pdf(data, "a.pdf", make_clients())
    monkeypatch.setattr(config, "STORE_SCHEMA_VERSION", current)
    pipeline._retrievers.clear()

    again = pipeline.ingest_pdf(data, "a.pdf", make_clients())

    assert again.paper_id == old.paper_id
    assert again.reused is False, "a paper stored by an older version must not be reused"
    assert fake_parse["calls"] == 2
    assert len(retrieve(pipeline.get_retriever(again.paper_id), "x", top_k=20)) == 5, "old vectors must be gone"


def test_unreadable_stored_papers_are_cleaned_up(fake_parse, monkeypatch):
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    pipeline._retrievers.clear()
    (store.paper_dir(info.paper_id) / "manifest.json").unlink()      # e.g. an interrupted ingestion
    monkeypatch.setattr(pipeline, "_STALE_TEMP_SECONDS", -1)         # treat it as old

    pipeline.cleanup_stale_papers()

    assert not store.paper_dir(info.paper_id).exists()
    assert embedder.load_retriever(info.paper_id) is None


def test_a_table_question_about_a_paper_without_tables_still_gets_context(fake_parse):
    fake_parse["result"] = {
        "text_chunks": [{"text": f"Passage {i}.", "page_number": 1, "category": "NarrativeText"} for i in range(6)],
        "tables": [],
        "images": [],
    }
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())

    results = retrieve(pipeline.get_retriever(info.paper_id), "What do the tables report?", prefer_type="table")

    assert len(results) == config.RETRIEVAL_TOP_K
    assert {r["type"] for r in results} == {"text"}


def test_a_table_question_reaches_the_model_with_a_table_end_to_end(fake_parse, monkeypatch):
    build = _paper_with_many_passages(fake_parse)
    monkeypatch.setattr(pipeline, "parse_pdf", lambda path, extract_dir: build(extract_dir))

    def reply(prompt):
        if prompt.startswith("Classify the following question"):
            return "table_query"
        return "The big model reaches 28.4 BLEU (page 8). That is the best score."

    clients = make_clients(agent_llm=FakeLLM(reply))
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    result = pipeline.answer_question("What are the key results in the tables?", info.paper_id, clients)

    assert result["retrieval_type"] == "table_query"
    assert [s["type"] for s in result["sources"][:2]] == ["table", "table"]
    answer_prompt = clients.agent_llm.prompts[-1]
    assert "[Table, page 8]" in answer_prompt and "The question is about a table" in answer_prompt


# --- the reference list, the paper's opening, retries ---
def _paper_with_references(monkeypatch):
    def parse(path, extract_dir):
        return {
            "text_chunks": [
                {"text": "Early Diagnosis of Ectopic Pregnancy. G. KOCUN, I. YELMEN. ABSTRACT We study EP.",
                 "page_number": 1, "page_end": 1, "category": "CompositeElement", "section": "body"},
                {"text": "I. INTRODUCTION The incidence of ectopic pregnancy has increased.",
                 "page_number": 1, "page_end": 2, "category": "CompositeElement", "section": "body"},
                {"text": "IV. RESULTS LightGBM DART reached an accuracy of 0.9161.",
                 "page_number": 11, "page_end": 11, "category": "CompositeElement", "section": "body"},
                {"text": "REFERENCES [1] Kupka, Assisted reproductive technology. [2] Backman, Pregnancy.",
                 "page_number": 18, "page_end": 19, "category": "CompositeElement", "section": "references"},
                {"text": "[30] Zhao, Development of a model. [31] Smith, Ectopic pregnancy diagnosis.",
                 "page_number": 19, "page_end": 20, "category": "CompositeElement", "section": "references"},
            ],
            "tables": [], "images": [], "images_skipped": 0,
        }

    monkeypatch.setattr(pipeline, "parse_pdf", parse)


def test_ordinary_questions_are_not_answered_from_the_reference_list(monkeypatch):
    _paper_with_references(monkeypatch)
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    results = retrieve(retriever, "What is this paper about and who wrote it?", top_k=10)

    assert len(results) == 3
    assert {r["section"] for r in results} == {"body"}


def test_a_question_about_citations_does_search_the_reference_list(monkeypatch):
    _paper_with_references(monkeypatch)
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients())
    retriever = pipeline.get_retriever(info.paper_id)

    for question in ("Which references discuss assisted reproduction?", "What prior work do the authors cite?"):
        results = retrieve(retriever, question, top_k=10)
        assert len(results) == 5
        assert sum(r["section"] == "references" for r in results) == 2


def test_the_title_and_abstract_reach_the_model_with_every_question(monkeypatch):
    """A question like 'who wrote it?' used to retrieve reference pages and come back unanswered."""
    _paper_with_references(monkeypatch)
    monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 1)      # search alone would return one chunk

    def reply(prompt):
        if prompt.startswith("Classify the following question"):
            return "text_query"
        return "It was written by G. Kocun and I. Yelmen (page 1). It studies ectopic pregnancy."

    clients = make_clients(agent_llm=FakeLLM(reply))
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    result = pipeline.answer_question("How was the model evaluated?", info.paper_id, clients)

    answer_prompt = clients.agent_llm.prompts[-1]
    assert "Early Diagnosis of Ectopic Pregnancy. G. KOCUN" in answer_prompt
    assert "[Title and abstract, page 1]" in answer_prompt
    opening = [s["content"][:5] for s in result["sources"] if s["page_number"] == 1]
    assert sorted(opening) == ["Early", "I. IN"], "both first-page records ride along"
    assert all(s["section"] == "body" for s in result["sources"])


def test_the_opening_is_not_listed_twice_when_search_finds_it_too(monkeypatch):
    _paper_with_references(monkeypatch)
    clients = answering_clients()
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)

    result = pipeline.answer_question("What is the paper about?", info.paper_id, clients)

    contents = [s["content"] for s in result["sources"]]
    assert len(contents) == len(set(contents)) == 3


def test_a_retry_searches_wider_than_the_first_attempt(fake_parse, monkeypatch):
    fake_parse["result"] = {
        "text_chunks": [
            {"text": f"Passage {i} about the method.", "page_number": i + 1, "page_end": i + 1,
             "category": "CompositeElement", "section": "body"}
            for i in range(12)
        ],
        "tables": [], "images": [], "images_skipped": 0,
    }
    monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 2)
    answers = ["There is insufficient information to answer. Sorry.", GOOD_ANSWER]

    def reply(prompt):
        if prompt.startswith("Classify the following question"):
            return "text_query"
        return answers.pop(0)

    clients = make_clients(agent_llm=FakeLLM(reply))
    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", clients)
    searched = []

    def on_step(node, state):
        if node == "retriever":
            searched.append(sum(1 for record in state["context"] if not record["pinned"]))

    result = pipeline.answer_question("How does it work?", info.paper_id, clients, on_step=on_step)

    assert result["retry_count"] == 1
    assert result["final_answer"] == GOOD_ANSWER
    assert searched == [2, 4], "the retry must look at twice as much of the paper"


def test_a_summary_that_fails_once_is_retried_before_falling_back(fake_parse):
    """Groq's per-minute limit clears if you wait; the summary should not be lost to it."""
    attempts = {"n": 0}

    def rate_limited_once(prompt):
        if "WMT 2014" in prompt:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("429 rate limit")
        return "Summary of a passage."

    info = pipeline.ingest_pdf(make_pdf_bytes(), "a.pdf", make_clients(text_summary_llm=FakeLLM(rate_limited_once)))

    assert attempts["n"] == 2
    assert info.warnings == [], "a summary recovered on the second attempt is not a failure"
