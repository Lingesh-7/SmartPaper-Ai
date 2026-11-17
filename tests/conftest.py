"""
Shared setup and fixtures for the offline test suite.

These tests never call a model provider and never run the layout
model: LLMs are replaced by helpers.FakeLLM, parse_pdf by a canned
result, and the embedding model by a deterministic fake. Chroma and
the on-disk paper store are real, pointed at a temporary directory.

The data directory must be redirected *before* config is imported,
which is why it happens at module import here rather than in a
fixture.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["SMARTPAPER_DATA_DIR"] = tempfile.mkdtemp(prefix="smartpaper-tests-")
# Tests must not depend on, or spend, real keys. An empty value also
# stops python-dotenv from filling these in from a local .env.
for _name in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
    os.environ[_name] = ""

# The packages have no __init__.py and rely on the repo root being on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64  # noqa: E402
import shutil  # noqa: E402

import pytest  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

import config  # noqa: E402
from helpers import jpeg_base64  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Fake embeddings and empty caches for every test; stored papers removed afterwards."""
    import pipeline
    from embeddings import embedder, store

    monkeypatch.setattr(
        embedder, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32)
    )
    # Fixed here so the tests do not move when the defaults are tuned:
    # the canned paper has 5 records, so a top-k of 4 means "most but
    # not all". And no test should wait out the summary retry pause.
    monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 4)
    monkeypatch.setattr(config, "SUMMARY_RETRY_PAUSE_SECONDS", 0)
    pipeline._retrievers.clear()
    yield
    pipeline._retrievers.clear()
    # Drop each paper's Chroma collection and files so tests cannot see
    # each other's papers. (The Chroma directory itself stays: its
    # database file is held open by the client for the whole session.)
    for manifest in store.list_manifests():
        embedder.delete_paper(manifest["paper_id"])
    for directory in (config.PAPERS_DIR, config.UPLOAD_DIR, config.EXTRACTED_IMAGES_DIR):
        shutil.rmtree(directory, ignore_errors=True)
    config.ensure_data_dirs()


@pytest.fixture
def fake_parse(monkeypatch):
    """
    Replace pipeline.parse_pdf with a canned result that also writes
    figure crops and a table snapshot into the extraction directory,
    the way unstructured does.

    Returns a dict the test can edit:
        "figures"  how many figures the canned paper has (default 1);
                   figure N is captioned "FIGURE N. ..." and labelled "Figure N"
        "result"   set to a parse_pdf-style dict to return that instead
        "calls"    number of times parse_pdf was called
        "extract_dirs"  the extraction directory of each call
    """
    import pipeline

    control = {"calls": 0, "extract_dirs": [], "result": None, "figures": 1}

    def _parse(file_path, extract_image_dir):
        control["calls"] += 1
        control["extract_dirs"].append(extract_image_dir)
        if control["result"] is not None:
            return control["result"]

        table = Path(extract_image_dir) / "table-8-1.jpg"
        table.write_bytes(base64.b64decode(jpeg_base64(color=(30, 30, 200))))

        images = []
        for index in range(1, control["figures"] + 1):
            figure = Path(extract_image_dir) / f"figure-3-{index}.jpg"
            figure.write_bytes(base64.b64decode(jpeg_base64()))
            images.append({
                "path": str(figure), "page_number": 3,
                "caption": f"FIGURE {index}. Diagram number {index} of the model.",
                "label": f"Figure {index}",
                "ocr_text": "Encoder Decoder Multi-Head Attention",
            })

        return {
            "text_chunks": [
                {"text": "The Transformer relies entirely on attention.", "page_number": 1, "page_end": 1, "category": "CompositeElement"},
                {"text": "We trained on the WMT 2014 English-German dataset.", "page_number": 7, "page_end": 8, "category": "CompositeElement"},
                {"text": "Self-attention connects all positions.", "page_number": 6, "page_end": 6, "category": "CompositeElement"},
            ],
            "tables": [
                {"text": "Table on page 8: TABLE 2. BLEU scores. Model BLEU Transformer 28.4", "page_number": 8,
                 "image_path": str(table), "caption": "TABLE 2. BLEU scores.", "label": "Table 2"},
            ],
            "images": images,
            "images_skipped": 4,
        }

    monkeypatch.setattr(pipeline, "parse_pdf", _parse)
    return control
