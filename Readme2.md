# SmartPaper AI

> A multi-agent, **multimodal** research-paper QA assistant. Upload an academic PDF, ask questions in natural language, and get answers grounded in the paper's **text, tables, _and_ figures**.

Built from three ideas layered on top of each other:

1. **Multimodal RAG** — a MultiVector retriever over ChromaDB that embeds *summaries* but returns *raw content*.
2. **An agentic LangGraph workflow** — planner → retriever → reasoner → critic, with a self-correcting retry loop.
3. **An optional QLoRA fine-tuned reader** — Phi-3 Mini trained on QASPER (present in the repo, not wired into the request path — see [Known gaps](#known-gaps)).

This document reflects the code as it actually is. Where a design choice looks odd, the reason is stated rather than glossed over.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
  - [Phase 1 — Ingestion](#phase-1--ingestion-upload)
  - [Phase 2 — Query](#phase-2--query-ask)
  - [The MultiVector pattern](#the-multivector-pattern)
  - [Per-paper retriever binding](#per-paper-retriever-binding)
- [Model routing](#model-routing)
- [Project layout](#project-layout)
- [HTTP API](#http-api)
- [Getting started](#getting-started)
- [Docker](#docker)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Fine-tuned reader](#fine-tuned-reader)
- [Known gaps](#known-gaps)
- [Conventions for contributors](#conventions-for-contributors)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Why this exists

Most "chat with your PDF" tools flatten a paper into a wall of text and lose exactly the parts a researcher cares about: the architecture diagram, the ablation table, the results plot. SmartPaper AI keeps all three modalities as first-class citizens through the entire pipeline — parsed separately, summarized by a model suited to each modality, embedded into one shared vector space, and handed back to a **vision-capable** reasoner when the retrieved evidence includes an image.

An agentic layer sits on top so a weak first answer isn't the final answer: a critic node inspects the draft, and if it looks like a refusal or is too thin, it rewrites the question and re-retrieves.

---

## How it works

Two phases, wired together in `app.py`.

### Phase 1 — Ingestion (`POST /upload`)

```
                            paper.pdf
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │  parsers/pdf_parser.py        │
                 │  unstructured.partition_pdf   │
                 │  strategy = "hi_res"          │
                 └──────────────┬───────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
     text_chunks             tables                images
  (NarrativeText only)  ("Table on page N: …")  (file paths on disk)
          │                     │                     │
          ▼                     ▼                     ▼
   ┌─────────────┐      ┌─────────────┐      ┌────────────────┐
   │ Llama-3.3   │      │ Llama-3.3   │      │ Gemini 2.5     │
   │ 70B         │      │ 70B         │      │ Flash (vision) │
   │ (OpenRouter)│      │ (Groq)      │      │ (Google)       │
   └──────┬──────┘      └──────┬──────┘      └────────┬───────┘
          │  summaries         │  summaries           │  summaries
          └─────────────────────┼──────────────────────┘
                                ▼
                   all-MiniLM-L6-v2 embeddings
                                │
                                ▼
              Chroma collection  mm_rag_<paper_id>     ← SUMMARY vectors
                                +
                      InMemoryStore docstore          ← RAW content
                 (text strings / table strings / base64 images)
```

`parse_pdf()` deliberately keeps **only `NarrativeText`** elements as text chunks. `Title`, `Header`, `Footer`, `ListItem`, and `FigureCaption` are parsed by `unstructured` and intentionally dropped, because the embedding step never consumed them — `text_chunks` is scoped to match exactly what downstream code uses.

Images are collected via `isinstance(el, ImageElement)` plus `el.metadata.image_path`. This matters: `unstructured` is configured with `extract_image_block_to_payload=False`, so it writes cropped image files to `extracted_data/` rather than embedding base64 in the element. Matching on the category *string* `"Image"` does **not** carry the file path — only the `isinstance` + metadata route does. Don't "simplify" that check.

Summarization is batched with bounded concurrency: `max_concurrency=12` for tables, `5` for text (OpenRouter is the tighter rate limit). Images are summarized serially, and any single image that fails to encode or summarize is skipped with a printed warning rather than aborting the upload.

### Phase 2 — Query (`POST /ask`)

A LangGraph `StateGraph` over a `TypedDict` state:

```
   question
       │
       ▼
  ┌─────────┐     ┌───────────┐     ┌──────────┐     ┌────────┐
  │ planner │────▶│ retriever │────▶│ reasoner │────▶│ critic │──┬──▶ END
  └─────────┘     └───────────┘     └──────────┘     └────────┘  │
                        ▲                                        │
                        └──────────── retry (max 2) ─────────────┘
```

| Node | File | What it does |
|---|---|---|
| **planner** | `agents/nodes.py` | Classifies the question as `text_query` / `figure_query` / `table_query` via a one-word Groq call. Anything unexpected falls back to `text_query`. |
| **retriever** | `agents/nodes.py` → `embeddings/retriever.py` | Calls `retrieve()`, which invokes the paper's MultiVectorRetriever and splits results into base64 images vs. text/table strings by decoding header bytes (JPEG/PNG/GIF/WebP magic numbers). Images are resized to 1300×600 before being passed on. |
| **reasoner** | `agents/nodes.py` | If images came back → **Gemini 2.5 Flash** with the images inlined as `data:image/jpeg;base64,…` parts. Otherwise → **Groq Llama-3.3-70B** with text-only context. Both prompts hard-constrain the model to the provided context and instruct it to say so when the evidence is insufficient. |
| **critic** | `agents/nodes.py` | Accepts the draft if it has ≥2 sentences *and* contains none of `REFUSAL_PHRASES` ("cannot answer", "does not mention", …). Otherwise it appends `" Please be more specific and detailed."` to the question, bumps `retry_count`, and loops back to **retriever**. After 2 retries it accepts whatever it has. |

The planner's classification is currently **informational** — it is returned to the client as `source_type` but does not change retrieval behavior. Retrieval is uniformly semantic across all three modalities.

The retry loop re-enters at **retriever**, not **planner**, so a retry re-searches with the augmented question but doesn't re-classify.

### The MultiVector pattern

This is the core idea of the whole system, so it's worth stating plainly:

> **Summaries are what gets embedded and searched. Raw content is what gets returned.**

A short, retrieval-optimized summary embeds far more cleanly than a raw table dump or a base64 image blob (which can't be embedded by a text model at all). So `_add_documents()` mints a `doc_id` per item, puts the **summary** in Chroma with that `doc_id` in its metadata, and puts the **raw content** in an `InMemoryStore` under the same key. At query time a summary vector is hit, its `doc_id` is looked up, and the reasoner receives the original paragraph, the original table string, or the original base64 image — never the summary.

### Per-paper retriever binding

`retriever_node(state, paper_retriever)` takes the retriever as an **explicit argument** rather than reading a module global, so a Flask process can serve several papers concurrently without cross-contamination. Because LangGraph requires nodes to be single-argument `state -> state` callables, `create_graph()` binds the paper's retriever via a closure:

```python
def _bound_retriever_node(state: AgentState) -> AgentState:
    return retriever_node(state, paper_retriever)
```

`run_agent(question, paper_id)` looks the retriever up in `agents.graph._paper_retrievers`, which `/upload` populates via `register_retriever()`.

> **A paper must have been uploaded in the *current process* before it can be queried.** `run_agent()` raises `ValueError` otherwise, and `/ask` surfaces that as a `400`. Restarting the server means re-uploading, even though the Chroma vectors persisted.

---

## Model routing

Every model choice lives in `config.py`. Nothing is hardcoded in a module.

| Stage | Model | Provider | Temp | Config key |
|---|---|---|---|---|
| Table summarization | `llama-3.3-70b-versatile` | Groq | 0 | `TABLE_SUMMARY_MODEL` |
| Image summarization | `gemini-2.5-flash` | Google | 0 | `IMAGE_SUMMARY_MODEL` |
| Text summarization | `meta-llama/llama-3.3-70b-instruct` | OpenRouter | 0.5 | `TEXT_SUMMARY_MODEL` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | local (HF) | — | `EMBEDDING_MODEL` |
| Agent planner / reasoner / critic | `llama-3.3-70b-versatile` | Groq | 0 | `AGENT_LLM_MODEL` |
| Agent vision reasoner | `gemini-2.5-flash` | Google | 0 | `AGENT_VISION_MODEL` |
| `/summary` route | `AGENT_LLM_MODEL` | Groq | 0 | — |
| Fine-tuned reader (unwired) | `Lingesh-7/smartpaper-phi3-qasper` | HF Hub | 0.3 | `FINETUNED_MODEL_REPO` |

Three different providers are used deliberately — tables and text hit different rate-limit pools, and images require a vision model. The embedding model runs **locally** via `sentence-transformers`, so the first run downloads ~90 MB of weights and no API key is involved in embedding.

**Why each client is named.** Every LLM client is instantiated once at module import with its own explicit variable (`_table_summary_llm`, `_image_summary_llm`, `_text_summary_llm`, `agent_llm`, `agent_vision_llm`). The original notebook this project was extracted from reused a single mutable `llm` variable, reassigning it three times — which only works because notebook cells run top-to-bottom in a fixed order. That pattern was deliberately removed; please don't reintroduce it.

---

## Project layout

```
AI_RESEARCHER_HELPER/
│
├── app.py                     Flask app — /, /upload, /ask, /summary
├── config.py                  ALL keys, model names, temps, paths
├── test_pipeline.py           End-to-end smoke script (not pytest)
│
├── parsers/
│   └── pdf_parser.py          unstructured hi_res → {text_chunks, images, tables}
│
├── embeddings/
│   ├── embedder.py            Summarize → embed → MultiVectorRetriever; ingest_paper(), delete_paper()
│   └── retriever.py           retrieve(), split_image_text_types(), base64/image helpers
│
├── agents/
│   ├── nodes.py               planner / retriever / reasoner / critic + should_continue
│   └── graph.py               StateGraph wiring, register_retriever(), run_agent()
│
├── models/
│   └── finetuned_reader.py    Lazy-loading Phi-3 + QLoRA reader (not imported anywhere)
│
├── templates/index.html       Three-panel UI: upload | chat | summary
├── static/css/style.css
├── static/js/main.js
│
├── Dockerfile                 python:3.11-slim + poppler-utils + tesseract-ocr
├── requirements.txt           pip path (see caveats below)
├── pyproject.toml + uv.lock   uv path (authoritative dependency set)
└── agents.png                 Rendered agent graph
```

> **No `__init__.py` files.** The `agents/`, `parsers/`, `embeddings/`, and `models/` packages rely on the current working directory being on `sys.path`. **Always run from the repo root** — `python app.py`, not `python ../app.py`.

---

## HTTP API

All responses are JSON. Max upload size is **20 MB** (`MAX_CONTENT_LENGTH`); exceeding it returns `413` with a JSON error rather than Flask's HTML page.

### `GET /`

Renders the single-page UI (`templates/index.html`) — a three-panel layout: drag-and-drop upload on the left, chat in the center, generated summary on the right.

### `POST /upload`

Multipart form with a `file` field. Must be a `.pdf`.

```bash
curl -F "file=@attention-is-all-you-need-Paper.pdf" http://localhost:7860/upload
```

```json
{
  "paper_id": "3f2b8c1e-...",
  "text_chunks": 42,
  "images": 5,
  "tables": 3
}
```

The PDF is saved to `UPLOAD_DIR/<paper_id>.pdf`, parsed, ingested, and its retriever registered. This is the slow call — every text chunk, table, and image gets an LLM round-trip. Expect tens of seconds to a few minutes for a real paper.

| Status | Meaning |
|---|---|
| `400` | No file, empty filename, or non-PDF extension |
| `413` | Over the 20 MB limit |
| `500` | Parsing or ingestion raised (message included) |

### `POST /ask`

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "3f2b8c1e-...", "question": "Explain the architecture diagram."}'
```

```json
{
  "final_answer": "The figure shows an encoder-decoder stack in which ...",
  "source_type": "figure_query",
  "retry_count": 0
}
```

`source_type` is the planner's classification. `retry_count` of `2` means the critic rejected the draft twice and you're seeing a best-effort answer — usually a signal that retrieval missed.

| Status | Meaning |
|---|---|
| `400` | Missing `paper_id` / `question`, or that `paper_id` has no registered retriever (e.g. after a restart) |
| `500` | The agent raised |

### `GET /summary?paper_id=<id>`

Takes the **first 5 text chunks** of the paper (a proxy for title + abstract + intro) and asks Groq for a 3-sentence overview, 5 key contributions, and 3 suggested follow-up questions. Returns `{"summary": "..."}`.

This reuses the cached `parsed_data` in `app.py`'s `_paper_parsed_data`, so it never re-parses the PDF. Returns `400` if the `paper_id` isn't in that cache.

---

## Getting started

### Prerequisites

- **Python 3.13** (`.python-version`; the Docker image pins 3.11 and also works)
- **Poppler** and **Tesseract** on `PATH` — `unstructured`'s `hi_res` strategy shells out to both for PDF rasterization and OCR. On Windows, install [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) and add their `bin` directories to `PATH`.
- API keys for **Groq**, **Google AI Studio**, and **OpenRouter**.

### 1. Clone

```bash
git clone https://github.com/Lingesh-7/Research-QA
cd Research-QA
```

### 2. Install

`uv` is the project's package manager and `uv.lock` is committed, so this is the reproducible path:

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install langchain-openrouter        # see caveat below
```

> ⚠️ **`requirements.txt` is incomplete.** `embeddings/embedder.py` imports `langchain_openrouter`, which is listed in `pyproject.toml` but **missing from `requirements.txt`**. Any pip- or Docker-based install will fail at import until you add it. `requirements.txt` also carries leftovers the app doesn't use (`streamlit`, `faiss-cpu`, `tavily-python`, `langchain-astradb`, `dataset`, `fpdf2`) and duplicates several entries. Prefer `uv sync`.

### 3. Environment

Create `.env` at the repo root:

```env
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-v1-...

# optional
PORT=7860
FINETUNED_MODEL_REPO=Lingesh-7/smartpaper-phi3-qasper
```

All three keys are **required**. `config.validate_config()` is called at `embeddings/embedder.py` **import time**, so a missing key fails immediately at startup with a clear `EnvironmentError` — not deep inside a chain at request time.

### 4. Windows path fix

`config.py` sets `CHROMA_PERSIST_DIR = "/tmp/chroma_db"` and `UPLOAD_DIR = "/tmp/uploads"`. Those are correct inside the Docker image but wrong on Windows. For a local Windows run, change them to relative paths:

```python
CHROMA_PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
```

Both directories are already in `.gitignore`.

### 5. Run

```bash
python app.py
```

Open **http://localhost:7860** (override with `PORT`).

---

## Docker

The image installs the system dependencies `unstructured`'s `hi_res` strategy needs (`poppler-utils`, `tesseract-ocr`, `libgl1`, `libglib2.0-0`):

```bash
docker build -t smartpaper .
docker run -p 7860:7860 --env-file .env smartpaper
```

Two caveats:

- The build runs `pip install -r requirements.txt`, so it inherits the missing `langchain-openrouter` problem above — add the line before building.
- `/tmp/chroma_db` and `/tmp/uploads` live inside the container. Mount a volume if you want them to survive `docker run` cycles — though note that persisted vectors alone aren't enough to query a paper after a restart (see [Known gaps](#known-gaps)).

---

## Configuration reference

Everything in `config.py`:

| Constant | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | env | Required |
| `GOOGLE_API_KEY` | env | Required |
| `OPENROUTER_API_KEY` | env | Required |
| `TABLE_SUMMARY_MODEL` | `llama-3.3-70b-versatile` | Groq |
| `TABLE_SUMMARY_TEMPERATURE` | `0` | |
| `IMAGE_SUMMARY_MODEL` | `gemini-2.5-flash` | Vision |
| `IMAGE_SUMMARY_TEMPERATURE` | `0` | |
| `TEXT_SUMMARY_MODEL` | `meta-llama/llama-3.3-70b-instruct` | OpenRouter |
| `TEXT_SUMMARY_TEMPERATURE` | `0.5` | |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally, 384-dim |
| `AGENT_LLM_MODEL` | `llama-3.3-70b-versatile` | Planner / reasoner / critic |
| `AGENT_LLM_TEMPERATURE` | `0` | |
| `AGENT_VISION_MODEL` | `gemini-2.5-flash` | Figure reasoning |
| `CHROMA_PERSIST_DIR` | `/tmp/chroma_db` | Change on Windows |
| `CHROMA_COLLECTION_NAME` | `mm_rag` | Per-paper suffix: `mm_rag_<paper_id>` |
| `EXTRACTED_IMAGES_DIR` | `./extracted_data` | Declared in config, but `parse_pdf()` defaults to its own `"extracted_data"` argument |
| `UPLOAD_DIR` | `/tmp/uploads` | Created at import |
| `FINETUNED_MODEL_REPO` | `Lingesh-7/smartpaper-phi3-qasper` | Env-overridable |

Keys are also re-exported into `os.environ` for libraries that read them directly rather than accepting a constructor argument.

---

## Testing

There is **no test framework, linter, or formatter configured**. What exists are two manual smoke scripts.

**End-to-end** — parses a PDF, ingests it, and runs 5 fixed questions through the agent:

```bash
python test_pipeline.py attention-is-all-you-need-Paper.pdf
```

It prints each answer along with its `retrieval_type` and `retry_count`. There is no way to run "a single test" other than editing the `test_questions` list in the script.

**Parser only** — fast sanity check that doesn't touch any API:

```bash
python parsers/pdf_parser.py attention-is-all-you-need-Paper.pdf
# Text chunks: 42
# Images: 5
# Tables: 3
```

If this prints `Images: 0` on a paper that obviously has figures, the problem is Poppler/Tesseract, not the pipeline.

---

## Fine-tuned reader

`models/finetuned_reader.py` loads **Phi-3 Mini (3.8B) + QLoRA adapters** trained on the [`allenai/qasper`](https://huggingface.co/datasets/allenai/qasper) dataset via Unsloth, published as a LoRA-only checkpoint at `Lingesh-7/smartpaper-phi3-qasper`. Training lives in `FT-For-RQA.ipynb` (untracked — `.gitignore` excludes `*.ipynb`).

| Parameter | Value |
|---|---|
| Base model | `microsoft/Phi-3-mini-4k-instruct` |
| Dataset | `allenai/qasper` |
| Method | QLoRA (4-bit) |
| Framework | Unsloth + PEFT |
| Max sequence length | 2048 |
| Inference temperature | 0.3 |

The module loads lazily on first `generate_answer()` call, since a 4-bit Unsloth load takes several seconds and most requests never need it.

**It is not wired into the request path.** Nothing imports it. To use it you'd call `generate_answer(question, context)` from the reasoner as an alternative branch — and you'd need a CUDA GPU, because the module hardcodes `.to("cuda")`. `unsloth` is in `pyproject.toml` but not `requirements.txt`.

---

## Known gaps

These are **documented and deliberate**, not accidental. Please don't "fix" them silently in a drive-by commit — each one has a real cost attached.

| Gap | Detail |
|---|---|
| **Fine-tuned reader unwired** | `models/finetuned_reader.py` is never imported. Hardcodes `.to("cuda")`; `unsloth` missing from `requirements.txt`. |
| **Page numbers are dropped** | `parse_pdf()` extracts `page_number` for every chunk, but `_add_documents()` stores **bare strings** in the docstore with no metadata. `retrieve()` therefore always returns `page_number: None`. Real citations require changing what the docstore holds — `(content, page_number)` tuples or dicts — and unpacking them in `retrieve()`. Flagged rather than silently faked. |
| **All state is in-process** | The `InMemoryStore` docstore, `agents.graph._paper_retrievers`, and `app.py`'s `_paper_parsed_data` don't survive a restart and don't work across multiple Gunicorn workers — even though Chroma persists the vectors to disk. |
| **`top_k` is accepted and ignored** | `retrieve(retriever, query, top_k=3)` takes the parameter for API compatibility but never wires `search_kwargs` through to the `MultiVectorRetriever`, so its own default `k` applies. |
| **`delete_paper()` is never called** | It's imported by `app.py` but unused, so Chroma collections accumulate one per upload, forever. |
| **`/tmp` paths on Windows** | `CHROMA_PERSIST_DIR` and `UPLOAD_DIR` are Docker-correct, Windows-wrong. |
| **`langchain-openrouter` missing from `requirements.txt`** | Present in `pyproject.toml`. Docker builds will fail on import. |
| **Planner output is informational** | The `text_query` / `figure_query` / `table_query` classification is returned to the client but doesn't steer retrieval. |
| **`pyproject.toml` template leftovers** | `name = "ats-freindly-resume-maker"` and `main.py` come from an unrelated template and are not part of this project. |
| **Uploaded PDFs are never deleted** | `UPLOAD_DIR` grows unbounded. |

---

## Conventions for contributors

- **All model names, temperatures, paths, and keys live in `config.py`.** Never hardcode a model string in a module.
- **Each LLM client is created once at module import with its own explicit name.** The notebook's reassigned-`llm` pattern was deliberately removed and should not come back.
- **Module docstrings explain *why* a piece diverges from the original notebook.** Preserve that reasoning when editing — it's the only record of several non-obvious decisions (the `isinstance(el, ImageElement)` check, the explicit `paper_retriever` argument, the `NarrativeText`-only filter).
- **Run everything from the repo root** — there are no `__init__.py` files.
- `.gitignore` excludes `*.ipynb`, `*.pdf`, `chroma_db/`, `extracted_data/`, and `uploads/`, so the training notebooks and sample PDFs in the working tree are untracked.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `EnvironmentError: Missing required environment variables` at startup | One of the three keys isn't in `.env`. All three are required, including `OPENROUTER_API_KEY`. |
| `400 No retriever registered for paper_id=...` | The server restarted since upload. Re-upload the PDF — retrievers live in process memory only. |
| `Images: 0` from the parser on a figure-heavy paper | Poppler and/or Tesseract not on `PATH`. |
| Upload hangs for minutes | Expected on long papers — every chunk, table, and image is an LLM round-trip. Watch the `[ingest_paper:<id>]` progress lines on stdout. |
| `ModuleNotFoundError: langchain_openrouter` | Installed via `requirements.txt`, which omits it. `pip install langchain-openrouter` or use `uv sync`. |
| `ModuleNotFoundError: agents` / `parsers` | You're not in the repo root. |
| Answers come back with `retry_count: 2` | The critic rejected two drafts — retrieval likely missed. Check that the paper actually ingested a nonzero number of chunks. |
| Rate-limit errors during ingestion | Lower `max_concurrency` in `summarize_texts` / `summarize_tables` in `embeddings/embedder.py`. |
| Chroma directory grows without bound | `delete_paper()` exists but is never called. Clean the persist directory manually. |

---

## Example queries

**Figures**
```
Explain Figure 3.
What trend is shown in the results plot?
What does the architecture diagram show?
```

**Tables**
```
Summarize Table 2.
Compare the BLEU scores across the reported models.
What are the key results reported in the tables?
```

**Research understanding**
```
What is the main contribution of this paper?
Why did the authors choose self-attention over recurrence?
What datasets were used in this work?
What limitations do the authors acknowledge?
```

---

## Doc drift note

The original `README.md` claims `BAAI/bge-base-en-v1.5` embeddings and Llama 3.1, and its Getting Started omits `OPENROUTER_API_KEY`. Neither matches the code. **`config.py` is the source of truth**; this document follows it.

---

## License

MIT — intended for research and educational use.

*© 2026 Lingesh R — AI & Machine Learning Engineer*
