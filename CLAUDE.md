# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**SmartPaper AI** — a multi-agent, multimodal research-paper QA assistant. Upload an academic PDF, ask natural-language questions, get answers grounded in the paper's text, tables, *and* figures, with page citations. Combines multimodal RAG (ChromaDB + MultiVectorRetriever) with an agentic LangGraph workflow, served as a Streamlit app and deployable to Streamlit Community Cloud (primary) or as a Hugging Face Docker Space. An optional QLoRA fine-tuned Phi-3 reader exists but is not wired in.

## Commands

```bash
# Setup (uv is the project's package manager — uv.lock is committed)
uv sync

# Run the app (http://localhost:8501 - Streamlit's default; no port is set in .streamlit/config.toml
# because Streamlit Community Cloud and the Dockerfile each supply their own)
uv run streamlit run streamlit_app.py

# Offline test suite — no API calls, no keys, fake LLMs + real Chroma (~1 min)
uv run pytest
uv run pytest tests/test_agent.py -k critic          # a single test / subset

# Manual end-to-end check against the REAL providers (needs keys in .env).
# Costs ~17 of a free OpenRouter key's 50 daily requests. --force re-ingests.
uv run python test_pipeline.py attention-is-all-you-need-Paper.pdf [--force]

# Parser-only smoke test — prints page/chunk/image/table counts
uv run python parsers/pdf_parser.py attention-is-all-you-need-Paper.pdf

# Docker (what the Hugging Face Space builds)
docker build -t smartpaper . && docker run -p 7860:7860 smartpaper
```

There is no linter or formatter configured. `test_pipeline.py` at the repo root is a manual script, not part of pytest (`testpaths = ["tests"]`).

Always run from the repo root. The `agents/`, `parsers/`, `embeddings/`, and `models/` packages have no `__init__.py` and rely on the repo root being on `sys.path`.

**Two deployment targets, two dependency files.** Streamlit Community Cloud installs from `uv.lock` (it takes priority over `requirements.txt` there) plus `packages.txt` for apt packages; the Dockerfile (Hugging Face Spaces) installs from `requirements.txt`. Both must give CPU-only PyTorch: `pyproject.toml` routes `torch`/`torchvision` to PyTorch's CPU index via `[tool.uv.sources]` — do not remove it, or `uv.lock` resolves the CUDA build on Linux (several GB) and the Community Cloud build fails. A consequence: `uv sync --extra finetune` gives a CPU torch that cannot run the fine-tuned reader; that model needs its own GPU environment (see README).

`requirements.txt` is **generated** (Linux, Python 3.13, CPU torch) for the Dockerfile — never edit it by hand; its header has the regeneration command. Local development uses `uv sync`. Adding a dependency means editing `pyproject.toml`, running `uv lock`, and regenerating `requirements.txt`.

## API keys — bring your own

Every visitor supplies three keys in the UI; all three are required (`config.ApiKeys`). They travel through the pipeline as explicit objects and live only in `st.session_state`. **Nothing copies a key into `os.environ`**, because that is process-wide and would leak one visitor's key into another's requests.

Keys in `.env` are a local-run fallback only (`config.keys_from_env()`): the Streamlit app adopts them if all three are present, and `test_pipeline.py` requires them. On the public Space they must stay unset.

| Provider | Job | Why this provider |
|---|---|---|
| **OpenRouter** (`nvidia/nemotron-3-ultra-550b-a55b:free`) | Agent text reasoning: contextualizer, planner, reasoner, paper overview | Free model |
| **Groq** (`openai/gpt-oss-120b`) | Text + table summaries at ingestion; fallback for the agent | Free tier is limited per minute, not per day |
| **Google** (`gemini-2.5-flash`) | Answers to figure and table questions, where the model must look at the image. Nothing at upload by default | Vision |

The split is forced by two hard free-tier limits. Design changes must respect both:

- **OpenRouter free models: 50 requests/day per key.** One paper needs ~75 summaries, so summarising cannot run there. `agent_llm` is `ChatOpenRouter.with_fallbacks([ChatGroq])` so a visitor who exhausts the 50 keeps getting answers.
- **Gemini 2.5 Flash free tier: about 20 requests/day** (measured: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, limit 20). So vision is spent only on answering figure/table questions - never on describing figures at upload (`config.IMAGE_SUMMARY_BUDGET = 0`), and never on a text question. When a vision call fails, the reasoner answers from captions and extracted text and sets `vision_fallback`.

## Architecture

`streamlit_app.py` is presentation only. Everything goes through **`pipeline.py`** — `ingest_pdf()`, `answer_question()`, `summarize_paper()`, `IngestJob` — which is also what `test_pipeline.py` and the tests call.

### Ingestion (`pipeline.ingest_pdf`)

```
bytes ─► paper_id = sha256(bytes)[:32] ─► already stored? ─► reuse (no parsing, no API calls)
      ─► size / %PDF / page-count / encryption checks
      ─► one ingestion at a time (process-wide lock)
parse_pdf()      partition(): unstructured partition_pdf(strategy="hi_res"), into a per-attempt temp dir
                 structure_elements(): section-aware text chunks, tables, real figures,
                 each with page numbers; captions attached to their figure/table
ingest_paper()   summarize: tables + text chunks → Groq
                 figures: index text = caption + text printed in the figure (no API call)
                 → embed those texts into a per-paper Chroma collection (`mm_rag_<paper_id>`)
                 → store the RAW content as typed records on disk (embeddings/store.py)
finally          delete the uploaded PDF and the extracted crops; evict LRU papers past the cap
```

The MultiVector pattern is the core idea: **summaries are what gets embedded and searched; raw content is what gets returned.**

A stored paper is `data/papers/<paper_id>/{docstore.json, manifest.json}`. A record is `{type: text|table|image, content, page_number, ...}` plus, by type: text `page_end`; table `image` (its snapshot), `caption`, `label`; image `caption`, `label`, `ocr_text`. `label` is `"Figure 5"` / `"Table 2"`, parsed from the caption, and is also in the vector metadata. Changing the record shape means bumping `config.STORE_SCHEMA_VERSION`: papers stored under another version are re-ingested on upload and removed by `pipeline.cleanup_stale_papers()`. The manifest is written last, so its presence means "ingested completely"; `load_retriever()` additionally checks that the Chroma vector count matches the record count and treats a mismatch as not ingested.

**How the parser structures a paper** (`parsers/pdf_parser.py::structure_elements`, testable without the layout model). This replaced "keep only `NarrativeText`, send every `Image` to a vision model", which mirrored the notebook and failed on a real 20-page IEEE paper: the layout model labels the *abstract* and every bulleted list as `ListItem` and the title and headings as `Title` (all dropped), labels page numbers and `VOLUME 14, 2026` as `NarrativeText` (kept), and 24 of its 37 `Image` crops were the journal logo or author portraits.

- **Text** = Title, NarrativeText, ListItem, uncategorized text, formulas. Dropped: Header/Footer, and short lines that recur across pages once digits are masked (running heads, page numbers). Captions are exempt from that. Then `chunk_by_title` groups by section (`config.CHUNK_*`); a chunk records `page_number`..`page_end`.
- **The reference list** — everything from a `REFERENCES`/`Bibliography` heading on, short of an `Appendix` heading, plus `[16] ...`-style entries the model lists just before that heading — is chunked separately and tagged `section="references"` (in the record and the vector metadata; every vector carries `section`).
- **Captions** (`FIGURE 5.`, `Fig. 3:`, `TABLE II.` — word, number, then `.` or `:`; a sentence starting "Table 4 summarizes" is not one) are removed from the text and attached to the nearest figure/table on the page **by position** (`metadata.coordinates`, gap under `CAPTION_MAX_GAP_FRACTION` of the page height). Reading order is not reliable in two-column layouts and is only the fallback when coordinates are missing. An unclaimed caption goes back into the text.
- **Figures**: a crop under `IMAGE_MIN_SIDE_PX` is dropped. A crop *without a caption* is dropped if it has almost no text in it, is a wide strip, or recurs at the same size on several pages. A crop *with* a caption is always kept — a paper's plots are often all exported at one size.

Images are collected via `isinstance(el, ImageElement)` + `el.metadata.image_path` — category-string matching on `"Image"` does *not* carry the file path, so don't "simplify" that check. `extract_image_dir` has no default on purpose: crops are named `figure-<page>-<n>.jpg`, so a shared directory lets concurrent uploads overwrite each other.

A text/table summary that fails is retried once at the end, one at a time, after `SUMMARY_RETRY_PAUSE_SECONDS` (Groq's per-minute limit clears with a wait; the client's own retries are seconds apart). If it fails again, ingestion still does not abort: the item's raw text is embedded instead and a warning is stored in the manifest. If *every* summary request fails, ingestion aborts and nothing is stored. A figure is never dropped because of the vision model: with `IMAGE_SUMMARY_BUDGET > 0` a description is added to its index text when the call succeeds, and after two failures in a row the rest are not attempted.

### Query (`pipeline.answer_question`) — LangGraph state machine

```
contextualizer → planner → retriever → reasoner → critic ─┬→ END
                               ▲                          │
                               └────────── retry ─────────┘  (max 2)
```

- **contextualizer** rewrites a follow-up into a standalone question using the chat history. No history → no LLM call. Any failure → the question as asked.
- **planner** classifies `text_query` / `figure_query` / `table_query` (falls back to `text_query`). The classification **steers retrieval and reasoning**.
- **retriever** calls `embeddings.retriever.retrieve(...)`, which builds the context in this order: (1) any figure/table the question names — "Explain Figure 9", "tables 4 and 5" — fetched by `label`; (2) for a table/figure question, the best-matching records of that type (`prefer_type`, up to `config.MODALITY_TOP_K`); (3) ordinary similarity results up to `RETRIEVAL_TOP_K`. Figures are excluded from (3) unless it is a `figure_query` (`exclude_type="image"`), and reference-list chunks unless the question is about references/citations (`asks_about_references`). The node then appends the paper's **front matter** — the first-page text records up to `FRONT_MATTER_MAX_CHARS`: title, authors, abstract — marked `pinned`, to *every* question. On a retry it searches `RETRIEVAL_TOP_K x (retry_count + 1)` wide, capped at `RETRY_WIDEN_MAX`.
- **reasoner** labels every context block with its page(s) and its number when it has one (`[Table 2, page 8]`, `[Text, pages 4-5]`) and asks for citations as `(page N)`. A figure also contributes its caption and printed text to the text context. Vision (Gemini) is used only for a `figure_query` (retrieved figures) or a `table_query` (retrieved tables' snapshots); if that call raises, the text model answers and `vision_fallback` is set. Everything else goes to the OpenRouter agent model.
- **critic** accepts the draft if it has ≥2 sentences (split on `.`) and contains none of `REFUSAL_PHRASES` (which include "insufficient information" — the wording the reasoner's own prompt asks for); otherwise appends `" Please be more specific and detailed."` and loops back to **retriever**. After 2 retries it accepts whatever it has. The citation format is `(page N)`, never `(p. N)`, precisely so a citation cannot fake a second sentence.

`create_graph(paper_retriever, clients)` binds both the paper's retriever and the visitor's LLM clients by closure, to satisfy LangGraph's `state -> state` node signature without module-level state.

### Streamlit specifics

- Ingestion runs in a worker thread (`pipeline.IngestJob`) and the page polls it from an `st.fragment(run_every=1.0)`. Running it inside the script would let any click or refresh kill it minutes in.
- "An answer is owed" is derived from state (last message is the user's), and every outcome appends exactly one assistant message — answer or error — so an interrupted run resumes and a failure never loops.
- Session keys: `api_keys`, `clients`, `paper`, `job`, `messages`. **Never name a session key `keys`** — `st.session_state.keys` is the mapping's built-in method, so `state.keys is None` is always false.

## Gotchas learned the hard way

- **`ChatOpenRouter(timeout=...)` is in milliseconds.** `ChatGroq` and `ChatGoogleGenerativeAI` take seconds. Passing seconds to OpenRouter makes every request time out, and the SDK's retry loop makes that look like a multi-minute hang.
- **Providers retire models.** Groq removed `llama-3.3-70b-versatile` (404 `model_not_found`), which the whole app originally ran on. Model names are overridable via `SMARTPAPER_*_MODEL`, and `check_keys()` reports a configured model the provider no longer lists.
- **Groq's free tier rate-limits hard on per-minute tokens.** Text and table summaries run at concurrency 2; at 3, a 76-passage paper drew more 429s than successes and lost 5 summaries, and at the notebook's 12 for tables, 2 of 6 tables lost theirs. Input tokens dominate (`reasoning_effort="low"` saved only ~8%), so the lever is pacing, not the prompt.
- **Similarity search is bad at questions about the paper as a whole.** "What is this paper about and who wrote it?" retrieved reference-list pages (dense with the paper's keywords) and never the title page. Hence the pinned front matter and the `references` section tag — do not remove either without re-testing that question.
- **Model output contains characters a Windows console cannot encode** (narrow no-break spaces, non-breaking hyphens). `test_pipeline.py` reconfigures stdout to UTF-8 for that reason.
- **Gemini's free quota is per day and tiny (~20 on 2.5 Flash), and Google does not publish per-model free limits** — they are only visible per account in AI Studio. The 429 body names the quota (`...PerDayPerProjectPerModel-FreeTier`) and a `retryDelay` of many hours; retrying is pointless. Other models (`gemini-3.5-flash-lite`, …) have separate quotas and can be selected with `SMARTPAPER_GEMINI_MODEL`.
- **One sample paper is not a test set.** The "NarrativeText only" parser looked fine on *Attention Is All You Need* (arXiv, single-column abstract) and fell apart on an IEEE Access paper. Check parser changes against at least both layouts.
- **Synthetic test data that differs only in digits is "boilerplate"** to `_find_boilerplate` (it masks digits). Use genuinely different sentences in parser tests.

## Known gaps (documented, not accidental — don't "fix" them silently)

- **The fine-tuned reader is not wired in.** `models/finetuned_reader.py` loads `Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER` (Phi-3 Mini + QLoRA on QASPER, trained in `FT-For-RQA.ipynb` via Unsloth) but nothing imports it. It hardcodes `.to("cuda")` and needs CUDA PyTorch in a separate environment. The deployment targets are CPU-only, and the owner decided (after its own evaluation showed no gain over the base model) to keep it out of the app and document it in the README instead.
- **The paper library is per-process and per-disk.** One Streamlit process is assumed: the ingestion lock and the retriever cache are in-process. On a free Space the disk is wiped on restart, so stored papers do not survive it unless `SMARTPAPER_DATA_DIR` points at persistent storage.
- **The library is shared between visitors by file hash.** Anyone uploading byte-identical PDF gets the stored result, including its cached overview. There is no listing of other visitors' papers, but there is also no per-user isolation.
- **Chat history is not persisted.** It lives in `st.session_state` and is gone when the tab closes.
- **Neither deployment has been built from this repo's development environment** (no Docker there; Community Cloud builds only on their side). `requirements.txt` and `uv.lock` resolve for Linux/py3.13, but the first real build is the test. Community Cloud's ~2.7 GB RAM is tight for torch + the hi_res layout model.
- `agents.png` shows the old four-node graph (no contextualizer) and is no longer referenced by the README.
- The untracked `chroma_db/`, `extracted_data/` and `uploads/` directories at the repo root are leftovers from the Flask version; the app now writes only under `data/`.

## Conventions

- All model names, temperatures, limits, paths, and key handling live in `config.py` — never hardcode a model string in a module.
- One explicitly named LLM client per job (`table_summary_llm`, `agent_vision_llm`, …), bundled in `LLMClients` and built per set of keys by `build_clients()`. The original notebook reused one mutable `llm` variable reassigned three times; that pattern was deliberately removed and should not come back. Clients are **not** created at module import any more — that cannot work with per-visitor keys.
- Exceptions meant for the visitor are `PaperError`, `PdfError`, `IngestionError`; everything shown in the UI goes through `pipeline.describe_error()`, which scrubs API keys and turns provider errors (401, 429, model-not-found, timeouts) into an action.
- Use `logging.getLogger("smartpaper.<area>")`, not `print`, and never log a key (`ApiKeys.__repr__` masks them).
- Module docstrings in this repo explain *why* a piece diverges from the original notebook. Preserve that reasoning when editing.
- `.gitignore` excludes `*.ipynb` (except `FT-For-RQA.ipynb`, which is tracked - keep tokens out of it), `*.pdf`, `data/`, `chroma_db/`, `extracted_data/`, and `uploads/`, so the notebooks and sample PDFs present in the working tree are untracked.
