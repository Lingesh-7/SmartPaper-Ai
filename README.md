---
title: SmartPaper AI
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Ask questions about a research paper, with page citations
---

# SmartPaper AI

**Live app:** https://huggingface.co/spaces/LINGESH-7/AI-RESEARCH-PAPER-QA
**Fine-tuned model:** https://huggingface.co/LINGESH-7/Phi-3-mini-4k-instruct-FT-ON-QASPER

> Upload a research paper, ask questions in plain language, and get answers grounded in the paper's text, tables **and** figures — each one citing the page it came from.

SmartPaper AI is a multi-agent, multimodal RAG assistant built with **Streamlit**, **LangGraph**, **ChromaDB** and **Unstructured**.

---

## What it does

| | |
|---|---|
| **Reads the whole paper** | A layout model (`unstructured`, `hi_res`) extracts sections, lists, tables and figures. Running headers, page numbers, journal logos and author portraits are discarded. |
| **Answers with citations** | Every answer cites pages, e.g. `(page 7)`, and shows the exact passages, tables and figures it used. |
| **Understands tables and figures** | A question about a table is answered from the table (the model is shown the table image); a question about a figure is answered from the figure. |
| **Knows figures and tables by number** | Captions are matched to their figure or table, so "Explain Figure 9" or "What does Table 4 report?" fetches exactly that one. |
| **Knows what the paper is** | The title, authors and abstract accompany every question, and the reference list is searched only when you ask about references. |
| **Follows the conversation** | "Explain that more simply" is rewritten into a standalone question before the paper is searched. |
| **Remembers papers** | A paper is processed once. Uploading the same PDF again — by anyone — opens it instantly. |
| **Bring your own keys** | Each visitor uses their own API keys. Keys stay in the browser session and are never stored. |

---

## Bring your own keys

The app needs three keys. All three providers have a free tier.

| Key | Used for | Get one |
|---|---|---|
| **OpenRouter** | Answering questions (free model `nvidia/nemotron-3-ultra-550b-a55b:free`) | https://openrouter.ai/keys |
| **Google AI Studio** | Looking at a figure or table image when you ask about one (`gemini-2.5-flash`) | https://aistudio.google.com/apikey |
| **Groq** | Summarising the paper at upload (`openai/gpt-oss-120b`) | https://console.groq.com/keys |

Why three — each free tier is used only for what it can carry:

- **OpenRouter** free models allow **50 requests per day** per key (1,000 once the account has bought 10 credits). A question costs 2–3 requests, but uploading one paper needs about 75 summaries — so summarising runs on Groq, whose free tier is limited per minute rather than per day. If the OpenRouter allowance runs out, questions are answered by the Groq model instead of failing.
- **Gemini 2.5 Flash**'s free tier allows about **20 requests per day**. So nothing is spent on it at upload — figures are indexed by their captions and the text printed in them — and it is used only when a question is about a figure or a table. If that allowance runs out, the question is still answered from captions and extracted text, and the answer says so.

---

## How it works

```
Upload                                           Ask
──────                                           ───
PDF                                              question (+ chat history)
 │  unstructured hi_res                           │
 ├── text by section ┐                            ▼
 ├── tables + caption┼─ summarise ─► embed     contextualizer  rewrite a follow-up as a standalone question
 └── figures+caption ┘  (Groq; figures  index      │
                         by caption)    texts      │
                                      │           ▼
                                      ▼         planner         text / table / figure question?
                              ChromaDB (vectors)   │
                              + paper store        ▼
                              (raw content,     retriever       named figure/table first ("Figure 9"),
                               page numbers)       │            then tables/figures for table/figure
                                                   ▼            questions, then the best passages
                                                reasoner        OpenRouter (text); Gemini looks at the
                                                   │            image for figure/table questions
                                                   │
                                                   ▼
                                                 critic ── too thin / a refusal? retry (max 2)
                                                   │
                                                   ▼
                                           answer + cited sources
```

The retrieval pattern is **multi-vector**: *summaries* are what gets embedded and searched, the *raw content* (passage, table, figure) is what gets returned to the model.

---

## Run it locally

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and for PDF parsing `tesseract` (and `poppler`) on your PATH.

```bash
git clone https://github.com/Lingesh-7/Research-QA
cd Research-QA

uv sync                       # install dependencies from uv.lock
cp .env.example .env          # optional: put your three keys in .env

uv run streamlit run streamlit_app.py
```

Open http://localhost:8501. With keys in `.env` the app uses them directly; without, it asks for them in the browser.

### Tests

```bash
uv run pytest                 # offline test suite: no API calls, no keys needed

# Manual end-to-end check against the real providers (needs keys in .env;
# uses ~17 of a free OpenRouter key's 50 daily requests):
uv run python test_pipeline.py attention-is-all-you-need-Paper.pdf
```

---

## Deploy to Streamlit Community Cloud

The repository is ready to deploy as it is.

1. Push the repository to GitHub.
2. Go to https://share.streamlit.io → **Create app** → **Deploy a public app from GitHub**.
3. Choose the repository and branch, and set **Main file path** to `streamlit_app.py`.
4. Open **Advanced settings** and choose **Python 3.13**. Leave **Secrets** empty.
5. Click **Deploy**. The first build takes several minutes (it installs PyTorch).

**Leave Secrets empty.** With no keys on the server, every visitor enters their own. If you add the three API keys as secrets, every visitor spends yours.

What the platform reads from this repository:

| File | Purpose |
|---|---|
| `streamlit_app.py` | The app |
| `uv.lock` (with `pyproject.toml`) | Python dependencies, pinned, with CPU-only PyTorch |
| `packages.txt` | System packages for PDF parsing: `poppler-utils`, `tesseract-ocr`, `libgl1`, `libglib2.0-0` |
| `.streamlit/config.toml` | Upload limit, theme |

Good to know:

- **Memory is the risk.** Community Cloud gives an app roughly 2.7 GB of RAM. PyTorch plus the layout model use a large share of that, so a long or image-heavy PDF can push the app over the limit and restart it. If that happens, lower `SMARTPAPER_MAX_PDF_PAGES` (set it under **Secrets** as `SMARTPAPER_MAX_PDF_PAGES = "20"`), or use the Docker route below, which has 16 GB.
- **First upload after a restart is slower**: the embedding and layout models are downloaded on first use.
- **Storage is temporary.** The paper library is emptied whenever the app restarts or goes to sleep.
- **One upload at a time:** a second visitor sees "yours is next in line".

---

## Deploy with Docker (Hugging Face Spaces or any host)

The repository is also a ready-to-build **Docker Space** (the block at the top of this file is the Space configuration), for when the app needs more memory than Community Cloud offers.

1. Create a new Space → **SDK: Docker** → blank template.
2. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. Wait for the build (several minutes: it installs PyTorch and pre-downloads the models).

**Do not add the three API keys as Space secrets.** With no keys on the server, every visitor enters their own. If you set them, every visitor spends yours.

Good to know:

- **Hardware:** the free CPU tier (2 vCPU, 16 GB) is enough. A 10–15 page paper takes roughly 3–5 minutes to process the first time.
- **Storage:** a free Space's disk is wiped whenever it restarts or goes to sleep, so the paper library starts empty again. To keep it, add persistent storage to the Space and set the variable `SMARTPAPER_DATA_DIR=/data`.
- **One upload at a time:** uploads are processed one after another; a second visitor sees "yours is next in line".

Any other Docker host works the same way:

```bash
docker build -t smartpaper .
docker run -p 7860:7860 -v smartpaper-data:/home/user/app/data smartpaper
```

---

## Configuration

Everything is optional; defaults are in [`config.py`](config.py) and listed in [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `SMARTPAPER_DATA_DIR` | `./data` | Where papers, vectors and temp files are stored |
| `SMARTPAPER_OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Model that answers questions |
| `SMARTPAPER_GROQ_MODEL` | `openai/gpt-oss-120b` | Model that summarises uploads (and the answer fallback) |
| `SMARTPAPER_GEMINI_MODEL` | `gemini-2.5-flash` | Vision model |
| `SMARTPAPER_VISION_SUMMARY_BUDGET` | `0` | Figures per paper the vision model may describe at upload to enrich search. Leave at 0 on free keys |
| `SMARTPAPER_MAX_UPLOAD_MB` | `20` | Largest PDF accepted |
| `SMARTPAPER_MAX_PDF_PAGES` | `50` | Longest PDF accepted |
| `SMARTPAPER_MAX_STORED_PAPERS` | `50` | Library size before the least recently used paper is removed |
| `SMARTPAPER_TOP_K` | `6` | How many passages/tables/figures a question retrieves |

Providers retire models. When one does, the key check in the app says so by name; change the matching variable — no code change needed.

---

## Project structure

```
├── streamlit_app.py        Streamlit UI (presentation only)
├── pipeline.py             ingest_pdf / answer_question / summarize_paper - what the UI calls
├── config.py               models, limits, paths, API-key handling
├── parsers/pdf_parser.py   unstructured hi_res -> text, tables, figures (+ page numbers)
├── embeddings/
│   ├── embedder.py         summarise, embed, build the multi-vector retriever
│   ├── retriever.py        typed retrieval, table/figure guarantee
│   └── store.py            on-disk paper store (raw content + manifest)
├── agents/
│   ├── nodes.py            contextualizer, planner, retriever, reasoner, critic
│   └── graph.py            LangGraph wiring
├── models/
│   ├── llm_clients.py      per-visitor LLM clients, key + model checks
│   └── finetuned_reader.py QLoRA Phi-3 reader (optional, needs a GPU - see below)
├── tests/                  offline test suite (pytest)
├── test_pipeline.py        manual end-to-end script against real providers
├── FT-For-RQA.ipynb        how the Phi-3 reader was fine-tuned (Colab)
├── Dockerfile              Hugging Face Spaces / any Docker host
├── requirements.txt        pinned Linux dependencies for the Dockerfile (generated)
└── pyproject.toml, uv.lock dependencies for local development
```

---

## Using the fine-tuned Phi-3 reader (optional)

The repository also contains a small research-QA model: **Phi-3 Mini (3.8B) fine-tuned with QLoRA on QASPER**. The app does **not** use it — answers come from OpenRouter, Gemini and Groq — because it needs a GPU and the app is built for CPU hosting. This section is for anyone who wants to run that model themselves.

- Adapter on the Hub: [`Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER`](https://huggingface.co/Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER) (LoRA weights only, 25 MB)
- Training notebook: [`FT-For-RQA.ipynb`](FT-For-RQA.ipynb)
- Loader: [`models/finetuned_reader.py`](models/finetuned_reader.py)

### What you need

- An **NVIDIA GPU with CUDA** and about 4 GB of free VRAM (the base model is loaded in 4-bit). It will not run on CPU or on a free Hugging Face Space. A free Google Colab T4 works.
- Linux or Colab is the easiest route; Unsloth's Windows support is limited.

### 1. Install

The simplest route is **Google Colab with a T4 GPU**: open `FT-For-RQA.ipynb` and run the install cell at the top.

On your own GPU machine, use a separate environment from the app's. The app's environment pins CPU-only PyTorch (so that it fits on free hosting), and this model needs the CUDA build:

```bash
python -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install unsloth python-dotenv       # installs CUDA PyTorch with it
```

### 2. Ask it a question

Run from the repository root:

```python
from models.finetuned_reader import generate_answer

context = (
    "Retrieval-Augmented Generation combines parametric and non-parametric memory "
    "for language generation using a pre-trained seq2seq model and a dense vector "
    "index of Wikipedia."
)
print(generate_answer("What does RAG stand for?", context))
# -> Retrieval-Augmented Generation
```

The first call downloads the base model (about 2.3 GB) and the adapter, then keeps them in memory. To load a different adapter, set `FINETUNED_MODEL_REPO` in `.env`.

### 3. Prompt format

The model was trained on plain text in exactly this shape, not on Phi-3's chat template — use the same shape if you call it without the loader:

```
Context: <text>

Question: <question>

Answer:
```

It tends to keep writing further `Question:` / `Answer:` pairs, so stop generation at the string `Question:` (the loader does this for you).

### How it was trained

| | |
|---|---|
| **Base model** | `unsloth/Phi-3-mini-4k-instruct`, 4-bit |
| **Method** | QLoRA via Unsloth — rank 16, alpha 32, dropout 0.05, on `q_proj` and `v_proj` (6.3M trainable parameters, 0.16%) |
| **Data** | `allenai/qasper` train split: 1,926 question–answer pairs from 888 NLP papers |
| **Context per example** | paper title + abstract, cut to 1,000 characters |
| **Training** | 1 epoch (241 steps), batch 2 × 4 accumulation, learning rate 2e-5, on a Tesla T4 (about 15 minutes) |

To retrain or change any of this, open `FT-For-RQA.ipynb` in Colab with a GPU runtime and run it top to bottom. Where it uploads to the Hub, put your own Hugging Face token in place of `YOUR_HF_TOKEN` and your own repository name.

### What to expect

- **Short, extractive answers** — a phrase or a sentence, in the style of QASPER's reference answers. No page citations.
- **Short contexts only.** It saw at most ~1,000 characters of context in training; give it one passage, not a whole paper.
- **Text only** — it cannot read tables or figures.
- **Measured result:** on 20 held-out QASPER questions, keyword overlap with the reference answer was 13.7% for the fine-tuned model and 14.6% for the base model, so this training run did not improve on the base. More epochs, full-paper context, and more LoRA target modules are the obvious next experiments.

---

## License

MIT License — intended for research and educational purposes.

© 2026 Lingesh R
