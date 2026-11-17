# Deployment

Two hosting options, plus how to run the fine-tuned model. For what the project is and how it works, see the [README](README.md).

**Whichever host you choose, do not set the API keys on the server.** With no keys present, each visitor enters their own and spends their own quota. If you set them, every visitor spends yours.

---

## Streamlit Community Cloud

The repository is ready to deploy as it is.

1. Push the repository to GitHub.
2. Go to https://share.streamlit.io → **Create app** → **Deploy a public app from GitHub**.
3. Choose the repository and branch, and set **Main file path** to `streamlit_app.py`.
4. Open **Advanced settings** and choose **Python 3.13**. Leave **Secrets** empty.
5. **Deploy.** The first build takes several minutes — it installs PyTorch.

What the platform reads from this repository:

| File | Purpose |
|---|---|
| `streamlit_app.py` | The app |
| `uv.lock` (with `pyproject.toml`) | Python dependencies, pinned, with CPU-only PyTorch |
| `packages.txt` | System packages for PDF parsing: `poppler-utils`, `tesseract-ocr`, `libgl1`, `libglib2.0-0` |
| `.streamlit/config.toml` | Upload limit and theme (no port — the host sets its own) |

### Things to expect

- **Memory is the constraint.** Community Cloud gives an app roughly 2.7 GB of RAM, and PyTorch plus the layout-detection model take a large share. A long or image-heavy PDF can push the app over the limit and restart it. If that happens, add a secret `SMARTPAPER_MAX_PDF_PAGES = "20"`, or use the Docker route below (16 GB on a free Hugging Face Space).
- **The first upload after a restart is slower** — the embedding and layout models download on first use.
- **Storage is temporary.** The paper library is emptied whenever the app restarts or sleeps.
- **One upload at a time.** A second visitor sees "yours is next in line".

---

## Docker — Hugging Face Spaces or any host

Use this when the app needs more memory than Community Cloud offers.

### Hugging Face Spaces

1. Create a Space → **SDK: Docker** → blank template.
2. **Add the Space configuration to the top of `README.md`.** Hugging Face reads it from there, and it is deliberately not in this repository's README (it renders as a stray table on GitHub):

   ```yaml
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
   ```

3. Push:

   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```

4. Wait for the build — several minutes, since it installs PyTorch and pre-downloads the models.

**Do not add the API keys as Space secrets.**

Notes:

- The free CPU tier (2 vCPU, 16 GB) is enough. A 20-page paper takes roughly 5–7 minutes to process the first time.
- A free Space's disk is wiped when it restarts or sleeps, so the paper library starts empty again. To keep it, add persistent storage and set the variable `SMARTPAPER_DATA_DIR=/data`.

### Any other Docker host

```bash
docker build -t smartpaper .
docker run -p 7860:7860 -v smartpaper-data:/home/user/app/data smartpaper
```

The image runs as a non-root user, pre-downloads the embedding and layout models at build time, and has a health check on `/_stcore/health`.

---

## Configuration

All optional; defaults are in [`config.py`](config.py) and listed in [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `SMARTPAPER_DATA_DIR` | `./data` | Where papers, vectors and temp files are stored |
| `SMARTPAPER_OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Model that answers questions |
| `SMARTPAPER_GROQ_MODEL` | `openai/gpt-oss-120b` | Model that summarises uploads (and the answer fallback) |
| `SMARTPAPER_GEMINI_MODEL` | `gemini-2.5-flash` | Vision model |
| `SMARTPAPER_VISION_SUMMARY_BUDGET` | `0` | Figures per paper the vision model may describe at upload. Leave at 0 on free keys |
| `SMARTPAPER_MAX_UPLOAD_MB` | `20` | Largest PDF accepted |
| `SMARTPAPER_MAX_PDF_PAGES` | `50` | Longest PDF accepted |
| `SMARTPAPER_MAX_STORED_PAPERS` | `50` | Library size before the least recently used paper is removed |
| `SMARTPAPER_TOP_K` | `6` | How many passages / tables / figures a question retrieves |
| `SMARTPAPER_AGENT_TIMEOUT` | `45` | Seconds before a question falls back from OpenRouter to Groq |

**Providers retire models.** When one does, the key check in the app names the missing model — change the matching variable, no code change needed.

---

## Running the fine-tuned Phi-3 reader

The app does not use this model (see [the README](README.md#the-fine-tuning-experiment) for why). This is for running it yourself.

- Adapter: [`Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER`](https://huggingface.co/Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER) — LoRA weights only, 25 MB
- Training notebook: [`FT-For-RQA.ipynb`](FT-For-RQA.ipynb)
- Loader: [`models/finetuned_reader.py`](models/finetuned_reader.py)

### What you need

An **NVIDIA GPU with CUDA** and about 4 GB of free VRAM — the base model loads in 4-bit. It will not run on CPU or on free hosting. A free Google Colab T4 works.

The simplest route is **Colab with a T4 GPU**: open `FT-For-RQA.ipynb` and run the install cell at the top.

On your own GPU machine, use a **separate environment** from the app's. The app pins CPU-only PyTorch so it fits on free hosting; this model needs the CUDA build:

```bash
python -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install unsloth python-dotenv       # brings CUDA PyTorch with it
```

### Ask it a question

From the repository root:

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

The first call downloads the base model (~2.3 GB) and the adapter, then keeps them in memory. Set `FINETUNED_MODEL_REPO` in `.env` to load a different adapter.

### Prompt format

It was trained on plain text in exactly this shape, not on Phi-3's chat template:

```
Context: <text>

Question: <question>

Answer:
```

It tends to keep writing further `Question:` / `Answer:` pairs, so stop generation at `Question:` — the loader does this for you.

### How it was trained

| | |
|---|---|
| **Base model** | `unsloth/Phi-3-mini-4k-instruct`, 4-bit |
| **Method** | QLoRA via Unsloth — rank 16, alpha 32, dropout 0.05, on `q_proj` and `v_proj` (6.3M trainable parameters, 0.16%) |
| **Data** | `allenai/qasper` train split: 1,926 question–answer pairs from 888 NLP papers |
| **Context per example** | Paper title + abstract, cut to 1,000 characters |
| **Training** | 1 epoch (241 steps), batch 2 × 4 accumulation, learning rate 2e-5, Tesla T4, ~15 minutes |

To retrain, open the notebook in Colab with a GPU runtime and run it top to bottom. Where it uploads to the Hub, put your own Hugging Face token in place of `YOUR_HF_TOKEN` and your own repository name.
