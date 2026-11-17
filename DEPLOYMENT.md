# Deployment

How to host the app, configure it, and run the fine-tuned model. For what the project is and how it works, see the [README](README.md).

**Do not set the API keys on the server.** With no keys present, each visitor enters their own and spends their own quota. If you set them, every visitor spends yours.

---

## Streamlit Community Cloud

The repository is ready to deploy as it is.

1. Push the repository to GitHub.
2. Go to https://share.streamlit.io, then **Create app**, then **Deploy a public app from GitHub**.
3. Choose the repository and branch, and set **Main file path** to `streamlit_app.py`.
4. Open **Advanced settings** and choose **Python 3.13**. Leave **Secrets** empty.
5. **Deploy.** The first build takes several minutes, because it installs PyTorch.

What the platform reads from this repository:

| File | Purpose |
|---|---|
| `streamlit_app.py` | The app |
| `uv.lock` (with `pyproject.toml`) | Python dependencies, pinned, with CPU-only PyTorch |
| `packages.txt` | System packages for PDF parsing: `poppler-utils`, `tesseract-ocr`, `libgl1`, `libglib2.0-0` |
| `.streamlit/config.toml` | Upload limit and theme. No port, because the host sets its own |

### Things to expect

- **Memory is the constraint.** Community Cloud gives an app roughly 2.7 GB of RAM, and PyTorch plus the layout-detection model take a large share of it. A long or image-heavy PDF can push the app over the limit and restart it. If that happens, add a secret `SMARTPAPER_MAX_PDF_PAGES = "20"`, or lower it further.
- **The first upload after a restart is slower,** because the embedding and layout models download on first use.
- **Storage is temporary.** The paper library is emptied whenever the app restarts or sleeps.
- **One upload at a time.** A second visitor sees "yours is next in line".

---

## Configuration

All of these are optional. The defaults are in [`config.py`](config.py).

| Variable | Default | Purpose |
|---|---|---|
| `SMARTPAPER_DATA_DIR` | `./data` | Where papers, vectors and temp files are stored |
| `SMARTPAPER_OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Model that answers questions |
| `SMARTPAPER_GROQ_MODEL` | `openai/gpt-oss-120b` | Model that summarises uploads, and the answer fallback |
| `SMARTPAPER_GEMINI_MODEL` | `gemini-2.5-flash` | Vision model |
| `SMARTPAPER_VISION_SUMMARY_BUDGET` | `0` | Figures per paper the vision model may describe at upload. Leave at 0 on free keys |
| `SMARTPAPER_MAX_UPLOAD_MB` | `20` | Largest PDF accepted |
| `SMARTPAPER_MAX_PDF_PAGES` | `50` | Longest PDF accepted |
| `SMARTPAPER_MAX_STORED_PAPERS` | `50` | Library size before the least recently used paper is removed |
| `SMARTPAPER_TOP_K` | `6` | How many passages, tables or figures a question retrieves |
| `SMARTPAPER_AGENT_TIMEOUT` | `45` | Seconds before a question falls back from OpenRouter to Groq |

For a local run, put them in a `.env` file at the repository root, alongside your three API keys:

```env
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
```

On a host, set them as environment variables or secrets. `.env` is gitignored.

**Providers retire models.** When one does, the key check in the app names the missing model. Change the matching variable; no code change is needed.

---

## Running the fine-tuned Phi-3 reader

The app does not use this model. The [README](README.md#the-fine-tuning-experiment) explains why. This section is for running it yourself.

- Adapter: [`Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER`](https://huggingface.co/Lingesh-7/phi-3-mini-4k-instruct-FT-ON-QASPER), LoRA weights only, 25 MB
- Training notebook: [`FT-For-RQA.ipynb`](FT-For-RQA.ipynb)
- Loader: [`models/finetuned_reader.py`](models/finetuned_reader.py)

### What you need

An **NVIDIA GPU with CUDA** and about 4 GB of free VRAM, since the base model loads in 4-bit. It will not run on CPU or on free hosting. A free Google Colab T4 works.

The simplest route is **Colab with a T4 GPU**: open `FT-For-RQA.ipynb` and run the install cell at the top.

On your own GPU machine, use a **separate environment** from the app's. The app pins CPU-only PyTorch so that it fits on free hosting, and this model needs the CUDA build:

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

The first call downloads the base model (about 2.3 GB) and the adapter, then keeps them in memory. Set `FINETUNED_MODEL_REPO` in `.env` to load a different adapter.

### Prompt format

It was trained on plain text in exactly this shape, not on Phi-3's chat template:

```
Context: <text>

Question: <question>

Answer:
```

It tends to keep writing further `Question:` and `Answer:` pairs, so stop generation at `Question:`. The loader does this for you.

### How it was trained

| | |
|---|---|
| **Base model** | `unsloth/Phi-3-mini-4k-instruct`, 4-bit |
| **Method** | QLoRA via Unsloth: rank 16, alpha 32, dropout 0.05, on `q_proj` and `v_proj` (6.3M trainable parameters, 0.16%) |
| **Data** | `allenai/qasper` train split: 1,926 question and answer pairs from 888 NLP papers |
| **Context per example** | Paper title + abstract, cut to 1,000 characters |
| **Training** | 1 epoch (241 steps), batch 2 x 4 accumulation, learning rate 2e-5, Tesla T4, about 15 minutes |

To retrain, open the notebook in Colab with a GPU runtime and run it top to bottom. Where it uploads to the Hub, put your own Hugging Face token in place of `YOUR_HF_TOKEN` and your own repository name.
