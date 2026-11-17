# SmartPaper AI - Streamlit app, built for a Hugging Face Space (Docker SDK).
# Also runs anywhere else:  docker build -t smartpaper . && docker run -p 7860:7860 smartpaper

FROM python:3.13-slim

# poppler-utils / tesseract-ocr: needed by unstructured's hi_res PDF parsing.
# libgl1 / libglib2.0-0: needed by OpenCV, which the layout model uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as user 1000, not root. Everything
# the app writes (data/, model caches) must live under that user's home.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app

# Dependencies first, so editing app code does not reinstall them.
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Download the embedding model and the layout-detection model at build
# time. Otherwise the first visitor's first upload waits for them, and
# a Space with no outbound access at that moment fails outright.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "from unstructured_inference.models.base import get_model; get_model('yolox')"

COPY --chown=user . .

# No API keys are set here on purpose: every visitor enters their own.
# Production settings (local runs keep Streamlit's defaults):
#  - errors are shown by the app itself, never as raw tracebacks
#  - CORS/XSRF checks are off because a Space is served inside an iframe
#    on another origin, where Streamlit's XSRF cookie is not sent and
#    file uploads fail with HTTP 403. The app has no login and no
#    cookies of its own, so there is no session for a forged request
#    to ride on.
ENV STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=none \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
