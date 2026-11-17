"""
Streamlit front end for SmartPaper AI.

Run with:  streamlit run streamlit_app.py

This file is presentation only. Everything it does goes through
pipeline.py (ingest_pdf / answer_question / summarize_paper), which is
also what test_pipeline.py exercises - so there is one implementation
of the product and this is a thin skin over it.

It replaces the Flask app (app.py + templates/ + static/). Things the
Flask version could not do, and why they are built the way they are:

- Bring your own keys. Each visitor enters their own OpenRouter,
  Google and Groq keys. They live in st.session_state - the memory of
  one browser tab's session on the server - and are never written to
  disk or to a log. Keys found in the server's environment (a local
  .env) are used as a convenience for local runs only.
- Per-visitor state. The current paper and the chat belong to the
  session, so two visitors never see each other's conversation.
- Ingestion runs as a background job (pipeline.IngestJob) that the
  page polls. Streamlit stops a running script whenever the visitor
  clicks anything or refreshes; an ingestion running inside the
  script would be killed minutes in.
- An unanswered question is derived from state, not from an event:
  if the last chat message is the visitor's, an answer is owed. So an
  interrupted run resumes instead of leaving a question hanging, and
  every outcome - answer or error - appends exactly one assistant
  message, which is what stops it from retrying forever.
"""

import base64
import logging

import streamlit as st

st.set_page_config(
    page_title="SmartPaper AI",
    page_icon=":material/menu_book:",
    layout="centered",
)

# The pipeline pulls in LangChain, Chroma and friends: several seconds
# on a cold start. Import under a spinner so the first visitor sees a
# page rather than a blank tab.
with st.spinner("Starting SmartPaper AI…"):
    import config
    import pipeline
    from models.llm_clients import build_clients, check_keys


@st.cache_resource
def _configure_logging() -> bool:
    """Once per server process: log to stdout, and clear stale temp files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    pipeline.cleanup_temp_files()
    pipeline.cleanup_stale_papers()
    return True


_configure_logging()

SUGGESTIONS = [
    "What is the main idea of this paper?",
    "What does the main figure show?",
    "What are the key results in the tables?",
    "What datasets were used?",
]

KEY_FIELDS = [
    # (ApiKeys field, label, what it is used for, where to get one)
    ("openrouter", "OpenRouter API key", "answers your questions", "https://openrouter.ai/keys"),
    ("google", "Google AI Studio API key", "reads figures and tables", "https://aistudio.google.com/apikey"),
    ("groq", "Groq API key", "summarises the paper when you upload it", "https://console.groq.com/keys"),
]

SOURCE_LABELS = {"text": "Passage", "table": "Table", "image": "Figure"}
QUESTION_TYPES = {"text_query": "Text question", "table_query": "Table question", "figure_query": "Figure question"}

MAX_MESSAGES = 60              # older chat messages are dropped
MESSAGES_KEEPING_SOURCES = 12  # only recent answers keep their (image-heavy) sources
MAX_SOURCE_CHARS = 1500


# --- State ------------------------------------------------------------------
def init_state() -> None:
    state = st.session_state
    state.setdefault("api_keys", None)        # config.ApiKeys ("keys" would shadow session_state.keys())
    state.setdefault("clients", None)         # models.llm_clients.LLMClients
    state.setdefault("keys_from_server", False)
    state.setdefault("key_notices", [])       # non-blocking warnings from the last key check
    state.setdefault("paper", None)           # pipeline.PaperInfo
    state.setdefault("job", None)             # pipeline.IngestJob while a paper is processing
    state.setdefault("upload_error", None)
    state.setdefault("messages", [])          # chat history, see answer_pending_question()
    state.setdefault("uploader_nonce", 0)     # bump to reset the file uploader

    if state.api_keys is None:
        # Local convenience: keys in the server's environment / .env.
        env_keys = config.keys_from_env()
        if env_keys.is_complete():
            state.api_keys = env_keys
            state.clients = build_clients(env_keys)
            state.keys_from_server = True


def md_escape(text: str) -> str:
    """Show text literally in st.markdown (file names, extracted passages)."""
    for char in "\\`*_{}[]<>#|$~":
        text = text.replace(char, "\\" + char)
    return text


# --- API keys ---------------------------------------------------------------
def render_key_form(form_key: str) -> None:
    """The three key inputs. On success, stores keys + clients and reruns."""
    with st.form(form_key, border=False):
        values = {}
        for field, label, purpose, url in KEY_FIELDS:
            values[field] = st.text_input(
                label,
                type="password",
                key=f"{form_key}_{field}",
                help=f"Used for: {purpose}. Get one at {url}",
                autocomplete="off",
            )
        submitted = st.form_submit_button("Save keys", type="primary", icon=":material/key:")

    if not submitted:
        return

    keys = config.ApiKeys(**values)
    if not keys.is_complete():
        names = [label for field, label, _, _ in KEY_FIELDS if not getattr(keys, field)]
        st.error("Still needed: " + ", ".join(names) + ".", icon=":material/error:")
        return

    with st.spinner("Checking your keys…"):
        checks = check_keys(keys)

    rejected = [check.message for check in checks.values() if check.ok is False]
    if rejected:
        for message in rejected:
            st.error(message, icon=":material/error:")
        return

    try:
        clients = build_clients(keys)
    except Exception as exc:  # noqa: BLE001
        st.error(pipeline.describe_error(exc, keys), icon=":material/error:")
        return

    st.session_state.api_keys = keys
    st.session_state.clients = clients
    st.session_state.keys_from_server = False
    # A key that could not be checked (no network, provider busy) is
    # accepted - it will fail later with its own message if it is bad.
    st.session_state.key_notices = [check.message for check in checks.values() if check.ok is None]
    st.rerun()


def render_key_help() -> None:
    st.markdown(
        "This app runs on **your own API keys**, so you stay in control of your usage. "
        "All three providers have a free tier:"
    )
    for _, label, purpose, url in KEY_FIELDS:
        st.markdown(f"- [{label}]({url}): {purpose}")
    st.caption(
        "Your keys are kept only in this browser tab's session on the server. "
        "They are not saved to disk or written to logs, and are gone when you close the tab."
    )


@st.dialog("API keys")
def key_dialog() -> None:
    render_key_help()
    render_key_form("keys_dialog")


# --- Upload and ingestion ---------------------------------------------------
def render_upload() -> None:
    st.subheader("Upload a paper")
    uploaded = st.file_uploader(
        "Research paper (PDF)",
        type=["pdf"],
        max_upload_size=config.MAX_UPLOAD_MB,
        key=f"uploader_{st.session_state.uploader_nonce}",
    )
    st.caption(
        f"Up to {config.MAX_UPLOAD_MB} MB and {config.MAX_PDF_PAGES} pages. "
        "A new 10-20 page paper takes about 3-5 minutes to read; "
        "a paper that was processed before opens straight away."
    )

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error, icon=":material/error:")

    if uploaded is None:
        return

    force = st.checkbox(
        "Process from scratch",
        key=f"force_{st.session_state.uploader_nonce}",
        help=(
            "A paper that was processed before is normally reused as it is. "
            "Tick this to read it again - for example if it was stored with warnings."
        ),
    )
    if st.button("Process this paper", type="primary", icon=":material/play_arrow:"):
        st.session_state.upload_error = None
        st.session_state.job = pipeline.IngestJob(
            uploaded.getvalue(),
            uploaded.name,
            st.session_state.clients,
            st.session_state.api_keys,
            force=force,
        )
        st.rerun()


def finish_job() -> None:
    """Move a finished job's outcome into session state."""
    job = st.session_state.job
    st.session_state.job = None
    if job.error:
        st.session_state.upload_error = job.error
        return
    st.session_state.paper = job.info
    st.session_state.messages = []
    st.session_state.upload_error = None
    st.session_state.uploader_nonce += 1
    if job.info.reused:
        st.toast("This paper was processed before - loaded from the library.", icon=":material/bolt:")


@st.fragment(run_every=1.0)
def render_job_progress() -> None:
    """Polls the background ingestion once a second."""
    job = st.session_state.job
    if job is None:
        return
    if job.done:
        finish_job()
        st.rerun()

    st.subheader("Reading your paper")
    st.progress(job.fraction, text=job.message)
    minutes, seconds = divmod(int(job.elapsed), 60)
    st.caption(
        f"{md_escape(job.filename)} · {minutes}:{seconds:02d} elapsed. "
        "You can leave this tab open; processing continues on the server."
    )


# --- Chat -------------------------------------------------------------------
def source_pages(source: dict) -> str:
    """ "page 4" or "pages 4-5" for one source; "" if the page is unknown."""
    page = source.get("page_number")
    if page is None:
        return ""
    page_end = source.get("page_end")
    if page_end is not None and page_end != page:
        return f"pages {page}-{page_end}"
    return f"page {page}"


def render_sources(sources: list) -> None:
    if not sources:
        return
    pages = set()
    for source in sources:
        if source.get("page_number") is not None:
            pages.update(range(source["page_number"], (source.get("page_end") or source["page_number"]) + 1))
    pages = sorted(pages)
    label = f"Sources ({len(sources)})"
    if pages:
        label += " · page" + ("s " if len(pages) > 1 else " ") + ", ".join(str(p) for p in pages)

    with st.expander(label, icon=":material/library_books:"):
        for source in sources:
            # "Figure 5" / "Table 2" when the paper numbers it, else the generic word.
            kind = source.get("label") or SOURCE_LABELS.get(source["type"], "Passage")
            if source.get("pinned"):
                kind = "Title and abstract"     # given to the model with every question
            where = source_pages(source)
            page_badge = f" :gray-badge[{where}]" if where else ""
            st.markdown(f":blue-badge[{kind}]{page_badge}")

            if source["type"] == "image":
                st.image(base64.b64decode(source["content"]))
                if source.get("caption"):
                    st.caption(md_escape(source["caption"]))
            elif source["type"] == "table" and source.get("image"):
                st.image(base64.b64decode(source["image"]))
                if source.get("caption"):
                    st.caption(md_escape(source["caption"]))
            else:
                text = source["content"]
                if len(text) > MAX_SOURCE_CHARS:
                    text = text[:MAX_SOURCE_CHARS].rstrip() + " …"
                st.markdown(md_escape(text))


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"], icon=":material/error:")
            return
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return

        details = [QUESTION_TYPES.get(message.get("retrieval_type"), "")]
        if message.get("understood_as"):
            details.append(f"understood as: “{md_escape(message['understood_as'])}”")
        if message.get("retry_count"):
            details.append(f"retried {message['retry_count']}×")
        details = [d for d in details if d]
        if details:
            st.caption(" · ".join(details))
        if message.get("vision_fallback"):
            st.caption(
                ":orange[:material/warning:] The vision model could not be reached "
                "(a free Google key allows about 20 requests a day), so this answer is based on "
                "captions and extracted text rather than on the image itself."
            )
        render_sources(message.get("sources") or [])


def trim_history() -> None:
    """Bound the session's memory: drop old messages, and old answers' sources."""
    messages = st.session_state.messages
    if len(messages) > MAX_MESSAGES:
        del messages[: len(messages) - MAX_MESSAGES]
    answers = [m for m in messages if m["role"] == "assistant" and m.get("sources")]
    for message in answers[:-MESSAGES_KEEPING_SOURCES]:
        message["sources"] = []


def answer_pending_question() -> None:
    """
    Answer the visitor's last message. Always appends exactly one
    assistant message - the answer, or the error - so the "an answer
    is owed" condition is false afterwards and nothing loops.
    """
    state = st.session_state
    question = state.messages[-1]["content"]
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in state.messages[:-1]
        if not m.get("error")
    ]

    with st.chat_message("assistant"):
        status = st.status("Reading the paper…", expanded=False)

        def on_step(node: str, agent_state: dict) -> None:
            if node == "contextualizer":
                standalone = agent_state.get("standalone_question", question)
                if standalone != question:
                    status.write(f"Understood as: “{md_escape(standalone)}”")
            elif node == "planner":
                kind = QUESTION_TYPES.get(agent_state.get("retrieval_type"), "Text question")
                status.update(label=f"{kind} - searching the paper…")
                status.write(f"Classified as a {kind.lower()}")
            elif node == "retriever":
                found = agent_state.get("context", [])
                kinds = ", ".join(sorted({SOURCE_LABELS.get(r["type"], "passage").lower() for r in found}))
                status.update(label="Writing the answer…")
                status.write(f"Found {len(found)} sources ({kinds})" if found else "Found no sources")
            elif node == "critic" and not agent_state.get("final_answer"):
                status.update(label="First answer was too thin - trying again…")
                status.write(f"Answer rejected, retry {agent_state.get('retry_count', 1)} of 2")

        try:
            result = pipeline.answer_question(
                question, state.paper.paper_id, state.clients,
                chat_history=history, on_step=on_step,
            )
        except Exception as exc:  # noqa: BLE001 - shown to the visitor as a chat message
            logging.getLogger("smartpaper.app").warning("Question failed: %s", type(exc).__name__)
            status.update(label="Could not answer", state="error")
            state.messages.append({
                "role": "assistant",
                "content": pipeline.describe_error(exc, state.api_keys),
                "error": True,
            })
            if pipeline.get_paper_info(state.paper.paper_id) is None:
                # The paper was evicted from the library (or the server
                # restarted without persistent storage).
                state.paper = None
                state.upload_error = "That paper is no longer on the server. Please upload it again."
            return

        status.update(label="Done", state="complete")
        understood_as = result["standalone_question"]
        state.messages.append({
            "role": "assistant",
            "content": result["final_answer"],
            "sources": result["sources"],
            "retrieval_type": result["retrieval_type"],
            "retry_count": result["retry_count"],
            "understood_as": understood_as if understood_as != question else "",
            "vision_fallback": result.get("vision_fallback", False),
        })
        trim_history()


def render_chat() -> None:
    state = st.session_state

    for message in state.messages:
        render_message(message)

    if state.messages and state.messages[-1]["role"] == "user":
        answer_pending_question()
        st.rerun()

    if not state.messages:
        st.caption("Ask anything about the paper, or start with one of these:")
        picked = st.pills(
            "Suggested questions", SUGGESTIONS, label_visibility="collapsed", key="suggestion"
        )
        if picked:
            state.messages.append({"role": "user", "content": picked})
            st.rerun()

    prompt = st.chat_input("Ask about this paper", max_chars=config.MAX_QUESTION_CHARS)
    if prompt and prompt.strip():
        state.messages.append({"role": "user", "content": prompt.strip()})
        st.rerun()


# --- Sidebar ----------------------------------------------------------------
@st.dialog("Paper overview", width="large")
def overview_dialog() -> None:
    state = st.session_state
    try:
        with st.spinner("Writing the overview…"):
            overview = pipeline.summarize_paper(state.paper.paper_id, state.clients)
    except Exception as exc:  # noqa: BLE001
        st.error(pipeline.describe_error(exc, state.api_keys), icon=":material/error:")
        return
    st.markdown(overview)


def render_sidebar() -> None:
    state = st.session_state
    with st.sidebar:
        st.header("SmartPaper AI")
        st.caption("Ask questions about a research paper. Answers cite the page, table or figure they come from.")

        paper = state.paper
        if paper is not None:
            with st.container(border=True):
                st.markdown(f"**{md_escape(paper.filename)}**")
                st.caption(
                    f"{paper.pages} pages · {paper.text_chunks} passages · "
                    f"{paper.tables} tables · {paper.images} figures"
                )
                for warning in paper.warnings:
                    st.caption(f":orange[:material/warning:] {warning}")
                if st.button("Paper overview", icon=":material/summarize:", width="stretch"):
                    overview_dialog()
                if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch",
                             disabled=not state.messages):
                    state.messages = []
                    st.rerun()
                if st.button("Use another paper", icon=":material/upload_file:", width="stretch"):
                    state.paper = None
                    state.messages = []
                    state.upload_error = None
                    st.rerun()

        if state.api_keys is not None:
            source = "set on the server" if state.keys_from_server else "saved for this session"
            st.caption(f":green[:material/check_circle:] API keys {source}")
            if st.button("Change keys", icon=":material/key:", width="stretch"):
                key_dialog()
            for notice in state.key_notices:
                st.caption(f":orange[:material/warning:] {notice}")
            st.caption(
                "Free-tier limits: an OpenRouter key allows 50 requests a day (a question uses 2-3; "
                "after that your Groq key answers). A Google key allows about 20 a day, "
                "used only for questions about a figure or a table."
            )

        st.caption("© 2026 Lingesh R · AI & Machine Learning Engineer")


# --- Page -------------------------------------------------------------------
def main() -> None:
    init_state()
    state = st.session_state
    render_sidebar()

    if state.api_keys is None:
        st.title("SmartPaper AI")
        st.markdown(
            "Upload a research paper and ask questions about it. "
            "Answers are grounded in the paper's text, tables and figures, and cite the page they come from."
        )
        with st.container(border=True):
            st.subheader("First, add your API keys")
            render_key_help()
            render_key_form("keys_main")
        return

    if state.job is not None:
        st.title("SmartPaper AI")
        render_job_progress()
        return

    if state.paper is None:
        st.title("SmartPaper AI")
        with st.container(border=True):
            render_upload()
        return

    render_chat()


main()
