"""
End-to-end smoke test for SmartPaper AI's modular pipeline.

Run with: python test_pipeline.py path/to/paper.pdf [--force]

Goes through the same pipeline.py functions the Streamlit app uses:
ingests a PDF, then runs 5 questions plus one follow-up through the
agent and prints the answers - mirroring the Day 5 guide's test
script. This is a manual script that calls the real model providers
(it needs OPENROUTER_API_KEY, GOOGLE_API_KEY and GROQ_API_KEY in .env).
The offline tests live in tests/.

Each question costs 2-3 of the 50 requests per day that a free
OpenRouter key allows, so one run of this script uses about a third
of a day's allowance.

A PDF that was ingested before is loaded from the library instead of
being processed again; pass --force to re-ingest it.
"""

import logging
import sys

import config
import pipeline
from models.llm_clients import build_clients


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv[1:]
    if not args:
        print("Usage: python test_pipeline.py <path_to_pdf> [--force]")
        sys.exit(1)

    # Model output contains characters (narrow spaces, non-breaking
    # hyphens) that a Windows console's default code page cannot
    # encode; without this, print() raises midway through the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one line per request is noise

    pdf_path = args[0]
    keys = config.validate_config()
    clients = build_clients(keys)

    print(f"[1/3] Ingesting {pdf_path} ...")
    with open(pdf_path, "rb") as handle:
        data = handle.read()

    last_message = {"text": None}

    def progress(message, fraction):
        if message != last_message["text"]:
            last_message["text"] = message
            print(f"  {fraction:5.0%}  {message}")

    info = pipeline.ingest_pdf(data, pdf_path, clients, progress=progress, force=force)
    print(f"  paper_id={info.paper_id} reused={info.reused} pages={info.pages} "
          f"text_chunks={info.text_chunks} images={info.images} tables={info.tables}")
    for warning in info.warnings:
        print(f"  WARNING: {warning}")

    print("\n[2/3] Generating the paper summary ...")
    print(pipeline.summarize_paper(info.paper_id, clients))

    print("\n[3/3] Running test questions through the agent ...")
    test_questions = [
        "What is the main idea behind this paper?",
        "What does the architecture diagram show?",
        "What are the key results reported in the tables?",
        "What datasets were used in this work?",
        "What are the limitations mentioned in the paper?",
        # A follow-up that only makes sense with the previous turn:
        "Can you explain that in simpler terms?",
    ]

    history = []
    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = pipeline.answer_question(q, info.paper_id, clients, chat_history=history)
        if result["standalone_question"] != q:
            print(f"(understood as: {result['standalone_question']})")
        pages = sorted({s["page_number"] for s in result["sources"] if s["page_number"] is not None})
        kinds = [s["type"] for s in result["sources"]]
        print(f"\nAnswer ({result['retrieval_type']}, retries={result['retry_count']}, "
              f"sources={kinds}, pages={pages}):\n{result['final_answer']}")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["final_answer"]})

    print("\n\nAll questions completed without uncaught errors.")


if __name__ == "__main__":
    main()
