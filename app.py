"""
Flask app for SmartPaper AI - upload, ask, and summary routes.
"""

import os
import uuid

from flask import Flask, request, jsonify, render_template

import config
from parsers.pdf_parser import parse_pdf
from embeddings.embedder import ingest_paper, delete_paper
from agents.graph import run_agent, register_retriever
from langchain_groq import ChatGroq

app = Flask(__name__)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_summary_llm = ChatGroq(model=config.AGENT_LLM_MODEL, temperature=0)

# tracks parsed_data per paper_id so /summary can reuse the first-page
# text chunks without re-parsing the PDF
_paper_parsed_data: dict = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    paper_id = str(uuid.uuid4())
    save_path = os.path.join(config.UPLOAD_DIR, f"{paper_id}.pdf")
    file.save(save_path)

    try:
        parsed_data = parse_pdf(save_path)
        retriever = ingest_paper(parsed_data, paper_id)
        print()
        register_retriever(paper_id, retriever)
        _paper_parsed_data[paper_id] = parsed_data
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

    return jsonify({
        "paper_id": paper_id,
        "text_chunks": len(parsed_data["text_chunks"]),
        "images": len(parsed_data["images"]),
        "tables": len(parsed_data["tables"]),
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    paper_id = data.get("paper_id")

    if not paper_id:
        return jsonify({"error": "No paper uploaded yet"}), 400
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = run_agent(question, paper_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Agent error: {str(e)}"}), 500

    return jsonify({
        "final_answer": result["final_answer"],
        "source_type": result["retrieval_type"],
        "retry_count": result["retry_count"],
    })


@app.route("/summary", methods=["GET"])
def summary():
    paper_id = request.args.get("paper_id")

    if not paper_id or paper_id not in _paper_parsed_data:
        return jsonify({"error": "No paper uploaded yet"}), 400

    parsed_data = _paper_parsed_data[paper_id]
    first_chunks = parsed_data["text_chunks"][:5]
    abstract_text = "\n".join(c["text"] for c in first_chunks)

    prompt = (
        "Based on the following text from the start of a research paper, provide:\n"
        "1. A 3-sentence overview\n"
        "2. 5 key contributions as bullet points\n"
        "3. 3 suggested follow-up questions a reader might ask\n\n"
        f"Text:\n{abstract_text}"
    )

    try:
        response = _summary_llm.invoke(prompt)
        summary_text = response.content
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

    return jsonify({"summary": summary_text})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "PDF exceeds the 20MB upload limit"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )