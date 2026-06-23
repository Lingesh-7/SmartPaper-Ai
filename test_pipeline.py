"""
End-to-end smoke test for SmartPaper AI's modular pipeline.

Run with: python test_pipeline.py path/to/paper.pdf

Parses a PDF, ingests it, registers the retriever, then runs 5
questions through the agent and prints the answers - mirroring the
Day 5 guide's test script.
"""

import sys

from parsers.pdf_parser import parse_pdf
from embeddings.embedder import ingest_paper
from agents.graph import run_agent, register_retriever


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    paper_id = "test-paper"

    print(f"[1/3] Parsing {pdf_path} ...")
    parsed_data = parse_pdf(pdf_path)
    print(f"  text_chunks={len(parsed_data['text_chunks'])} "
          f"images={len(parsed_data['images'])} "
          f"tables={len(parsed_data['tables'])}")

    print(f"\n[2/3] Ingesting paper (summarizing + embedding) ...")
    retriever = ingest_paper(parsed_data, paper_id)
    register_retriever(paper_id, retriever)

    print(f"\n[3/3] Running test questions through the agent ...")
    test_questions = [
        "What is the main idea behind this paper?",
        "What does the architecture diagram show?",
        "What are the key results reported in the tables?",
        "What datasets were used in this work?",
        "What are the limitations mentioned in the paper?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = run_agent(q, paper_id)
        print(f"\nAnswer ({result['retrieval_type']}, "
              f"retries={result['retry_count']}):\n{result['final_answer']}")

    print("\n\nAll questions completed without uncaught errors.")


if __name__ == "__main__":
    main()
