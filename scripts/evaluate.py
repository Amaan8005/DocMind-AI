"""Basic RAG evaluation: baseline (vector-only + default prompt) vs
improved (hybrid vector+BM25 + grounded prompt) on the test_docs corpus.

Run from the repository root:  pipenv run python scripts/evaluate.py
"""

import csv
import os
import sys
import time

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

TEST_DOCS = os.path.join(ROOT, "test_docs")
FILES = ["sample.txt", "sample.csv", "sample.pptx"]
OLLAMA_ENDPOINT = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:0.5b"
EMBED_MODEL = "nomic-embed-text"
OUT_CSV = os.path.join(ROOT, "scripts", "eval_results.csv")

st.session_state["ollama_endpoint"] = OLLAMA_ENDPOINT
st.session_state["selected_model"] = CHAT_MODEL
st.session_state["system_prompt"] = None
st.session_state["top_k"] = 3
st.session_state["similarity_cutoff"] = 0.3

from utils import llama_index as lix
from utils.llama_index import TEXT_QA_TEMPLATE
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine

QUESTIONS = [
    ("What is DocMind AI?", "Offline/local RAG assistant", "txt"),
    ("Which chat models does DocMind use for generation?", "llama3, qwen2.5", "txt"),
    ("Which embedding model does DocMind use?", "nomic-embed-text", "txt"),
    ("What file types are supported?", "TXT, CSV, DOCX, PPTX, PDF, Markdown", "txt"),
    ("What is the maximum number of documents per ingestion run?", "300", "txt"),
    ("What license is DocMind AI released under?", "MIT", "txt"),
    ("What was the revenue for SmartChip X9 in Q4?", "$555,000 (Asia Pacific)", "csv"),
    ("How many units of Alphabot 3000 were sold in Q2 North America?", "1350", "csv"),
    ("What was the total revenue of DocMind Pro in North America?", "$705,000 (Q3+Q4)", "csv"),
    ("Which region had the highest revenue for SmartChip X9?", "Asia Pacific", "csv"),
    ("What is the revenue of SmartChip X9 in Q1?", "$450,000", "csv"),
    ("Which product sold the most total units?", "SmartChip X9", "csv"),
    ("What is the title of the presentation?", "DocMind AI Launch Plan", "pptx"),
    ("What are the four launch milestones?", "Q1 local files, Q2 GitHub/website, Q3 hybrid retrieval, Q4 caching", "pptx"),
    ("What is the target answer latency?", "Under 5 seconds", "pptx"),
    ("Why is local RAG good for privacy?", "Documents never leave the device", "pptx"),
    ("Who prepared the presentation?", "The Product Team", "pptx"),
    ("Who is the current President of the United States?", "Refuse / not in documents", "negative"),
    ("What is the capital of France?", "Refuse / not in documents", "negative"),
    ("Give me the recipe for pasta carbonara.", "Refuse / not in documents", "negative"),
]


def _source_files(nodes):
    out = []
    for ns in nodes:
        meta = ns.node.metadata or {}
        name = meta.get("file_name", meta.get("source", "?"))
        out.append(f"{name}:{round(ns.score, 3)}")
    return out


def ask_baseline(engine, question):
    resp = engine.query(question)
    return str(resp.response), _source_files(resp.source_nodes)


def ask_improved(hybrid, llm, question):
    nodes = hybrid.retrieve(question)
    if not nodes:
        return "I could not find this information in the documents.", []
    numbered = []
    for i, ns in enumerate(nodes, 1):
        numbered.append(f"[{i}]:\n{ns.node.get_content()}")
    context = "\n\n".join(numbered)
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=TEXT_QA_TEMPLATE.format(context_str=context, query_str=question),
        )
    ]
    resp = llm.chat(messages)
    return resp.message.content, _source_files(nodes)


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    lix.setup_embedding_model(
        EMBED_MODEL, chunk_size=256, chunk_overlap=32, ollama_endpoint=OLLAMA_ENDPOINT
    )
    files = [os.path.join(TEST_DOCS, f) for f in FILES]
    docs = lix.load_documents(TEST_DOCS, input_files=files)
    lix.create_query_engine(docs)

    hybrid = st.session_state["retriever"]
    vector_only = hybrid.vector_retriever

    llm = Ollama(
        model=CHAT_MODEL,
        base_url=OLLAMA_ENDPOINT,
        request_timeout=300,
        context_window=2048,
        temperature=0.4,
        keep_alive="2m",
        additional_kwargs={"num_predict": 512},
    )
    Settings.llm = llm
    baseline_engine = RetrieverQueryEngine(retriever=vector_only)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "n", "type", "question", "expected",
                "baseline_answer", "baseline_sources", "baseline_time_s",
                "improved_answer", "improved_sources", "improved_time_s",
            ]
        )
        for n, (question, expected, qtype) in enumerate(QUESTIONS, 1):
            row = [n, qtype, question, expected]

            t0 = time.time()
            base_ans, base_src = ask_baseline(baseline_engine, question)
            row += [base_ans, "; ".join(base_src), round(time.time() - t0, 1)]

            t0 = time.time()
            impr_ans, impr_src = ask_improved(hybrid, llm, question)
            row += [impr_ans, "; ".join(impr_src), round(time.time() - t0, 1)]

            writer.writerow(row)
            fh.flush()
            print(f"[{n:02d}/20] {qtype}: {question[:60]}", flush=True)

    print(f"\nDone. Results: {OUT_CSV}")


if __name__ == "__main__":
    main()
