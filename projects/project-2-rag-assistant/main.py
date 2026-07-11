"""Project 2: Local Document Q&A with RAG + Citations

An in-memory RAG assistant that:
- Ingests documents from a local text corpus
- Embeds each chunk with text-embedding-3-small
- Retrieves top-K chunks per query
- Generates grounded answers with citations
- Works fully in mock mode

Run:
  uv run python main.py
  uv run python main.py --demo
"""

NL = chr(10)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any

from shared.mock import get_client, is_mock

SAMPLE_CORPUS = [
    ("intro-1",   "Python is a high-level, general-purpose programming language created by Guido van Rossum."),
    ("intro-2",   "Python 3.12 introduced type parameter syntax and improvements to f-strings."),
    ("openai-1",  "OpenAI provides GPT-4o and other LLMs via a REST API and official Python SDK."),
    ("openai-2",  "The openai Python SDK v1.0+ uses a client object: client = openai.OpenAI()."),
    ("embed-1",   "Embeddings represent text as dense high-dimensional vectors (1536 dims for text-embedding-3-small)."),
    ("embed-2",   "Cosine similarity measures the angle between two embedding vectors, ranging from -1 to 1."),
    ("batch-1",   "The Batch API offers a 50% cost discount for offline, non-latency-sensitive workloads."),
    ("ft-1",      "Fine-tuning adapts a base model on domain data; requires at least 10 examples."),
    ("rag-1",     "RAG (Retrieval Augmented Generation) combines retrieval of relevant chunks with LLM generation."),
    ("rag-2",     "Chunking strategies: fixed-size, sentence-level, or paragraph-level splits."),
]


@dataclass
class Document:
    doc_id: str
    text: str
    embedding: list[float] = field(default_factory=list)


def embed_texts(client: object, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)  # type: ignore[attr-defined]
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_index(client: object, corpus: list[tuple[str, str]]) -> list[Document]:
    texts = [text for _, text in corpus]
    embeddings = embed_texts(client, texts)
    return [Document(doc_id=did, text=text, embedding=emb)
            for (did, text), emb in zip(corpus, embeddings)]


def retrieve(index: list[Document], query_emb: list[float], top_k: int = TOP_K) -> list[Document]:
    return sorted(index, key=lambda d: cosine_similarity(d.embedding, query_emb), reverse=True)[:top_k]


def answer_with_citations(client: object, index: list[Document], question: str) -> dict[str, Any]:
    [q_emb] = embed_texts(client, [question])
    chunks = retrieve(index, q_emb)
    context = NL.join(f"[{doc.doc_id}] {doc.text}" for doc in chunks)
    system = (
        "Answer using ONLY the provided context. Include the document IDs you used in citations. "
        f"Return JSON: {{\"answer\": \"...\", \"citations\": [\"id1\", ...]}}{NL}{NL}Context:{NL}{context}"
    )
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw, "citations": []}


def demo_mode(client: object, index: list[Document]) -> None:
    questions = [
        "What is RAG?",
        "How much can I save with the Batch API?",
        "What Python version introduced type parameter syntax?",
    ]
    for q in questions:
        result = answer_with_citations(client, index, q)
        print(f"Q: {q}")
        print(f"A: {result.get('answer', '')!r}")
        print(f"   Citations: {result.get('citations', [])}{NL}")


def interactive_mode(client: object, index: list[Document]) -> None:
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"RAG Assistant [{mode}] | {len(index)} documents indexed | Type 'quit' to exit{NL}")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"{NL}Goodbye!")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break
        result = answer_with_citations(client, index, question)
        print(f"A: {result.get('answer', '')}")
        citations = result.get("citations", [])
        if citations:
            print(f"   [sources: {', '.join(citations)}]{NL}")
        else:
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Document Q&A")
    parser.add_argument("--demo", action="store_true", help="Run scripted demo")
    args = parser.parse_args()

    client = get_client()
    print("Building index...")
    index = build_index(client, SAMPLE_CORPUS)
    print(f"Indexed {len(index)} documents ({len(index[0].embedding)}-dim vectors){NL}")

    if args.demo:
        demo_mode(client, index)
    else:
        interactive_mode(client, index)
