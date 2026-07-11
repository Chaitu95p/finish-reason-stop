"""Exercise 2 - Task 2.1: Ingest and Embed a Document Corpus

GOAL: Build an in-memory vector store from a list of text chunks.
For each chunk: compute its embedding, store (chunk_text, embedding) pairs.

SKILLS PRACTICED:
  - client.embeddings.create()
  - Cosine similarity from scratch
  - In-memory semantic index

Run:
  uv run python 01_ingest_and_embed.py
"""

NL = chr(10)
EMBED_MODEL = "text-embedding-3-small"

import math
from dataclasses import dataclass

from shared.mock import get_client, is_mock

CORPUS = [
    "Python is a high-level interpreted programming language.",
    "OpenAI provides GPT-4o and other large language models via API.",
    "Embeddings represent text as high-dimensional vectors.",
    "Vector similarity search enables semantic matching without exact keyword matching.",
    "Fine-tuning adapts a base model on domain-specific data.",
    "Batch API provides 50% cost savings over synchronous calls.",
    "The Responses API is OpenAI's primary stateful conversation API.",
    "RAG combines retrieval and generation for grounded answers.",
]


@dataclass
class Document:
    text: str
    embedding: list[float]


def embed_texts(client: object, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; return list of embedding vectors."""
    response = client.embeddings.create(  # type: ignore[attr-defined]
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_index(client: object, texts: list[str]) -> list[Document]:
    """Build a document index by embedding all texts."""
    embeddings = embed_texts(client, texts)
    return [Document(text=t, embedding=e) for t, e in zip(texts, embeddings)]


def search(index: list[Document], query_embedding: list[float], top_k: int = 3) -> list[Document]:
    """Return top_k most similar documents."""
    scored = [(cosine_similarity(doc.embedding, query_embedding), doc) for doc in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 2 - Task 2.1: Ingest & Embed [{mode}]{NL}{sep}")

    client = get_client()

    print(f"{NL}  Building index from {len(CORPUS)} documents...")
    index = build_index(client, CORPUS)
    print(f"  Index built: {len(index)} documents, {len(index[0].embedding)}-dim vectors")

    queries = ["What is an embedding?", "How can I save money on API calls?"]
    for query in queries:
        print(f"{NL}  Query: {query!r}")
        [query_emb] = embed_texts(client, [query])
        results = search(index, query_emb, top_k=2)
        for i, doc in enumerate(results):
            print(f"    [{i+1}] {doc.text!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Embed once at ingest time; search many times at query time")
    print("  2. Cosine similarity works without numpy — just dot product and norms")
    print("  3. Batch all texts in one embeddings.create() call for efficiency")
