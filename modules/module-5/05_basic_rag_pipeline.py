"""Domain 5 - Task 5.5: Basic RAG Pipeline

CONCEPTS:
  1. Ingest — chunk documents, embed, store in memory
  2. Retrieve — embed query, find similar chunks
  3. Generate — provide context + query to model
  4. Grounding — response is anchored to retrieved text

Mnemonic: IRGR — Ingest, Retrieve, Generate, Retrieval_grounded

Run:
  uv run python 05_basic_rag_pipeline.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Pure-Python math primitives
# ---------------------------------------------------------------------------


def dot_product(a: list[float], b: list[float]) -> float:
    """Sum of element-wise products."""
    return sum(x * y for x, y in zip(a, b))


def l2_norm(vector: list[float]) -> float:
    """Square root of sum of squares."""
    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for zero vectors."""
    na, nb = l2_norm(a), l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot_product(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A document chunk with its source metadata and embedding."""
    chunk_id: int
    source: str
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its similarity score."""
    rank: int
    chunk_id: int
    source: str
    score: float
    text: str


# ---------------------------------------------------------------------------
# Chunking helper (reused from task 5.4 — inline for self-containment)
# ---------------------------------------------------------------------------


def _sentence_chunk(text: str) -> list[str]:
    """Split on sentence boundaries and return non-empty strings."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# DocumentStore — ingest, embed, retrieve
# ---------------------------------------------------------------------------


@dataclass
class DocumentStore:
    """
    In-memory store for document chunks with embedding-based retrieval.

    Attributes:
        chunks: All ingested chunks with their embeddings.
    """
    chunks: list[Chunk] = field(default_factory=list)
    _next_id: int = field(default=0, init=False, repr=False)

    def ingest(self, text: str, source: str = "doc") -> int:
        """
        Chunk text into sentences, embed all chunks in one batch call,
        and append them to the store.  Returns the number of chunks added.
        """
        client = get_client()
        sentences = _sentence_chunk(text)
        if not sentences:
            return 0

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=sentences)

        for i, item in enumerate(response.data):
            self.chunks.append(
                Chunk(
                    chunk_id=self._next_id,
                    source=source,
                    text=sentences[i],
                    embedding=item.embedding,
                )
            )
            self._next_id += 1

        return len(sentences)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        """
        Embed query, rank all stored chunks by cosine similarity, return top-k.
        """
        if not self.chunks:
            return []

        client = get_client()
        q_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        query_vec = q_response.data[0].embedding

        scored: list[tuple[float, Chunk]] = [
            (cosine_similarity(query_vec, chunk.embedding), chunk)
            for chunk in self.chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(scored[:k], start=1):
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    score=score,
                    text=chunk.text,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Generator — chat completion grounded in retrieved context
# ---------------------------------------------------------------------------


def generate_answer(query: str, context_chunks: list[RetrievedChunk]) -> str:
    """
    Build a grounded prompt from retrieved chunks and call chat completions.

    Args:
        query:          The user's question.
        context_chunks: Top-k chunks from the document store.

    Returns:
        The model's answer as a string.
    """
    client = get_client()

    context_parts: list[str] = []
    for rc in context_chunks:
        context_parts.append(f"[Source: {rc.source}, chunk {rc.chunk_id}]{NL}{rc.text}")

    context_str = (NL + "-" * 40 + NL).join(context_parts)

    system_msg = (
        "You are a helpful assistant. Answer the user's question using only "
        "the provided context. If the context does not contain the answer, "
        "say so explicitly."
    )
    user_msg = (
        f"Context:{NL}{context_str}{NL}{NL}"
        f"Question: {query}"
    )

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    # suppress unused-variable warning for second client call above
    _ = client
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

DOCS: list[tuple[str, str]] = [
    (
        "python_intro",
        (
            "Python is a high-level, interpreted programming language. "
            "It was created by Guido van Rossum and released in 1991. "
            "Python emphasizes code readability and simplicity. "
            "It supports multiple programming paradigms including procedural, "
            "object-oriented, and functional styles."
        ),
    ),
    (
        "ml_basics",
        (
            "Machine learning is a branch of artificial intelligence. "
            "Supervised learning trains models on labelled examples. "
            "Unsupervised learning finds patterns in unlabelled data. "
            "Reinforcement learning optimises agents through reward signals."
        ),
    ),
    (
        "rag_overview",
        (
            "Retrieval-Augmented Generation combines retrieval with generation. "
            "A retriever fetches relevant documents given a query. "
            "A generator reads the retrieved context and produces an answer. "
            "RAG reduces hallucinations by grounding responses in source text."
        ),
    ),
]

# ---------------------------------------------------------------------------
# ANTI-PATTERN 1: generating without retrieved context (pure parametric memory)
# ---------------------------------------------------------------------------


def anti_pattern_no_context(query: str) -> str:
    """
    WRONG: send the question directly to the model with no retrieved context.
    The model answers from parametric memory — may hallucinate or be outdated.
    Fix: always retrieve relevant chunks first and include them in the prompt.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    sep = "=" * 60
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"

    print(f"Domain 5 - Task 5.5: Basic RAG Pipeline {mock_label}")
    print(sep)

    # DEMO 1: Ingest
    print("DEMO 1: Ingest documents — chunk, embed, store")
    print(sep)
    store = DocumentStore()
    for source, text in DOCS:
        added = store.ingest(text, source=source)
        print(f"  Ingested '{source}': {added} chunks added")
    print(f"  Total chunks in store: {len(store.chunks)}")

    print()
    print(sep)

    # DEMO 2: Retrieve
    print("DEMO 2: Retrieve top-3 chunks for a query")
    print(sep)
    query = "How does RAG reduce hallucinations?"
    results = store.retrieve(query, k=3)
    print(f"  Query: {repr(query)}")
    print()
    for r in results:
        print(f"  Rank {r.rank}  score={r.score:.4f}  source={r.source}")
        print(f"    {repr(r.text[:80])}")
    if is_mock():
        print(f"{NL}  [Mock mode] All cosine scores are 1.0 (identical mock vectors).")

    print()
    print(sep)

    # DEMO 3: Generate grounded answer
    print("DEMO 3: Generate answer grounded in retrieved context")
    print(sep)
    answer = generate_answer(query, results)
    print(f"  Query  : {repr(query)}")
    print(f"  Answer : {answer}")

    print()
    print(sep)

    # DEMO 4: Full pipeline end-to-end
    print("DEMO 4: Full pipeline — ingest, retrieve, generate")
    print(sep)
    q2 = "What programming paradigms does Python support?"
    top_chunks = store.retrieve(q2, k=3)
    answer2 = generate_answer(q2, top_chunks)
    print(f"  Query  : {repr(q2)}")
    print(f"  Top chunks retrieved: {len(top_chunks)}")
    print(f"  Answer : {answer2}")

    print()
    print(sep)

    # ANTI-PATTERN 1
    print("ANTI-PATTERN 1: Generating without retrieved context")
    print(sep)
    ap_answer = anti_pattern_no_context("What is RAG?")
    print(f"  Answer (no context): {ap_answer}")
    print()
    print("  Problem: the model answers from parametric memory only.")
    print("  It cannot cite sources, may be outdated, and can hallucinate.")
    print("  Fix: always retrieve relevant chunks and inject them as context.")

    print()
    print(sep)
    print("KEY TAKEAWAYS:")
    print(f"  - Ingest once: chunk documents, embed each chunk, store vectors.{NL}"
          f"    Only the query needs embedding at query time.")
    print(f"  - Retrieve by embedding the query and ranking stored chunks by{NL}"
          f"    cosine similarity; return the top-k most relevant chunks.")
    print(f"  - Generate with context: prepend retrieved chunks to the prompt{NL}"
          f"    so the model's answer is grounded in source text.")
    print(f"  - RAG reduces hallucinations because the model cites retrieved{NL}"
          f"    passages rather than relying on potentially stale parametric memory.")
