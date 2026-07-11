"""Domain 5 - Task 5.3: Semantic Search

CONCEPTS:
  1. Corpus embedding — embed all documents once
  2. Query embedding — embed search query
  3. Similarity ranking — sort by cosine similarity
  4. Top-K retrieval — return N most similar

Mnemonic: CQSR — Corpus, Query, Similarity, Rank

Run:
  uv run python 03_semantic_search.py
"""

import math
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Pure-Python math (same primitives, no numpy)
# ---------------------------------------------------------------------------


def dot_product(a: list[float], b: list[float]) -> float:
    """Sum of element-wise products."""
    return sum(x * y for x, y in zip(a, b))


def l2_norm(vector: list[float]) -> float:
    """Square root of the sum of squares."""
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
class Document:
    """A corpus document with its pre-computed embedding."""
    doc_id: int
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class SearchResult:
    """A ranked search result."""
    rank: int
    doc_id: int
    score: float
    text: str


# ---------------------------------------------------------------------------
# Corpus embedding (DEMO 1)
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    "Python is a versatile programming language used for web development.",
    "Machine learning algorithms learn patterns from training data.",
    "The Eiffel Tower is a famous landmark in Paris, France.",
    "Neural networks are inspired by the structure of the human brain.",
    "French cuisine is known for its rich sauces and elegant presentation.",
]


def embed_corpus(texts: list[str]) -> list[Document]:
    """
    Embed all corpus texts in a single batch API call.
    Returns a list of Document objects with populated embeddings.
    """
    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    docs: list[Document] = []
    for i, item in enumerate(response.data):
        docs.append(Document(doc_id=i, text=texts[i], embedding=item.embedding))
    return docs


# ---------------------------------------------------------------------------
# Query embedding (DEMO 2)
# ---------------------------------------------------------------------------


def embed_query(query: str) -> list[float]:
    """Embed a single query string and return its float vector."""
    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Similarity ranking (DEMO 3)
# ---------------------------------------------------------------------------


def rank_documents(
    query_vec: list[float],
    docs: list[Document],
) -> list[tuple[float, Document]]:
    """Return (score, doc) pairs sorted by cosine similarity descending."""
    scored = [(cosine_similarity(query_vec, d.embedding), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Top-K retrieval (DEMO 4)
# ---------------------------------------------------------------------------


def top_k(
    query: str,
    docs: list[Document],
    k: int = 3,
) -> list[SearchResult]:
    """Embed query, rank all docs, return top-k as SearchResult objects."""
    query_vec = embed_query(query)
    ranked = rank_documents(query_vec, docs)
    results: list[SearchResult] = []
    for rank, (score, doc) in enumerate(ranked[:k], start=1):
        results.append(SearchResult(rank=rank, doc_id=doc.doc_id, score=score, text=doc.text))
    return results


# ---------------------------------------------------------------------------
# Hand-crafted vectors for meaningful mock-mode demo
# ---------------------------------------------------------------------------
#
# The mock API returns [0.1]*1536 for every input, so all cosine similarities
# are 1.0 in mock mode — the ranking call pattern still works but the scores
# are uninformative.  We build a small set of sparse vectors that encode
# topic clusters so the ranking produces visible differences.
#
#   dims 0-1  : programming / Python
#   dims 2-3  : machine learning / AI
#   dims 4-5  : geography / landmarks
#   dims 6-7  : food / cuisine
#


def _make_topic_vector(dims: int, hot: list[int], val: float = 1.0) -> list[float]:
    v = [0.0] * dims
    for i in hot:
        v[i] = val
    return v


def build_mock_corpus() -> list[Document]:
    """Return Document objects with hand-crafted topic vectors (8 dims)."""
    DIM = 8
    raw: list[tuple[str, list[int]]] = [
        (CORPUS[0], [0, 1]),        # Python / web dev
        (CORPUS[1], [2, 3]),        # ML / data
        (CORPUS[2], [4, 5]),        # Paris / landmarks
        (CORPUS[3], [2, 3, 6]),     # neural networks (AI + slight overlap)
        (CORPUS[4], [4, 5, 7]),     # French cuisine (France + food)
    ]
    docs: list[Document] = []
    for i, (text, hot) in enumerate(raw):
        docs.append(Document(doc_id=i, text=text, embedding=_make_topic_vector(DIM, hot)))
    return docs


def top_k_crafted(
    query_vec: list[float],
    docs: list[Document],
    k: int = 3,
) -> list[SearchResult]:
    """Same as top_k but accepts a pre-built query vector."""
    ranked = rank_documents(query_vec, docs)
    results: list[SearchResult] = []
    for rank, (score, doc) in enumerate(ranked[:k], start=1):
        results.append(SearchResult(rank=rank, doc_id=doc.doc_id, score=score, text=doc.text))
    return results


# ---------------------------------------------------------------------------
# ANTI-PATTERN 1: re-embedding the corpus on every query
# ---------------------------------------------------------------------------


def anti_pattern_reembed_every_query(query: str, texts: list[str]) -> list[str]:
    """
    WRONG: embeds the corpus inside the search function.
    With 1 M documents this is catastrophically slow and costly.
    Fix: embed corpus once at ingest time; embed only the query at search time.
    """
    docs = embed_corpus(texts)   # should be done once, outside search
    query_vec = embed_query(query)
    ranked = rank_documents(query_vec, docs)
    return [d.text for _, d in ranked[:3]]


if __name__ == "__main__":
    sep = "=" * 60

    # ----------------------------------------------------------------
    # DEMO 1: Batch-embed corpus
    # ----------------------------------------------------------------
    print(sep)
    print("DEMO 1: Embed corpus in a single batch call")
    print(sep)
    docs = embed_corpus(CORPUS)
    for d in docs:
        print(f"  doc {d.doc_id}: dims={len(d.embedding)}  text={repr(d.text[:50])}")

    # ----------------------------------------------------------------
    # DEMO 2: Query embedding
    # ----------------------------------------------------------------
    print()
    print(sep)
    print("DEMO 2: Embed a query")
    print(sep)
    q = "deep learning and artificial intelligence"
    qv = embed_query(q)
    print(f"  Query : {repr(q)}")
    print(f"  Dims  : {len(qv)}")

    # ----------------------------------------------------------------
    # DEMO 3: Ranking with real API embeddings
    # ----------------------------------------------------------------
    print()
    print(sep)
    print("DEMO 3: Full ranking (real API or identical mock scores)")
    print(sep)
    results = top_k(q, docs, k=3)
    for r in results:
        print(f"  Rank {r.rank}  score={r.score:.4f}  doc_id={r.doc_id}  {repr(r.text[:60])}")
    if is_mock():
        print(f"{NL}  [Mock mode] All API scores are 1.0 (vectors are identical).")
        print("  See DEMO 4 for meaningful ranking via hand-crafted vectors.")

    # ----------------------------------------------------------------
    # DEMO 4: Hand-crafted vectors for meaningful mock-mode ranking
    # ----------------------------------------------------------------
    print()
    print(sep)
    print("DEMO 4: Hand-crafted topic vectors — meaningful ranking")
    print(sep)
    mock_docs = build_mock_corpus()
    # Query vector aligned with ML/AI topic (dims 2 and 3)
    ai_query_vec = _make_topic_vector(8, [2, 3])
    ai_results = top_k_crafted(ai_query_vec, mock_docs, k=3)
    print("  Query topic: 'machine learning / AI'  (dims 2-3 hot)")
    print("  Top 3:")
    for r in ai_results:
        print(f"    Rank {r.rank}  score={r.score:.4f}  {repr(r.text[:60])}")

    print()
    geo_query_vec = _make_topic_vector(8, [4, 5])
    geo_results = top_k_crafted(geo_query_vec, mock_docs, k=3)
    print("  Query topic: 'geography / landmarks'  (dims 4-5 hot)")
    print("  Top 3:")
    for r in geo_results:
        print(f"    Rank {r.rank}  score={r.score:.4f}  {repr(r.text[:60])}")

    # ----------------------------------------------------------------
    # ANTI-PATTERN 1
    # ----------------------------------------------------------------
    print()
    print(sep)
    print("ANTI-PATTERN 1: Re-embedding corpus on every query (expensive)")
    print(sep)
    q2 = "programming language"
    _ = anti_pattern_reembed_every_query(q2, CORPUS)
    print(f"  Called embed_corpus() inside search for: {repr(q2)}")
    print("  Fix: embed corpus once at startup and cache the Document objects.")

    print()
    print(sep)
    print("KEY TAKEAWAYS:")
    print(sep)
    print(f"  - Embed the corpus once in a batch call and store the vectors;{NL}"
          f"    only the query needs embedding at search time.")
    print(f"  - Cosine similarity ranking is a simple sort — no specialised{NL}"
          f"    data structure required for small corpora.")
    print(f"  - Top-K retrieval returns the N documents whose vectors are most{NL}"
          f"    directionally aligned with the query vector.")
    print(f"  - Mock embeddings are identical for all texts; use hand-crafted{NL}"
          f"    topic vectors to verify ranking logic without a real API key.")
