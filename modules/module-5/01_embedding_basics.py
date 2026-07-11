"""Domain 5 - Task 5.1: Embedding Basics

CONCEPTS:
  1. client.embeddings.create() — embed text to float vector
  2. text-embedding-3-small — 1536 dimensions default
  3. data[0].embedding — the float list
  4. Normalization — unit vectors for cosine similarity

Mnemonic: EDNU — Embed, Dimensions, Normalize, Use

Run:
  uv run python 01_embedding_basics.py
"""

import math

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Embed a single text string and return the float vector."""
    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def l2_norm(vector: list[float]) -> float:
    """Compute the L2 (Euclidean) norm of a vector."""
    return math.sqrt(sum(x * x for x in vector))


def normalize(vector: list[float]) -> list[float]:
    """Return a unit vector by dividing each element by the L2 norm."""
    norm = l2_norm(vector)
    if norm == 0.0:
        return vector[:]
    return [x / norm for x in vector]


def vector_stats(vector: list[float]) -> dict[str, object]:
    """Return dimension count, first 5 values, and L2 norm."""
    return {
        "dimensions": len(vector),
        "first_5": vector[:5],
        "norm": l2_norm(vector),
    }


# ---------------------------------------------------------------------------
# ANTI-PATTERN 1: using the raw (un-normalised) vector for similarity
# ---------------------------------------------------------------------------


def anti_pattern_raw_similarity(a: list[float], b: list[float]) -> float:
    """
    WRONG: dot product without normalising. Result depends on vector magnitude,
    not just direction — large vectors dominate regardless of semantic content.
    """
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# DEMO 1: embed, inspect, normalise
# ---------------------------------------------------------------------------


def demo_embed_and_inspect(text: str) -> None:
    """Embed text, print stats, then print stats after normalisation."""
    print(f"Text: {repr(text)}")
    vector = embed_text(text)
    stats = vector_stats(vector)
    print(f"Dimensions : {stats['dimensions']}")
    print(f"First 5    : {stats['first_5']}")
    print(f"L2 norm    : {stats['norm']:.6f}")

    unit = normalize(vector)
    unit_norm = l2_norm(unit)
    print(f"Norm after normalisation: {unit_norm:.6f}  (should be ~1.0)")


if __name__ == "__main__":
    sep = "=" * 60

    print(sep)
    print("DEMO 1: Embed and inspect a sentence")
    print(sep)
    sentence = "The quick brown fox jumps over the lazy dog."
    demo_embed_and_inspect(sentence)

    print()
    print(sep)
    print("DEMO 2: Embed multiple texts and compare norms")
    print(sep)
    texts = [
        "Python is a high-level programming language.",
        "Machine learning models learn from data.",
    ]
    for t in texts:
        vec = embed_text(t)
        print(f"Text  : {repr(t)}")
        print(f"Dims  : {len(vec)}  |  Norm: {l2_norm(vec):.6f}")
        unit = normalize(vec)
        print(f"Unit norm: {l2_norm(unit):.6f}")
        print()

    if is_mock():
        print("[Mock mode] Embeddings are synthetic [0.1]*1536 vectors.")
        print(f"Mock norm = sqrt(1536 * 0.01) = {math.sqrt(1536 * 0.01):.6f}")

    print()
    print(sep)
    print("ANTI-PATTERN 1: Raw dot product without normalisation")
    print(sep)
    v1 = embed_text("cat")
    v2 = embed_text("dog")
    raw = anti_pattern_raw_similarity(v1, v2)
    print(f"Raw dot product: {raw:.4f}  (misleading — depends on magnitude)")
    print("Fix: always normalise vectors before computing cosine similarity.")

    print()
    print(sep)
    print("KEY TAKEAWAYS:")
    print(sep)
    print(f"  - client.embeddings.create() returns a float vector via data[0].embedding{NL}"
          f"    that encodes semantic meaning as a point in high-dimensional space.")
    print(f"  - text-embedding-3-small produces 1536-dimensional vectors by default;{NL}"
          f"    dimensions can be reduced via the dimensions= parameter.")
    print(f"  - Normalising to unit length (divide by L2 norm) is required before{NL}"
          f"    cosine similarity so results reflect direction, not magnitude.")
    print("  - The L2 norm of a correctly normalised vector is exactly 1.0.")
