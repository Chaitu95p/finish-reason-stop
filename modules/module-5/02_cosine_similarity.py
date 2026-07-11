"""Domain 5 - Task 5.2: Cosine Similarity

CONCEPTS:
  1. Dot product — sum of element-wise products
  2. L2 norm — sqrt of sum of squares
  3. Cosine similarity — dot product of unit vectors
  4. Range — -1 (opposite) to 1 (identical)

Mnemonic: DNCS — Dot_product, Norm, Cosine, Similarity

Run:
  uv run python 02_cosine_similarity.py
"""

import math

from shared.mock import get_client

NL = chr(10)
MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Pure-Python math primitives
# ---------------------------------------------------------------------------


def dot_product(a: list[float], b: list[float]) -> float:
    """Return the dot product of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(vector: list[float]) -> float:
    """Return the L2 (Euclidean) norm of a vector."""
    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Return cosine similarity in [-1, 1].

    Equal to dot(a, b) / (||a|| * ||b||).
    Returns 0.0 if either vector is the zero vector.
    """
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product(a, b) / (norm_a * norm_b)


def normalize(vector: list[float]) -> list[float]:
    """Return the unit vector in the same direction."""
    norm = l2_norm(vector)
    if norm == 0.0:
        return vector[:]
    return [x / norm for x in vector]


# ---------------------------------------------------------------------------
# Helpers for building illustrative hand-crafted vectors
# ---------------------------------------------------------------------------


def make_vector(size: int, hot_indices: list[int], hot_value: float = 1.0) -> list[float]:
    """
    Return a zero-padded vector of length `size` with `hot_value` at each
    index in `hot_indices`.  Used to craft vectors with known similarity.
    """
    v: list[float] = [0.0] * size
    for idx in hot_indices:
        v[idx] = hot_value
    return v


# ---------------------------------------------------------------------------
# DEMO 1: identical texts → similarity ≈ 1.0 via API
# ---------------------------------------------------------------------------


def demo_identical_texts() -> None:
    """Embed the same text twice; cosine similarity must be 1.0."""
    client = get_client()
    text = "Embeddings encode meaning as vectors."
    r1 = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    r2 = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    v1 = r1.data[0].embedding
    v2 = r2.data[0].embedding
    sim = cosine_similarity(v1, v2)
    print(f"Identical text similarity : {sim:.6f}  (expected ~1.0)")


# ---------------------------------------------------------------------------
# DEMO 2: hand-crafted vectors that model related / unrelated semantics
# ---------------------------------------------------------------------------


def demo_crafted_vectors() -> None:
    """
    Use hand-crafted sparse vectors so similarity differences are visible
    even in mock mode (where the API returns identical vectors for all inputs).

    Interpretation:
      - identical_a / identical_b  : same vector  → sim = 1.0
      - related_a  / related_b     : large overlap → sim close to 1.0
      - unrelated_a / unrelated_b  : no overlap   → sim = 0.0
      - opposite_a / opposite_b    : mirror image → sim = -1.0
    """
    DIM = 8

    identical_a = make_vector(DIM, [0, 1, 2])
    identical_b = make_vector(DIM, [0, 1, 2])

    related_a = make_vector(DIM, [0, 1, 2, 3])
    related_b = make_vector(DIM, [0, 1, 2, 4])   # share 3 of 4 active dims

    unrelated_a = make_vector(DIM, [0, 1])
    unrelated_b = make_vector(DIM, [4, 5])        # zero overlap

    opposite_a = [1.0, 0.0, 0.0]
    opposite_b = [-1.0, 0.0, 0.0]

    pairs: list[tuple[str, list[float], list[float]]] = [
        ("identical",  identical_a,  identical_b),
        ("related",    related_a,    related_b),
        ("unrelated",  unrelated_a,  unrelated_b),
        ("opposite",   opposite_a,   opposite_b),
    ]
    for label, va, vb in pairs:
        sim = cosine_similarity(va, vb)
        print(f"  {label:<12}  dot={dot_product(va, vb):.2f}  "
              f"||a||={l2_norm(va):.4f}  ||b||={l2_norm(vb):.4f}  "
              f"cosine={sim:+.4f}")


# ---------------------------------------------------------------------------
# ANTI-PATTERN 1: comparing raw dot products across vectors of different norms
# ---------------------------------------------------------------------------


def anti_pattern_raw_dot() -> None:
    """
    WRONG: using dot product directly as a similarity measure.
    A longer vector will score higher even if it points in a different direction.
    """
    short = [1.0, 0.0]
    long_similar = [100.0, 0.0]    # same direction as 'short', but 100x longer
    long_different = [0.0, 100.0]  # perpendicular to 'short'

    print(f"  dot(short, long_similar)   = {dot_product(short, long_similar):.1f}")
    print(f"  dot(short, long_different) = {dot_product(short, long_different):.1f}")
    print(f"  cosine(short, long_similar)   = {cosine_similarity(short, long_similar):.4f}")
    print(f"  cosine(short, long_different) = {cosine_similarity(short, long_different):.4f}")
    print("  cosine correctly scores similar direction=1.0, perpendicular=0.0")


if __name__ == "__main__":
    sep = "=" * 60

    print(sep)
    print("DEMO 1: Identical text via API (cosine should be ~1.0)")
    print(sep)
    demo_identical_texts()

    print()
    print(sep)
    print("DEMO 2: Hand-crafted vectors — four similarity regimes")
    print(sep)
    print("  label        dot    ||a||   ||b||   cosine")
    print("  " + "-" * 54)
    demo_crafted_vectors()

    print()
    print(sep)
    print("DEMO 3: Dot-product formula components")
    print(sep)
    a = [3.0, 4.0]
    b = [1.0, 0.0]
    print(f"  a = {a}  b = {b}")
    print(f"  dot(a, b)         = {dot_product(a, b):.4f}")
    print(f"  l2_norm(a)        = {l2_norm(a):.4f}  (expected 5.0)")
    print(f"  l2_norm(b)        = {l2_norm(b):.4f}  (expected 1.0)")
    print(f"  cosine_similarity = {cosine_similarity(a, b):.4f}  (expected 0.6)")

    print()
    print(sep)
    print("ANTI-PATTERN 1: Raw dot product ignores vector magnitude")
    print(sep)
    anti_pattern_raw_dot()

    print()
    print(sep)
    print("KEY TAKEAWAYS:")
    print(sep)
    print(f"  - Cosine similarity = dot(a,b) / (||a||*||b||) and lies in [-1, 1];{NL}"
          f"    1.0 means identical direction, 0.0 orthogonal, -1.0 opposite.")
    print(f"  - Always normalise before comparing: unit vectors make cosine equal{NL}"
          f"    to the plain dot product, removing magnitude bias.")
    print(f"  - Mock embeddings return identical vectors for all inputs; craft{NL}"
          f"    sparse hand-built vectors to demonstrate meaningful similarity differences.")
    print(f"  - The dot_product(), l2_norm(), and cosine_similarity() primitives{NL}"
          f"    can be implemented in pure Python with no external libraries.")
