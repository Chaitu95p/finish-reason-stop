"""Domain 5 - Task 5.4: Chunking Strategies

CONCEPTS:
  1. Fixed-size chunking — split by character count with overlap
  2. Sentence chunking — split on punctuation boundaries
  3. Paragraph chunking — split on double newlines
  4. Chunk size tradeoffs — large=context, small=precision

Mnemonic: FSPC — Fixed, Sentence, Paragraph, Context_tradeoff

Run:
  uv run python 04_chunking_strategies.py
"""

from __future__ import annotations

import re

from shared.mock import is_mock

NL = chr(10)
MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Sample text used across all demos
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Python is a high-level, general-purpose programming language. "
    "It was created by Guido van Rossum and first released in 1991. "
    "Python's design philosophy emphasizes code readability. "
    "The language uses significant indentation to delimit blocks."
    f"{NL}{NL}"
    "Machine learning is a subset of artificial intelligence. "
    "It focuses on building systems that learn from data. "
    "Supervised learning, unsupervised learning, and reinforcement learning "
    "are the three main paradigms. Each paradigm suits different problems."
    f"{NL}{NL}"
    "Large language models are trained on vast text corpora. "
    "They predict the next token given the preceding context. "
    "Fine-tuning adapts a pre-trained model to a specific domain. "
    "Prompt engineering shapes model behaviour without changing weights."
)


# ---------------------------------------------------------------------------
# Chunking functions
# ---------------------------------------------------------------------------


def fixed_chunk(text: str, size: int = 200, overlap: int = 20) -> list[str]:
    """
    Split text into fixed-size character chunks with optional overlap.

    Args:
        text:    Input text to split.
        size:    Maximum characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of text chunks, each at most `size` characters long.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be < size ({size})")

    chunks: list[str] = []
    step = size - overlap
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


def sentence_chunk(text: str) -> list[str]:
    """
    Split text at sentence boundaries (.  !  ?  followed by whitespace or EOS).

    Returns a list of non-empty sentence strings.
    """
    # Split on . ! ? followed by whitespace or end-of-string
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def paragraph_chunk(text: str) -> list[str]:
    """
    Split text on double newlines (blank lines between paragraphs).

    Returns a list of non-empty paragraph strings.
    """
    raw = re.split(r"\n{2,}", text)
    return [p.strip() for p in raw if p.strip()]


# ---------------------------------------------------------------------------
# Analysis helper
# ---------------------------------------------------------------------------


def chunk_stats(chunks: list[str], strategy: str) -> None:
    """Print summary stats for a list of chunks."""
    if not chunks:
        print(f"  {strategy}: 0 chunks")
        return
    sizes = [len(c) for c in chunks]
    print(f"  {strategy}:")
    print(f"    chunk count : {len(chunks)}")
    print(f"    min chars   : {min(sizes)}")
    print(f"    max chars   : {max(sizes)}")
    print(f"    avg chars   : {sum(sizes) / len(sizes):.1f}")


# ---------------------------------------------------------------------------
# ANTI-PATTERN 1: splitting on whitespace with no overlap — context loss
# ---------------------------------------------------------------------------


def anti_pattern_naive_split(text: str, size: int = 200) -> list[str]:
    """
    WRONG: split on whitespace only; no overlap means a sentence spanning
    a chunk boundary loses context on both sides.
    """
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > size and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


if __name__ == "__main__":
    sep = "=" * 60
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"

    print(f"Domain 5 - Task 5.4: Chunking Strategies {mock_label}")
    print(sep)

    # DEMO 1: Fixed-size chunking
    print("DEMO 1: Fixed-size chunking (size=150, overlap=20)")
    print(sep)
    fixed_chunks = fixed_chunk(SAMPLE_TEXT, size=150, overlap=20)
    chunk_stats(fixed_chunks, "fixed_chunk")
    print()
    for i, c in enumerate(fixed_chunks[:3], start=1):
        print(f"  Chunk {i} [{len(c)} chars]: {repr(c[:80])}...")
    if len(fixed_chunks) > 3:
        print(f"  ... ({len(fixed_chunks) - 3} more chunks)")

    print()
    print(sep)

    # DEMO 2: Sentence chunking
    print("DEMO 2: Sentence chunking (split on .  !  ?)")
    print(sep)
    sent_chunks = sentence_chunk(SAMPLE_TEXT)
    chunk_stats(sent_chunks, "sentence_chunk")
    print()
    for i, c in enumerate(sent_chunks[:4], start=1):
        print(f"  Sentence {i}: {repr(c)}")
    if len(sent_chunks) > 4:
        print(f"  ... ({len(sent_chunks) - 4} more sentences)")

    print()
    print(sep)

    # DEMO 3: Paragraph chunking
    print("DEMO 3: Paragraph chunking (split on blank lines)")
    print(sep)
    para_chunks = paragraph_chunk(SAMPLE_TEXT)
    chunk_stats(para_chunks, "paragraph_chunk")
    print()
    for i, c in enumerate(para_chunks, start=1):
        print(f"  Paragraph {i} [{len(c)} chars]:{NL}    {repr(c[:100])}...")

    print()
    print(sep)

    # DEMO 4: Tradeoff comparison
    print("DEMO 4: Size tradeoffs — large vs small fixed chunks")
    print(sep)
    large_chunks = fixed_chunk(SAMPLE_TEXT, size=400, overlap=0)
    small_chunks = fixed_chunk(SAMPLE_TEXT, size=80, overlap=10)
    chunk_stats(large_chunks, "large (size=400, overlap=0 )")
    chunk_stats(small_chunks, "small (size=80,  overlap=10)")
    print()
    print("  Large chunks: more context per retrieval hit, lower precision.")
    print("  Small chunks: precise retrieval, risk losing surrounding context.")
    print("  Overlap mitigates boundary loss — sentences split across chunks")
    print("  are fully represented in at least one chunk.")

    print()
    print(sep)

    # ANTI-PATTERN 1
    print("ANTI-PATTERN 1: Naive word-split with no overlap")
    print(sep)
    naive_chunks = anti_pattern_naive_split(SAMPLE_TEXT, size=200)
    chunk_stats(naive_chunks, "naive_split")
    print()
    print("  Problem: a sentence crossing a boundary is split mid-way.")
    print("  The embedding of each half loses the other half's context.")
    print("  Fix: use character-level overlap so boundary content appears")
    print("  in full in at least one adjacent chunk.")

    print()
    print(sep)
    print("KEY TAKEAWAYS:")
    print(f"  - Fixed-size chunking is simple and predictable; add overlap{NL}"
          f"    (10-20% of chunk size) to avoid losing context at boundaries.")
    print(f"  - Sentence chunking preserves semantic units but produces{NL}"
          f"    variable-length chunks that may be too small for dense context.")
    print(f"  - Paragraph chunking is natural for well-structured documents;{NL}"
          f"    chunk size varies widely depending on paragraph length.")
    print(f"  - Smaller chunks improve retrieval precision; larger chunks{NL}"
          f"    provide richer context for generation — tune to your use case.")
