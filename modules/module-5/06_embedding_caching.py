"""Domain 5 - Task 5.6: Embedding Caching

CONCEPTS:
  1. Cache key — hash of text + model name
  2. Cache hit — return stored embedding, skip API call
  3. Cache miss — embed and store
  4. Persistence — save/load cache from JSON file

Mnemonic: KHMP — Key, Hit, Miss, Persist

Run:
  uv run python 06_embedding_caching.py
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Cache key helper
# ---------------------------------------------------------------------------


def make_cache_key(text: str, model: str) -> str:
    """
    Return a deterministic hex digest for (text, model).

    Uses SHA-256 over 'model:text' so keys are collision-resistant and
    safe as JSON dict keys regardless of the original text content.
    """
    raw = f"{model}:{text}".encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingCache:
    """
    In-memory embedding cache with optional JSON persistence.

    Attributes:
        model:      Embedding model name used for all requests.
        cache_path: Optional file path for persistent JSON cache.
        _store:     Internal dict mapping cache key -> embedding vector.
        _hits:      Number of cache hits since creation.
        _misses:    Number of cache misses since creation.
        _api_calls: Number of actual API calls made (equals _misses).
    """
    model: str = EMBEDDING_MODEL
    cache_path: Path | None = None
    _store: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)
    _api_calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_path is not None and self.cache_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """
        Return the embedding for text.  Serves from cache on hit;
        calls the API and stores the result on miss.
        """
        key = make_cache_key(text, self.model)
        if key in self._store:
            self._hits += 1
            return self._store[key]

        # Cache miss — call the API
        self._misses += 1
        self._api_calls += 1
        client = get_client()
        response = client.embeddings.create(model=self.model, input=text)
        vector = response.data[0].embedding
        self._store[key] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Return embeddings for a list of texts.

        Texts already in cache are served immediately.
        New texts are embedded in a single batch API call.
        """
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            key = make_cache_key(text, self.model)
            if key in self._store:
                self._hits += 1
                results[i] = self._store[key]
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            self._misses += len(missing_texts)
            self._api_calls += 1
            client = get_client()
            response = client.embeddings.create(model=self.model, input=missing_texts)
            for j, item in enumerate(response.data):
                idx = missing_indices[j]
                text = missing_texts[j]
                key = make_cache_key(text, self.model)
                self._store[key] = item.embedding
                results[idx] = item.embedding

        # All slots filled — narrow type
        return [v for v in results if v is not None]

    def save(self) -> None:
        """Persist the cache to JSON at cache_path (no-op if path is None)."""
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "entries": self._store,
        }
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def stats(self) -> dict[str, int]:
        """Return a snapshot of hit/miss/api_call counters."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "api_calls": self._api_calls,
            "cached_entries": len(self._store),
        }

    def reset_counters(self) -> None:
        """Reset hit/miss/api_call counters (does not clear the store)."""
        self._hits = 0
        self._misses = 0
        self._api_calls = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load cache entries from JSON file."""
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("model") == self.model:
                self._store = payload.get("entries", {})
        except (json.JSONDecodeError, KeyError):
            # Corrupt or incompatible file — start fresh
            self._store = {}


# ---------------------------------------------------------------------------
# ANTI-PATTERN 1: embedding the same text repeatedly without caching
# ---------------------------------------------------------------------------


def anti_pattern_no_cache(texts: list[str]) -> list[list[float]]:
    """
    WRONG: call the API for every text on every request.
    If the same sentence appears in 100 documents you pay 100x the cost.
    Fix: use EmbeddingCache so repeated texts are served from memory.
    """
    client = get_client()
    vectors: list[list[float]] = []
    for text in texts:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        vectors.append(response.data[0].embedding)
    return vectors


if __name__ == "__main__":
    sep = "=" * 60
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"

    print(f"Domain 5 - Task 5.6: Embedding Caching {mock_label}")
    print(sep)

    # DEMO 1: Cache miss then hit
    print("DEMO 1: Cache miss followed by cache hit")
    print(sep)
    cache = EmbeddingCache(model=EMBEDDING_MODEL)
    text_a = "The speed of light is approximately 299,792 km/s."

    print(f"  First call  (miss): {repr(text_a[:50])}")
    t0 = time.perf_counter()
    vec1 = cache.embed(text_a)
    t1 = time.perf_counter()
    s1 = cache.stats()
    print(f"    dims={len(vec1)}  elapsed={((t1 - t0) * 1000):.2f}ms")
    print(f"    hits={s1['hits']}  misses={s1['misses']}  api_calls={s1['api_calls']}")

    print()
    print(f"  Second call (hit) : {repr(text_a[:50])}")
    t2 = time.perf_counter()
    vec2 = cache.embed(text_a)
    t3 = time.perf_counter()
    s2 = cache.stats()
    print(f"    dims={len(vec2)}  elapsed={((t3 - t2) * 1000):.2f}ms")
    print(f"    hits={s2['hits']}  misses={s2['misses']}  api_calls={s2['api_calls']}")
    print(f"    Vectors identical: {vec1 == vec2}")

    print()
    print(sep)

    # DEMO 2: Batch embedding with partial cache
    print("DEMO 2: Batch embedding — partial cache reuse")
    print(sep)
    cache2 = EmbeddingCache(model=EMBEDDING_MODEL)
    warm_texts = [
        "Python is a programming language.",
        "Machine learning learns from data.",
    ]
    # Pre-warm two entries
    for t in warm_texts:
        cache2.embed(t)
    cache2.reset_counters()

    batch_texts = warm_texts + [
        "RAG combines retrieval with generation.",   # new
        "Embeddings encode meaning as vectors.",      # new
    ]
    print(f"  Pre-warmed entries : {len(warm_texts)}")
    print(f"  Batch size         : {len(batch_texts)}")
    vectors = cache2.embed_batch(batch_texts)
    s3 = cache2.stats()
    print(f"  Results            : {len(vectors)} vectors")
    print(f"  Cache hits         : {s3['hits']}  (served from cache)")
    print(f"  Cache misses       : {s3['misses']}  (needed API)")
    print(f"  API calls made     : {s3['api_calls']}  (1 batch for new texts)")

    print()
    print(sep)

    # DEMO 3: Persistence — save and reload
    print("DEMO 3: Persistence — save cache to JSON and reload")
    print(sep)
    tmp_path = Path("/tmp/embedding_cache_demo.json")
    cache3 = EmbeddingCache(model=EMBEDDING_MODEL, cache_path=tmp_path)
    persist_texts = [
        "Gravity pulls objects toward the Earth.",
        "Water molecules consist of two hydrogen atoms and one oxygen atom.",
    ]
    for t in persist_texts:
        cache3.embed(t)
    cache3.save()
    file_size = tmp_path.stat().st_size
    print(f"  Saved {len(persist_texts)} entries to: {tmp_path}")
    print(f"  File size: {file_size} bytes")

    # Reload from disk
    cache4 = EmbeddingCache(model=EMBEDDING_MODEL, cache_path=tmp_path)
    cache4.reset_counters()
    for t in persist_texts:
        cache4.embed(t)
    s4 = cache4.stats()
    print(f"  After reload — hits={s4['hits']}  misses={s4['misses']}  api_calls={s4['api_calls']}")
    print("  All entries served from disk — zero API calls on reload.")

    # Clean up temp file
    tmp_path.unlink(missing_ok=True)

    print()
    print(sep)

    # DEMO 4: Cost savings illustration
    print("DEMO 4: Cost savings — repeated embeds without caching")
    print(sep)
    repeated_texts = ["The same sentence repeated many times."] * 10
    cache5 = EmbeddingCache(model=EMBEDDING_MODEL)
    for t in repeated_texts:
        cache5.embed(t)
    s5 = cache5.stats()
    print(f"  Requested embeddings : {len(repeated_texts)}")
    print(f"  API calls made       : {s5['api_calls']}  (only the first miss)")
    print(f"  Cache hits           : {s5['hits']}  (9 free lookups)")
    print(f"  Savings              : {len(repeated_texts) - s5['api_calls']} avoided API calls")

    print()
    print(sep)

    # ANTI-PATTERN 1
    print("ANTI-PATTERN 1: Calling API for every text with no caching")
    print(sep)
    ap_texts = ["Repeated text for every call."] * 5
    _ = anti_pattern_no_cache(ap_texts)
    print(f"  Embedded {len(ap_texts)} texts — made {len(ap_texts)} API calls.")
    print("  Problem: identical texts trigger a separate API call every time.")
    print("  Fix: use EmbeddingCache.embed() to serve repeated texts from memory.")

    print()
    print(sep)
    print("KEY TAKEAWAYS:")
    print(f"  - A cache key is the SHA-256 hash of model+text, ensuring{NL}"
          f"    identical inputs always resolve to the same stored vector.")
    print(f"  - Cache hits skip the API entirely, reducing latency to microseconds{NL}"
          f"    and eliminating cost for repeated or overlapping documents.")
    print(f"  - Batch embedding with partial cache reuse issues one API call for{NL}"
          f"    new texts only — warm entries are served locally.")
    print(f"  - Persisting to JSON lets the cache survive process restarts,{NL}"
          f"    so ingest costs are paid once even across multiple runs.")
