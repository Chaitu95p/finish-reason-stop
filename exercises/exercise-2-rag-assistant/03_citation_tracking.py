"""Exercise 2 - Task 2.3: Citation Tracking

GOAL: Extend the RAG pipeline to include source citations in the answer.
The model outputs citations referencing the specific chunks it used.

SKILLS PRACTICED:
  - Structured output for citation extraction
  - Grounding answers with source attribution

Run:
  uv run python 03_citation_tracking.py
"""

NL = chr(10)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

import json
import math
from dataclasses import dataclass

from pydantic import BaseModel
from shared.mock import get_client, is_mock

CORPUS = [
    ("doc-1", "Python 3.12 introduced type parameter syntax for generics."),
    ("doc-2", "gpt-4o-mini costs $0.15 per million input tokens."),
    ("doc-3", "The Batch API offers 50% discount on all supported models."),
    ("doc-4", "Fine-tuning requires consistent system prompt across all examples."),
    ("doc-5", "Vector stores enable semantic search across uploaded files."),
]


class CitedAnswer(BaseModel):
    answer: str
    citations: list[str]


@dataclass
class Document:
    doc_id: str
    text: str
    embedding: list[float]


def embed_texts(client: object, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)  # type: ignore[attr-defined]
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(index: list[Document], q_emb: list[float], top_k: int = 3) -> list[Document]:
    return sorted(index, key=lambda d: cosine_similarity(d.embedding, q_emb), reverse=True)[:top_k]


def rag_with_citations(client: object, index: list[Document], question: str) -> CitedAnswer:
    """Generate an answer with cited source document IDs."""
    [q_emb] = embed_texts(client, [question])
    chunks = retrieve(index, q_emb, top_k=3)
    context = NL.join(f"[{doc.doc_id}] {doc.text}" for doc in chunks)
    system = (
        "Answer the question using the provided context. "
        "Include the document IDs (e.g. doc-1) you referenced in the citations field."
        f"{NL}{NL}Context:{NL}{context}"
    )
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"{question} Respond with JSON: {{\"answer\": \"...\", \"citations\": [\"doc-N\", ...]}}"},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        return CitedAnswer(answer=data.get("answer", ""), citations=data.get("citations", []))
    except Exception:
        return CitedAnswer(answer=raw, citations=[])


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 2 - Task 2.3: Citation Tracking [{mode}]{NL}{sep}")

    client = get_client()
    texts = [text for _, text in CORPUS]
    embeddings = embed_texts(client, texts)
    index = [Document(doc_id=did, text=text, embedding=emb) for (did, text), emb in zip(CORPUS, embeddings)]

    questions = [
        "What is the cost of gpt-4o-mini?",
        "How can I save on API costs for batch jobs?",
    ]
    for q in questions:
        print(f"{NL}  Q: {q!r}")
        result = rag_with_citations(client, index, q)
        print(f"  A: {result.answer!r}")
        print(f"  Citations: {result.citations}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Tag each context chunk with an ID so the model can cite it")
    print("  2. Request JSON output with 'answer' and 'citations' fields")
    print("  3. Validate Pydantic model against the JSON output for type safety")
