"""Exercise 2 - Task 2.2: Retrieve Context and Generate Answer

GOAL: Full RAG pipeline — embed query, retrieve top chunks,
construct a grounded prompt, generate an answer.

SKILLS PRACTICED:
  - Embedding + retrieval pipeline
  - Context injection into system prompt
  - Grounded generation

Run:
  uv run python 02_retrieve_and_answer.py
"""

NL = chr(10)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

import math
from dataclasses import dataclass

from shared.mock import get_client, is_mock

CORPUS = [
    "Python 3.12 introduced type parameter syntax and f-string improvements.",
    "OpenAI's Batch API allows submitting up to 50,000 requests per batch.",
    "The gpt-4o-mini model is 17× cheaper than gpt-4o for input tokens.",
    "Vector stores enable persistent semantic search across uploaded files.",
    "The Responses API supports stateful conversations via previous_response_id.",
    "Prompt caching reduces cost for repeated long system prompts.",
    "Fine-tuning requires at least 10 examples but 50-100 is recommended.",
]


@dataclass
class Document:
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


def retrieve(index: list[Document], query_embedding: list[float], top_k: int = 3) -> list[Document]:
    scored = sorted(index, key=lambda d: cosine_similarity(d.embedding, query_embedding), reverse=True)
    return scored[:top_k]


def rag_answer(client: object, index: list[Document], question: str) -> str:
    """Retrieve relevant chunks then generate a grounded answer."""
    [q_emb] = embed_texts(client, [question])
    chunks = retrieve(index, q_emb, top_k=3)
    context = NL.join(f"- {doc.text}" for doc in chunks)
    system = f"Answer using ONLY the provided context. If the answer is not in the context, say so.{NL}{NL}Context:{NL}{context}"
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 2 - Task 2.2: Retrieve & Answer [{mode}]{NL}{sep}")

    client = get_client()

    embeddings = embed_texts(client, CORPUS)
    index = [Document(text=t, embedding=e) for t, e in zip(CORPUS, embeddings)]
    print(f"  Index ready: {len(index)} documents")

    questions = [
        "How much cheaper is gpt-4o-mini than gpt-4o?",
        "What is the minimum number of examples for fine-tuning?",
    ]
    for q in questions:
        print(f"{NL}  Q: {q!r}")
        answer = rag_answer(client, index, q)
        print(f"  A: {answer!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Retrieval narrows context — only relevant chunks reach the model")
    print("  2. System prompt explicitly instructs: answer from context only")
    print("  3. top_k=3 is a good starting point; tune based on answer quality")
