"""Domain 8 - Task 8.4: RAG with Vector Store

CONCEPTS:
  1. Full RAG: ingest → store → retrieve → generate
  2. Vector store as managed retrieval layer
  3. Model automatically retrieves relevant context
  4. Grounded answers with citations

Mnemonic: ISRG — Ingest, Store, Retrieve, Generate

Run:
  uv run python 04_rag_with_vector_store.py
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NL = chr(10)
MODEL = "gpt-4o"
MAX_POLL_ITERATIONS = 15
POLL_INTERVAL_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A document to be ingested into the RAG pipeline."""

    name: str
    content: str
    category: str = "general"


@dataclass
class IngestedDocument:
    """A document after upload to the Files API."""

    document: Document
    file_id: str


@dataclass
class RAGCorpus:
    """A collection of ingested documents stored in a vector store."""

    vector_store_id: str
    vector_store_name: str
    documents: list[IngestedDocument] = field(default_factory=list)
    status: str = "pending"


@dataclass
class RAGAnswer:
    """An answer produced by the RAG pipeline."""

    question: str
    answer: str
    response_id: str
    vector_store_id: str


# ---------------------------------------------------------------------------
# Phase 1: Ingest — upload documents to Files API
# ---------------------------------------------------------------------------


def ingest_documents(
    client: object,
    documents: list[Document],
) -> list[IngestedDocument]:
    """Upload all documents to the Files API. Returns IngestedDocument list."""
    ingested: list[IngestedDocument] = []
    for doc in documents:
        file_tuple = (doc.name, io.BytesIO(doc.content.encode("utf-8")), "text/plain")
        result = client.files.create(file=file_tuple, purpose="assistants")  # type: ignore[union-attr]
        ingested.append(IngestedDocument(document=doc, file_id=result.id))
        print(f"  Ingested: {doc.name} → file_id={result.id}")
    return ingested


# ---------------------------------------------------------------------------
# Phase 2: Store — create vector store and add files
# ---------------------------------------------------------------------------


def create_rag_corpus(
    client: object,
    name: str,
    ingested_docs: list[IngestedDocument],
) -> RAGCorpus:
    """Create a vector store and attach all ingested documents."""
    vs = client.vector_stores.create(name=name)  # type: ignore[union-attr]
    corpus = RAGCorpus(
        vector_store_id=vs.id,
        vector_store_name=vs.name,
        documents=ingested_docs,
    )

    for idoc in ingested_docs:
        client.vector_stores.files.create(  # type: ignore[union-attr]
            vector_store_id=vs.id, file_id=idoc.file_id
        )
        print(f"  Attached {idoc.document.name} to store {vs.id}")

    corpus.status = "indexing"
    return corpus


def wait_for_corpus_ready(
    client: object,
    corpus: RAGCorpus,
    max_iterations: int = MAX_POLL_ITERATIONS,
    interval: float = POLL_INTERVAL_SECONDS,
) -> RAGCorpus:
    """Poll until the vector store finishes indexing all files."""
    print(f"  Polling for completion (max {max_iterations} iterations)...")
    for i in range(1, max_iterations + 1):
        vs = client.vector_stores.retrieve(corpus.vector_store_id)  # type: ignore[union-attr]
        status = vs.status
        fc = vs.file_counts
        print(
            f"  Iteration {i}: status={status}, "
            f"completed={fc.completed}, in_progress={fc.in_progress}, failed={fc.failed}"
        )
        if status in ("completed", "failed", "expired"):
            corpus.status = status
            return corpus
        if not is_mock():
            time.sleep(interval)
    corpus.status = "timeout"
    return corpus


# ---------------------------------------------------------------------------
# Phase 3: Retrieve + Phase 4: Generate
# ---------------------------------------------------------------------------


def rag_query(
    client: object,
    corpus: RAGCorpus,
    question: str,
    system_instructions: str | None = None,
) -> RAGAnswer:
    """Ask a question. The model retrieves relevant chunks and generates an answer."""
    file_search_tool = {
        "type": "file_search",
        "vector_store_ids": [corpus.vector_store_id],
    }
    instructions = system_instructions or (
        "You are a precise assistant. Answer questions strictly from the provided documents. "
        "If the answer is not found, say so explicitly."
    )
    response = client.responses.create(  # type: ignore[union-attr]
        model=MODEL,
        input=question,
        tools=[file_search_tool],
        instructions=instructions,
    )
    return RAGAnswer(
        question=question,
        answer=getattr(response, "output_text", ""),
        response_id=getattr(response, "id", ""),
        vector_store_id=corpus.vector_store_id,
    )


def batch_rag_query(
    client: object,
    corpus: RAGCorpus,
    questions: list[str],
    system_instructions: str | None = None,
) -> list[RAGAnswer]:
    """Run multiple questions against the RAG corpus."""
    answers: list[RAGAnswer] = []
    for q in questions:
        answer = rag_query(client, corpus, q, system_instructions)
        answers.append(answer)
    return answers


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_corpus(client: object, corpus: RAGCorpus) -> None:
    """Delete the vector store (files are cleaned up separately)."""
    client.vector_stores.delete(corpus.vector_store_id)  # type: ignore[union-attr]
    print(f"  Deleted vector store: {corpus.vector_store_id}")
    for idoc in corpus.documents:
        client.files.delete(idoc.file_id)  # type: ignore[union-attr]
        print(f"  Deleted file: {idoc.file_id} ({idoc.document.name})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    client = get_client()
    sep = "=" * 60

    print(sep)
    print("Domain 8 - Task 8.4: RAG with Vector Store")
    print(f"Mode: {'MOCK' if is_mock() else 'LIVE'}")
    print(sep)

    # Define the corpus: a small technical documentation set
    corpus_documents = [
        Document(
            name="authentication.txt",
            content=(
                "Authentication Guide" + NL
                + "API keys are passed via the Authorization header: Bearer <key>." + NL
                + "Never expose API keys in client-side code." + NL
                + "Use environment variables: OPENAI_API_KEY." + NL
                + "Keys can be rotated in the dashboard under API settings."
            ),
            category="security",
        ),
        Document(
            name="rate_limits.txt",
            content=(
                "Rate Limits Reference" + NL
                + "Tier 1: 10,000 tokens per minute (TPM)." + NL
                + "Tier 2: 40,000 TPM." + NL
                + "Tier 3: 100,000 TPM." + NL
                + "429 errors indicate rate limit exceeded; implement exponential backoff."
            ),
            category="operations",
        ),
        Document(
            name="models.txt",
            content=(
                "Model Reference" + NL
                + "gpt-4o: 128k context, best for complex reasoning, $5/1M input tokens." + NL
                + "gpt-4o-mini: 128k context, fast and cheap, $0.15/1M input tokens." + NL
                + "o1: reasoning model, $15/1M input tokens." + NL
                + "Use gpt-4o-mini for high-volume tasks to reduce cost."
            ),
            category="models",
        ),
    ]

    # DEMO 1: Phase 1 — Ingest
    print(NL + "DEMO 1: Phase 1 — Ingest documents")
    print("-" * 40)
    ingested = ingest_documents(client, corpus_documents)
    print(f"  Ingested {len(ingested)} document(s)")

    # DEMO 2: Phase 2 — Store
    print(NL + "DEMO 2: Phase 2 — Create vector store and attach files")
    print("-" * 40)
    corpus = create_rag_corpus(client, name="OpenAI Docs RAG", ingested_docs=ingested)
    print(f"  Vector store created: {corpus.vector_store_id}")
    print(f"  Initial status: {corpus.status}")

    # DEMO 3: Wait for indexing
    print(NL + "DEMO 3: Wait for indexing to complete")
    print("-" * 40)
    corpus = wait_for_corpus_ready(client, corpus)
    print(f"  Corpus ready. Final status: {corpus.status}")

    # DEMO 4: Phase 3+4 — Retrieve + Generate (single query)
    print(NL + "DEMO 4: Phase 3+4 — Single RAG query")
    print("-" * 40)
    question = "How should I authenticate to the OpenAI API?"
    print(f"  Question: {question}")
    answer = rag_query(client, corpus, question)
    print(f"  Answer:   {answer.answer}")
    print(f"  Response ID: {answer.response_id}")

    # DEMO 5: Batch RAG queries
    print(NL + "DEMO 5: Batch RAG queries across topics")
    print("-" * 40)
    qa_pairs = [
        "What is the TPM limit for Tier 2?",
        "Which model should I use for high-volume tasks?",
        "What HTTP status code means rate limit exceeded?",
        "Where do I rotate my API keys?",
    ]
    results = batch_rag_query(client, corpus, qa_pairs)
    for r in results:
        print(f"  Q: {r.question}")
        print(f"  A: {r.answer[:120]}")
        print()

    # DEMO 6: Full pipeline summary
    print(NL + "DEMO 6: Full RAG pipeline summary")
    print("-" * 40)
    pipeline_steps = [
        ("Ingest", "Upload documents to Files API with purpose='assistants'"),
        ("Store",  "Create vector store; attach files with vector_stores.files.create()"),
        ("Retrieve", "Responses API + file_search tool fetches relevant chunks automatically"),
        ("Generate", "Model synthesizes grounded answer from retrieved context"),
    ]
    for step, description in pipeline_steps:
        print(f"  {step:<10}: {description}")

    # DEMO 7: Cleanup
    print(NL + "DEMO 7: Cleanup — delete vector store and files")
    print("-" * 40)
    cleanup_corpus(client, corpus)

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - RAG has four phases: Ingest, Store, Retrieve, Generate (ISRG)")
    print("  - The Files API ingests raw documents; vector stores index them for search")
    print("  - The Responses API with file_search performs retrieval automatically")
    print("  - Batch multiple questions with a single corpus for efficiency")
    print("  - Always clean up both the vector store and the underlying files")
    print("  - Production RAG: reuse one vector store across many queries")
    print(sep)
