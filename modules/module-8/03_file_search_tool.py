"""Domain 8 - Task 8.3: Built-in file_search Tool

CONCEPTS:
  1. {"type": "file_search"} — enable file search tool
  2. vector_store_ids in tool config — which stores to search
  3. Automatic retrieval — model searches and cites
  4. Annotations — citations in response output

Mnemonic: FVAC — File_search, Vector_store, Automatic, Citations

Run:
  uv run python 03_file_search_tool.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NL = chr(10)
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """Represents a citation/annotation from a file_search response."""

    file_id: str
    filename: str
    quote: str
    start_index: int = 0
    end_index: int = 0


@dataclass
class FileSearchResult:
    """Encapsulates a response from file_search-enabled query."""

    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    response_id: str = ""


# ---------------------------------------------------------------------------
# Tool configuration helpers
# ---------------------------------------------------------------------------


def build_file_search_tool(vector_store_ids: list[str]) -> dict:  # type: ignore[type-arg]
    """Build the file_search tool definition for the Responses API."""
    return {
        "type": "file_search",
        "vector_store_ids": vector_store_ids,
    }


def build_file_search_tool_no_stores() -> dict:  # type: ignore[type-arg]
    """Build file_search tool without specifying stores (uses all attached stores)."""
    return {"type": "file_search"}


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def parse_annotations(output_item: object) -> list[Citation]:
    """Extract Citation objects from a response output item's annotations."""
    citations: list[Citation] = []
    annotations = getattr(output_item, "annotations", None)
    if not annotations:
        return citations
    for ann in annotations:
        ann_type = getattr(ann, "type", "")
        if ann_type == "file_citation":
            fc = getattr(ann, "file_citation", None)
            citations.append(Citation(
                file_id=getattr(fc, "file_id", "") if fc else "",
                filename=getattr(ann, "filename", getattr(fc, "file_id", "unknown")),
                quote=getattr(fc, "quote", "") if fc else "",
                start_index=getattr(ann, "start_index", 0),
                end_index=getattr(ann, "end_index", 0),
            ))
    return citations


def parse_file_search_response(response: object) -> FileSearchResult:
    """Parse the Responses API result into a FileSearchResult."""
    answer = getattr(response, "output_text", "")
    all_citations: list[Citation] = []

    output_items = getattr(response, "output", [])
    for item in output_items:
        item_type = getattr(item, "type", "")
        if item_type == "output_text":
            text_obj = getattr(item, "text", None)
            if text_obj is not None:
                citations = parse_annotations(text_obj)
                all_citations.extend(citations)
            # Also check annotations directly on the item
            citations = parse_annotations(item)
            all_citations.extend(citations)

    return FileSearchResult(
        answer_text=answer,
        citations=all_citations,
        response_id=getattr(response, "id", ""),
    )


def query_with_file_search(
    client: object,
    question: str,
    vector_store_ids: list[str],
    instructions: str | None = None,
) -> FileSearchResult:
    """Run a question against the Responses API with file_search enabled."""
    tool = build_file_search_tool(vector_store_ids)
    kwargs: dict = {  # type: ignore[type-arg]
        "model": MODEL,
        "input": question,
        "tools": [tool],
    }
    if instructions:
        kwargs["instructions"] = instructions

    response = client.responses.create(**kwargs)  # type: ignore[union-attr]
    return parse_file_search_response(response)


def print_file_search_result(result: FileSearchResult, label: str | None = None) -> None:
    """Print a FileSearchResult."""
    if label:
        print(f"  [{label}]")
    print(f"  Response ID: {result.response_id}")
    print(f"  Answer:      {result.answer_text}")
    if result.citations:
        print(f"  Citations ({len(result.citations)}):")
        for i, c in enumerate(result.citations, 1):
            print(f"    {i}. file_id={c.file_id}, file={c.filename}")
            if c.quote:
                print(f"       quote: {c.quote[:80]}...")
    else:
        print("  Citations: none (or mock mode)")


# ---------------------------------------------------------------------------
# Setup helpers (upload + create vector store)
# ---------------------------------------------------------------------------


def setup_demo_vector_store(client: object) -> str:
    """Create a demo vector store with one document. Returns vector_store_id."""
    # Upload a document
    doc = (
        "OpenAI API Limits" + NL
        + "Rate limits: 10,000 TPM for tier 1." + NL
        + "Context window: 128,000 tokens for gpt-4o." + NL
        + "Batch API discount: 50% off standard pricing." + NL
        + "File size limit: 512 MB per file."
    )
    file_tuple = ("api_limits.txt", io.BytesIO(doc.encode("utf-8")), "text/plain")
    uploaded = client.files.create(file=file_tuple, purpose="assistants")  # type: ignore[union-attr]

    # Create vector store and attach the file
    vs = client.vector_stores.create(name="API Limits Knowledge Base")  # type: ignore[union-attr]
    client.vector_stores.files.create(  # type: ignore[union-attr]
        vector_store_id=vs.id, file_id=uploaded.id
    )
    return str(vs.id)


def teardown_demo_vector_store(client: object, vector_store_id: str) -> None:
    """Delete the demo vector store."""
    client.vector_stores.delete(vector_store_id)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    client = get_client()
    sep = "=" * 60

    print(sep)
    print("Domain 8 - Task 8.3: Built-in file_search Tool")
    print(f"Mode: {'MOCK' if is_mock() else 'LIVE'}")
    print(sep)

    # DEMO 1: file_search tool schema
    print(NL + "DEMO 1: file_search tool schema")
    print("-" * 40)
    import json
    vs_id_example = "vs_abc123"
    tool_with_stores = build_file_search_tool([vs_id_example])
    tool_without_stores = build_file_search_tool_no_stores()
    print("  Tool with explicit vector_store_ids:")
    print("  " + json.dumps(tool_with_stores, indent=2).replace("\n", NL + "  "))
    print(NL + "  Tool without stores (uses all attached):")
    print("  " + json.dumps(tool_without_stores, indent=2).replace("\n", NL + "  "))

    # DEMO 2: Set up a demo vector store
    print(NL + "DEMO 2: Setup — upload doc and create vector store")
    print("-" * 40)
    vs_id = setup_demo_vector_store(client)
    print(f"  Vector store ready: {vs_id}")

    # DEMO 3: Query with file_search tool
    print(NL + "DEMO 3: Query using file_search tool")
    print("-" * 40)
    question = "What is the context window size for gpt-4o?"
    print(f"  Question: {question}")
    result = query_with_file_search(
        client,
        question=question,
        vector_store_ids=[vs_id],
        instructions="Answer only from the provided documents.",
    )
    print_file_search_result(result, "file_search query")

    # DEMO 4: Multiple queries on the same store
    print(NL + "DEMO 4: Multiple questions on same vector store")
    print("-" * 40)
    questions = [
        "What is the rate limit for tier 1?",
        "How much discount does the Batch API offer?",
        "What is the maximum file size?",
    ]
    for q in questions:
        r = query_with_file_search(client, question=q, vector_store_ids=[vs_id])
        print(f"  Q: {q}")
        print(f"  A: {r.answer_text[:100]}")
        print()

    # DEMO 5: Annotations structure explanation
    print(NL + "DEMO 5: Annotation/citation structure in responses")
    print("-" * 40)
    print("  When using file_search with a real API key, response output items")
    print("  contain annotations of type 'file_citation':")
    print()
    example_annotation = {
        "type": "file_citation",
        "text": "【4:0†source】",
        "start_index": 42,
        "end_index": 55,
        "file_citation": {
            "file_id": "file-abc123",
            "quote": "Context window: 128,000 tokens for gpt-4o.",
        },
    }
    print("  " + json.dumps(example_annotation, indent=2).replace("\n", NL + "  "))

    # ANTI-PATTERN 1: Using chat completions instead of responses API for file_search
    print(NL + "ANTI-PATTERN 1: file_search belongs in the Responses API")
    print("-" * 40)
    print("  BAD:  client.chat.completions.create(model=..., tools=[file_search_tool])")
    print("        chat.completions does not support file_search or vector_store_ids")
    print("  GOOD: client.responses.create(model=..., tools=[file_search_tool])")
    print("  Reason: The Responses API has native retrieval-augmented generation support.")

    # Cleanup
    print(NL + "Cleanup: deleting demo vector store")
    teardown_demo_vector_store(client, vs_id)
    print(f"  Deleted vector store: {vs_id}")

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - Add {'type': 'file_search', 'vector_store_ids': [...]} to tools list")
    print("  - The Responses API (not chat.completions) supports file_search natively")
    print("  - The model automatically retrieves relevant chunks before answering")
    print("  - Citations appear as annotations of type 'file_citation' in output items")
    print("  - vector_store_ids scopes retrieval to specific stores per query")
    print("  - Parse output[].text.annotations to extract citation metadata")
    print(sep)
