# Project 2: RAG Document Q&A Assistant

## What it does
An in-memory RAG assistant that embeds a document corpus, retrieves relevant chunks per query, and generates grounded answers with cited source IDs.

## How to run
```
uv run python main.py --demo
```

## With real API
```
OPENAI_API_KEY=sk-... uv run python main.py
```

## Architecture
- `build_index()`: embeds all corpus documents at startup
- `retrieve()`: cosine similarity search over embedded documents
- `answer_with_citations()`: retrieves top-K, injects as context, generates JSON answer with citations
- In-memory index — no external vector DB required

## Key SDK patterns used
- `client.embeddings.create()` with batched input list
- Cosine similarity without numpy (pure Python)
- `response_format={"type": "json_object"}` for structured citation output
