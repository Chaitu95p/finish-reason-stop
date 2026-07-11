# Check Coverage Skill

Builds a coverage matrix showing which OpenAI SDK surfaces are covered across all modules.

**Trigger:** User asks about coverage, completeness, or what's been taught.

**Example:** "Check coverage" or "What SDK features are covered?"

## What to do

1. Read the first 30 lines (docstring) of every `0*.py` script across all modules
2. Extract the CONCEPTS list from each script's docstring
3. Map concepts to SDK surfaces:
   - Responses API, Chat Completions, Function Calling
   - Structured Outputs, Streaming, Async
   - Embeddings, Audio, Images, Files, Vector Stores
   - Batch API, Fine-Tuning, Realtime API
   - Error Handling, Production, Testing
4. Build a coverage matrix

## Output format

```
# SDK Coverage Matrix

| SDK Surface         | Module | Scripts | Key Concepts |
|---------------------|--------|---------|--------------|
| Responses API       | 1      | 5       | create, output_text, status |
| Chat Completions    | 1      | 5       | messages, finish_reason |
| ...                 | ...    | ...     | ...          |

## Coverage Summary
Total scripts: N
Total concepts: M
Uncovered surfaces: [list]

## Depth Rating
- Responses API: ████████ 80%
- Embeddings: ██████ 60%
```

**Context:** Run in fork mode. Read files only, no modifications.
