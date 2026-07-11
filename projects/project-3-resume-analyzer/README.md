# Project 3: Resume Analyzer

## What it does
Analyzes resume text to extract structured candidate information and scores job fit with strengths/gaps analysis.

## How to run
```
uv run python main.py --demo
```

## With real API
```
OPENAI_API_KEY=sk-... uv run python main.py --demo
```

## Architecture
- `extract_resume()`: structured extraction with Pydantic `ResumeExtraction` model
- `analyze_fit()`: scores candidate against job description with `JobFitAnalysis` model
- Two-step pipeline: extract → score
- `json_object` response format + Pydantic validation for type safety

## Key SDK patterns used
- `response_format={"type": "json_object"}` for structured extraction
- Pydantic `BaseModel.model_validate()` for response validation
- Two sequential API calls in a pipeline pattern
