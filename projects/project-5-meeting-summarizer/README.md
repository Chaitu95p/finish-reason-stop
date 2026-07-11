# Project 5: Meeting Summarizer

## What it does
Transcribes meeting audio and extracts a structured summary with attendees, topics, action items, and decisions.

## How to run
```
uv run python main.py --demo
```

## With real API + audio file
```
OPENAI_API_KEY=sk-... uv run python main.py --audio meeting.mp3
```

## Architecture
- `transcribe()`: wraps `client.audio.transcriptions.create()` with BytesIO
- `extract_summary()`: structured extraction with `MeetingSummary` Pydantic model
- `format_report()`: renders the Pydantic model as a human-readable report
- Pipeline: audio → transcript → structured summary → formatted report

## Key SDK patterns used
- `client.audio.transcriptions.create()` with `io.BytesIO` + `.name` attribute
- `response_format={"type": "json_object"}` for structured meeting extraction
- Pydantic `BaseModel` with `list[str]` and `str | None` fields
