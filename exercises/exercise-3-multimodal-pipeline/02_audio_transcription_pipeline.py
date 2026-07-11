"""Exercise 3 - Task 3.2: Audio Transcription Pipeline

GOAL: STT → summarize → TTS pipeline. Transcribe audio,
extract key points, synthesize a spoken summary.

SKILLS PRACTICED:
  - audio.transcriptions.create()
  - audio.speech.create()
  - Chaining API calls into a pipeline

Run:
  uv run python 02_audio_transcription_pipeline.py
"""

NL = chr(10)
TRANSCRIPTION_MODEL = "whisper-1"
CHAT_MODEL = "gpt-4o"
TTS_MODEL = "tts-1"

import io
import time

from shared.mock import get_client, is_mock


def transcribe(client: object, audio_bytes: bytes, filename: str = "audio.mp3") -> str:
    """Transcribe audio bytes to text."""
    f = io.BytesIO(audio_bytes)
    f.name = filename
    result = client.audio.transcriptions.create(  # type: ignore[attr-defined]
        model=TRANSCRIPTION_MODEL,
        file=f,
        language="en",
    )
    return result.text


def summarize(client: object, transcript: str) -> str:
    """Summarize transcript into 2-3 bullet points."""
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": f"Summarize this in 2-3 bullet points:{NL}{transcript}",
        }],
        max_tokens=150,
    )
    return response.choices[0].message.content or ""


def synthesize(client: object, text: str, voice: str = "alloy") -> bytes:
    """Convert text to speech; return audio bytes."""
    response = client.audio.speech.create(  # type: ignore[attr-defined]
        model=TTS_MODEL,
        voice=voice,
        input=text,
    )
    return response.content


def run_pipeline(client: object, audio_bytes: bytes) -> dict:
    """STT → summarize → TTS pipeline. Returns all intermediate results."""
    t0 = time.monotonic()
    transcript = transcribe(client, audio_bytes)
    t1 = time.monotonic()
    summary = summarize(client, transcript)
    t2 = time.monotonic()
    spoken_summary = synthesize(client, summary)
    t3 = time.monotonic()
    return {
        "transcript": transcript,
        "summary": summary,
        "audio_bytes": len(spoken_summary),
        "transcription_ms": (t1 - t0) * 1000,
        "summarization_ms": (t2 - t1) * 1000,
        "synthesis_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
    }


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 3 - Task 3.2: Audio Pipeline [{mode}]{NL}{sep}")

    client = get_client()
    # In real use: load actual audio file bytes
    mock_audio = b"mock_audio_bytes"
    result = run_pipeline(client, mock_audio)

    print(f"{NL}  Pipeline results:")
    print(f"  transcript:       {result['transcript']!r}")
    print(f"  summary:          {result['summary']!r}")
    print(f"  audio_bytes:      {result['audio_bytes']}")
    print(f"  total_latency_ms: {result['total_ms']:.1f}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. STT → process → TTS is a common voice assistant pattern")
    print("  2. Set f.name on BytesIO so whisper-1 knows the format (mp3, wav, etc.)")
    print("  3. Time each stage separately to identify bottlenecks")
