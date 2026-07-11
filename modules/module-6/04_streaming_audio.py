"""Domain 6 - Task 6.4: Streaming Audio Output

CONCEPTS:
  1. Streaming TTS — avoid loading full audio into memory at once
  2. Chunk-by-chunk writing — process bytes as they arrive
  3. Use case — large TTS output, real-time playback
  4. Buffer accumulation — write chunks to BytesIO for mock demo

Mnemonic: SCBR — Stream, Chunks, Buffer, Real_time

Run:
  uv run python 04_streaming_audio.py
"""

NL = chr(10)
TTS_MODEL = "tts-1"

import io

from shared.mock import get_client, is_mock


def demo_chunked_tts() -> None:
    """DEMO 1: Simulate streaming TTS by chunking mock audio bytes."""
    client = get_client()
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice="alloy",
        input="This is a long text that would be streamed in chunks for real-time playback.",
        response_format="pcm",
    )
    full_audio = response.content
    chunk_size = max(1, len(full_audio) // 4)  # simulate 4 chunks
    buffer = io.BytesIO()

    print(f"  Total audio: {len(full_audio)} bytes, chunk size: {chunk_size}")
    for i, start in enumerate(range(0, len(full_audio), chunk_size)):
        chunk = full_audio[start:start + chunk_size]
        buffer.write(chunk)
        print(f"    Chunk {i+1}: {len(chunk)} bytes written")

    buffer.seek(0)
    print(f"  Total buffered: {len(buffer.getvalue())} bytes")


def demo_streaming_pattern() -> None:
    """DEMO 2: Show the real streaming pattern (live API only)."""
    print("  Real streaming pattern (live API with httpx streaming):")
    print("    response = client.audio.speech.create(model=..., stream=True, ...)")
    print("    with response as stream:")
    print("        for chunk in stream.iter_bytes(chunk_size=4096):")
    print("            audio_player.write(chunk)   # or buffer.write(chunk)")
    print(f"{NL}  Mock mode: use response.content and simulate chunking manually")
    print("  (see DEMO 1 above for the mock approach)")


def anti_pattern_full_buffer() -> None:
    """ANTI-PATTERN 1: Loading full audio into memory before playing.

    For long TTS output (minutes of speech), this wastes memory and
    adds a full wait before the user hears any audio.
    Streaming starts playback immediately with each chunk.
    """
    print("  ANTI-PATTERN: buffer = b''")
    print("    for chunk in stream: buffer += chunk  # builds full audio in RAM")
    print("    play_audio(buffer)                    # user waits for full download")
    print(f"{NL}  CORRECT: play each chunk as it arrives")
    print("    for chunk in stream: audio_player.write(chunk)  # immediate playback")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 6 - Task 6.4: Streaming Audio [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Chunked Audio Processing ---")
    demo_chunked_tts()

    print(f"{NL}--- DEMO 2: Real Streaming Pattern ---")
    demo_streaming_pattern()

    print(f"{NL}--- ANTI-PATTERN 1: Full Buffer Before Play ---")
    anti_pattern_full_buffer()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Stream TTS for long audio to reduce memory usage and time-to-first-byte")
    print("  2. response_format='pcm' is best for real-time streaming (no codec overhead)")
    print("  3. Use iter_bytes(chunk_size=4096) to process audio as it streams")
    print("  4. Mock mode: simulate streaming by chunking response.content manually")
