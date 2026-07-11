"""Domain 6 - Task 6.3: Text to Speech

CONCEPTS:
  1. client.audio.speech.create() — TTS API
  2. voice options — alloy, echo, fable, onyx, nova, shimmer
  3. response_format options — mp3 (default), opus, aac, flac, wav, pcm
  4. response.content — raw audio bytes to write to file or play

Mnemonic: VRAF — Voice, Response_format, Audio_bytes, File_write

Run:
  uv run python 03_text_to_speech.py
"""

NL = chr(10)
TTS_MODEL = "tts-1"
TTS_HD_MODEL = "tts-1-hd"

import io

from shared.mock import get_client, is_mock

VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


def demo_basic_tts() -> None:
    """DEMO 1: Generate speech from text."""
    client = get_client()
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice="alloy",
        input="Hello! This is a text-to-speech demonstration.",
    )
    audio_bytes = response.content
    print("  Voice: alloy")
    print(f"  Audio bytes: {len(audio_bytes)} bytes")
    print("  (In live mode: playable mp3 audio)")


def demo_all_voices() -> None:
    """DEMO 2: Show all available voices."""
    client = get_client()
    text = "The quick brown fox jumps over the lazy dog."
    print(f"  Generating speech with all {len(VOICES)} voices...")
    for voice in VOICES:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
        )
        print(f"    {voice:<10}: {len(response.content)} bytes")


def demo_save_to_file() -> None:
    """DEMO 3: Save audio bytes to file using pathlib.Path."""
    client = get_client()
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice="nova",
        input="Saving audio to a file.",
        response_format="mp3",
    )
    # In demo: write to BytesIO instead of disk
    buffer = io.BytesIO(response.content)
    buffer.seek(0)
    size = len(buffer.getvalue())
    print(f"  Saved {size} bytes to buffer (would be: output.mp3)")
    print("  Real usage: Path('output.mp3').write_bytes(response.content)")


def demo_hd_vs_standard() -> None:
    """DEMO 4: Standard (tts-1) vs HD (tts-1-hd) quality."""
    client = get_client()
    text = "Quality comparison test."
    std = client.audio.speech.create(model=TTS_MODEL, voice="alloy", input=text)
    hd = client.audio.speech.create(model=TTS_HD_MODEL, voice="alloy", input=text)
    print(f"  tts-1 (standard): {len(std.content)} bytes — faster, cheaper")
    print(f"  tts-1-hd (HD):    {len(hd.content)} bytes — higher quality, slower")
    print("  Use standard for: real-time, chatbots | HD for: final content, podcasts")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 6 - Task 6.3: Text to Speech [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Basic TTS ---")
    demo_basic_tts()

    print(f"{NL}--- DEMO 2: All Voices ---")
    demo_all_voices()

    print(f"{NL}--- DEMO 3: Save to File ---")
    demo_save_to_file()

    print(f"{NL}--- DEMO 4: Standard vs HD ---")
    demo_hd_vs_standard()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. response.content is raw audio bytes — write with Path.write_bytes()")
    print("  2. 6 voices available: alloy, echo, fable, onyx, nova, shimmer")
    print("  3. tts-1 for real-time/low-latency; tts-1-hd for quality audio")
    print("  4. response_format='mp3' is default; use 'pcm' for real-time playback pipelines")
