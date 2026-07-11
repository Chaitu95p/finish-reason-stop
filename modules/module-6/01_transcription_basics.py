"""Domain 6 - Task 6.1: Audio Transcription Basics

CONCEPTS:
  1. client.audio.transcriptions.create() — Whisper API for STT
  2. model="whisper-1" — the transcription model name
  3. file parameter — file-like object (BytesIO or open file handle)
  4. response.text — the transcription as a plain string

Mnemonic: WFTR — Whisper, File_object, Transcribe, Response_text

Run:
  uv run python 01_transcription_basics.py
"""

NL = chr(10)
MODEL = "gpt-4o"
WHISPER_MODEL = "whisper-1"

import io

from shared.mock import get_client, is_mock


def demo_basic_transcription() -> None:
    """DEMO 1: Transcribe mock audio bytes."""
    client = get_client()
    mock_audio = io.BytesIO(b"mock_audio_data_representing_speech")
    mock_audio.name = "speech.mp3"  # Whisper uses filename extension to detect format

    result = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=mock_audio,
    )
    print(f"  Transcription: {result.text!r}")


def demo_with_options() -> None:
    """DEMO 2: Transcription with language hint and response format."""
    client = get_client()
    mock_audio = io.BytesIO(b"mock_french_audio")
    mock_audio.name = "french_speech.mp3"

    result = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=mock_audio,
        language="fr",             # hint: ISO 639-1 language code
        prompt="Meeting notes",    # hint to Whisper about context/vocabulary
        response_format="text",    # "text" | "json" | "verbose_json" | "srt" | "vtt"
        temperature=0.0,           # 0.0 = deterministic transcription
    )
    print(f"  Transcription (with options): {result.text!r}")
    print("  language='fr' hint — Whisper skips language detection")


def demo_supported_formats() -> None:
    """DEMO 3: Show supported audio formats."""
    supported = ["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"]
    print(f"  Supported audio formats: {', '.join(supported)}")
    print("  Max file size: 25 MB")
    print("  File extension matters — Whisper uses it to detect format")


def anti_pattern_string_instead_of_file() -> None:
    """ANTI-PATTERN 1: Passing a file path string instead of a file object.

    client.audio.transcriptions.create(file="speech.mp3") is WRONG.
    The 'file' param requires a file-like object (BytesIO or open()), not a path.
    """
    print("  ANTI-PATTERN: file='speech.mp3'  # TypeError — needs file-like object")
    print("  CORRECT:      file=open('speech.mp3', 'rb')")
    print("  CORRECT:      file=io.BytesIO(audio_bytes)")
    print("  → In mock mode we use io.BytesIO(b'mock_audio')")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 6 - Task 6.1: Audio Transcription Basics [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Basic Transcription ---")
    demo_basic_transcription()

    print(f"{NL}--- DEMO 2: With Options ---")
    demo_with_options()

    print(f"{NL}--- DEMO 3: Supported Formats ---")
    demo_supported_formats()

    print(f"{NL}--- ANTI-PATTERN 1: String Instead of File ---")
    anti_pattern_string_instead_of_file()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. file param is a file-like object — use io.BytesIO or open(path, 'rb')")
    print("  2. Set .name attribute on BytesIO so Whisper knows the file format")
    print("  3. language hint speeds up transcription by skipping auto-detection")
    print("  4. response.text is the transcription string — no further parsing needed")
