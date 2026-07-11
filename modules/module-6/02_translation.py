"""Domain 6 - Task 6.2: Audio Translation

CONCEPTS:
  1. client.audio.translations.create() — translate audio to English
  2. Always outputs English — no target language parameter
  3. Handles non-English source audio automatically
  4. vs transcription — transcription preserves original language

Mnemonic: TAFE — Translate, Always_English, From_any, End_result

Run:
  uv run python 02_translation.py
"""

NL = chr(10)
WHISPER_MODEL = "whisper-1"

import io

from shared.mock import get_client, is_mock


def demo_translation() -> None:
    """DEMO 1: Translate non-English audio to English."""
    client = get_client()
    mock_french_audio = io.BytesIO(b"mock_french_audio_bonjour_monde")
    mock_french_audio.name = "french_speech.mp3"

    result = client.audio.translations.create(
        model=WHISPER_MODEL,
        file=mock_french_audio,
    )
    print(f"  Translation (French → English): {result.text!r}")
    print("  (In live mode: would detect French, translate to English)")


def demo_transcription_vs_translation() -> None:
    """DEMO 2: Side-by-side comparison of transcription vs translation."""
    client = get_client()
    mock_audio = io.BytesIO(b"mock_spanish_audio")
    mock_audio.name = "spanish.mp3"

    transcription = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=mock_audio,
        language="es",
    )

    mock_audio2 = io.BytesIO(b"mock_spanish_audio")
    mock_audio2.name = "spanish.mp3"
    translation = client.audio.translations.create(
        model=WHISPER_MODEL,
        file=mock_audio2,
    )

    print("  Comparison:")
    print(f"    transcriptions.create (preserves language): {transcription.text!r}")
    print(f"    translations.create   (always English):     {translation.text!r}")


def demo_use_cases() -> None:
    """DEMO 3: When to use translation vs transcription."""
    print("  Use transcriptions when:")
    print("    - You need the text in the original language")
    print("    - You'll translate yourself (custom pipeline)")
    print("    - Language preservation matters (e.g., subtitles)")
    print(f"{NL}  Use translations when:")
    print("    - You only need English output")
    print("    - Downstream processing expects English")
    print("    - Multilingual audio with English-only infrastructure")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 6 - Task 6.2: Audio Translation [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Translation ---")
    demo_translation()

    print(f"{NL}--- DEMO 2: Transcription vs Translation ---")
    demo_transcription_vs_translation()

    print(f"{NL}--- DEMO 3: Use Cases ---")
    demo_use_cases()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. translations.create() always outputs English — no target_language param")
    print("  2. transcriptions.create() preserves the source language")
    print("  3. Both use model='whisper-1' and the same file parameter format")
    print("  4. Translation is a one-step alternative to STT + separate translate call")
