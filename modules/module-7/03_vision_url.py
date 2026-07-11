"""Domain 7 - Task 7.3: Vision with Image URLs

CONCEPTS:
  1. image_url content block — {"type":"image_url","image_url":{"url":...}}
  2. detail param — "low" (85 tokens), "high" (variable), "auto"
  3. Multiple images — list multiple content blocks in one message
  4. Cost impact — detail="high" uses more tokens for large images

Mnemonic: UDMA — Url, Detail, Multiple, Auto

Run:
  uv run python 03_vision_url.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock

SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"


def demo_vision_url() -> None:
    """DEMO 1: Send an image URL for visual analysis."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image? Describe it briefly."},
                {"type": "image_url", "image_url": {"url": SAMPLE_URL}},
            ],
        }],
        max_tokens=300,
    )
    print(f"  Response: {completion.choices[0].message.content!r}")
    print(f"  finish_reason: {completion.choices[0].finish_reason!r}")


def demo_detail_levels() -> None:
    """DEMO 2: Show detail parameter options and their token costs."""
    print("  detail parameter options:")
    print("    'low':  always 85 tokens — fast, cheap, good for thumbnails")
    print("    'high': 170 base + 85 per 512x512 tile — detailed analysis")
    print("    'auto': model chooses based on image size (default)")
    print(f"{NL}  Example cost calculation for a 1024x1024 image:")
    print("    low:  85 tokens")
    print("    high: 170 + (4 tiles × 85) = 510 tokens")


def demo_multiple_images() -> None:
    """DEMO 3: Send multiple images in one message."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images:"},
                {"type": "image_url", "image_url": {"url": SAMPLE_URL, "detail": "low"}},
                {"type": "image_url", "image_url": {"url": SAMPLE_URL, "detail": "low"}},
            ],
        }],
        max_tokens=300,
    )
    print(f"  Multi-image response: {completion.choices[0].message.content!r}")
    print("  (Sent 2 images; model compares them)")


def anti_pattern_missing_text() -> None:
    """ANTI-PATTERN 1: Sending image without a text instruction.

    The model needs a text prompt to know what to analyze.
    An image-only message gives unpredictable results.
    """
    print("  ANTI-PATTERN: content=[image_url block] with no text instruction")
    print("  → Model may describe, caption, or ignore the image unpredictably")
    print("  CORRECT: always pair image with a specific text instruction")
    print("    content=[{text: 'What is in this image?'}, {image_url: ...}]")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 7 - Task 7.3: Vision with URLs [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Vision URL ---")
    demo_vision_url()

    print(f"{NL}--- DEMO 2: Detail Levels ---")
    demo_detail_levels()

    print(f"{NL}--- DEMO 3: Multiple Images ---")
    demo_multiple_images()

    print(f"{NL}--- ANTI-PATTERN 1: Missing Text ---")
    anti_pattern_missing_text()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. image_url block: {\"type\":\"image_url\",\"image_url\":{\"url\":\"...\",\"detail\":\"auto\"}}")
    print("  2. detail='low' = 85 tokens flat; detail='high' = variable by image size")
    print("  3. Multiple images: add multiple image_url content blocks in one message")
    print("  4. Always include a text instruction alongside image content blocks")
