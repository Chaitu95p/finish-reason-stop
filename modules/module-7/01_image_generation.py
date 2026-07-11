"""Domain 7 - Task 7.1: Image Generation

CONCEPTS:
  1. client.images.generate() — DALL-E image generation
  2. model options — "dall-e-3" (better quality) vs "dall-e-2"
  3. size options — 1024x1024, 1792x1024, 1024x1792 (DALL-E 3)
  4. response.data[0].url — the generated image URL (expires in ~1hr)

Mnemonic: GSUR — Generate, Size, Url, Revised_prompt

Run:
  uv run python 01_image_generation.py
"""

NL = chr(10)

from shared.mock import get_client, is_mock


def demo_basic_generation() -> None:
    """DEMO 1: Generate an image from a text prompt."""
    client = get_client()
    response = client.images.generate(
        model="dall-e-3",
        prompt="A serene mountain lake at sunset with pine trees reflecting in the water",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    url = response.data[0].url
    print(f"  Generated image URL: {url!r}")
    print(f"  Image count: {len(response.data)}")


def demo_sizes_and_quality() -> None:
    """DEMO 2: Different sizes and quality settings."""
    client = get_client()
    configs = [
        ("1024x1024", "standard", "Square, standard — cheapest"),
        ("1792x1024", "standard", "Landscape, standard"),
        ("1024x1024", "hd", "Square, HD — highest quality"),
    ]
    for size, quality, note in configs:
        response = client.images.generate(
            model="dall-e-3",
            prompt="A red apple on a wooden table",
            size=size,
            quality=quality,
        )
        print(f"  {note}: {response.data[0].url!r}")


def demo_style_options() -> None:
    """DEMO 3: Style parameter — vivid vs natural."""
    client = get_client()
    for style in ["vivid", "natural"]:
        response = client.images.generate(
            model="dall-e-3",
            prompt="A futuristic cityscape at night",
            size="1024x1024",
            style=style,
        )
        print(f"  style={style!r}: {response.data[0].url!r}")
    print("  vivid: hyper-real, dramatic colors")
    print("  natural: more realistic, less stylized")


def anti_pattern_dall_e_2_size() -> None:
    """ANTI-PATTERN 1: Using DALL-E 3 sizes with DALL-E 2.

    DALL-E 2 only supports: 256x256, 512x512, 1024x1024.
    DALL-E 3 supports: 1024x1024, 1792x1024, 1024x1792.
    Using 1792x1024 with dall-e-2 causes a 400 error.
    """
    print("  ANTI-PATTERN: images.generate(model='dall-e-2', size='1792x1024')")
    print("  → API 400: size '1792x1024' not supported for dall-e-2")
    print("  DALL-E 2 sizes: 256x256, 512x512, 1024x1024")
    print("  DALL-E 3 sizes: 1024x1024, 1792x1024, 1024x1792")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 7 - Task 7.1: Image Generation [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Basic Generation ---")
    demo_basic_generation()

    print(f"{NL}--- DEMO 2: Sizes and Quality ---")
    demo_sizes_and_quality()

    print(f"{NL}--- DEMO 3: Style Options ---")
    demo_style_options()

    print(f"{NL}--- ANTI-PATTERN 1: Wrong Model/Size Combo ---")
    anti_pattern_dall_e_2_size()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. response.data[0].url contains the image URL (expires in ~1 hour)")
    print("  2. DALL-E 3 supports landscape/portrait; DALL-E 2 only square sizes")
    print("  3. quality='hd' produces sharper images at 2x the cost")
    print("  4. style='vivid' is dramatic; 'natural' is more realistic")
