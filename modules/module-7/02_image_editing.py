"""Domain 7 - Task 7.2: Image Editing

CONCEPTS:
  1. client.images.edit() — in-painting with a mask
  2. mask — PNG with transparent (alpha=0) area defining the edit zone
  3. image + mask must be same square dimensions (DALL-E 2 only)
  4. prompt describes what to put in the masked area

Mnemonic: MIEN — Mask, Image_edit, Edit_zone, N_variations

Run:
  uv run python 02_image_editing.py
"""

NL = chr(10)

import io

from shared.mock import get_client, is_mock


def create_mock_image(size: int = 100) -> bytes:
    """Create mock PNG bytes (simplified header for demo)."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * (size * size * 4)


def demo_image_edit() -> None:
    """DEMO 1: Edit an image using a mask (in-painting)."""
    client = get_client()
    image_bytes = create_mock_image()
    mask_bytes = create_mock_image()  # transparent area = edit zone

    image_file = io.BytesIO(image_bytes)
    image_file.name = "image.png"
    mask_file = io.BytesIO(mask_bytes)
    mask_file.name = "mask.png"

    response = client.images.edit(
        model="dall-e-2",
        image=image_file,
        mask=mask_file,
        prompt="Replace the sky with a dramatic sunset",
        size="1024x1024",
        n=1,
    )
    print(f"  Edited image URL: {response.data[0].url!r}")
    print("  (Transparent pixels in mask = area to replace)")


def demo_create_variation() -> None:
    """DEMO 2: Generate variations of an existing image."""
    client = get_client()
    image_bytes = create_mock_image()
    image_file = io.BytesIO(image_bytes)
    image_file.name = "original.png"

    response = client.images.create_variation(
        model="dall-e-2",
        image=image_file,
        n=2,
        size="512x512",
    )
    print(f"  Generated {len(response.data)} variations:")
    for i, img in enumerate(response.data):
        print(f"    Variation {i+1}: {img.url!r}")


def demo_mask_convention() -> None:
    """DEMO 3: Explain the mask convention."""
    print("  Mask conventions for images.edit():")
    print("    Transparent pixels (alpha=0): area to REPLACE (edit zone)")
    print("    Opaque pixels (alpha=255):    area to KEEP unchanged")
    print(f"{NL}  Requirements:")
    print("    - image and mask must be the same size")
    print("    - Both must be square (DALL-E 2 limitation)")
    print("    - Both must be PNG format with RGBA channels")
    print("    - Max size: 4MB each")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 7 - Task 7.2: Image Editing [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Image Edit ---")
    demo_image_edit()

    print(f"{NL}--- DEMO 2: Create Variation ---")
    demo_create_variation()

    print(f"{NL}--- DEMO 3: Mask Convention ---")
    demo_mask_convention()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Transparent pixels in mask = edit zone; opaque = preserved area")
    print("  2. images.edit() is DALL-E 2 only — DALL-E 3 doesn't support editing")
    print("  3. Both image and mask must be same square dimensions in PNG format")
    print("  4. images.create_variation() generates N variations without a mask")
