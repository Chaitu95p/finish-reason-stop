"""Domain 7 - Task 7.4: Vision with Base64 Images

CONCEPTS:
  1. base64 encoding — encode local image bytes to ASCII string
  2. data URL format — "data:image/jpeg;base64,<encoded>"
  3. image_url type — same block, uses data URL instead of http URL
  4. When to use — local files, private images, no public URL available

Mnemonic: EDPL — Encode, Data_url, Private_images, Local_files

Run:
  uv run python 04_vision_base64.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import base64

from shared.mock import get_client, is_mock


def encode_image_to_base64(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode image bytes to a base64 data URL."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def demo_base64_vision() -> None:
    """DEMO 1: Encode local image and send as base64 data URL."""
    client = get_client()
    mock_image_bytes = b"\xff\xd8\xff" + b"\x00" * 100  # mock JPEG header + data
    data_url = encode_image_to_base64(mock_image_bytes, "image/jpeg")

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
            ],
        }],
        max_tokens=200,
    )
    print(f"  Data URL prefix: {data_url[:50]}...")
    print(f"  Data URL length: {len(data_url)} chars")
    print(f"  Response: {completion.choices[0].message.content!r}")


def demo_mime_types() -> None:
    """DEMO 2: Supported MIME types for base64 images."""
    supported = [
        ("image/jpeg", ".jpg/.jpeg"),
        ("image/png", ".png"),
        ("image/gif", ".gif"),
        ("image/webp", ".webp"),
    ]
    print("  Supported MIME types for base64 images:")
    for mime, exts in supported:
        data_url = encode_image_to_base64(b"mock", mime)
        prefix = data_url[:35]
        print(f"    {mime:<15} ({exts}): {prefix}...")


def demo_url_vs_base64() -> None:
    """DEMO 3: When to use URL vs base64."""
    print("  Use image URL when:")
    print("    - Image is publicly accessible")
    print("    - Image is large (URL avoids sending bytes in request)")
    print("    - URL is stable (doesn't expire before API call)")
    print(f"{NL}  Use base64 when:")
    print("    - Image is local / private")
    print("    - No public URL available")
    print("    - Image is generated in-memory (BytesIO)")
    print("    - You need deterministic behavior (URL might expire)")


def anti_pattern_file_path() -> None:
    """ANTI-PATTERN 1: Passing a file path string as the image URL.

    File paths (file:///path/to/img.jpg) are not valid for the API.
    The API can only access http/https URLs or data URLs.
    """
    print("  ANTI-PATTERN: image_url={'url': 'file:///home/user/img.jpg'}")
    print("  → API error: URL scheme not supported")
    print("  CORRECT for local files: read → base64 encode → data URL")
    print("    image_bytes = Path('img.jpg').read_bytes()")
    print("    data_url = encode_image_to_base64(image_bytes, 'image/jpeg')")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 7 - Task 7.4: Vision Base64 [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Base64 Vision ---")
    demo_base64_vision()

    print(f"{NL}--- DEMO 2: MIME Types ---")
    demo_mime_types()

    print(f"{NL}--- DEMO 3: URL vs Base64 ---")
    demo_url_vs_base64()

    print(f"{NL}--- ANTI-PATTERN 1: File Path ---")
    anti_pattern_file_path()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. base64.b64encode(bytes).decode('utf-8') encodes image bytes")
    print("  2. Data URL format: 'data:<mime>;base64,<encoded_string>'")
    print("  3. Use same image_url block structure — just swap http:// for data:")
    print("  4. Prefer http URLs for large images; base64 adds ~33% size overhead")
