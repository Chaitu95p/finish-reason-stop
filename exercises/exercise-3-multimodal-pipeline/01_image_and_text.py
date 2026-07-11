"""Exercise 3 - Task 3.1: Image Analysis and Text Generation Pipeline

GOAL: Build a pipeline that analyzes an image and generates a structured
caption + alt-text description using vision + structured output.

SKILLS PRACTICED:
  - base64 image encoding for vision
  - response_format=json_object with vision messages
  - Pydantic validation of vision output

Run:
  uv run python 01_image_and_text.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import base64
import json

from pydantic import BaseModel
from shared.mock import get_client, is_mock


class ImageDescription(BaseModel):
    caption: str
    alt_text: str
    tags: list[str]
    suitable_for_children: bool


def encode_image_bytes(image_bytes: bytes) -> str:
    """Base64-encode image bytes for use in vision messages."""
    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_image(client: object, image_bytes: bytes, mime_type: str = "image/jpeg") -> ImageDescription:
    """Analyze image bytes and return structured description."""
    b64_image = encode_image_bytes(image_bytes)
    data_url = f"data:{mime_type};base64,{b64_image}"

    prompt = (
        "Analyze this image and return JSON with fields: "
        "caption (string), alt_text (string, accessibility-friendly), "
        "tags (list of strings), suitable_for_children (boolean)."
    )
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
            ],
        }],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or json.dumps({
        "caption": "A colorful landscape with mountains and a clear blue sky.",
        "alt_text": "Scenic mountain landscape under blue sky.",
        "tags": ["landscape", "mountains", "nature"],
        "suitable_for_children": True,
    })
    data = json.loads(raw)
    return ImageDescription.model_validate(data)


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 3 - Task 3.1: Image & Text Pipeline [{mode}]{NL}{sep}")

    client = get_client()

    # In real use: load from file with Path("image.jpg").read_bytes()
    mock_image_bytes = b"mock_image_bytes_would_be_real_jpeg_data"
    print(f"{NL}  Analyzing image ({len(mock_image_bytes)} bytes)...")

    result = analyze_image(client, mock_image_bytes)
    print(f"  caption:              {result.caption!r}")
    print(f"  alt_text:             {result.alt_text!r}")
    print(f"  tags:                 {result.tags}")
    print(f"  suitable_for_children: {result.suitable_for_children}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Encode image as base64; wrap in data URL for vision messages")
    print("  2. Request JSON output in the prompt; validate with Pydantic")
    print("  3. detail='low' reduces token cost for description tasks")
