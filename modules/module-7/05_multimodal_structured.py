"""Domain 7 - Task 7.5: Vision + Structured Output

CONCEPTS:
  1. Vision input combined with structured output format
  2. ImageAnalysis Pydantic model — typed extraction from visual content
  3. response_format with vision messages
  4. Typed access to extracted visual information

Mnemonic: VIPS — Vision, Image_analysis, Pydantic, Structured

Run:
  uv run python 05_multimodal_structured.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import base64
import json
from typing import Literal

from pydantic import BaseModel, Field
from shared.mock import get_client, is_mock


class ImageAnalysis(BaseModel):
    scene_type: Literal["indoor", "outdoor", "abstract", "document", "other"]
    objects: list[str]
    dominant_colors: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    description: str


class ProductAnalysis(BaseModel):
    product_name: str
    category: str
    visible_features: list[str]
    estimated_price_range: str
    condition: Literal["new", "used", "damaged", "unknown"]


def demo_vision_structured() -> None:
    """DEMO 1: Vision input → structured Pydantic output."""
    client = get_client()
    mock_image = base64.b64encode(b"mock_image_bytes").decode("utf-8")
    data_url = f"data:image/jpeg;base64,{mock_image}"

    # Build multimodal message
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Analyze this image and return JSON with: "
                    "scene_type (indoor/outdoor/abstract/document/other), "
                    "objects (list of objects), dominant_colors (list), "
                    "confidence (0-1), description (string)."
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
        ],
    }]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content or json.dumps({
        "scene_type": "outdoor",
        "objects": ["tree", "sky", "mountain"],
        "dominant_colors": ["blue", "green"],
        "confidence": 0.87,
        "description": "A scenic outdoor landscape.",
    })

    try:
        analysis = ImageAnalysis.model_validate(json.loads(raw))
        print(f"  scene_type: {analysis.scene_type!r}")
        print(f"  objects: {analysis.objects}")
        print(f"  colors: {analysis.dominant_colors}")
        print(f"  confidence: {analysis.confidence:.2f}")
        print(f"  description: {analysis.description!r}")
    except Exception as e:
        print(f"  Parse error: {e}")
        print(f"  Raw response: {raw!r}")


def demo_schema_display() -> None:
    """DEMO 2: Show the JSON schema that would be sent for vision analysis."""
    schema = ImageAnalysis.model_json_schema()
    print("  ImageAnalysis schema:")
    print(f"    required fields: {schema.get('required', [])}")
    for field_name, field_def in schema.get("properties", {}).items():
        ftype = field_def.get("type") or str(field_def.get("anyOf", ""))[:30]
        print(f"    [{field_name}]: {ftype}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 7 - Task 7.5: Vision + Structured Output [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Vision → Structured ---")
    demo_vision_structured()

    print(f"{NL}--- DEMO 2: Schema ---")
    demo_schema_display()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Combine multimodal messages (text + image_url) with response_format")
    print("  2. Pydantic validation ensures type safety on vision extraction results")
    print("  3. Use json_object mode + model_validate() for robust parsing")
    print("  4. detail='low' reduces token cost when high spatial detail isn't needed")
