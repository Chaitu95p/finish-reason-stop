"""Domain 3 - Task 3.2: JSON Schema with Strict Mode

CONCEPTS:
  1. response_format with json_schema — schema-enforced output
  2. strict: true — no extra fields; all properties required
  3. additionalProperties: false — required companion to strict mode
  4. Schema design rules — no $ref, simple types, no optional in strict

Mnemonic: SANE — Strict, AdditionalProperties_false, No_optional, Enforced

Run:
  uv run python 02_json_schema_strict.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json

from shared.mock import get_client, is_mock

SENTIMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                },
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["sentiment", "confidence", "reasoning"],
            "additionalProperties": False,
        },
    },
}

RECIPE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "recipe",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
                "steps": {"type": "array", "items": {"type": "string"}},
                "prep_minutes": {"type": "integer"},
            },
            "required": ["name", "ingredients", "steps", "prep_minutes"],
            "additionalProperties": False,
        },
    },
}


def demo_strict_schema() -> None:
    """DEMO 1: Strict JSON schema enforced output."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "Analyze: 'This coffee is absolutely terrible and cold.'",
        }],
        response_format=SENTIMENT_SCHEMA,
    )
    raw = completion.choices[0].message.content or '{"sentiment":"neutral","confidence":0.5,"reasoning":"mock"}'
    try:
        data = json.loads(raw)
        print(f"  sentiment:  {data.get('sentiment')!r}")
        print(f"  confidence: {data.get('confidence')}")
        print(f"  reasoning:  {data.get('reasoning')!r}")
    except json.JSONDecodeError as e:
        print(f"  Parse error: {e}")


def demo_schema_structure() -> None:
    """DEMO 2: Print the full JSON schema structure for inspection."""
    print("  Sentiment schema:")
    print(f"    strict: {SENTIMENT_SCHEMA['json_schema']['strict']}")
    print(f"    required fields: {SENTIMENT_SCHEMA['json_schema']['schema']['required']}")
    print(f"    additionalProperties: {SENTIMENT_SCHEMA['json_schema']['schema']['additionalProperties']}")


def anti_pattern_missing_additional_properties() -> None:
    """ANTI-PATTERN 1: strict=True without additionalProperties: false.

    OpenAI requires additionalProperties: false when strict is enabled.
    Without it, the API returns a 400 error.
    """
    bad_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "bad_example",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                # MISSING: "additionalProperties": False
            },
        },
    }
    print("  ANTI-PATTERN: strict=True without additionalProperties: false")
    print("  → API returns 400: strict mode requires additionalProperties: false")
    print("  FIX: always add 'additionalProperties': false to strict schemas")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 3 - Task 3.2: JSON Schema Strict Mode [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Strict Schema Enforcement ---")
    demo_strict_schema()

    print(f"{NL}--- DEMO 2: Schema Structure ---")
    demo_schema_structure()

    print(f"{NL}--- ANTI-PATTERN 1: Missing additionalProperties ---")
    anti_pattern_missing_additional_properties()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. strict:true guarantees the model follows the schema exactly")
    print("  2. additionalProperties: false is REQUIRED with strict mode")
    print("  3. All properties must be in 'required' — no optional fields in strict mode")
    print("  4. Prefer Pydantic integration (Task 3.3) for Python-native schema definition")
