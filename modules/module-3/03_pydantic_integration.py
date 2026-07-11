"""Domain 3 - Task 3.3: Pydantic Integration with .parse()

CONCEPTS:
  1. client.beta.chat.completions.parse() — native Pydantic support
  2. response_format=PydanticModel — auto schema generation from model
  3. message.parsed — the validated Pydantic instance (typed access)
  4. Pydantic v2 — BaseModel, Field with constraints, validators

Mnemonic: PPBV — Parse, Pydantic_model, Beta_client, Validated_instance

Run:
  uv run python 03_pydantic_integration.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from typing import Literal

from pydantic import BaseModel, Field
from shared.mock import get_client, is_mock


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_phrases: list[str]


class RecipeExtraction(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    prep_minutes: int = Field(ge=1)
    difficulty: Literal["easy", "medium", "hard"]


def demo_parse_sentiment() -> None:
    """DEMO 1: Parse sentiment analysis into a typed Pydantic model."""
    client = get_client()
    # beta.chat.completions.parse() accepts a Pydantic model class as response_format
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "Analyze: 'The new update is fantastic and works perfectly!'",
        }],
        response_format=SentimentAnalysis,
    )
    # message.parsed is the validated SentimentAnalysis instance
    parsed = completion.choices[0].message
    print(f"  finish_reason: {completion.choices[0].finish_reason!r}")
    # In live mode: parsed.parsed is a SentimentAnalysis instance
    # In mock mode: we show the schema that would be generated
    schema = SentimentAnalysis.model_json_schema()
    print(f"  Schema name: {schema.get('title')!r}")
    print(f"  Schema fields: {list(schema.get('properties', {}).keys())}")
    print("  (Live mode: completion.choices[0].message.parsed → SentimentAnalysis instance)")


def demo_model_schema() -> None:
    """DEMO 2: Inspect auto-generated JSON schema from Pydantic model."""
    schema = SentimentAnalysis.model_json_schema()
    print("  SentimentAnalysis schema:")
    print(f"    type: {schema['type']!r}")
    print(f"    required: {schema.get('required', [])}")
    for field_name, field_def in schema.get("properties", {}).items():
        print(f"    [{field_name}]: {field_def}")


def demo_model_validation() -> None:
    """DEMO 3: Pydantic validation catches bad data before it reaches your code."""
    try:
        bad = SentimentAnalysis(
            sentiment="very_positive",  # not in Literal
            confidence=1.5,  # exceeds le=1.0
            reasoning="test",
            key_phrases=[],
        )
        print(f"  Unexpectedly passed: {bad}")
    except Exception as e:
        print(f"  Validation caught bad data: {type(e).__name__}")
        print("  (This is the correct behavior)")

    good = SentimentAnalysis(
        sentiment="positive",
        confidence=0.95,
        reasoning="Uses positive adjectives",
        key_phrases=["fantastic", "perfectly"],
    )
    print(f"  Valid instance: sentiment={good.sentiment!r}, confidence={good.confidence}")


def anti_pattern_manual_json_parse() -> None:
    """ANTI-PATTERN 1: Manually json.loads() instead of using .parse().

    Manual parsing skips type validation and gives you raw dicts.
    .parse() gives you a typed, validated Python object.
    """
    import json
    raw_json = '{"sentiment": "positive", "confidence": 0.9, "reasoning": "good", "key_phrases": []}'
    # WRONG:
    data = json.loads(raw_json)
    print(f"  ANTI-PATTERN: raw dict, no type safety: {type(data)}")
    print(f"  confidence type: {type(data['confidence'])}")  # float, but no bounds check

    # CORRECT: validate with Pydantic
    model = SentimentAnalysis.model_validate(data)
    print(f"  CORRECT: typed model: {type(model).__name__}")
    print(f"  confidence: {model.confidence} (ge=0.0, le=1.0 enforced)")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 3 - Task 3.3: Pydantic Integration [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Parse Sentiment ---")
    demo_parse_sentiment()

    print(f"{NL}--- DEMO 2: Model Schema ---")
    demo_model_schema()

    print(f"{NL}--- DEMO 3: Validation ---")
    demo_model_validation()

    print(f"{NL}--- ANTI-PATTERN 1: Manual JSON Parse ---")
    anti_pattern_manual_json_parse()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. client.beta.chat.completions.parse(response_format=Model) → typed output")
    print("  2. message.parsed holds the validated Pydantic instance — no json.loads() needed")
    print("  3. Pydantic v2 Field(ge=0.0, le=1.0) enforces numeric bounds automatically")
    print("  4. model_json_schema() generates the JSON Schema that gets sent to the API")
