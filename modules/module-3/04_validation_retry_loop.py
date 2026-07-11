"""Domain 3 - Task 3.4: Validation and Retry Loop

CONCEPTS:
  1. pydantic.ValidationError — raised on schema mismatch
  2. Retry with error context — re-prompt model with the validation error
  3. Max retry limit — prevent infinite loops on persistent failures
  4. model_validate vs model_construct — strict vs lenient

Mnemonic: VRML — Validate, Retry_with_error, Max_retries, Lenient_fallback

Run:
  uv run python 04_validation_retry_loop.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError
from shared.mock import get_client, is_mock

T = TypeVar("T", bound=BaseModel)


class ProductInfo(BaseModel):
    name: str
    price_usd: float = Field(ge=0.0)
    in_stock: bool
    categories: list[str]


def parse_with_retry[T: BaseModel](
    model_class: type[T],
    prompt: str,
    max_retries: int = 3,
) -> T | None:
    """Call chat completions with JSON mode, validate with Pydantic, retry on failure."""
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": prompt}]
    schema_str = json.dumps(model_class.model_json_schema(), indent=2)

    for attempt in range(1, max_retries + 1):
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        print(f"  Attempt {attempt}: raw={raw!r}")

        try:
            data = json.loads(raw)
            instance = model_class.model_validate(data)
            print(f"  Attempt {attempt}: validation PASSED")
            return instance
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {e}"
        except ValidationError as e:
            error_msg = f"Schema mismatch: {e.errors()}"

        print(f"  Attempt {attempt}: FAILED — {error_msg}")
        # Re-prompt with error context
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                f"Your response failed validation: {error_msg}. "
                f"Please return valid JSON matching this schema: {schema_str}"
            ),
        })

    print(f"  All {max_retries} attempts failed")
    return None


def demo_retry_loop() -> None:
    """DEMO 1: Retry loop with validation feedback."""
    prompt = (
        "Extract product info from: 'Laptop Pro X, $999.99, in stock, categories: electronics, computers. "
        "Return JSON with: name, price_usd, in_stock, categories."
    )
    result = parse_with_retry(ProductInfo, prompt, max_retries=2)
    if result:
        print(f"  Success: name={result.name!r} price={result.price_usd}")
    else:
        print("  Failed after retries (expected in mock mode)")


def demo_validate_vs_construct() -> None:
    """DEMO 2: model_validate (strict) vs model_construct (lenient)."""
    good_data = {"name": "Widget", "price_usd": 9.99, "in_stock": True, "categories": ["tools"]}
    bad_data = {"name": "Widget", "price_usd": -5.0, "in_stock": "yes", "categories": "tools"}

    # Strict validation
    try:
        strict = ProductInfo.model_validate(bad_data)
        print(f"  model_validate passed (unexpected): {strict}")
    except ValidationError as e:
        print(f"  model_validate caught {len(e.errors())} error(s): {[err['msg'] for err in e.errors()]}")

    # Lenient construction (skips validators)
    lenient = ProductInfo.model_construct(**good_data)
    print(f"  model_construct (no validation): {lenient}")


def anti_pattern_silent_failure() -> None:
    """ANTI-PATTERN 1: Accepting invalid data without validation.

    Using raw dict values without validation means type errors
    propagate silently downstream — hard to debug.
    """
    raw = '{"name": "Widget", "price_usd": "free", "in_stock": "yes", "categories": "all"}'
    data = json.loads(raw)
    # WRONG: no validation
    print(f"  ANTI-PATTERN: price_usd is {data['price_usd']!r} (string, not float)")
    print("  This causes runtime errors when you try to use it in arithmetic")

    # CORRECT: validate immediately
    try:
        ProductInfo.model_validate(data)
    except ValidationError as e:
        print(f"  CORRECT: caught before it propagates: {len(e.errors())} error(s)")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 3 - Task 3.4: Validation Retry Loop [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Retry Loop ---")
    demo_retry_loop()

    print(f"{NL}--- DEMO 2: Validate vs Construct ---")
    demo_validate_vs_construct()

    print(f"{NL}--- ANTI-PATTERN 1: Silent Failure ---")
    anti_pattern_silent_failure()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Retry with error context: append the error to messages before re-calling")
    print("  2. Max retry limit is essential — don't loop forever on a bad prompt")
    print("  3. model_validate() raises on bad data; model_construct() skips validators")
    print("  4. Validate at the boundary: catch ValidationError right after json.loads()")
