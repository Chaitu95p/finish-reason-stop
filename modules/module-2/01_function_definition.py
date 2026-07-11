"""Domain 2 - Task 2.1: Function Definition and Schema Design

CONCEPTS:
  1. Tool schema — name, description, parameters (JSON Schema object)
  2. Description quality — the model uses it to decide when/how to call
  3. Parameter types — string, number, boolean, enum via "enum" key
  4. Required vs optional — "required" array controls what model must fill

Mnemonic: SEEB — Schema, Exact_types, Enough_description, Behavior

Run:
  uv run python 01_function_definition.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json
from dataclasses import dataclass

from shared.mock import is_mock

# ---------------------------------------------------------------------------
# Mock tool implementations (pure Python, no external I/O)
# ---------------------------------------------------------------------------

@dataclass
class WeatherResult:
    city: str
    temperature_c: float
    condition: str
    humidity_pct: int


def get_weather(city: str, unit: str = "celsius") -> WeatherResult:
    """Mock weather lookup."""
    return WeatherResult(city=city, temperature_c=22.5, condition="sunny", humidity_pct=55)


def search_products(query: str, max_results: int = 5, category: str = "all") -> list[str]:
    """Mock product search."""
    return [f"Product {i+1} matching '{query}'" for i in range(min(max_results, 3))]


def send_email(to: str, subject: str, body: str, cc: str = "") -> bool:
    """Mock email sender."""
    print(f"    [MOCK EMAIL] To={to!r} Subject={subject!r}")
    return True


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

GOOD_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather for a specific city. "
            "Returns temperature, sky condition, and humidity. "
            "Use when the user asks about weather in a location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'London' or 'Tokyo'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

BAD_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Get weather",  # ANTI-PATTERN: too vague
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "string"},  # ANTI-PATTERN: unclear param name
            },
            "required": [],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_products",
        "description": (
            "Search the product catalog for items matching a query. "
            "Returns up to max_results product names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "max_results": {"type": "integer", "description": "Max items to return (1-20)"},
                "category": {
                    "type": "string",
                    "enum": ["all", "electronics", "clothing", "food"],
                    "description": "Product category filter",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def demo_tool_schemas() -> None:
    """DEMO 1: Print and compare tool schema structure."""
    print("  Good weather tool schema:")
    print(f"    name: {GOOD_WEATHER_TOOL['function']['name']!r}")
    print(f"    description: {GOOD_WEATHER_TOOL['function']['description']!r}")
    params = GOOD_WEATHER_TOOL["function"]["parameters"]["properties"]
    for pname, pdef in params.items():
        required = pname in GOOD_WEATHER_TOOL["function"]["parameters"].get("required", [])
        print(f"    param [{pname}]: type={pdef['type']!r} required={required}")

    print(f"{NL}  Search tool — required vs optional params:")
    required_params = SEARCH_TOOL["function"]["parameters"]["required"]
    all_params = SEARCH_TOOL["function"]["parameters"]["properties"].keys()
    for pname in all_params:
        req = pname in required_params
        print(f"    [{pname}]: {'required' if req else 'optional'}")


def anti_pattern_vague_schema() -> None:
    """ANTI-PATTERN 1: Vague descriptions and unclear parameter names.

    The model uses descriptions to understand WHEN and HOW to call the tool.
    Vague schemas lead to incorrect argument values or tool being skipped.
    """
    print("  BAD tool schema:")
    print(f"    name: {BAD_WEATHER_TOOL['function']['name']!r}")
    print(f"    description: {BAD_WEATHER_TOOL['function']['description']!r}")
    print("    → Model won't know city goes in 'x', may hallucinate or skip tool")
    print(f"{NL}  GOOD equivalent:")
    print(f"    description: {GOOD_WEATHER_TOOL['function']['description']!r}")
    print("    → Clear when to use, what 'city' means, enum for 'unit'")


def demo_schema_as_json() -> None:
    """DEMO 2: Serialize tool schema to JSON — what gets sent to the API."""
    schema_json = json.dumps(GOOD_WEATHER_TOOL, indent=2)
    print(f"  JSON schema (first 300 chars):{NL}{schema_json[:300]}...")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 2 - Task 2.1: Function Definition [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Tool Schema Structure ---")
    demo_tool_schemas()

    print(f"{NL}--- DEMO 2: Schema as JSON ---")
    demo_schema_as_json()

    print(f"{NL}--- ANTI-PATTERN 1: Vague Schema ---")
    anti_pattern_vague_schema()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Descriptions are prompts — write them for the model, not for humans")
    print("  2. Use 'enum' for categorical params — prevents model from hallucinating values")
    print("  3. Mark only truly required params as 'required'; rest are optional")
    print("  4. additionalProperties: false is required for strict mode structured outputs")
