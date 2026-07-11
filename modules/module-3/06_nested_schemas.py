"""Domain 3 - Task 3.6: Nested Schemas and Discriminated Unions

CONCEPTS:
  1. Nested Pydantic models — model fields that reference other models
  2. list[Model] — arrays of typed objects
  3. Literal types for discriminated unions — type-safe variant selection
  4. model_dump() and model_json_schema() — serialization and introspection

Mnemonic: NALD — Nested, Arrays, Literal_union, Dump

Run:
  uv run python 06_nested_schemas.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from shared.mock import is_mock

# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str


class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    email: str
    address: Address


class Team(BaseModel):
    name: str
    department: str
    members: list[Person]
    budget_usd: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    type: Literal["text"]
    content: str
    language: str = "en"


class ImageBlock(BaseModel):
    type: Literal["image"]
    url: str
    alt_text: str
    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)


class CodeBlock(BaseModel):
    type: Literal["code"]
    code: str
    language: str
    filename: str | None = None


ContentBlock = Annotated[
    TextBlock | ImageBlock | CodeBlock,
    Field(discriminator="type"),
]


class Document(BaseModel):
    title: str
    blocks: list[ContentBlock]


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_nested_model() -> None:
    """DEMO 1: Create and serialize a nested Pydantic model."""
    team = Team(
        name="ML Platform",
        department="Engineering",
        budget_usd=500000.0,
        members=[
            Person(
                name="Alice Chen",
                age=32,
                email="alice@example.com",
                address=Address(street="123 Main St", city="San Francisco", country="US", postal_code="94102"),
            ),
            Person(
                name="Bob Kim",
                age=28,
                email="bob@example.com",
                address=Address(street="456 Oak Ave", city="Seattle", country="US", postal_code="98101"),
            ),
        ],
    )
    print(f"  Team: {team.name!r}, {len(team.members)} members")
    print(f"  First member: {team.members[0].name!r} in {team.members[0].address.city!r}")
    dumped = team.model_dump()
    print(f"  Serialized keys: {list(dumped.keys())}")
    print(f"  Member[0] address keys: {list(dumped['members'][0]['address'].keys())}")


def demo_discriminated_union() -> None:
    """DEMO 2: Create a document with mixed content block types."""
    doc = Document(
        title="API Guide",
        blocks=[
            TextBlock(type="text", content="Welcome to our API.", language="en"),
            CodeBlock(type="code", code="client.chat.completions.create(...)", language="python", filename="example.py"),
            ImageBlock(type="image", url="https://example.com/diagram.png", alt_text="Architecture", width_px=800, height_px=600),
            TextBlock(type="text", content="See the diagram above.", language="en"),
        ],
    )
    print(f"  Document: {doc.title!r}, {len(doc.blocks)} blocks")
    for i, block in enumerate(doc.blocks):
        print(f"    block[{i}]: type={block.type!r}")


def demo_json_schema() -> None:
    """DEMO 3: Inspect auto-generated JSON schema for the Team model."""
    schema = Team.model_json_schema()
    print(f"  Team schema title: {schema.get('title')!r}")
    print(f"  Team properties: {list(schema.get('properties', {}).keys())}")
    members_schema = schema.get("properties", {}).get("members", {})
    print(f"  members field: {members_schema.get('type')!r} of items")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 3 - Task 3.6: Nested Schemas [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Nested Model ---")
    demo_nested_model()

    print(f"{NL}--- DEMO 2: Discriminated Union ---")
    demo_discriminated_union()

    print(f"{NL}--- DEMO 3: JSON Schema Introspection ---")
    demo_json_schema()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Nested Pydantic models auto-generate nested JSON schemas")
    print("  2. list[Model] maps to 'array' with 'items' referencing the model schema")
    print("  3. Discriminated unions use Literal[...] on a shared 'type' field")
    print("  4. model_dump() → dict, model_json_schema() → the schema sent to the API")
