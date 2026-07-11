# Demo Script Rules

**Applies to:** `modules/module-*/0*.py` and `exercises/exercise-*/0*.py`

These rules are automatically injected when working on demo scripts.

## Required Conventions

### Module-level constants (always first after imports)
```python
NL = chr(10)   # NEVER use \n inline in f-strings
MODEL = "gpt-4o"
```

### Imports
```python
from shared.mock import get_client, is_mock
# NEVER: from openai import OpenAI  (except in anti-pattern sections)
```

### Docstring format (exact)
```python
"""Domain N - Task N.M: <Task Name>

CONCEPTS:
  1. concept_one — description
  2. concept_two — description
  3. concept_three — description

Mnemonic: WORD — W=thing, O=other, R=result, D=done

Run:
  uv run python NN_name.py
"""
```

### Function signatures
- Type hints required on ALL parameters and return types
- `Optional[X]` or `X | None` for optional values
- No bare `Any` types

### Data structures
- `dataclass` or `Pydantic BaseModel` for domain objects
- `TypedDict` for dict-shaped data with known keys
- Never raw `dict` for structured domain data

### Section labels
```python
def demo_basic_usage() -> None:
    """DEMO 1: Description of what this demonstrates."""
    ...

def anti_pattern_common_mistake() -> None:
    """ANTI-PATTERN 1: What NOT to do and why."""
    ...
```

### Main block
```python
if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain N - Task N.M: Name [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Name ---")
    demo_1()

    print(f"{NL}--- ANTI-PATTERN 1: Name ---")
    anti_pattern_1()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. First takeaway")
    print("  2. Second takeaway")
    print("  3. Third takeaway")
```

## Anti-Patterns to Avoid in Script Code
- `\n` inside f-strings (use `NL = chr(10)`)
- Hardcoded API keys
- `except Exception` swallowing errors silently
- `json.loads()` without try/except
- Fixed iteration cap as primary stopping condition
- Not checking `finish_reason` or `response.status`
