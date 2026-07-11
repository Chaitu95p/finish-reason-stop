# New Demo

Scaffold a new numbered demo script in a module directory.

**Usage:** `/new-demo <module-dir> <domain.task>`

**Example:** `/new-demo modules/module-1 1.6` creates `modules/module-1/06_new_demo.py`

**What it does:**
1. Finds the next available script number in the module dir
2. Creates a new script file from the canonical template
3. Fills in the docstring with the domain/task number
4. Leaves `# TODO` markers for content

**Template applied:**
```python
"""Domain N - Task N.M: <Task Name>

CONCEPTS:
  1. concept_1
  2. concept_2
  3. concept_3

Mnemonic: WORD — W=thing, O=other, R=result, D=done

Run:
  uv run python NN_name.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock


def demo_1() -> None:
    """DEMO 1: Description."""
    client = get_client()
    # TODO: implement demo


def anti_pattern_1() -> None:
    """ANTI-PATTERN 1: What not to do."""
    # TODO: implement anti-pattern


if __name__ == "__main__":
    sep = "=" * 60
    print(f"{NL}{sep}{NL}Domain N - Task N.M: Name{NL}{sep}")
    demo_1()
    anti_pattern_1()
    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. ...")
    print("  2. ...")
    print("  3. ...")
```
