# Review Script

Check a demo script for convention compliance.

**Usage:** `/review-script <path>`

**Example:** `/review-script modules/module-1/01_responses_api_basics.py`

**Checks performed:**
1. Docstring format — Domain N - Task N.M, CONCEPTS, Mnemonic, Run
2. `NL = chr(10)` present at module level
3. `MODEL = "gpt-4o"` present at module level
4. `from shared.mock import get_client` (not raw openai import)
5. Type hints on all function signatures
6. `if __name__ == "__main__":` block present
7. `KEY TAKEAWAYS:` printed in main block
8. No `\n` inline in f-strings
9. Ruff check passes

**Command:**
Read the file and check each convention, then run `uv run ruff check <path>`.
