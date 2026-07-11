# Run Script

Run a single demo script and print a concept summary.

**Usage:** `/run-script <path>`

**Example:** `/run-script modules/module-1/01_responses_api_basics.py`

**What it does:**
1. Prints the script's docstring (domain, concepts, mnemonic)
2. Runs the script with `uv run python`
3. Shows the output including KEY TAKEAWAYS

**Command:**
```bash
python -c "import ast, sys; src=open('$ARGUMENTS').read(); tree=ast.parse(src); print(ast.get_docstring(tree))" && echo "---" && uv run python $ARGUMENTS
```

**Mock mode:** Scripts auto-detect missing `OPENAI_API_KEY` and use mock responses.
