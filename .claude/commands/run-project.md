# Run Project

Run a mini-project's main entry point.

**Usage:** `/run-project N`

**Example:** `/run-project 1` runs `projects/project-1-chatbot/main.py`

**What it does:**
1. Finds `projects/project-N-*/main.py`
2. Prints the project README summary
3. Runs `uv run python main.py` from the project directory
4. Shows mock mode banner if no API key

**Command:**
```bash
PROJECT=$(ls -d projects/project-$ARGUMENTS-*/ | head -1) && echo "=== $(cat ${PROJECT}README.md | head -5) ===" && cd "$PROJECT" && uv run python main.py
```

**Mock mode:** All projects support mock mode — runnable without OPENAI_API_KEY.
