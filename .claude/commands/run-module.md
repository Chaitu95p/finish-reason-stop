# Run Module

Run all scripts in a module in order.

**Usage:** `/run-module N`

**Example:** `/run-module 1` runs all scripts in modules/module-1/

**What it does:**
1. Lists all `0*.py` scripts in `modules/module-N/` sorted by filename
2. Runs each with `uv run python <script>` from the module directory
3. Prints a separator between each script
4. Reports PASS/FAIL for each

**Command:**
```bash
cd modules/module-$ARGUMENTS && for f in $(ls 0*.py | sort); do echo "=== $f ==="; uv run python "$f"; done
```

**Mock mode:** If `OPENAI_API_KEY` is not set, all scripts run in mock mode automatically. No API calls are made.

**Notes:**
- Scripts are numbered `01_`, `02_`, etc. and run in that order
- Each script is self-contained and runnable independently
- Use `/run-script` to run a single script with more detail
