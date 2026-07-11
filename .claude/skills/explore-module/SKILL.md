# Explore Module Skill

Reads all scripts in a module and produces a structured study summary.

**Trigger:** User asks to explore, study, or summarize a module.

**Example:** "Explore module 2" or "Give me a study guide for module 4"

## What to do

1. List all `0*.py` scripts in `modules/module-N/`
2. Read each script's docstring (concepts, mnemonic)
3. Identify the key patterns demonstrated
4. Produce a study summary with:
   - Module overview (1 paragraph)
   - Script-by-script concept table
   - Key SDK patterns covered
   - Common anti-patterns to avoid
   - Suggested learning order
   - Quiz-ready questions (3-5 per script)

## Output format

```
# Module N Study Guide: <Module Name>

## Overview
[1 paragraph]

## Scripts
| Script | Concept | Mnemonic |
|--------|---------|----------|
| 01_... | ...     | ...      |

## Key Patterns
- Pattern 1: ...
- Pattern 2: ...

## Anti-Patterns to Avoid
- ...

## Self-Test Questions
1. ...
```

**Context:** Run in fork mode to keep main context clean.
