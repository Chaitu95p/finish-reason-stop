#!/usr/bin/env bash
# Pre-tool-use hook: block dangerous commands

TOOL_INPUT="${CLAUDE_TOOL_INPUT:-}"

check_blocked() {
  local input="$1"
  if echo "$input" | grep -qE 'rm\s+-rf|git\s+push\s+--force|pip\s+install|pip3\s+install|curl.*\|\s*bash|wget.*\|\s*bash'; then
    echo "[pre-tool-use] BLOCKED: dangerous command pattern detected." >&2
    echo "[pre-tool-use] Input: $input" >&2
    exit 2
  fi
}

check_blocked "$TOOL_INPUT"
