#!/usr/bin/env bash
# Post-tool-use hook: log tool usage with timestamp

LOG_FILE="$(dirname "$0")/../tool-usage.log"
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
TOOL_NAME="${CLAUDE_TOOL_NAME:-unknown}"
SUMMARY="${CLAUDE_TOOL_SUMMARY:-no summary}"

echo "${TIMESTAMP} | ${TOOL_NAME} | ${SUMMARY}" >> "$LOG_FILE"
