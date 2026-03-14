#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION_NAME="${SESSION_NAME:-amiros_operator}"
POLL_SECONDS="${POLL_SECONDS:-300}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ok] tmux session '$SESSION_NAME' already exists"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" "cd '$REPO_ROOT' && POLL_SECONDS='$POLL_SECONDS' bash scripts/amiros_operator_loop.sh"
echo "[ok] started tmux session '$SESSION_NAME'"
