#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOURS="${HOURS:-8}"
SESSION="${SESSION:-amiros_overnight_autonomy}"
CAFFEINATE_SESSION="${CAFFEINATE_SESSION:-amiros_overnight_autonomy_caffeinate}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
else
  tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && python3 scripts/amiros_overnight_operator.py --max-hours '$HOURS'"
  echo "started tmux session: $SESSION"
fi

if command -v caffeinate >/dev/null 2>&1; then
  if tmux has-session -t "$CAFFEINATE_SESSION" 2>/dev/null; then
    echo "tmux session $CAFFEINATE_SESSION already exists"
  else
    SECONDS_TOTAL="$(python3 - <<PY
hours = float(${HOURS})
print(int(hours * 3600))
PY
)"
    tmux new-session -d -s "$CAFFEINATE_SESSION" "caffeinate -dimsu -t '$SECONDS_TOTAL'"
    echo "started caffeinate session: $CAFFEINATE_SESSION"
  fi
fi

echo "operator log: docs/status/AMIROS_OVERNIGHT_AUTONOMY_LOG.md"
echo "operator status: docs/status/AMIROS_OVERNIGHT_AUTONOMY_STATUS.md"
