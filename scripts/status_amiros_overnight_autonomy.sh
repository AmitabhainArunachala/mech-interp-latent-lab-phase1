#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== tmux =="
tmux ls 2>/dev/null | grep 'amiros_overnight_autonomy' || true
echo
echo "== status =="
if [[ -f docs/status/AMIROS_OVERNIGHT_AUTONOMY_STATUS.md ]]; then
  tail -n 80 docs/status/AMIROS_OVERNIGHT_AUTONOMY_STATUS.md
else
  echo "no status file yet"
fi
echo
echo "== log =="
if [[ -f docs/status/AMIROS_OVERNIGHT_AUTONOMY_LOG.md ]]; then
  tail -n 80 docs/status/AMIROS_OVERNIGHT_AUTONOMY_LOG.md
else
  echo "no log file yet"
fi
