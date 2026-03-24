#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8000}"
MAX_PORT="${MAX_PORT:-8100}"

find_free_port() {
  local port="$1"
  while [[ "$port" -le "$MAX_PORT" ]]; do
    if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "$port"
      return 0
    fi
    port=$((port + 1))
  done
  return 1
}

cd "$REPO_ROOT"
python3 scripts/build_mistral_control_atlas.py >/dev/null
cd website

FREE_PORT="$(find_free_port "$PORT")" || {
  echo "No free port found in range ${PORT}-${MAX_PORT}" >&2
  exit 1
}

if [[ "$FREE_PORT" != "$PORT" ]]; then
  echo "Port ${PORT} is busy, using ${FREE_PORT} instead."
fi

echo "Serving Mistral Control Atlas at http://127.0.0.1:${FREE_PORT}"
python3 -m http.server "$FREE_PORT"
