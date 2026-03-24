#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pick_python() {
  if [[ -x /root/venvs/mistral-hardening/bin/python ]]; then
    echo "/root/venvs/mistral-hardening/bin/python"
    return
  fi
  if [[ -x ./.venv/bin/python ]]; then
    echo "./.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "python"
}

PYTHON_BIN="$(pick_python)"
export PYTHONPATH="${PYTHONPATH:-.}"

SCALE="${SCALE:-1.0}"
TOKEN_WINDOW="${TOKEN_WINDOW:-2}"

export SCALE
export TOKEN_WINDOW

bash scripts/runpod_mistral_soft_break_factorized_v1_queue.sh

FACTOR_SUMMARY="$(ls -1dt results/mistral_soft_break_factorized_v1/* | head -n 1)/summary.json"
CONDITION_NAME="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"]["best_condition"])' "$FACTOR_SUMMARY")"
export CONDITION_NAME

bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh
