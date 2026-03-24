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
WAIT_LOG="${WAIT_LOG:-$REPO_ROOT/results/soft_break_targeted_confirm_waiter.log}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WAIT_LOG"
}

latest_summary() {
  local root="$1"
  local latest
  latest="$(ls -1dt "$root"/* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest:-}" && -f "$latest/summary.json" ]]; then
    echo "$latest/summary.json"
  fi
}

wait_for_summary() {
  local root="$1"
  local label="$2"
  while true; do
    local summary
    summary="$(latest_summary "$root")"
    if [[ -n "${summary:-}" ]]; then
      echo "$summary"
      return 0
    fi
    log "waiting_for_${label}"
    sleep 20
  done
}

mkdir -p "$(dirname "$WAIT_LOG")"
log "targeted_confirm_waiter_started"

TOKEN_SUMMARY="$(wait_for_summary "$REPO_ROOT/results/mistral_soft_break_tokenwindow_v1" tokenwindow)"
FACTOR_SUMMARY="$(wait_for_summary "$REPO_ROOT/results/mistral_soft_break_factorized_v1" factorized)"

TOKEN_WINDOW="$("$PYTHON_BIN" -c 'import json,sys; name=json.load(open(sys.argv[1]))["verdict"]["best_condition"]; print("full" if name=="anti_late_full" else name.replace("anti_late_last",""))' "$TOKEN_SUMMARY")"
SCALE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scale"])' "$FACTOR_SUMMARY")"
CONDITION_NAME="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"]["best_condition"])' "$FACTOR_SUMMARY")"

log "selected_scale=${SCALE}"
log "selected_token_window=${TOKEN_WINDOW}"
log "selected_condition=${CONDITION_NAME}"

tmux new-session -d -s mistral_soft_break_targeted_confirm \
  "cd $REPO_ROOT && export AMIROS_POD_NAME=${AMIROS_POD_NAME:-$(hostname)} AMIROS_HOST=${AMIROS_HOST:-$(hostname)} AMIROS_PORT=${AMIROS_PORT:-22} AMIROS_SESSION=mistral_soft_break_targeted_confirm SCALE=$SCALE TOKEN_WINDOW=$TOKEN_WINDOW CONDITION_NAME=$CONDITION_NAME; bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh"

log "targeted_confirm_launched"
