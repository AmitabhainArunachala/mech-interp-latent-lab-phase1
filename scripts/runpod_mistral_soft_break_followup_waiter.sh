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
WAIT_LOG="${WAIT_LOG:-$REPO_ROOT/results/soft_break_followup_waiter.log}"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/results/mistral_soft_break_latebundle_sweep_v1}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WAIT_LOG"
}

latest_dir() {
  ls -1dt "$1"/* 2>/dev/null | head -n 1
}

wait_for_summary() {
  local root="$1"
  local label="$2"
  while true; do
    local latest
    latest="$(latest_dir "$root")"
    if [[ -n "${latest:-}" && -f "$latest/summary.json" ]]; then
      echo "$latest/summary.json"
      return 0
    fi
    log "waiting_for_${label}"
    sleep 20
  done
}

run_and_wait() {
  local session_name="$1"
  local command="$2"
  local result_root="$3"
  tmux new-session -d -s "$session_name" "$command"
  log "launched_${session_name}"
  wait_for_summary "$result_root" "$session_name" >/dev/null
  log "finished_${session_name}"
}

mkdir -p "$(dirname "$WAIT_LOG")"
log "waiter_started"

SWEEP_SUMMARY="$(wait_for_summary "$SWEEP_ROOT" "soft_break_sweep")"
BEST_SCALE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"]["best_scale"])' "$SWEEP_SUMMARY")"
log "best_scale=${BEST_SCALE}"

run_and_wait \
  "mistral_soft_break_tokenwindow" \
  "cd $REPO_ROOT && export AMIROS_POD_NAME=${AMIROS_POD_NAME:-$(hostname)} AMIROS_HOST=${AMIROS_HOST:-$(hostname)} AMIROS_PORT=${AMIROS_PORT:-22} AMIROS_SESSION=mistral_soft_break_tokenwindow SCALE=$BEST_SCALE; bash scripts/runpod_mistral_soft_break_tokenwindow_v1_queue.sh" \
  "$REPO_ROOT/results/mistral_soft_break_tokenwindow_v1"

run_and_wait \
  "mistral_soft_break_factorized" \
  "cd $REPO_ROOT && export AMIROS_POD_NAME=${AMIROS_POD_NAME:-$(hostname)} AMIROS_HOST=${AMIROS_HOST:-$(hostname)} AMIROS_PORT=${AMIROS_PORT:-22} AMIROS_SESSION=mistral_soft_break_factorized SCALE=$BEST_SCALE; bash scripts/runpod_mistral_soft_break_factorized_v1_queue.sh" \
  "$REPO_ROOT/results/mistral_soft_break_factorized_v1"

log "waiter_done"
