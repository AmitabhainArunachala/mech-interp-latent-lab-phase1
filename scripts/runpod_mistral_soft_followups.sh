#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WATCH_PID="${1:-}"

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
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_soft_followups/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "watch_pid=${WATCH_PID:-none}" | tee -a "$STATUS_FILE"
echo "provenance_note=base_v01_soft_followups_l4_mlp_then_l5_soft_residual" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

wait_for_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return
  fi
  echo "waiting_for_pid=$pid" | tee -a "$STATUS_FILE"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
  echo "watch_pid_completed=$pid" | tee -a "$STATUS_FILE"
}

run_step() {
  local name="$1"
  shift
  echo "" | tee -a "$STATUS_FILE"
  echo ">>> START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo ">>> DONE  $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  else
    local rc=$?
    echo ">>> FAIL  $name rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
    exit "$rc"
  fi
}

wait_for_pid "$WATCH_PID"

run_step l4_mlp_l25_bridge_soft \
  "$PYTHON_BIN" -u -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l4_mlp_l25_bridge_soft.json

run_step l5_soft_gate_l25_bridge \
  "$PYTHON_BIN" -u -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l5_soft_gate_l25_bridge.json

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_soft_followups_complete=1" | tee -a "$STATUS_FILE"
