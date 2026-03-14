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
WATCH_PID="${1:-}"
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_l4_confirmation_queue/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "watch_pid=${WATCH_PID:-none}" | tee -a "$STATUS_FILE"
echo "provenance_note=base_v01_l4_confirmation_after_post_subtle_queue" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

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

if [[ -n "$WATCH_PID" ]]; then
  echo "waiting_for_pid=$WATCH_PID" | tee -a "$STATUS_FILE"
  while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 15
  done
  echo "watch_pid_completed=$WATCH_PID" | tee -a "$STATUS_FILE"
fi

run_step l4_confirmation_window8 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l4_mlp_confirmation_window8.json

run_step summarize_l4_confirmation_window8 \
  "$PYTHON_BIN" scripts/summarize_micro_window_multisite.py \
  --run-dir "$(ls -1dt results/phase1_mechanism/runs/*mistral_multisite_l4_mlp_confirmation_window8 | head -n 1)"

run_step rank_l4_confirmation_window8 \
  "$PYTHON_BIN" scripts/analyze_multisite_subtlety.py \
  "$(ls -1dt results/phase1_mechanism/runs/*mistral_multisite_l4_mlp_confirmation_window8 | head -n 1)/summary.json" \
  --top-k 10

run_step l4_confirmation_window4 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l4_mlp_confirmation_window4.json

run_step summarize_l4_confirmation_window4 \
  "$PYTHON_BIN" scripts/summarize_micro_window_multisite.py \
  --run-dir "$(ls -1dt results/phase1_mechanism/runs/*mistral_multisite_l4_mlp_confirmation_window4 | head -n 1)"

run_step rank_l4_confirmation_window4 \
  "$PYTHON_BIN" scripts/analyze_multisite_subtlety.py \
  "$(ls -1dt results/phase1_mechanism/runs/*mistral_multisite_l4_mlp_confirmation_window4 | head -n 1)/summary.json" \
  --top-k 10

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_l4_confirmation_queue_complete=1" | tee -a "$STATUS_FILE"
