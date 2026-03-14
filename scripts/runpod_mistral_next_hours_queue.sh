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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_next_hours_queue/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "watch_pid=${WATCH_PID:-none}" | tee -a "$STATUS_FILE"
echo "provenance_note=base_v01_next_hours_queue_l4_mlp_then_targeted_head_to_head" | tee -a "$STATUS_FILE"
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

run_step targeted_head_to_head_l27h10_bundle \
  env \
  MODEL="mistralai/Mistral-7B-v0.1" \
  DEVICE="cuda" \
  HEAD_SWEEP_FILE="results/full_head_sweep/full_head_sweep_20260312_052013.json" \
  N_PROMPTS="40" \
  TOP_K="8" \
  MAX_PAIRS="15" \
  PAIR_SOURCE="top_effect" \
  PAIR_POOL_SIZE="8" \
  RANKING_METRIC="rank_d" \
  SINGLE_HEAD_THRESHOLD="0.3" \
  MANUAL_HEADS="L27.H10,L27.H2,L27.H18,L27.H26,L27.H5,L18.H1,L28.H3,L19.H1" \
  bash scripts/runpod_head_to_head_base.sh

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_next_hours_queue_complete=1" | tee -a "$STATUS_FILE"
