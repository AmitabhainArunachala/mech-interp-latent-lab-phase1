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
OUT_DIR="$REPO_ROOT/results/mistral_base_priority_queue/$RUN_ID"
mkdir -p "$OUT_DIR"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
DEVICE="${DEVICE:-cuda}"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "model=$MODEL" | tee -a "$STATUS_FILE"
echo "device=$DEVICE" | tee -a "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "watch_pid=${WATCH_PID:-none}" | tee -a "$STATUS_FILE"
echo "provenance_note=base_v01_priority_queue_after_soft_followups" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

wait_for_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return
  fi
  echo "waiting_for_pid=$pid" | tee -a "$STATUS_FILE"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
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

run_step p0_canonical_n100 \
  "$PYTHON_BIN" -u scripts/p0_canonical_pipeline.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n 100

run_step full_head_sweep_n100 \
  "$PYTHON_BIN" -u scripts/full_head_sweep.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-prompts 100 \
    --batch-layers 8

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_base_priority_queue_complete=1" | tee -a "$STATUS_FILE"
