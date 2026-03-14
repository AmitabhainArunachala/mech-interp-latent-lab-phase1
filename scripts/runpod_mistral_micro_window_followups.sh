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
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_micro_window_followups/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "provenance_note=base_v01_micro_window4_l4_mlp_and_l5_residual_with_l25_bridge" | tee -a "$STATUS_FILE"
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

run_step l4_mlp_micro4_l25_bridge \
  "$PYTHON_BIN" -u -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l4_mlp_micro4_l25_bridge.json

run_step summarize_l4_mlp_micro4_l25_bridge \
  "$PYTHON_BIN" scripts/summarize_micro_window_multisite.py \
  --run-dir "$(ls -1dt results/phase1_mechanism/runs/*mistral_multisite_l4_mlp_micro4_l25_bridge | head -n 1)"

run_step l5_resid_micro4_l25_bridge \
  "$PYTHON_BIN" -u -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l5_resid_micro4_l25_bridge.json

run_step summarize_l5_resid_micro4_l25_bridge \
  "$PYTHON_BIN" scripts/summarize_micro_window_multisite.py \
  --run-dir "$(ls -1dt results/phase1_mechanism/runs/*mistral_multisite_l5_resid_micro4_l25_bridge | head -n 1)"

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_micro_window_followups_complete=1" | tee -a "$STATUS_FILE"
