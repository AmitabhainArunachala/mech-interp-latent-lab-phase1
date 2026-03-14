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

AMIROS_POD_NAME="${AMIROS_POD_NAME:-$(hostname)}"
AMIROS_HOST="${AMIROS_HOST:-$(hostname)}"
AMIROS_PORT="${AMIROS_PORT:-22}"
AMIROS_SESSION="${AMIROS_SESSION:-mistral_l4_micro4_confirm_v1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_l4_micro4_confirm_v1/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group mistral_l4_micro4_confirm_v1 \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Focused L4 micro4 confirmatory lane against bridge-only baselines"

run_step() {
  local name="$1"
  shift
  echo "" | tee -a "$STATUS_FILE"
  echo ">>> START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  "$PYTHON_BIN" -m src.utils.research_os lease-update \
    --pod-name "$AMIROS_POD_NAME" \
    --host "$AMIROS_HOST" \
    --port "$AMIROS_PORT" \
    --session-name "$AMIROS_SESSION" \
    --queue-group mistral_l4_micro4_confirm_v1 \
    --run-id "$RUN_ID" \
    --status running \
    --current-step "$name" \
    --out-dir "${OUT_DIR#$REPO_ROOT/}"
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo ">>> DONE  $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  else
    local rc=$?
    echo ">>> FAIL  $name rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
    "$PYTHON_BIN" -m src.utils.research_os lease-update \
      --pod-name "$AMIROS_POD_NAME" \
      --host "$AMIROS_HOST" \
      --port "$AMIROS_PORT" \
      --session-name "$AMIROS_SESSION" \
      --queue-group mistral_l4_micro4_confirm_v1 \
      --run-id "$RUN_ID" \
      --status failed \
      --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

capture_run_dir() {
  local run_name="$1"
  "$PYTHON_BIN" - "$run_name" <<'PY'
import sys
from pathlib import Path

run_name = sys.argv[1]
runs_root = Path("results/phase1_mechanism/runs")
matches = sorted([p for p in runs_root.glob(f"*{run_name}*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit(f"no run dir found for {run_name}")
print(matches[-1])
PY
}

run_step l4_micro4_confirmatory_focus_v1 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l4_mlp_micro4_confirmatory_focus_v1.json

RUN_DIR="$(capture_run_dir mistral_multisite_l4_mlp_micro4_confirmatory_focus_v1)"
printf '%s\n' "$RUN_DIR" | tee "$OUT_DIR/run_dir.log"

run_step summarize_l4_micro4_confirmatory_focus_v1 \
  "$PYTHON_BIN" scripts/summarize_micro_window_multisite.py \
  --run-dir "$RUN_DIR"

run_step rank_l4_micro4_confirmatory_focus_v1 \
  "$PYTHON_BIN" scripts/analyze_multisite_subtlety.py \
  "$RUN_DIR/summary.json" \
  --top-k 10

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group mistral_l4_micro4_confirm_v1 \
  --experiment-id l4_micro4_confirmatory_focus_v1 \
  --status completed \
  --artifact-path "$RUN_DIR/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path configs/canonical/causal_state_benchmark_v4_multisite_mistral_l4_mlp_micro4_confirmatory_focus_v1.json \
  --prompt-contract heldout_causal_slice \
  --metric-path "causal_state_benchmark_v4_multisite + summarize_micro_window_multisite" \
  --claim-id L4_MICRO4_CONFIRM_V1

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_l4_micro4_confirm_v1_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group mistral_l4_micro4_confirm_v1 \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
