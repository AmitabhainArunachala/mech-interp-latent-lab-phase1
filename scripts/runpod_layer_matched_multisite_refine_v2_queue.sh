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
AMIROS_SESSION="${AMIROS_SESSION:-layer_matched_multisite_refine_v2}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/layer_matched_multisite_refine_v2/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "experiment=layer_matched_multisite_refine_v2" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group layer_matched_multisite_refine_v2 \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Refine the winning low-dose layer-matched bundle with a softer L25 residual bridge sweep and low-dose mean-diff control."

echo ">>> START layer_matched_multisite_refine_v2 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"

if "$PYTHON_BIN" scripts/layer_matched_multisite_refine_v2.py \
    --device cuda \
    --out-dir "$OUT_DIR" \
    --seeds 101 202 303 404 505 606 707 808 \
    2>&1 | tee "$OUT_DIR/experiment.log"; then
  echo ">>> DONE  layer_matched_multisite_refine_v2 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
else
  rc=$?
  echo ">>> FAIL  layer_matched_multisite_refine_v2 rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  "$PYTHON_BIN" -m src.utils.research_os lease-update \
    --pod-name "$AMIROS_POD_NAME" \
    --host "$AMIROS_HOST" \
    --port "$AMIROS_PORT" \
    --session-name "$AMIROS_SESSION" \
    --queue-group layer_matched_multisite_refine_v2 \
    --run-id "$RUN_ID" \
    --status failed \
    --current-step layer_matched_multisite_refine_v2 \
    --out-dir "${OUT_DIR#$REPO_ROOT/}"
  exit "$rc"
fi

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group layer_matched_multisite_refine_v2 \
  --experiment-id layer_matched_multisite_refine_v2 \
  --status completed \
  --artifact-path "$OUT_DIR/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path "scripts/layer_matched_multisite_refine_v2.py" \
  --prompt-contract standard_3x6_train_3x6_test \
  --metric-path "layer_matched_multisite_refine_v2" \
  --claim-id LAYER_MATCHED_MULTISITE_REFINE_V2

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "layer_matched_multisite_refine_v2_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group layer_matched_multisite_refine_v2 \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
