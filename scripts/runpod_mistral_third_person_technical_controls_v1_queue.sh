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

OUTPUT_STEM="${OUTPUT_STEM:-third_person_technical_controls_v1}"
QUEUE_GROUP="${QUEUE_GROUP:-mistral_third_person_technical_controls_v1}"
EXPERIMENT_ID="${EXPERIMENT_ID:-third_person_technical_controls_v1}"
CLAIM_ID="${CLAIM_ID:-THIRD_PERSON_TECHNICAL_CONTROLS_V1}"
NOTES="${NOTES:-Direct third-person technical self-reference control against recursive and baseline prompt-pass R_V.}"
PER_GROUP="${PER_GROUP:-10}"
WINDOW="${WINDOW:-16}"
EARLY_LAYER="${EARLY_LAYER:-5}"
LATE_LAYER="${LATE_LAYER:-27}"

AMIROS_POD_NAME="${AMIROS_POD_NAME:-$(hostname)}"
AMIROS_HOST="${AMIROS_HOST:-$(hostname)}"
AMIROS_PORT="${AMIROS_PORT:-22}"
AMIROS_SESSION="${AMIROS_SESSION:-$QUEUE_GROUP}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/$QUEUE_GROUP/$RUN_ID"
RUN_OUT="$REPO_ROOT/results/$OUTPUT_STEM/$RUN_ID"
mkdir -p "$OUT_DIR" "$RUN_OUT"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "$NOTES"

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
    --queue-group "$QUEUE_GROUP" \
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
      --queue-group "$QUEUE_GROUP" \
      --run-id "$RUN_ID" \
      --status failed \
      --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

run_step "$EXPERIMENT_ID" \
  "$PYTHON_BIN" scripts/third_person_technical_controls_v1.py \
  --device cuda \
  --output-dir "$RUN_OUT" \
  --experiment-name "$EXPERIMENT_ID" \
  --per-group "$PER_GROUP" \
  --window "$WINDOW" \
  --early-layer "$EARLY_LAYER" \
  --late-layer "$LATE_LAYER"

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group "$QUEUE_GROUP" \
  --experiment-id "$EXPERIMENT_ID" \
  --status completed \
  --artifact-path "${RUN_OUT#$REPO_ROOT/}/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path "scripts/third_person_technical_controls_v1.py --per-group $PER_GROUP --window $WINDOW --early-layer $EARLY_LAYER --late-layer $LATE_LAYER" \
  --prompt-contract third_person_technical_controls \
  --metric-path "prompt-pass compute_rv_with_components on recursive/baseline/pseudo/same-vocab/surreal/third-person-technical groups" \
  --claim-id "$CLAIM_ID"

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "${QUEUE_GROUP}_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
