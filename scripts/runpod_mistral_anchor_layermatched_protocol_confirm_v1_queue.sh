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

QUEUE_GROUP="${QUEUE_GROUP:-mistral_anchor_layermatched_protocol_confirm_v1}"
EXPERIMENT_ID="${EXPERIMENT_ID:-anchor_layermatched_protocol_confirm_v1}"
OUTPUT_STEM="${OUTPUT_STEM:-$EXPERIMENT_ID}"
CLAIM_ID="${CLAIM_ID:-ANCHOR_LAYERMATCHED_PROTOCOL_CONFIRM_V1}"
NOTES="${NOTES:-Focused confirmation of the anchor+layer-matched low-dose winner against the static anchor champions.}"
TRAIN_PER_GROUP="${TRAIN_PER_GROUP:-6}"
TEST_PER_GROUP="${TEST_PER_GROUP:-6}"
CONDITION_NAMES="${CONDITION_NAMES:-control,anchor_bridge_2,anchor_single_mlp_0p125_bridge_3,anchor_layermatched_low_bridge_2,anchor_layermatched_low_bridge_3,anchor_meandiff_low_bridge_2}"
GENERATION_SEEDS="${GENERATION_SEEDS:-101 202 303 404 505 606 707 808}"

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

read -r -a SEED_ARGS <<< "$GENERATION_SEEDS"

run_step "$EXPERIMENT_ID" \
  "$PYTHON_BIN" scripts/anchor_layermatched_protocol_v1.py \
  --device cuda \
  --out-dir "$RUN_OUT" \
  --experiment-name "$EXPERIMENT_ID" \
  --train-per-group "$TRAIN_PER_GROUP" \
  --test-per-group "$TEST_PER_GROUP" \
  --condition-names "$CONDITION_NAMES" \
  --generation-seeds "${SEED_ARGS[@]}"

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group "$QUEUE_GROUP" \
  --experiment-id "$EXPERIMENT_ID" \
  --status completed \
  --artifact-path "${RUN_OUT#$REPO_ROOT/}/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path "scripts/anchor_layermatched_protocol_v1.py --experiment-name $EXPERIMENT_ID --train-per-group $TRAIN_PER_GROUP --test-per-group $TEST_PER_GROUP --condition-names $CONDITION_NAMES --generation-seeds $GENERATION_SEEDS" \
  --prompt-contract heldout_anchor_layermatched_protocol_confirm_v1 \
  --metric-path "anchor_layermatched_protocol_v1 + classify_output + compute_rv" \
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
