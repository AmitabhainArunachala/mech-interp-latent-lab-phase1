#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pick_python() {
  if [[ -x /root/venvs/mistral-hardening/bin/python ]]; then echo "/root/venvs/mistral-hardening/bin/python"; return; fi
  if [[ -x ./.venv/bin/python ]]; then echo "./.venv/bin/python"; return; fi
  if command -v python3 >/dev/null 2>&1; then echo "python3"; return; fi
  echo "python"
}

PYTHON_BIN="$(pick_python)"
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

QUEUE_GROUP="llama3_8b_p0_canonical_v1"
EXPERIMENT_ID="llama3_8b_p0_canonical_v1"
CLAIM_ID="LLAMA3_8B_P0_CANONICAL_V1"
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
P0_N=80
PATH_N=20
# Llama-3-8B has 32 layers; early ~15% = L5, late ~84% = L27
PATH_LAYERS="0 2 4 6 8 10 12 14 16 18 20 22 24 26 27 28 30"
NOTES="Frozen-contract P0 canonical + full path patching on Llama-3-8B base. Upgrades provisional Jan 2026 phase-2 result to locked provenance."

AMIROS_POD_NAME="${AMIROS_POD_NAME:-$(hostname)}"
AMIROS_HOST="${AMIROS_HOST:-$(hostname)}"
AMIROS_PORT="${AMIROS_PORT:-22}"
AMIROS_SESSION="${AMIROS_SESSION:-$QUEUE_GROUP}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/$QUEUE_GROUP/$RUN_ID"
mkdir -p "$OUT_DIR"
STATUS_FILE="$OUT_DIR/STATUS.txt"

echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "model_name=$MODEL_NAME" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" --host "$AMIROS_HOST" --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" --status running --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" --notes "$NOTES"

run_step() {
  local name="$1"; shift
  echo "" | tee -a "$STATUS_FILE"
  echo ">>> START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  "$PYTHON_BIN" -m src.utils.research_os lease-update \
    --pod-name "$AMIROS_POD_NAME" --host "$AMIROS_HOST" --port "$AMIROS_PORT" \
    --session-name "$AMIROS_SESSION" --queue-group "$QUEUE_GROUP" \
    --run-id "$RUN_ID" --status running --current-step "$name" \
    --out-dir "${OUT_DIR#$REPO_ROOT/}"
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo ">>> DONE  $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  else
    local rc=$?
    echo ">>> FAIL  $name rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
    "$PYTHON_BIN" -m src.utils.research_os lease-update \
      --pod-name "$AMIROS_POD_NAME" --host "$AMIROS_HOST" --port "$AMIROS_PORT" \
      --session-name "$AMIROS_SESSION" --queue-group "$QUEUE_GROUP" \
      --run-id "$RUN_ID" --status failed --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

SAFE_MODEL="$("$PYTHON_BIN" -c "print('$MODEL_NAME'.replace('/','__').replace('.','-'))")"
P0_ARTIFACT="results/p0_canonical/${SAFE_MODEL}_p0_result.json"

run_step llama3_8b_p0_canonical \
  "$PYTHON_BIN" scripts/p0_canonical_pipeline.py \
  --model "$MODEL_NAME" --n "$P0_N" --device cuda

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID-p0" --queue-group "$QUEUE_GROUP" \
  --experiment-id llama3_8b_p0_canonical --status completed \
  --artifact-path "$P0_ARTIFACT" --model-family BASE_V01 \
  --model-name "$MODEL_NAME" \
  --config-path "scripts/p0_canonical_pipeline.py --model $MODEL_NAME --n $P0_N --device cuda" \
  --prompt-contract mistral_hardening_v1/core_measurement \
  --metric-path "P0 canonical pipeline" --claim-id "$CLAIM_ID"

BEFORE_LATEST="$(ls -1t results/path_patching/path_patching_summary_*.json 2>/dev/null | head -n 1 || true)"
run_step llama3_8b_full_path_patching \
  "$PYTHON_BIN" scripts/full_path_patching.py \
  --model "$MODEL_NAME" --device cuda --n-prompts "$PATH_N" --layers $PATH_LAYERS
AFTER_LATEST="$(ls -1t results/path_patching/path_patching_summary_*.json 2>/dev/null | head -n 1 || true)"

if [[ -n "$AFTER_LATEST" && "$AFTER_LATEST" != "$BEFORE_LATEST" ]]; then
  "$PYTHON_BIN" -m src.utils.research_os result-upsert \
    --run-id "$RUN_ID-path" --queue-group "$QUEUE_GROUP" \
    --experiment-id llama3_8b_full_path_patching --status completed \
    --artifact-path "$AFTER_LATEST" --model-family BASE_V01 \
    --model-name "$MODEL_NAME" \
    --config-path "scripts/full_path_patching.py --model $MODEL_NAME --device cuda --n-prompts $PATH_N --layers $PATH_LAYERS" \
    --prompt-contract mistral_hardening_v1/core_measurement \
    --metric-path "full_path_patching" --claim-id "$CLAIM_ID"
fi

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "${QUEUE_GROUP}_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" --host "$AMIROS_HOST" --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" --status completed --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
