#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f /root/venvs/mistral-hardening/bin/activate ]]; then
  echo "Missing Runpod venv at /root/venvs/mistral-hardening" >&2
  exit 1
fi

source /root/venvs/mistral-hardening/bin/activate

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/overnight_mistral_hardening/$RUN_ID"
mkdir -p "$OUT_DIR"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
DEVICE="${DEVICE:-cuda}"
CANONICAL_GIT_COMMIT="${CANONICAL_GIT_COMMIT:-unknown}"

echo "run_id=$RUN_ID" | tee "$OUT_DIR/STATUS.txt"
echo "model=$MODEL" | tee -a "$OUT_DIR/STATUS.txt"
echo "device=$DEVICE" | tee -a "$OUT_DIR/STATUS.txt"
echo "git_commit=$CANONICAL_GIT_COMMIT" | tee -a "$OUT_DIR/STATUS.txt"
echo "provenance_note=base_v01_default_use_explicit_MODEL_for_instruct" | tee -a "$OUT_DIR/STATUS.txt"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT_DIR/STATUS.txt"

run_step() {
  local name="$1"
  shift
  echo "" | tee -a "$OUT_DIR/STATUS.txt"
  echo ">>> START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT_DIR/STATUS.txt"
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo ">>> DONE  $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT_DIR/STATUS.txt"
  else
    local rc=$?
    echo ">>> FAIL  $name rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT_DIR/STATUS.txt"
    exit $rc
  fi
}

LAYER_ARGS=()
for layer in $(seq 0 31); do
  LAYER_ARGS+=("$layer")
done

export CANONICAL_GIT_COMMIT

run_step full_head_sweep_n20 \
  python scripts/full_head_sweep.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-prompts 20 \
    --batch-layers 8

run_step full_path_patching_all_layers_n20 \
  python scripts/full_path_patching.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-prompts 20 \
    --layers "${LAYER_ARGS[@]}"

run_step dual_layer_bridge_smoke \
  python scripts/persistent_patching_v3_dual.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-sessions 2 \
    --max-turns 10

run_step dual_layer_bridge_full \
  python scripts/persistent_patching_v3_dual.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-sessions 10 \
    --max-turns 30

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT_DIR/STATUS.txt"
echo "overnight_complete=1" | tee -a "$OUT_DIR/STATUS.txt"
