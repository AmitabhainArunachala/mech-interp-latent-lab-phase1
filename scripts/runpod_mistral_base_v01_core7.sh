#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f /root/venvs/mistral-hardening/bin/activate ]]; then
  echo "Missing Runpod venv at /root/venvs/mistral-hardening" >&2
  exit 1
fi

source /root/venvs/mistral-hardening/bin/activate

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_base_v01_core7/$RUN_ID"
mkdir -p "$OUT_DIR"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
DEVICE="${DEVICE:-cuda}"
CANONICAL_GIT_COMMIT="${CANONICAL_GIT_COMMIT:-unknown}"

echo "run_id=$RUN_ID" | tee "$OUT_DIR/STATUS.txt"
echo "model=$MODEL" | tee -a "$OUT_DIR/STATUS.txt"
echo "device=$DEVICE" | tee -a "$OUT_DIR/STATUS.txt"
echo "git_commit=$CANONICAL_GIT_COMMIT" | tee -a "$OUT_DIR/STATUS.txt"
echo "hf_hub_disable_xet=$HF_HUB_DISABLE_XET" | tee -a "$OUT_DIR/STATUS.txt"
echo "hf_hub_offline=$HF_HUB_OFFLINE" | tee -a "$OUT_DIR/STATUS.txt"
echo "transformers_offline=$TRANSFORMERS_OFFLINE" | tee -a "$OUT_DIR/STATUS.txt"
echo "note=mode_atlas uses frozen mode_atlas_v1 and remains capped at 20 prompts per mode" | tee -a "$OUT_DIR/STATUS.txt"
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

run_step p0_canonical_n100 \
  python scripts/p0_canonical_pipeline.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n 100

run_step full_head_sweep_n100 \
  python scripts/full_head_sweep.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-prompts 100 \
    --batch-layers 8

run_step svd_circuit_decomposition_n100 \
  python scripts/svd_circuit_decomposition.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-prompts 100

run_step full_path_patching_n100 \
  python scripts/full_path_patching.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-prompts 100 \
    --layers "${LAYER_ARGS[@]}"

run_step mediation_2x2_n100 \
  python scripts/mediation_2x2.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-pairs 100

run_step persistent_dual_patch_full \
  python scripts/persistent_patching_v3_dual.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-sessions 10 \
    --max-turns 30

run_step mode_atlas_base_contract \
  python scripts/computational_mode_atlas.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --n-prompts 20

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT_DIR/STATUS.txt"
echo "mistral_base_v01_core7_complete=1" | tee -a "$OUT_DIR/STATUS.txt"
