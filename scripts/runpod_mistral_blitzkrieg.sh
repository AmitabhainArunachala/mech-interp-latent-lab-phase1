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
OUT_DIR="$REPO_ROOT/results/mistral_blitzkrieg/$RUN_ID"
mkdir -p "$OUT_DIR"

CANONICAL_MODEL="${CANONICAL_MODEL:-mistralai/Mistral-7B-v0.1}"
LEGACY_MODEL="${LEGACY_MODEL:-mistralai/Mistral-7B-v0.1}"
DEVICE="${DEVICE:-cuda}"
CANONICAL_GIT_COMMIT="${CANONICAL_GIT_COMMIT:-unknown}"

echo "run_id=$RUN_ID" | tee "$OUT_DIR/STATUS.txt"
echo "canonical_model=$CANONICAL_MODEL" | tee -a "$OUT_DIR/STATUS.txt"
echo "legacy_model=$LEGACY_MODEL" | tee -a "$OUT_DIR/STATUS.txt"
echo "device=$DEVICE" | tee -a "$OUT_DIR/STATUS.txt"
echo "git_commit=$CANONICAL_GIT_COMMIT" | tee -a "$OUT_DIR/STATUS.txt"
echo "provenance_note=base_v01_defaults_use_explicit_overrides_for_instruct" | tee -a "$OUT_DIR/STATUS.txt"
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

run_step canonical_full_head_sweep_n40 \
  python scripts/full_head_sweep.py \
    --model "$CANONICAL_MODEL" \
    --device "$DEVICE" \
    --n-prompts 40 \
    --batch-layers 8

run_step canonical_svd_circuit_n40 \
  python scripts/svd_circuit_decomposition.py \
    --model "$CANONICAL_MODEL" \
    --device "$DEVICE" \
    --n-prompts 40

run_step canonical_full_path_patching_n40 \
  python scripts/full_path_patching.py \
    --model "$CANONICAL_MODEL" \
    --device "$DEVICE" \
    --n-prompts 40 \
    --layers "${LAYER_ARGS[@]}"

run_step canonical_dual_patch_smoke \
  python scripts/persistent_patching_v3_dual.py \
    --model "$CANONICAL_MODEL" \
    --device "$DEVICE" \
    --n-sessions 2 \
    --max-turns 10

run_step canonical_dual_patch_full \
  python scripts/persistent_patching_v3_dual.py \
    --model "$CANONICAL_MODEL" \
    --device "$DEVICE" \
    --n-sessions 10 \
    --max-turns 30

run_step canonical_mediation_smoke \
  python scripts/mediation_2x2.py \
    --model "$CANONICAL_MODEL" \
    --device "$DEVICE" \
    --n-pairs 10

run_step canonical_mediation_full \
  python scripts/mediation_2x2.py \
    --model "$CANONICAL_MODEL" \
    --device "$DEVICE" \
    --n-pairs 40

run_step exploratory_sufficiency_ladder_smoke \
  python scripts/sufficiency_ladder.py \
    --model "$LEGACY_MODEL" \
    --device "$DEVICE" \
    --n-sessions 2 \
    --max-turns 10 \
    --tag blitz_smoke

run_step exploratory_sufficiency_ladder_full \
  python scripts/sufficiency_ladder.py \
    --model "$LEGACY_MODEL" \
    --device "$DEVICE" \
    --n-sessions 10 \
    --max-turns 30 \
    --tag blitz_full

run_step exploratory_hardening_battery \
  python scripts/hardening_battery.py \
    --model "$LEGACY_MODEL" \
    --device "$DEVICE"

run_step cpu_within_session_bridge \
  python scripts/within_session_bridge.py

run_step cpu_bridge_battery \
  python scripts/bridge_battery.py

run_step cpu_classifier_evaluation \
  python scripts/classifier_evaluation.py

run_step cpu_perplexity_repairing \
  python scripts/perplexity_repairing.py

run_step cpu_per_token_rv_analysis \
  python scripts/per_token_rv_analysis.py

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT_DIR/STATUS.txt"
echo "blitzkrieg_complete=1" | tee -a "$OUT_DIR/STATUS.txt"
