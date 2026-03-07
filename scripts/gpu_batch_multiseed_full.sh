#!/bin/bash
# Hardening Track 3: multi-seed full 2x2 replication
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs results/sufficiency_ladder
LOG_FILE="logs/hardening_multiseed_full_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "${LOG_FILE}"
echo "HARDENING TRACK 3: MULTI-SEED FULL — $(date)" | tee -a "${LOG_FILE}"
echo "Repo: ${REPO_DIR}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

SEEDS=("42" "123" "777")

for SEED in "${SEEDS[@]}"; do
  TAG="full_seed_${SEED}"
  echo "" | tee -a "${LOG_FILE}"
  echo "[seed=${SEED}] start: $(date)" | tee -a "${LOG_FILE}"
  python3 scripts/sufficiency_ladder.py \
    --n-sessions 10 \
    --max-turns 30 \
    --seed "${SEED}" \
    --rv-window 16 \
    --min-new-tokens 24 \
    --max-new-tokens 150 \
    --temperature 0.7 \
    --rep-penalty 1.3 \
    --dual-alpha 1.0 \
    --conditions clean_baseline,kv_only,dual_patch,kv_plus_dual,clean_recursive \
    --induce-min-lift 0.15 \
    --induce-alpha 0.01 \
    --tag "${TAG}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo "[seed=${SEED}] done: $(date)" | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "Track 3 complete: $(date)" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
