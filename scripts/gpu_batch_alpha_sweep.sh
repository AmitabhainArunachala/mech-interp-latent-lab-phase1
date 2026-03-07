#!/bin/bash
# Hardening Track 1: dual-alpha sweep for KV x dual interaction
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs results/sufficiency_ladder
LOG_FILE="logs/hardening_alpha_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "${LOG_FILE}"
echo "HARDENING TRACK 1: ALPHA SWEEP — $(date)" | tee -a "${LOG_FILE}"
echo "Repo: ${REPO_DIR}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

ALPHAS=("0.00" "0.25" "0.50" "0.75" "1.00" "1.25")

for ALPHA in "${ALPHAS[@]}"; do
  TAG="alpha_${ALPHA//./p}"
  echo "" | tee -a "${LOG_FILE}"
  echo "[alpha=${ALPHA}] start: $(date)" | tee -a "${LOG_FILE}"
  python3 scripts/sufficiency_ladder.py \
    --n-sessions 8 \
    --max-turns 30 \
    --seed 42 \
    --rv-window 16 \
    --min-new-tokens 24 \
    --max-new-tokens 150 \
    --temperature 0.7 \
    --rep-penalty 1.3 \
    --dual-alpha "${ALPHA}" \
    --conditions clean_baseline,kv_only,dual_patch,kv_plus_dual \
    --induce-min-lift 0.15 \
    --induce-alpha 0.01 \
    --tag "${TAG}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo "[alpha=${ALPHA}] done: $(date)" | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "Track 1 complete: $(date)" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
