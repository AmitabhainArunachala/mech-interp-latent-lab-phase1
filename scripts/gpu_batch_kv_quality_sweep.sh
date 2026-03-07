#!/bin/bash
# Hardening Track 2: KV-only quality sweep over sampling settings
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs results/sufficiency_ladder
LOG_FILE="logs/hardening_kv_quality_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "${LOG_FILE}"
echo "HARDENING TRACK 2: KV QUALITY SWEEP — $(date)" | tee -a "${LOG_FILE}"
echo "Repo: ${REPO_DIR}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Format: temp rep_penalty
SETTINGS=(
  "0.60 1.15"
  "0.70 1.20"
  "0.70 1.30"
  "0.80 1.30"
  "0.85 1.40"
)

for PAIR in "${SETTINGS[@]}"; do
  TEMP="$(echo "${PAIR}" | awk '{print $1}')"
  REP="$(echo "${PAIR}" | awk '{print $2}')"
  TAG="kvq_t${TEMP//./p}_r${REP//./p}"
  echo "" | tee -a "${LOG_FILE}"
  echo "[temp=${TEMP}, rep=${REP}] start: $(date)" | tee -a "${LOG_FILE}"
  python3 scripts/sufficiency_ladder.py \
    --n-sessions 8 \
    --max-turns 30 \
    --seed 42 \
    --rv-window 16 \
    --min-new-tokens 24 \
    --max-new-tokens 150 \
    --temperature "${TEMP}" \
    --rep-penalty "${REP}" \
    --dual-alpha 1.0 \
    --conditions clean_baseline,kv_only,clean_recursive \
    --induce-min-lift 0.15 \
    --induce-alpha 0.01 \
    --tag "${TAG}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo "[temp=${TEMP}, rep=${REP}] done: $(date)" | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "Track 2 complete: $(date)" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
