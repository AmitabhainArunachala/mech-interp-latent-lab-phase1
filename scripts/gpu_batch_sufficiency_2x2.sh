#!/bin/bash
# GPU batch runner for sufficiency ladder 2x2 (KV x dual patch)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p results/sufficiency_ladder
LOG_FILE="results/sufficiency_ladder/batch_sufficiency_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "${LOG_FILE}"
echo "SUFFICIENCY 2x2 BATCH — $(date)" | tee -a "${LOG_FILE}"
echo "Repo: ${REPO_DIR}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

python3 scripts/sufficiency_ladder.py \
  --n-sessions 10 \
  --max-turns 30 \
  --seed 42 \
  --rv-window 16 \
  --min-new-tokens 24 \
  --induce-min-lift 0.15 \
  --induce-alpha 0.01 \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Batch finished: $(date)" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
