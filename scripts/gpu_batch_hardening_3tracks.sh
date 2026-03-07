#!/bin/bash
# Launch all hardening tracks sequentially.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs
MASTER_LOG="logs/hardening_3tracks_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "${MASTER_LOG}"
echo "HARDENING 3-TRACK PACK — $(date)" | tee -a "${MASTER_LOG}"
echo "Repo: ${REPO_DIR}" | tee -a "${MASTER_LOG}"
echo "==========================================" | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
echo "[1/3] Alpha sweep" | tee -a "${MASTER_LOG}"
bash scripts/gpu_batch_alpha_sweep.sh 2>&1 | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
echo "[2/3] KV quality sweep" | tee -a "${MASTER_LOG}"
bash scripts/gpu_batch_kv_quality_sweep.sh 2>&1 | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
echo "[3/3] Multi-seed full replication" | tee -a "${MASTER_LOG}"
bash scripts/gpu_batch_multiseed_full.sh 2>&1 | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
echo "HARDENING PACK COMPLETE — $(date)" | tee -a "${MASTER_LOG}"
echo "Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
