#!/bin/bash
set -o pipefail
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets

cd /workspace/mech-interp-latent-lab-phase1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Use /tmp for logs to avoid NFS quota issues, copy results at end
LOGDIR=/tmp/phase3_${TIMESTAMP}
mkdir -p ${LOGDIR}

# Results go to workspace
RESDIR=results/rv_masterplan/phase3_${TIMESTAMP}
mkdir -p ${RESDIR} 2>/dev/null || true

MODEL="mistralai/Mistral-7B-v0.1"

echo "=== Phase 3 Relaunch v2: $(date) ==="
echo "Model: ${MODEL}"
echo "Log dir: ${LOGDIR}"
echo "HF_HOME: ${HF_HOME}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"

echo ""
echo "[E3.1] SAE Feature Analysis — starting $(date)"
python scripts/sae_feature_analysis.py --model ${MODEL} --device cuda 2>&1 | tee ${LOGDIR}/e3.1_sae.log
E31_EXIT=$?
echo "[E3.1] Exit code: ${E31_EXIT} — $(date)"

echo ""
echo "[E3.2] Circuit Tracing — starting $(date)"
python scripts/circuit_tracing_analysis.py --model ${MODEL} --device cuda 2>&1 | tee ${LOGDIR}/e3.2_circuit.log
E32_EXIT=$?
echo "[E3.2] Exit code: ${E32_EXIT} — $(date)"

echo ""
echo "[E2.4] DII Intervention — starting $(date)"
python scripts/dii_intervention.py --model ${MODEL} --device cuda 2>&1 | tee ${LOGDIR}/e2.4_dii.log
E24_EXIT=$?
echo "[E2.4] Exit code: ${E24_EXIT} — $(date)"

echo ""
echo "=== All Phase 3 experiments complete: $(date) ==="
echo "Exit codes: E3.1=${E31_EXIT} E3.2=${E32_EXIT} E2.4=${E24_EXIT}"

# Copy logs to results dir
cp ${LOGDIR}/*.log ${RESDIR}/ 2>/dev/null || true
echo "Logs copied to ${RESDIR}"
