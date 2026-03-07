#!/usr/bin/env bash
set -o pipefail

export HF_HOME=/workspace/hf_cache
export PYTHONPATH=/workspace/mech-interp
export TRANSFORMERS_VERBOSITY=error
LOG=/tmp/remaining_v2.log

echo ">>> E4.2 CONCEPT ERASURE START: $(date)" | tee -a $LOG
cd /tmp
python3 /workspace/mech-interp/scripts/linear_probe_selfref.py \
    --device cuda \
    --model mistralai/Mistral-7B-v0.1 \
    --n-prompts 20 2>&1 | tee -a $LOG
echo ">>> E4.2 CONCEPT ERASURE DONE: $(date)" | tee -a $LOG

echo ">>> E1.3 QWEN2.5-3B START: $(date)" | tee -a $LOG

# Use /tmp for HF cache to avoid NFS quota on new model downloads
export HF_HOME=/tmp/hf_cache
mkdir -p /tmp/hf_cache

cd /tmp
python3 /workspace/mech-interp/scripts/scaling_gap_sweep.py \
    --device cuda \
    --single-model qwen2.5-3b \
    --n-prompts 20 2>&1 | tee -a $LOG

echo ">>> E1.3 QWEN2.5-3B DONE: $(date)" | tee -a $LOG
echo "=== ALL V2 DONE: $(date) ===" | tee -a $LOG
ls -la /tmp/results/linear_probe/ /tmp/results/power_up/ 2>&1 | tee -a $LOG
