#!/bin/bash
cd /workspace/mech-interp-latent-lab-phase1
LOG=/workspace/experiment_batch.log

echo "=== BATCH START: $(date) ===" > $LOG

echo "--- [1/4] Mode Atlas ---" >> $LOG 2>&1
python3 scripts/computational_mode_atlas.py --device cuda >> $LOG 2>&1
echo "--- Mode Atlas DONE: $(date) ---" >> $LOG

echo "--- [2/4] Per-Head Attention ---" >> $LOG 2>&1
python3 scripts/per_head_attention_decomposition.py --device cuda >> $LOG 2>&1
echo "--- Per-Head DONE: $(date) ---" >> $LOG

echo "--- [3/4] Statistical Hardening ---" >> $LOG 2>&1
python3 scripts/statistical_hardening.py --device cuda >> $LOG 2>&1
echo "--- Statistical Hardening DONE: $(date) ---" >> $LOG

echo "--- [4/4] Full Path Patching ---" >> $LOG 2>&1
python3 scripts/full_path_patching.py --device cuda >> $LOG 2>&1
echo "--- Path Patching DONE: $(date) ---" >> $LOG

echo "=== ALL COMPLETE: $(date) ===" >> $LOG
