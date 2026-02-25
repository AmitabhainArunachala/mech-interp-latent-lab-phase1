#!/bin/bash
# GPU Batch Runner: v3 dual-layer + C2 experiment
# Expected runtime: ~2-3 hours total on RTX PRO 6000
set -euo pipefail

cd /workspace/mech-interp-latent-lab-phase1/mech-interp-latent-lab-phase1
export HF_HOME=/workspace/.hf
export HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub

echo "=========================================="
echo "GPU BATCH v3 — $(date)"
echo "=========================================="

# ── Experiment 1: Dual-layer patching (L18 residual + L27 V-proj) ──
echo ""
echo "[1/2] DUAL-LAYER PATCHING v3 (4 conditions × 10 sessions × 30 turns)"
echo "      Expected: ~60-90 min"
echo "      Start: $(date)"
python3 scripts/persistent_patching_v3_dual.py
echo "      Done: $(date)"

# ── Experiment 2: C2 measurement suite (n=50) ──
echo ""
echo "[2/2] C2 MEASUREMENT SUITE (n=50 prompts × 3 configs)"
echo "      Expected: ~30-45 min"
echo "      Start: $(date)"
python3 scripts/run_c2_rv_measurement.py --n_prompts 50
echo "      Done: $(date)"

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE — $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  v3 dual: results/persistent_patching_v3/"
echo "  C2:      results/phase1_mechanism/runs/"
