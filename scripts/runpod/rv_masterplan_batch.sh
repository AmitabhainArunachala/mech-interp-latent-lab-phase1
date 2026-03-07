#!/bin/bash
set -euo pipefail
# ═════════════════════════════════════════════════════════════════
# R_V MASTER PLAN v2 — RunPod Batch Runner
# Updated 2026-03-03 with GQA fix, checkpoint fix, auth instructions.
# Run on A100-80GB pod. Complete all remaining GPU experiments.
# ═════════════════════════════════════════════════════════════════
# BEFORE RUNNING:
#   1. SSH into RunPod
#   2. cd /workspace/mech-interp-latent-lab-phase1
#   3. Create a NEW HuggingFace token at:
#      https://huggingface.co/settings/tokens
#      -> Type: Read (classic) OR fine-grained with "Access gated repos"
#   4. huggingface-cli login  (paste the NEW token)
#   5. Verify: python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
#   6. Accept model licenses:
#      - https://huggingface.co/google/gemma-2-2b (click "Agree")
#      - https://huggingface.co/meta-llama/Llama-3.2-3B (click "Agree")
#   7. git pull  (to get GQA fix and new scripts)
#   8. bash scripts/runpod/rv_masterplan_batch.sh 2>&1 | tee batch_log.txt
#
# ESTIMATED TIME: ~8-10h on A100-80GB
# ═════════════════════════════════════════════════════════════════

PROJ=/workspace/mech-interp-latent-lab-phase1
cd "$PROJ"

# Suppress HF download progress bars to save context window
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

echo "=== [$(date)] R_V Master Plan v2 Batch Start ==="

# Ensure result dirs exist
mkdir -p results/rv_masterplan/{E1.3_scaling_gap,E1.1_power_up,E2.1_svd_fixed,E1.4_checkpoints_fixed,E3.1_sae}

# Clean model cache if disk is tight
df -h /workspace | tail -1
echo "If disk >80% full, run: rm -rf ~/.cache/huggingface/hub/models--EleutherAI--pythia-*"

# ── 1. E1.3: Re-run failed gated models (~2h) ───────────────
echo ""
echo "=== [$(date)] E1.3: Gemma-2-2B + Llama-3.2-3B ==="
python3 scripts/scaling_gap_sweep.py --device cuda --models gemma-2-2b llama-3.2-3b 2>&1 | tee results/rv_masterplan/E1.3_scaling_gap/run_rerun.log
echo "[$(date)] E1.3 re-run complete"

# ── 2. E2.1: SVD with GQA fix (~1h) ───────────────────────
echo ""
echo "=== [$(date)] E2.1: SVD Circuit Decomposition (GQA fixed) ==="
python3 scripts/svd_circuit_decomposition.py --device cuda --n-prompts 20 2>&1 | tee results/rv_masterplan/E2.1_svd_fixed/run.log
echo "[$(date)] E2.1 SVD complete"

# ── 3. E1.1: Power-up to n≥100 (~4h) ─────────────────────
echo ""
echo "=== [$(date)] E1.1: Power-Up (n=120 target) ==="
python3 scripts/power_up_multiseed.py --device cuda --n-prompts 120 2>&1 | tee results/rv_masterplan/E1.1_power_up/run.log
echo "[$(date)] E1.1 power-up complete"

# ── 4. E1.4: Fix Pythia-2.8B checkpoints (~2h) ──────────────
echo ""
echo "=== [$(date)] E1.4: Pythia-2.8B checkpoints (force-download) ==="
python3 scripts/training_checkpoint_sweep.py --device cuda --models pythia-2.8b --force-download 2>&1 | tee results/rv_masterplan/E1.4_checkpoints_fixed/run.log
echo "[$(date)] E1.4 checkpoint fix complete"

# ── 5. E3.1+E3.4: SAE on Gemma-2-2B (if available) (~4h) ─────
echo ""
echo "=== [$(date)] E3.1+E3.4: SAE Feature Analysis ==="
pip install -q sae-lens transformer-lens 2>/dev/null || true
python3 scripts/sae_feature_analysis.py --device cuda --n-prompts 20 2>&1 | tee results/rv_masterplan/E3.1_sae/run.log || echo "[WARN] E3.1 SAE failed (may need Gemma-2-2B access)"
echo "[$(date)] E3.1+E3.4 complete (or skipped)"

echo ""
echo "=== [$(date)] ALL GPU EXPERIMENTS COMPLETE ==="
echo ""
echo "SYNC COMMAND (run on LOCAL machine):"
echo "  rsync -avz --no-owner --no-group -e 'ssh -p PORT' root@HOST:/workspace/mech-interp-latent-lab-phase1/results/ ~/mech-interp-latent-lab-phase1/results/"
echo ""
echo "Disk usage:"
df -h /workspace | tail -1
