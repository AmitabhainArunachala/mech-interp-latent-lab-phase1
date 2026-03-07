#!/bin/bash
# DEPLOY & RUN ALL GPU EXPERIMENTS ON RUNPOD
# Usage: bash scripts/deploy_runpod.sh
set -euo pipefail

REMOTE="root@82.221.170.234"
PORT=27988
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -p $PORT -i $KEY $REMOTE"
SCP="scp -P $PORT -i $KEY"
PROJECT="mech-interp-latent-lab-phase1"
REMOTE_DIR="/workspace/$PROJECT"

echo "══════════════════════════════════════════════════════════"
echo "  RUNPOD DEPLOYMENT — 5 GPU EXPERIMENTS"
echo "══════════════════════════════════════════════════════════"

# ── 1. Sync code to RunPod ──
echo ""
echo "▸ [1/6] Syncing code to RunPod..."
rsync -avz --delete \
    -e "ssh -p $PORT -i $KEY" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'results/' \
    --exclude '.venv' \
    --exclude 'wandb/' \
    "$HOME/$PROJECT/" \
    "$REMOTE:$REMOTE_DIR/"

echo "  ✅ Code synced"

# ── 2. Install deps on RunPod ──
echo ""
echo "▸ [2/6] Installing dependencies..."
$SSH "cd $REMOTE_DIR && pip install -q transformers accelerate torch numpy scipy scikit-learn 2>/dev/null || true"
echo "  ✅ Dependencies ready"

# ── 3. Run experiments ──
echo ""
echo "▸ [3/6] Running computational mode atlas (~45 min)..."
$SSH "cd $REMOTE_DIR && python3 scripts/computational_mode_atlas.py --device cuda 2>&1" | tee results/mode_atlas_remote.log || echo "  ⚠️ Mode atlas failed, continuing..."

echo ""
echo "▸ [4/6] Running per-head attention decomposition (~30 min)..."
$SSH "cd $REMOTE_DIR && python3 scripts/per_head_attention_decomposition.py --device cuda 2>&1" | tee results/per_head_remote.log || echo "  ⚠️ Per-head failed, continuing..."

echo ""
echo "▸ [5/6] Running statistical hardening (~20 min)..."
$SSH "cd $REMOTE_DIR && python3 scripts/statistical_hardening.py --device cuda 2>&1" | tee results/stat_hardening_remote.log || echo "  ⚠️ Statistical hardening failed, continuing..."

echo ""
echo "▸ [6/6] Running full path patching (~1 hour)..."
$SSH "cd $REMOTE_DIR && python3 scripts/full_path_patching.py --device cuda 2>&1" | tee results/path_patching_remote.log || echo "  ⚠️ Path patching failed, continuing..."

# NOTE: Scaling law sweep is the longest (~2+ hours) and downloads multiple models.
# Run it separately if needed:
#   ssh -p 27988 -i ~/.ssh/id_ed25519 root@82.221.170.234
#   cd /workspace/mech-interp-latent-lab-phase1
#   nohup python3 scripts/scaling_law_sweep.py --device cuda > scaling_law.log 2>&1 &

# ── Sync results back ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SYNCING RESULTS BACK..."
echo "═══════════════════════════════════════════════════════════"
rsync -avz \
    -e "ssh -p $PORT -i $KEY" \
    "$REMOTE:$REMOTE_DIR/results/" \
    "$HOME/$PROJECT/results/"

echo ""
echo "  ✅ ALL EXPERIMENTS COMPLETE — results synced to local"
echo "  Run: python3 scripts/orchestrator.py"
echo "═══════════════════════════════════════════════════════════"
