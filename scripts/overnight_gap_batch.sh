#!/bin/bash
# OVERNIGHT BATCH: Gap Experiments B, C, D
# =========================================
# One-shot launcher — runs all remaining experiments sequentially.
# Re-runs Gap C from scratch (previous run lost to filesystem hang).
#
# Usage: tmux new-session -d -s overnight 'bash /workspace/mech-interp-latent-lab-phase1/mech-interp-latent-lab-phase1/scripts/overnight_gap_batch.sh'
#
# Estimated runtime: ~10-14 hours total
#   Gap C: ~4 hrs (5 conditions × 8 sessions × 30 turns)
#   Gap D: ~4 hrs (6 conditions × 8 sessions × 30 turns)
#   Gap B: ~6 hrs (7 conditions × 5 sessions × 30 turns)

set -o pipefail

PROJ="/workspace/mech-interp-latent-lab-phase1/mech-interp-latent-lab-phase1"
SCRIPT="$PROJ/scripts/gap_experiments_a_through_d.py"
LOGDIR="$PROJ/logs"
RESDIR="$PROJ/results/gap_experiments"

mkdir -p "$LOGDIR" "$RESDIR"

TS=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "OVERNIGHT GAP BATCH — Started $TS"
echo "=========================================="
echo ""

# Kill any stale python processes (not jupyter)
echo "[$(date)] Cleaning stale processes..."
pkill -f "exp2_multi_arch" 2>/dev/null
pkill -f "exp3_hebbian" 2>/dev/null
pkill -f "exp_multi_arch" 2>/dev/null
pkill -f "hebbian_experiment" 2>/dev/null
sleep 2

# Verify GPU is free
echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

# ── Gap C (most informative — L0 MLP × KV interaction) ───────────────────────
echo "=========================================="
echo "[$(date)] Starting Gap C: L0 MLP × KV Interaction"
echo "=========================================="
python "$SCRIPT" --experiment c 2>&1 | tee "$LOGDIR/gap_c_${TS}.log"
C_EXIT=$?
echo "[$(date)] Gap C exit code: $C_EXIT"
echo ""

# ── Gap D (KV layer-band ablation) ───────────────────────────────────────────
echo "=========================================="
echo "[$(date)] Starting Gap D: KV Layer-Band Ablation"
echo "=========================================="
python "$SCRIPT" --experiment d 2>&1 | tee "$LOGDIR/gap_d_${TS}.log"
D_EXIT=$?
echo "[$(date)] Gap D exit code: $D_EXIT"
echo ""

# ── Gap B (intermediate-layer behavioral patching — slowest) ──────────────────
echo "=========================================="
echo "[$(date)] Starting Gap B: Intermediate-Layer Behavioral Patching"
echo "=========================================="
python "$SCRIPT" --experiment b 2>&1 | tee "$LOGDIR/gap_b_${TS}.log"
B_EXIT=$?
echo "[$(date)] Gap B exit code: $B_EXIT"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE — $(date)"
echo "  Gap C exit: $C_EXIT"
echo "  Gap D exit: $D_EXIT"
echo "  Gap B exit: $B_EXIT"
echo "=========================================="
echo ""
echo "Results in: $RESDIR"
ls -la "$RESDIR"/*.json 2>/dev/null
echo ""
echo "Logs in: $LOGDIR"
ls -la "$LOGDIR"/gap_*${TS}*.log 2>/dev/null
