#!/bin/bash
# GPU batch launcher for Gap Experiments A-D
# Run on: root@198.13.252.19 -p 15104
# 
# Gap A (fast, ~10 min): KV cache corruption measurement
# Gap B (slow, ~6 hrs): Intermediate-layer behavioral patching (7 conditions × 5 sessions × 30 turns)
# Gap C (medium, ~4 hrs): L0 MLP × KV interaction (5 conditions × 8 sessions × 30 turns)  
# Gap D (medium, ~4 hrs): KV layer-band ablation (6 conditions × 8 sessions × 30 turns)

set -o pipefail

PROJ="/workspace/mech-interp-latent-lab-phase1/mech-interp-latent-lab-phase1"
SCRIPT="$PROJ/scripts/gap_experiments_a_through_d.py"
LOGDIR="$PROJ/logs"
mkdir -p "$LOGDIR"

TS=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "GAP EXPERIMENTS A-D — $TS"
echo "=========================================="

# Run A first (fast, diagnostic)
echo "[$(date)] Starting Gap A: KV Corruption Test..."
python "$SCRIPT" --experiment a 2>&1 | tee "$LOGDIR/gap_a_${TS}.log"
echo "[$(date)] Gap A complete."

# Run C next (L0 MLP × KV — most informative)
echo "[$(date)] Starting Gap C: L0 MLP × KV Interaction..."
python "$SCRIPT" --experiment c 2>&1 | tee "$LOGDIR/gap_c_${TS}.log"
echo "[$(date)] Gap C complete."

# Run D (KV layer bands — second most informative)
echo "[$(date)] Starting Gap D: KV Layer-Band Ablation..."
python "$SCRIPT" --experiment d 2>&1 | tee "$LOGDIR/gap_d_${TS}.log"
echo "[$(date)] Gap D complete."

# Run B last (slowest — intermediate layer behavioral)
echo "[$(date)] Starting Gap B: Intermediate-Layer Behavioral Patching..."
python "$SCRIPT" --experiment b 2>&1 | tee "$LOGDIR/gap_b_${TS}.log"
echo "[$(date)] Gap B complete."

echo "=========================================="
echo "ALL GAP EXPERIMENTS COMPLETE — $(date)"
echo "=========================================="
