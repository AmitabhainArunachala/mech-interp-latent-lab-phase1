#!/bin/bash
echo "🚀 Initiating Protocol: THE SLIPPERY SLOPE"
echo "Target: runpod-dec10"

# 1. Sync Code
echo "📦 Syncing code..."
./sync_to_runpod.sh

# 2. Run Remote Experiment
echo "🧪 Running Experiment on GPU..."
ssh runpod-dec10 "cd /workspace/mech-interp-latent-lab-phase1 && \
    export PYTHONPATH=. && \
    python3 experiment_causal_sweep.py"

# 3. Pull Results
echo "📥 Retrieving Evidence..."
rsync -avz runpod-dec10:/workspace/mech-interp-latent-lab-phase1/results/dec13_slope/ ./results/dec13_slope/

echo "✅ DONE. Check results/dec13_slope/l5_l27_sweep.csv"

