#!/bin/bash
# Script to run Circuit Hunt V2 on RunPod
# Usage: ./run_circuit_hunt_on_runpod.sh

set -e

echo "============================================================"
echo "Circuit Hunt V2: RunPod Execution"
echo "============================================================"

# Check if we're on RunPod (or have GPU)
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" || {
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
}

echo "✓ CUDA available"
echo ""

# Run the focused experiment
echo "Starting experiment_circuit_hunt_v2_focused.py..."
echo ""

python3 experiment_circuit_hunt_v2_focused.py

echo ""
echo "============================================================"
echo "Experiment complete!"
echo "Check results in: results/circuit_hunt_v2_focused/"
echo "============================================================"

