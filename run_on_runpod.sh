#!/bin/bash
# Script to run Circuit Hunt V2 on RunPod with GPU

echo "============================================================"
echo "Circuit Hunt V2: RunPod Execution"
echo "============================================================"
echo ""

# Check if we're already on RunPod
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found - checking GPU..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
    echo "Running experiment..."
    python3 experiment_circuit_hunt_v2_focused.py
else
    echo "⚠️  Not on RunPod (nvidia-smi not found)"
    echo ""
    echo "To run on RunPod:"
    echo "1. SSH into RunPod:"
    echo "   ssh -p 18147 root@198.13.252.9"
    echo ""
    echo "2. Navigate to project:"
    echo "   cd /workspace/mech-interp-latent-lab-phase1"
    echo ""
    echo "3. Run this script:"
    echo "   ./run_on_runpod.sh"
    echo ""
    echo "Or run directly:"
    echo "   python3 experiment_circuit_hunt_v2_focused.py"
fi

