#!/bin/bash
# RunPod Setup Script
# Run this on the RunPod instance via web terminal or after SSH is configured

set -e

echo "=== RunPod Setup Script ==="
echo ""

# Check if we're in the right directory
if [ ! -d "/root/mech-interp-latent-lab-phase1" ]; then
    echo "Creating repository directory..."
    mkdir -p /root/mech-interp-latent-lab-phase1
    cd /root/mech-interp-latent-lab-phase1
else
    cd /root/mech-interp-latent-lab-phase1
    echo "Repository directory exists"
fi

echo ""
echo "=== Checking Python ==="
python3 --version || echo "⚠️  Python3 not found"

echo ""
echo "=== Checking GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "⚠️  nvidia-smi not found"

echo ""
echo "=== Installing Dependencies ==="
pip install --upgrade pip --quiet
pip install transformers torch numpy pandas scipy tqdm --quiet
pip install scikit-learn --quiet  # For stats

echo ""
echo "=== Verifying Installation ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Sync code: scp -r src/ configs/ scripts/ prompts/ root@<ip>:/root/mech-interp-latent-lab-phase1/"
echo "  2. Or clone from git if repository is available"
echo "  3. Run experiments: python3 scripts/run_mlp_vproj_combined.py"
