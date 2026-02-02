#!/bin/bash
# Quick setup script for RunPod GPU - installs deps and verifies setup

SSH_HOST="root@213.173.111.30"
SSH_PORT="26212"
SSH_KEY="~/.ssh/id_ed25519"
SSH_CMD="ssh -o StrictHostKeyChecking=no -p $SSH_PORT -i $SSH_KEY $SSH_HOST"

echo "🚀 Setting up RunPod GPU workspace..."

# Test connection
echo "[1/5] Testing connection..."
if ! $SSH_CMD "echo 'Connection OK'" 2>/dev/null; then
    echo "❌ Connection failed! Is the RunPod running?"
    exit 1
fi

# Check GPU
echo "[2/5] Checking GPU..."
$SSH_CMD "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

# Check disk space
echo "[3/5] Checking disk space..."
$SSH_CMD "df -h /workspace 2>/dev/null || df -h / | head -2"

# Install dependencies
echo "[4/5] Installing dependencies..."
$SSH_CMD "cd /workspace/mech-interp-latent-lab-phase1 && pip3 install -q --upgrade pip && pip3 install -q -r env.txt"

# Verify setup
echo "[5/5] Verifying setup..."
$SSH_CMD "cd /workspace/mech-interp-latent-lab-phase1 && export CUDA_VISIBLE_DEVICES=0 && python3 -c 'import torch; import transformers; print(\"✅ Setup complete!\"); print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\"); print(f\"Transformers: {transformers.__version__}\"); print(f\"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")'"

echo ""
echo "✅ RunPod GPU workspace ready!"
echo ""
echo "Connection command:"
echo "  ssh $SSH_HOST -p $SSH_PORT -i $SSH_KEY"
echo ""
echo "Repo location:"
echo "  cd /workspace/mech-interp-latent-lab-phase1"

