#!/bin/bash
# Quick setup script to sync repo to GPU and install dependencies

SSH_HOST="root@195.26.233.61"
SSH_PORT="53317"
SSH_KEY="~/.ssh/id_ed25519"
SSH_CMD="ssh -o StrictHostKeyChecking=no -p $SSH_PORT -i $SSH_KEY $SSH_HOST"

echo "🚀 Setting up GPU workspace..."

# Check if repo exists on remote
echo "[1/4] Checking remote workspace..."
$SSH_CMD "cd /workspace && if [ -d 'mech-interp-latent-lab-phase1' ]; then echo 'Repo exists'; else echo 'Repo not found'; fi"

# Sync repo using rsync (exclude large files)
echo "[2/4] Syncing repository..."
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='results/' --exclude='*.png' --exclude='*.csv' \
  -e "ssh -p $SSH_PORT -i $SSH_KEY" \
  /Users/dhyana/mech-interp-latent-lab-phase1/ \
  $SSH_HOST:/workspace/mech-interp-latent-lab-phase1/

# Install dependencies
echo "[3/4] Installing dependencies..."
$SSH_CMD "cd /workspace/mech-interp-latent-lab-phase1 && pip3 install -q transformers accelerate scipy pandas matplotlib seaborn"

# Verify setup
echo "[4/4] Verifying setup..."
$SSH_CMD "cd /workspace/mech-interp-latent-lab-phase1 && python3 -c 'import torch; import transformers; print(\"✅ Setup complete!\"); print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"Transformers: {transformers.__version__}\")'"

echo "✅ GPU workspace ready!"









