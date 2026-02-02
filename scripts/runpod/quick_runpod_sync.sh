#!/bin/bash
# All-in-one RunPod sync and setup script

SSH_HOST="root@213.173.111.30"
SSH_PORT="26212"
SSH_KEY="~/.ssh/id_ed25519"
SSH_CMD="ssh -o StrictHostKeyChecking=no -p $SSH_PORT -i $SSH_KEY $SSH_HOST"
REPO_PATH="/workspace/mech-interp-latent-lab-phase1"

echo "🚀 RunPod Quick Sync & Setup"
echo "=============================="
echo "Host: $SSH_HOST:$SSH_PORT"
echo ""

# Step 1: Test connection
echo "[1/6] Testing connection..."
if ! $SSH_CMD "echo 'Connection OK'" 2>/dev/null; then
    echo "❌ Connection failed!"
    echo ""
    echo "Please ensure:"
    echo "  1. RunPod is started from the dashboard"
    echo "  2. SSH is enabled"
    echo "  3. Port $SSH_PORT is correct"
    echo ""
    echo "Trying again in 3 seconds..."
    sleep 3
    if ! $SSH_CMD "echo 'Connection OK'" 2>/dev/null; then
        echo "❌ Still failing. Please check RunPod status."
        exit 1
    fi
fi
echo "✅ Connected!"

# Step 2: Check GPU
echo ""
echo "[2/6] Checking GPU..."
$SSH_CMD "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader" || echo "⚠️  GPU check failed"

# Step 3: Check disk space
echo ""
echo "[3/6] Checking disk space..."
$SSH_CMD "df -h /workspace 2>/dev/null || df -h / | head -2"

# Step 4: Sync repo
echo ""
echo "[4/6] Syncing repository..."
rsync -avz --progress \
  --exclude='*.csv' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='*.log' \
  --exclude='*.png' \
  --exclude='results/' \
  --exclude='models/' \
  --exclude='*.npz' \
  -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT -i $SSH_KEY" \
  /Users/dhyana/mech-interp-latent-lab-phase1/ \
  $SSH_HOST:$REPO_PATH/

# Step 5: Install dependencies
echo ""
echo "[5/6] Installing/updating dependencies..."
$SSH_CMD "cd $REPO_PATH && pip3 install -q --upgrade pip && pip3 install -q -r env.txt"

# Step 6: Verify setup
echo ""
echo "[6/6] Verifying setup..."
$SSH_CMD "cd $REPO_PATH && export CUDA_VISIBLE_DEVICES=0 && python3 << 'PYEOF'
import torch
import transformers
print('✅ Setup complete!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)')
print(f'Transformers: {transformers.__version__}')
PYEOF
"

echo ""
echo "=============================="
echo "✅ RunPod sync complete!"
echo ""
echo "Next steps:"
echo "  1. Open Cursor on RunPod"
echo "  2. File → Open Folder → $REPO_PATH"
echo ""
echo "Or SSH directly:"
echo "  ssh $SSH_HOST -p $SSH_PORT -i $SSH_KEY"
echo "  cd $REPO_PATH"

