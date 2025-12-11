#!/bin/bash
# Quick sync to RunPod - bypassing GitHub for now

echo "Syncing to RunPod..."
rsync -avz --progress \
  --exclude='*.csv' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='*.log' \
  -e "ssh" \
  ~/mech-interp-latent-lab-phase1/ \
  runpod-dec10:/workspace/mech-interp-latent-lab-phase1/

echo ""
echo "✅ Sync complete!"
echo ""
echo "Now in your RunPod Cursor window:"
echo "  File → Open Folder → /workspace/mech-interp-latent-lab-phase1"

