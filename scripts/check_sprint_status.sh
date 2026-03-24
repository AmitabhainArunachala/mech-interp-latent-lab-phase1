#!/bin/bash
# Quick status check for the cross-arch sprint
# Usage: bash scripts/check_sprint_status.sh
POD="root@154.54.102.57"
PORT=15120
KEY="$HOME/.ssh/id_ed25519"

echo "=== Sprint Status $(date) ==="
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $POD -p $PORT -i $KEY '
echo "--- Running processes ---"
ps aux | grep p0_canonical | grep -v grep | awk "{print \$NF}" | head -5
echo ""
echo "--- P0 Results ---"
ls -1t /workspace/mech-interp-latent-lab-phase1/results/p0_canonical/*.json 2>/dev/null
echo ""
echo "--- HF Cache Size ---"
du -sh /workspace/hf_cache/ 2>/dev/null
echo ""
echo "--- GPU Usage ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
' 2>&1
