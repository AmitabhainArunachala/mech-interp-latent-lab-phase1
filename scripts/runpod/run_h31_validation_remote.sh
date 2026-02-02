#!/bin/bash
# Quick script to run H31 validation on RunPod GPU

ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 << 'EOF'
cd /workspace/mech-interp-latent-lab-phase1
echo "=========================================="
echo "Running H31 Validation (n=50 prompts)"
echo "=========================================="
python3 h31_validation_n50.py
EOF

echo ""
echo "Results saved on remote. Downloading CSV..."
scp -P 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44:/workspace/mech-interp-latent-lab-phase1/results/h31_validation/h31_validation_n50.csv ./results/h31_validation/ 2>/dev/null || echo "CSV will be in remote results/ directory"









