#!/bin/bash
# DEC9 RunPod Setup Script
# Automatically uploads files and sets up the confound experiment

# Your RunPod connection details
POD_HOST="root@69.19.136.34"
POD_PORT="12777"
SSH_KEY="~/.ssh/id_ed25519"

echo "=================================="
echo "DEC9 Confound Test - RunPod Setup"
echo "=================================="
echo ""
echo "Pod: $POD_HOST"
echo "Port: $POD_PORT"
echo ""

# Test connection
echo "Testing SSH connection..."
ssh -p $POD_PORT -i $SSH_KEY $POD_HOST "echo 'Connection successful!'" || {
    echo "ERROR: Could not connect to RunPod"
    echo "Make sure your pod is running and SSH key is set up"
    exit 1
}

echo ""
echo "✅ Connection successful!"
echo ""

# Create directory on RunPod
echo "Creating project directory on RunPod..."
ssh -p $POD_PORT -i $SSH_KEY $POD_HOST "mkdir -p /workspace/mech-interp-confounds/results"

# Upload files
echo ""
echo "Uploading files..."
echo "  1. Confound prompts..."
scp -P $POD_PORT -i $SSH_KEY \
    ~/mech-interp-latent-lab-phase1/REUSABLE_PROMPT_BANK/confounds.py \
    $POD_HOST:/workspace/mech-interp-confounds/

echo "  2. Test script..."
scp -P $POD_PORT -i $SSH_KEY \
    ~/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_TRACE/run_confound_tests.py \
    $POD_HOST:/workspace/mech-interp-confounds/

echo ""
echo "✅ Files uploaded!"
echo ""

# Install dependencies
echo "Installing dependencies on RunPod..."
ssh -p $POD_PORT -i $SSH_KEY $POD_HOST << 'ENDSSH'
cd /workspace/mech-interp-confounds
pip install -q transformers torch pandas scipy tqdm numpy
echo "✅ Dependencies installed"
ENDSSH

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To run the experiment:"
echo ""
echo "  ssh -p $POD_PORT -i $SSH_KEY $POD_HOST"
echo "  cd /workspace/mech-interp-confounds"
echo "  python run_confound_tests.py"
echo ""
echo "Or run directly (will take ~2-3 hours):"
echo ""
echo "  ssh -p $POD_PORT -i $SSH_KEY $POD_HOST 'cd /workspace/mech-interp-confounds && python run_confound_tests.py'"
echo ""
echo "To download results when done:"
echo ""
echo "  scp -P $POD_PORT -i $SSH_KEY $POD_HOST:/workspace/mech-interp-confounds/results/*.csv ~/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_TRACE/results/"
echo ""

