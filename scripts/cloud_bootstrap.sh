#!/bin/bash
# =============================================================================
# CLOUD BOOTSTRAP - Run this ONE command on any GPU instance
# =============================================================================
#
# Usage (copy-paste this entire block to your cloud terminal):
#
#   curl -sSL https://raw.githubusercontent.com/YOUR_USER/mech-interp-latent-lab-phase1/main/scripts/cloud_bootstrap.sh | bash
#
# Or if repo is private:
#   git clone https://github.com/YOUR_USER/mech-interp-latent-lab-phase1.git
#   cd mech-interp-latent-lab-phase1 && bash scripts/cloud_bootstrap.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "CIRCUIT MAPPING EXPERIMENT - CLOUD BOOTSTRAP"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the repo or need to clone
if [ ! -f "experiments/full_circuit_mapping.py" ]; then
    echo -e "${YELLOW}Cloning repository...${NC}"
    git clone https://github.com/AmitabhainArunachala/mech-interp-latent-lab-phase1.git 2>/dev/null || \
    git clone git@github.com:AmitabhainArunachala/mech-interp-latent-lab-phase1.git
    cd mech-interp-latent-lab-phase1
fi

echo -e "${GREEN}✓ In repository${NC}"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠ No GPU detected - will run on CPU (slow)${NC}"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch transformers accelerate
pip install --quiet sentence-transformers lancedb
pip install --quiet numpy pandas scipy matplotlib tqdm

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Set HF token if provided
if [ -n "$HF_TOKEN" ]; then
    echo -e "${GREEN}✓ HF_TOKEN is set${NC}"
else
    echo -e "${YELLOW}⚠ HF_TOKEN not set - may fail on gated models${NC}"
    echo "  Set it with: export HF_TOKEN=your_token"
fi

# Create results directory
mkdir -p results/circuit_mapping

echo ""
echo "=============================================="
echo "READY TO RUN"
echo "=============================================="
echo ""
echo "Starting experiment in tmux session 'circuit'..."
echo "You can safely disconnect after this starts."
echo ""
echo "To reconnect later: tmux attach -t circuit"
echo "To check progress:  tail -f results/circuit_mapping/*/experiment.log"
echo ""

# Kill existing tmux session if exists
tmux kill-session -t circuit 2>/dev/null || true

# Start in tmux
tmux new-session -d -s circuit "cd $(pwd) && python experiments/full_circuit_mapping.py --model mistralai/Mistral-7B-Instruct-v0.2 --n-prompts 30 2>&1 | tee experiment_output.log; echo 'EXPERIMENT COMPLETE'; bash"

echo -e "${GREEN}✓ Experiment started in tmux session 'circuit'${NC}"
echo ""
echo "Commands:"
echo "  tmux attach -t circuit     # Watch live output"
echo "  Ctrl+B then D              # Detach (keeps running)"
echo "  tail -f experiment_output.log  # Quick check"
echo ""
echo "Results will be in: results/circuit_mapping/"
echo ""
echo -e "${GREEN}You can now disconnect. Experiment will continue.${NC}"
