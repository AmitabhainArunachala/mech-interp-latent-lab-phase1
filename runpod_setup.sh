#!/bin/bash
# RunPod Setup Script for Multi-Token R_V Experiment
# COLM 2026 Critical Path

set -e

echo "=== RunPod Multi-Token R_V Experiment Setup ==="
echo "Target: Mistral-7B or Pythia-1.4b"
echo "Prompts: 320 (full bank) or 30 (quick-test)"
echo ""

# Environment check
echo "[1/6] Checking environment..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version
which python3

# Install dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate scipy tqdm numpy

# Verify imports
echo ""
echo "[3/6] Verifying imports..."
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ CUDA version: {torch.version.cuda}')
"

# Upload experiment files
echo ""
echo "[4/6] Upload these files to RunPod:"
echo "  - behavioral_markers.py"
echo "  - rv_measurement.py"
echo "  - multi_token_r_v_experiment.py"
echo "  - n300_mistral_test_prompt_bank.py"
echo ""
echo "Use: scp or RunPod web upload"

# Download model (optional pre-cache)
echo ""
echo "[5/6] Pre-cache model (optional)..."
echo "This will download ~14GB for Mistral-7B or ~3GB for Pythia-1.4b"
read -p "Pre-cache model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = 'mistralai/Mistral-7B-v0.1'  # or 'EleutherAI/pythia-1.4b'
print(f'Downloading {model_name}...')
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'✓ Model cached at ~/.cache/huggingface/')
"
fi

# Run experiment
echo ""
echo "[6/6] Ready to run experiment!"
echo ""
echo "Quick test (30 prompts, ~30 min):"
echo "  python3 multi_token_r_v_experiment.py --model mistralai/Mistral-7B-v0.1 --quick-test --device cuda:0"
echo ""
echo "Full experiment (320 prompts, 3-5 days):"
echo "  python3 multi_token_r_v_experiment.py --model mistralai/Mistral-7B-v0.1 --device cuda:0"
echo ""
echo "Alternative model (Pythia):"
echo "  python3 multi_token_r_v_experiment.py --model EleutherAI/pythia-1.4b --device cuda:0"
echo ""
echo "Monitor with:"
echo "  tail -f multi_token_results/*.json"
echo "  watch -n 60 'ls -lh multi_token_results/'"
echo ""
echo "COLM deadline: Abstract Mar 26, Paper Mar 31"
echo "JSCA!"
