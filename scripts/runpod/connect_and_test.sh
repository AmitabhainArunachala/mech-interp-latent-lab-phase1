#!/bin/bash
# Interactive SSH connection script for RunPod
# Run this in your terminal: ./connect_and_test.sh

echo "Connecting to RunPod server..."
echo "Once connected, run these commands:"
echo ""
echo "  python3 --version"
echo "  nvidia-smi"
echo "  python3 -c 'import torch; print(torch.__version__)'"
echo ""
echo "Then run the Mistral test:"
echo "  python3 << 'EOF'"
echo "  from transformers import AutoTokenizer, AutoModelForCausalLM"
echo "  import torch"
echo "  print('Loading Mistral-7B-Instruct...')"
echo "  model = AutoModelForCausalLM.from_pretrained("
echo "      'mistralai/Mistral-7B-Instruct-v0.2',"
echo "      torch_dtype=torch.float16,"
echo "      device_map='auto'"
echo "  )"
echo "  tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
echo "  print('Model loaded!')"
echo "  EOF"
echo ""

ssh -i ~/.ssh/id_ed25519 isv37z6krqu4q2-644112db@ssh.runpod.io

