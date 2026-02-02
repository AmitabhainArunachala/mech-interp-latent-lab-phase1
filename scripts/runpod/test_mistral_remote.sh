#!/bin/bash
# Quick test script to load Mistral-7B-Instruct on remote server

ssh -i ~/.ssh/id_ed25519 isv37z6krqu4q2-644112db@ssh.runpod.io << 'EOF'
set -e
echo "=== Environment Check ==="
python3 --version
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"
echo ""
echo "=== Python Packages Check ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>&1 || echo "PyTorch not installed"
echo ""
echo "=== Loading Mistral-7B-Instruct ==="
python3 << 'PYEOF'
import sys
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\nLoading Mistral-7B-Instruct...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✓ Model loaded")
    
    print("\n=== Quick Test ===")
    test_prompt = "What is 2+2?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
    print("\n✓ Model test successful!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install: pip install transformers torch accelerate")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

EOF

