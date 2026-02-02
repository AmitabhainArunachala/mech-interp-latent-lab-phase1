#!/usr/bin/env python3
"""Direct SSH test with output capture."""
import subprocess
import sys

SSH_CMD = [
    "ssh",
    "-i", "~/.ssh/id_ed25519",
    "-o", "StrictHostKeyChecking=no",
    "-o", "LogLevel=ERROR",
    "isv37z6krqu4q2-644112db@ssh.runpod.io"
]

def test_connection():
    """Test basic connection."""
    cmd = SSH_CMD + ["python3 --version"]
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    return result.returncode == 0

def load_mistral():
    """Load Mistral model."""
    python_code = '''
import sys
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("Loading Mistral-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully!")
    test_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**test_input, max_new_tokens=10)
    print("Test inference successful!")
    print("Response:", tokenizer.decode(output[0]))
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
    
    cmd = SSH_CMD + [f'python3 -c "{python_code.replace(chr(34), chr(92)+chr(34))}"']
    print("Running Mistral load test...")
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
    print(f"\nReturn code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")
    return result.returncode == 0

if __name__ == "__main__":
    print("Testing SSH connection...")
    if test_connection():
        print("\n✓ Connection works!")
        print("\nLoading Mistral-7B-Instruct...")
        if load_mistral():
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Model loading failed")
            sys.exit(1)
    else:
        print("\n✗ Connection failed")
        sys.exit(1)

