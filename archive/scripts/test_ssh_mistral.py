#!/usr/bin/env python3
"""Test SSH connection and load Mistral-7B-Instruct on remote server."""
import subprocess
import sys
import os

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
SSH_HOST = "isv37z6krqu4q2-644112db@ssh.runpod.io"

def run_remote_command(cmd):
    """Run a command on remote server via SSH."""
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        SSH_HOST,
        cmd
    ]
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode

def main():
    print("🔌 Testing SSH connection...")
    
    # Test 1: Python version
    print("\n1. Checking Python version...")
    stdout, stderr, code = run_remote_command("python3 --version")
    if code == 0:
        print(f"   ✓ {stdout.strip()}")
    else:
        print(f"   ✗ Error: {stderr}")
        return 1
    
    # Test 2: GPU check
    print("\n2. Checking GPU...")
    stdout, stderr, code = run_remote_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if code == 0:
        print(f"   ✓ {stdout.strip()}")
    else:
        print(f"   ⚠ GPU check failed (may not be critical): {stderr.strip()}")
    
    # Test 3: PyTorch check
    print("\n3. Checking PyTorch...")
    pytorch_check = """
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"""
    stdout, stderr, code = run_remote_command(f'python3 -c "{pytorch_check}"')
    if code == 0:
        print(f"   ✓ {stdout.strip().replace(chr(10), chr(10) + '   ')}")
    else:
        print(f"   ✗ PyTorch not available: {stderr.strip()}")
        print("   Installing transformers and torch...")
        install_cmd = "pip3 install transformers torch accelerate --quiet"
        stdout, stderr, code = run_remote_command(install_cmd)
        if code != 0:
            print(f"   ✗ Installation failed: {stderr}")
            return 1
    
    # Test 4: Load Mistral model
    print("\n4. Loading Mistral-7B-Instruct...")
    mistral_test = """
import sys
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print('Loading tokenizer...', end=' ', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    print('✓')
    
    print('Loading model...', end=' ', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    print('✓')
    
    print('Running test inference...', end=' ', flush=True)
    test_prompt = 'What is 2+2?'
    inputs = tokenizer(test_prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('✓')
    print(f'Test: {test_prompt}')
    print(f'Response: {response}')
    print('\\n✅ All tests passed!')
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # Escape the command properly
    mistral_test_escaped = mistral_test.replace('"', '\\"').replace('\n', '\\n')
    stdout, stderr, code = run_remote_command(f'python3 -c "{mistral_test_escaped}"')
    
    if code == 0:
        print(stdout)
        print("\n🎉 SSH setup complete and Mistral-7B-Instruct loaded successfully!")
        return 0
    else:
        print(f"✗ Model loading failed:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

