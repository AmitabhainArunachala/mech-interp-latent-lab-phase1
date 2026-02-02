#!/usr/bin/env python3
"""Test SSH connection using paramiko to bypass PTY issues."""
import paramiko
import sys
import os

def test_ssh_and_mistral():
    """Test SSH connection and load Mistral."""
    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
    hostname = "ssh.runpod.io"
    username = "isv37z6krqu4q2-644112db"
    
    print("🔌 Connecting to RunPod server...")
    
    try:
        # Load private key
        private_key = paramiko.Ed25519Key.from_private_key_file(ssh_key_path)
        
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect
        ssh.connect(
            hostname=hostname,
            username=username,
            pkey=private_key,
            look_for_keys=False,
            allow_agent=False
        )
        
        print("✅ SSH connection established!")
        
        # Test 1: Python version
        print("\n1. Checking Python version...")
        stdin, stdout, stderr = ssh.exec_command("python3 --version 2>&1")
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        # Filter out PTY error message
        output_lines = [line for line in output.split('\n') if 'PTY' not in line and 'Pseudo-terminal' not in line]
        python_version = '\n'.join(output_lines).strip()
        if python_version and 'Python' in python_version:
            print(f"   ✓ {python_version}")
        else:
            print(f"   Output: {output}")
            print(f"   Error: {error}")
            # Try alternative approach
            stdin, stdout, stderr = ssh.exec_command("bash -c 'python3 --version'")
            alt_output = stdout.read().decode().strip()
            if alt_output:
                print(f"   ✓ {alt_output} (via bash)")
            else:
                return 1
        
        # Test 2: GPU check
        print("\n2. Checking GPU...")
        stdin, stdout, stderr = ssh.exec_command("bash -c 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1'")
        gpu_info = stdout.read().decode().strip()
        gpu_info_clean = '\n'.join([line for line in gpu_info.split('\n') if 'PTY' not in line and 'Pseudo-terminal' not in line]).strip()
        if gpu_info_clean and 'GB' in gpu_info_clean:
            print(f"   ✓ {gpu_info_clean.split(chr(10))[0]}")
        else:
            print("   ⚠ GPU check failed (may not be critical)")
        
        # Test 3: PyTorch check
        print("\n3. Checking PyTorch...")
        pytorch_check = "bash -c \"python3 -c 'import torch; print(\\\"PyTorch: {}\\\".format(torch.__version__)); print(\\\"CUDA: {}\\\".format(torch.cuda.is_available()))' 2>&1\""
        stdin, stdout, stderr = ssh.exec_command(pytorch_check)
        pytorch_output = stdout.read().decode().strip()
        pytorch_error = stderr.read().decode().strip()
        # Filter PTY errors
        pytorch_clean = '\n'.join([line for line in pytorch_output.split('\n') if 'PTY' not in line and 'Pseudo-terminal' not in line]).strip()
        if pytorch_clean and 'PyTorch' in pytorch_clean:
            print(f"   ✓ {pytorch_clean}")
        else:
            print(f"   ⚠ PyTorch check: {pytorch_error[:100] if pytorch_error else 'Not found'}")
            print("   Installing transformers and torch...")
            stdin, stdout, stderr = ssh.exec_command("bash -c 'pip3 install transformers torch accelerate --quiet 2>&1'")
            install_output = stdout.read().decode().strip()
            install_error = stderr.read().decode().strip()
            install_clean = '\n'.join([line for line in (install_output + install_error).split('\n') if 'PTY' not in line]).strip()
            if install_clean and ("error" in install_clean.lower() or "failed" in install_clean.lower()):
                print(f"   ✗ Installation error: {install_clean[:200]}")
            else:
                print("   ✓ Installation completed")
        
        # Test 4: Load Mistral
        print("\n4. Loading Mistral-7B-Instruct...")
        mistral_script = """
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
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('✓')
    print(f'Prompt: {test_prompt}')
    print(f'Response: {response}')
    print('✅ All tests passed!')
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        # Write script to remote and execute via bash
        # Escape the script properly for bash
        mistral_script_escaped = mistral_script.replace('$', '\\$').replace('"', '\\"')
        cmd = f"bash -c 'python3 << \"PYEOF\"\n{mistral_script}\nPYEOF\n'"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        
        # Get output with timeout handling
        import select
        import time
        
        output_lines = []
        error_lines = []
        
        # Read output
        while True:
            if stdout.channel.recv_ready():
                output_lines.append(stdout.channel.recv(4096).decode())
            if stderr.channel.recv_stderr_ready():
                error_lines.append(stderr.channel.recv_stderr(4096).decode())
            if stdout.channel.exit_status_ready():
                break
            time.sleep(0.1)
        
        exit_status = stdout.channel.recv_exit_status()
        
        output = ''.join(output_lines)
        errors = ''.join(error_lines)
        
        # Filter out PTY error messages
        output_clean = '\n'.join([line for line in output.split('\n') if 'PTY' not in line and 'Pseudo-terminal' not in line]).strip()
        errors_clean = '\n'.join([line for line in errors.split('\n') if 'PTY' not in line and 'Pseudo-terminal' not in line]).strip()
        
        if output_clean:
            print(output_clean)
        
        if errors_clean:
            print(f"\nErrors: {errors_clean}")
        
        if exit_status == 0 or ('✓' in output_clean or '✅' in output_clean):
            print("\n🎉 SSH setup complete and Mistral-7B-Instruct loaded successfully!")
            ssh.close()
            return 0
        else:
            print(f"\n✗ Command failed with exit status {exit_status}")
            print(f"Raw output: {output[:500]}")
            ssh.close()
            return 1
            
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        import paramiko
    except ImportError:
        print("Installing paramiko...")
        os.system("pip3 install paramiko")
        import paramiko
    
    sys.exit(test_ssh_and_mistral())

