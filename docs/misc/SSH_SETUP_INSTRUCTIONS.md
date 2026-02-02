# SSH Setup Instructions for RunPod Server

## Connection Details
```bash
ssh isv37z6krqu4q2-644112db@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## Quick Test: Load Mistral-7B-Instruct

Once connected to the remote server, run:

```bash
# 1. Check environment
python3 --version
nvidia-smi

# 2. Install dependencies (if needed)
pip3 install transformers torch accelerate

# 3. Run the test script
# Option A: Copy-paste the code below, or
# Option B: Use the remote_mistral_test.py script

python3 << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading Mistral-7B-Instruct...")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Running test...")
test_prompt = "What is 2+2? Answer:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.7)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
print("✅ Test successful!")
EOF
```

## Alternative: Use the Test Script

If you can copy `remote_mistral_test.py` to the server (via manual copy-paste or other method):

```bash
python3 remote_mistral_test.py
```

## Troubleshooting

If you encounter PTY errors:
- Try connecting with: `ssh -t isv37z6krqu4q2-644112db@ssh.runpod.io -i ~/.ssh/id_ed25519`
- Or use an interactive terminal session

If model loading fails:
- Check GPU memory: `nvidia-smi`
- Try loading with `torch_dtype=torch.bfloat16` instead of `float16`
- Reduce memory usage: `device_map="cpu"` (slower but works)

