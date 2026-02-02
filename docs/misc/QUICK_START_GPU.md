# Quick Start - GPU Server

## Connect
```bash
ssh -p 18748 root@216.81.151.42
```

## Setup (one-time)
```bash
cd /root/mech-interp-latent-lab-phase1
export HF_TOKEN="HF_TOKEN_REDACTED"
export PYTHONPATH=/root/mech-interp-latent-lab-phase1:$PYTHONPATH
```

## Run Experiment
```bash
nohup python3 scripts/run_cross_arch_llama.py > /tmp/cross_arch_llama.log 2>&1 &
tail -f /tmp/cross_arch_llama.log
```

## Monitor
```bash
# Log
tail -f /tmp/cross_arch_llama.log

# GPU
nvidia-smi

# Process
ps aux | grep run_cross_arch_llama
```

## Current Issue
Module import error: `prompts.loader` not found
Fix: Ensure `prompts/loader.py` exists on GPU server
