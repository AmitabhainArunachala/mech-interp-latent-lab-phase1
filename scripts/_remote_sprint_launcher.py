#!/usr/bin/env python3
"""Remote sprint launcher - writes script and launches with correct env for RunPod containers."""
import os, subprocess, sys

WS = "/workspace"
os.environ["HOME"] = WS
os.environ["TMPDIR"] = os.path.join(WS, "tmp")
os.environ["XDG_CACHE_HOME"] = os.path.join(WS, ".cache")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN must be set")

for d in ["tmp", ".cache/huggingface", "hf_cache/hub", "hf_cache/transformers"]:
    os.makedirs(os.path.join(WS, d), exist_ok=True)

# Write HF token
with open(os.path.join(WS, ".cache/huggingface/token"), "w") as f:
    f.write(HF_TOKEN)

# Write sprint script
lines = [
    "#!/bin/bash",
    "export HOME=/workspace",
    "export TMPDIR=/workspace/tmp",
    "export XDG_CACHE_HOME=/workspace/.cache",
    "export HF_HOME=/workspace/hf_cache",
    "export HF_HUB_DISABLE_XET=1",
    "export PYTHONPATH=/workspace/mech-interp-latent-lab-phase1",
    "export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub",
    "export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers",
    "cd /workspace/mech-interp-latent-lab-phase1",
    'echo "=== SPRINT START ===" && date -u',
    'echo "=== STEP 1/3: Llama-3-8B ===" && bash scripts/runpod_llama3_8b_p0_canonical_v1_queue.sh',
    'echo "=== STEP 2/3: Gemma-2-9B ===" && bash scripts/runpod_gemma9b_p0_canonical_v1_queue.sh',
    'echo "=== STEP 3/3: Mixtral-8x7B ===" && bash scripts/runpod_mixtral8x7b_p0_canonical_v1_queue.sh',
    'echo "=== SPRINT DONE ===" && date -u',
]
script_path = os.path.join(WS, "run_sprint.sh")
with open(script_path, "w") as f:
    f.write("\n".join(lines) + "\n")
os.chmod(script_path, 0o755)
print(f"Script written: {os.path.getsize(script_path)} bytes")

# Launch
env = dict(os.environ)
env["HF_TOKEN"] = HF_TOKEN
env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
log_path = os.path.join(WS, "sprint_full.log")
log = open(log_path, "w")
p = subprocess.Popen(
    ["bash", script_path],
    stdout=log, stderr=subprocess.STDOUT,
    env=env, cwd=os.path.join(WS, "mech-interp-latent-lab-phase1"),
    start_new_session=True,
)
print(f"LAUNCHED PID={p.pid}")
print(f"Log: {log_path}")
print("Monitor with: tail -f /workspace/sprint_full.log")
