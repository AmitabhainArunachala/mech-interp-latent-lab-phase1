#!/usr/bin/env bash
# Cross-Architecture Replication Sprint
# Launches P0 canonical + path patching on Gemma-2-9B, Mixtral-8x7B, Llama-3-8B
#
# Usage:
#   Option A: Run all 3 sequentially on one A100 80GB pod (~10h total)
#     ssh runpod 'cd /workspace/mech-interp-latent-lab-phase1 && bash scripts/launch_cross_arch_replication_sprint.sh'
#
#   Option B: Run on 3 separate pods in parallel (~3-4h total, ~$12)
#     Pod 1 (any A100): bash scripts/runpod_gemma9b_p0_canonical_v1_queue.sh
#     Pod 2 (A100 80GB): bash scripts/runpod_mixtral8x7b_p0_canonical_v1_queue.sh
#     Pod 3 (any A100): bash scripts/runpod_llama3_8b_p0_canonical_v1_queue.sh
#
# Mixtral REQUIRES A100 80GB (47B params, ~35GB in bfloat16 with KV cache)
# Gemma-2-9B and Llama-3-8B work on A100 40GB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Cross-Architecture Replication Sprint ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

# Run sequentially (safe, works on one pod)
echo "=== Step 1/3: Gemma-2-9B (~3h) ==="
bash scripts/runpod_gemma9b_p0_canonical_v1_queue.sh
echo ""

echo "=== Step 2/3: Llama-3-8B (~3h) ==="
bash scripts/runpod_llama3_8b_p0_canonical_v1_queue.sh
echo ""

echo "=== Step 3/3: Mixtral-8x7B (~4h, needs 80GB VRAM) ==="
bash scripts/runpod_mixtral8x7b_p0_canonical_v1_queue.sh
echo ""

echo "=== Sprint Complete ==="
echo "Finish: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Results:"
ls -la results/gemma9b_p0_canonical_v1/*/STATUS.txt 2>/dev/null
ls -la results/llama3_8b_p0_canonical_v1/*/STATUS.txt 2>/dev/null
ls -la results/mixtral8x7b_p0_canonical_v1/*/STATUS.txt 2>/dev/null
echo ""
echo "P0 canonical results:"
ls -la results/p0_canonical/*gemma* results/p0_canonical/*Llama* results/p0_canonical/*Mixtral* 2>/dev/null
echo ""
echo "Path patching results:"
ls -1t results/path_patching/path_patching_summary_*.json 2>/dev/null | head -3
