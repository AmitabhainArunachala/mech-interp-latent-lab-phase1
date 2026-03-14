#!/usr/bin/env bash
# =============================================================================
# P0 RunPod Launch Guide — R_V Paper (COLM 2026)
# =============================================================================
#
# PURPOSE
# -------
# This script documents the exact commands to reproduce the P0 canonical run
# on RunPod GPU instances.  It is both a reference and a direct executor:
#
#   bash p0_runpod_launch.sh [model_key]
#
# If called with a model_key argument it runs that model.  If called with no
# argument it prints the launch plan for all 5 models.
#
# INSTANCE RECOMMENDATIONS (2026-03 RunPod spot pricing)
# -------------------------------------------------------
# For 7B-class models (Mistral, OPT, Qwen):
#   Recommended : A100 SXM 40GB    — $1.19/hr on-demand, ~$0.70/hr spot
#   Fallback    : A5000 24GB       — $0.44/hr on-demand (tight on OPT peak)
#   Avoid       : V100 16GB        — insufficient VRAM for 7B + activations
#
# For GPT-2-XL (1.6B params, 1600 hidden dim):
#   Recommended : RTX 3090 24GB    — $0.44/hr; comfortable VRAM headroom
#   Fallback    : RTX 4090 24GB    — same price bracket, faster CUDA cores
#
# For Pythia-1.4B (null result, smoke test):
#   Recommended : RTX 3090 24GB    — overkill but cheap and available
#   Alternative : Run on CPU (AGNI VPS) — ~30-60s/prompt, fine for n=20
#
# ESTIMATED COST PER MODEL (spot pricing, A100 40GB at $0.70/hr)
# ---------------------------------------------------------------
#   Model load time           : ~30s
#   Measurement rate          : ~4 prompts/min (2 forward passes/prompt)
#   n=100 prompts × 2 cond.   : ~50 min
#   Total GPU time per model  : ~55 min
#   Cost per model            : ~$0.64 at $0.70/hr spot
#   Cost for all 5 models     : ~$3.20 (if run in parallel on 5 pods)
#
# VRAM BUDGET (7B models at bfloat16)
# ------------------------------------
#   Model weights    : ~14 GB
#   Activations      : ~3-4 GB peak (V-projection tensors, 512 tokens)
#   PyTorch overhead : ~1-2 GB
#   Total peak       : ~18-19 GB  →  A100 40GB has 21 GB headroom
#
# PARALLELIZATION STRATEGY
# -------------------------
# Launch 5 independent RunPod pods, one per model.  Each pod:
#   1. Clones the repo
#   2. Installs deps
#   3. Runs p0_canonical_pipeline.py --model <name> --n 100
#   4. Writes result JSON to results/p0_canonical/
#   5. Copies result back (scp or RunPod volume mount)
#
# After all 5 finish (~55 min wall time), run p0_aggregate_results.py locally.
#
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/YOUR_ORG/mech-interp-latent-lab-phase1.git"
REPO_DIR="mech-interp-latent-lab-phase1"
RESULTS_DIR="$REPO_DIR/results/p0_canonical"

# Map of model keys → HF model names.
# The paper's primary Mistral claim is base v0.1, not Instruct v0.2.
declare -A MODELS
MODELS[mistral]="mistralai/Mistral-7B-v0.1"
MODELS[opt]="facebook/opt-6.7b"
MODELS[gpt2xl]="openai-community/gpt2-xl"
MODELS[qwen]="Qwen/Qwen2.5-7B-Instruct"
MODELS[pythia]="EleutherAI/pythia-1.4b"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [model_key]

model_key must be one of: ${!MODELS[@]}

Examples:
    $0 mistral    # Run Mistral-7B-v0.1 (base)
    $0 opt        # Run OPT-6.7B
    $0 gpt2xl     # Run GPT-2-XL
    $0 qwen       # Run Qwen2.5-7B-Instruct
    $0 pythia     # Run Pythia-1.4B

Without argument: print the full launch plan.
EOF
}

# ---------------------------------------------------------------------------
# Install dependencies (run once per pod, idempotent)
# ---------------------------------------------------------------------------
install_deps() {
    echo "=== Installing dependencies ==="
    pip install --quiet --upgrade pip
    pip install --quiet \
        torch torchvision torchaudio \
        transformers>=4.38.0 \
        accelerate>=0.27.0 \
        scipy>=1.11.0 \
        numpy>=1.24.0 \
        tqdm \
        sentencepiece \
        protobuf \
        tiktoken \
        einops
    echo "=== Dependencies installed ==="
}

# ---------------------------------------------------------------------------
# Clone repo (if not already present)
# ---------------------------------------------------------------------------
clone_repo() {
    if [[ ! -d "$REPO_DIR" ]]; then
        echo "=== Cloning repo ==="
        git clone "$REPO_URL" "$REPO_DIR"
    else
        echo "=== Repo already present, pulling latest ==="
        git -C "$REPO_DIR" pull --ff-only
    fi
}

# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------
run_model() {
    local model_key="$1"
    if [[ -z "${MODELS[$model_key]+_}" ]]; then
        echo "ERROR: Unknown model key '$model_key'"
        echo "Valid keys: ${!MODELS[*]}"
        exit 1
    fi

    local model_name="${MODELS[$model_key]}"
    echo ""
    echo "================================================================"
    echo "RUNNING: $model_name"
    echo "================================================================"

    install_deps
    clone_repo

    # Verify CUDA is available
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    total = torch.cuda.get_device_properties(0).total_memory
    print(f'VRAM: {total / 1024**3:.1f} GB')
print(f'PyTorch version: {torch.__version__}')
"

    mkdir -p "$RESULTS_DIR"

    python3 "$REPO_DIR/scripts/p0_canonical_pipeline.py" \
        --model "$model_name" \
        --n 100 \
        --device cuda

    echo ""
    echo "=== DONE: $model_name ==="
    local safe_name
    safe_name=$(echo "$model_name" | tr '/' '__' | tr '.' '-')
    local result_file="$RESULTS_DIR/${safe_name}_p0_result.json"
    if [[ -f "$result_file" ]]; then
        echo "Result saved to: $result_file"
        echo "--- Summary ---"
        python3 -c "
import json, sys
with open('$result_file') as f:
    r = json.load(f)
print(f\"Model     : {r['model_name']}\")
print(f\"Layers    : early={r['early_layer']}, late={r['late_layer']}\")
print(f\"n valid   : selfref={r['n_selfref_valid']}, baseline={r['n_baseline_valid']}\")
print(f\"Mean R_V  : selfref={r.get('selfref_rv_mean', 'N/A'):.4f}  baseline={r.get('baseline_rv_mean', 'N/A'):.4f}\")
print(f\"Hedges g  : {r.get('hedges_g', 'N/A')}\")
print(f\"p (MWU)   : {r.get('p_value_mwu', 'N/A')}\")
print(f\"Direction : {r.get('direction', 'N/A')}\")
print(f\"Sign      : {r.get('sign_label', 'N/A')}\")
"
    else
        echo "WARNING: Expected result file not found at $result_file"
    fi
}

# ---------------------------------------------------------------------------
# Print full launch plan (no argument mode)
# ---------------------------------------------------------------------------
print_plan() {
    cat <<'PLAN'
=============================================================================
P0 CANONICAL RUN — FULL LAUNCH PLAN
=============================================================================

STEP 1: Prepare RunPod instances
---------------------------------
Launch 5 separate pods (or fewer if running sequentially):

  Pod A — Mistral-7B-v0.1 (base)
    Instance : A100 SXM 40GB (or A5000 24GB)
    Template : PyTorch 2.x + CUDA 12.x
    Disk     : 50 GB (model cache)

  Pod B — facebook/opt-6.7b
    Instance : A100 SXM 40GB
    Disk     : 50 GB

  Pod C — openai-community/gpt2-xl
    Instance : RTX 3090 24GB (or A100; GPT-2-XL is only 6 GB at bf16)
    Disk     : 20 GB

  Pod D — Qwen/Qwen2.5-7B-Instruct
    Instance : A100 SXM 40GB
    Disk     : 50 GB

  Pod E — EleutherAI/pythia-1.4b
    Instance : RTX 3090 24GB (or A5000)
    Disk     : 20 GB

STEP 2: On each pod, run:
--------------------------
  # Replace <model_key> with: mistral | opt | gpt2xl | qwen | pythia
  curl -sSL https://raw.githubusercontent.com/YOUR_ORG/mech-interp-latent-lab-phase1/main/scripts/p0_runpod_launch.sh | bash -s -- <model_key>

  OR if you have the repo cloned:
  bash mech-interp-latent-lab-phase1/scripts/p0_runpod_launch.sh <model_key>

STEP 3: Copy results back to Mac
---------------------------------
  # From Mac, after all pods finish:
  scp root@<pod_ip>:mech-interp-latent-lab-phase1/results/p0_canonical/*.json \
      ~/mech-interp-latent-lab-phase1/results/p0_canonical/

  # OR mount the RunPod volume and rsync

STEP 4: Aggregate results
--------------------------
  python3 ~/mech-interp-latent-lab-phase1/scripts/p0_aggregate_results.py

STEP 5: Interpret output
-------------------------
  The aggregate script prints:
    - Per-model: Hedges' g, 95% CI, p (MWU), direction
    - Cross-arch claim verdict: "DEFENSIBLE" or "NOT DEFENSIBLE"
    - Sign reversal resolution: which models had it and whether it is fixed

=============================================================================
CRITICAL REMINDERS
=============================================================================
  1. bfloat16 ONLY. Never float16. The pipeline enforces this automatically.
  2. attn_implementation=eager. Required for V-projection hook registration.
  3. Prompt bank is CANONICAL_CODE/n300_mistral_test_prompt_bank.py.
     Do NOT use any other prompt file for this run.
  4. L3_deeper vs composite-baseline is the ONLY valid comparison for P0.
  5. Effect size is Hedges' g (not raw Cohen's d).
     g < 0 = contraction = expected. g > 0 = sign reversal = problem.
  6. Run BOTH conditions per pod. Do not mix pods across conditions.
=============================================================================
PLAN
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    print_plan
    exit 0
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

run_model "$1"
