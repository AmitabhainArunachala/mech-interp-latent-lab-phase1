#!/bin/bash
# Sync pipeline fixes to RunPod
# Usage: ./sync_to_runpod.sh user@runpod-host /workspace/mech-interp-latent-lab-phase1

RUNPOD_HOST="${1:-}"
REMOTE_PATH="${2:-/workspace/mech-interp-latent-lab-phase1}"

if [ -z "$RUNPOD_HOST" ]; then
    echo "Usage: ./sync_to_runpod.sh user@runpod-host [remote_path]"
    echo "Example: ./sync_to_runpod.sh root@123.45.67.89 /workspace/repo"
    exit 1
fi

echo "=== Syncing Pipeline Fixes to RunPod ==="
echo "Host: $RUNPOD_HOST"
echo "Path: $REMOTE_PATH"
echo ""

# Files to sync
FILES=(
    "src/pipelines/confound_validation.py"
    "src/pipelines/rv_l27_causal_validation.py"
    "src/pipelines/head_ablation_validation.py"
    "src/pipelines/registry.py"
    "configs/gold/04_head_validation.json"
    "QUICK_START.md"
    "GOLD_STANDARD_SUITE.md"
    "PIPELINE_COMPOSER_RERUN.md"
)

for file in "${FILES[@]}"; do
    echo "Syncing: $file"
    scp "$file" "$RUNPOD_HOST:$REMOTE_PATH/$file"
done

echo ""
echo "=== Sync Complete ==="
echo ""
echo "Now run on RunPod:"
echo "  python -m src.pipelines.run --config configs/gold/01_existence.json"
echo "  python -m src.pipelines.run --config configs/gold/02_causality.json"
echo "  python -m src.pipelines.run --config configs/gold/04_head_validation.json"
