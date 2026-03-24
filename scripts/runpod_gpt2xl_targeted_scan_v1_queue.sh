#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_GROUP="${QUEUE_GROUP:-gpt2xl_targeted_scan_v1}" \
EXPERIMENT_ID="${EXPERIMENT_ID:-gpt2xl_targeted_scan_v1}" \
CLAIM_ID="${CLAIM_ID:-GPT2XL_TARGETED_SCAN_V1}" \
CONFIG_PATH="${CONFIG_PATH:-configs/canonical/causal_state_targeted_scan_v1_gpt2xl_base_v1_deterministic.json}" \
bash scripts/runpod_qwen_targeted_scan_v1_queue.sh
