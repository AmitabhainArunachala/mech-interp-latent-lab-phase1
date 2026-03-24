#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_GROUP="${QUEUE_GROUP:-gpt2xl_core_replication_v1}" \
EXPERIMENT_ID="${EXPERIMENT_ID:-gpt2xl_core_replication_v1}" \
CLAIM_ID="${CLAIM_ID:-GPT2XL_CORE_REPLICATION_V1}" \
MODEL_NAME="${MODEL_NAME:-openai-community/gpt2-xl}" \
P0_N="${P0_N:-80}" \
PATH_N="${PATH_N:-20}" \
PATH_LAYERS="${PATH_LAYERS:-0 4 8 12 16 20 24 28 32 36 40 44}" \
NOTES="${NOTES:-Narrow automated core replication bundle on GPT-2 XL base: canonical P0 plus full path patching on the frozen core contract.}" \
bash scripts/runpod_qwen_core_replication_v1_queue.sh
