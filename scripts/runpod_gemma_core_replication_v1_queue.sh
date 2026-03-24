#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_GROUP="${QUEUE_GROUP:-gemma_core_replication_v1}" \
EXPERIMENT_ID="${EXPERIMENT_ID:-gemma_core_replication_v1}" \
CLAIM_ID="${CLAIM_ID:-GEMMA_CORE_REPLICATION_V1}" \
MODEL_NAME="${MODEL_NAME:-google/gemma-2-9b}" \
P0_N="${P0_N:-80}" \
PATH_N="${PATH_N:-20}" \
PATH_LAYERS="${PATH_LAYERS:-0 3 6 9 12 15 18 21 24 27 30 33 35 38 41}" \
NOTES="${NOTES:-Narrow automated core replication bundle on Gemma base: canonical P0 plus full path patching on the frozen core contract.}" \
bash scripts/runpod_qwen_core_replication_v1_queue.sh
