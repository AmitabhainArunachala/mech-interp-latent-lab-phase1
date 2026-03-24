#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_GROUP="${QUEUE_GROUP:-opt_core_replication_v1}" \
EXPERIMENT_ID="${EXPERIMENT_ID:-opt_core_replication_v1}" \
CLAIM_ID="${CLAIM_ID:-OPT_CORE_REPLICATION_V1}" \
MODEL_NAME="${MODEL_NAME:-facebook/opt-6.7b}" \
P0_N="${P0_N:-80}" \
PATH_N="${PATH_N:-20}" \
PATH_LAYERS="${PATH_LAYERS:-0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30}" \
NOTES="${NOTES:-Narrow automated core replication bundle on OPT-6.7B base: canonical P0 plus full path patching on the frozen core contract.}" \
bash scripts/runpod_qwen_core_replication_v1_queue.sh
