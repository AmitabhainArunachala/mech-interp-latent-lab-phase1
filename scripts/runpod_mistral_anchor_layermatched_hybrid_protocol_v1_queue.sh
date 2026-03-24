#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export QUEUE_GROUP="${QUEUE_GROUP:-mistral_anchor_layermatched_hybrid_protocol_v1}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-anchor_layermatched_hybrid_protocol_v1}"
export CLAIM_ID="${CLAIM_ID:-ANCHOR_LAYERMATCHED_HYBRID_PROTOCOL_V1}"
export NOTES="${NOTES:-Hybrid static protocol that combines the stronger layer-matched inducer with the older subtle-L4 maintenance assist.}"
export TRAIN_PER_GROUP="${TRAIN_PER_GROUP:-6}"
export TEST_PER_GROUP="${TEST_PER_GROUP:-6}"
export CONDITION_NAMES="${CONDITION_NAMES:-control,anchor_bridge_2,anchor_single_mlp_0p125_bridge_3,anchor_layermatched_low_bridge_2,anchor_layermatched_low_bridge_3,anchor_single_mlp_0p125_layermatched_low_bridge_2,anchor_single_mlp_0p125_layermatched_low_bridge_3}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-101 202 303 404 505 606 707 808}"
export AMIROS_SESSION="${AMIROS_SESSION:-mistral_anchor_layermatched_hybrid_protocol_v1}"

exec bash "$REPO_ROOT/scripts/runpod_mistral_anchor_layermatched_protocol_confirm_v1_queue.sh"
