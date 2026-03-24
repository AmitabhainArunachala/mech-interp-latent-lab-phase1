#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export QUEUE_GROUP="${QUEUE_GROUP:-mistral_anchor_reduced_latebundle_confirm_v1}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-anchor_reduced_latebundle_confirm_v1}"
export CLAIM_ID="${CLAIM_ID:-ANCHOR_REDUCED_LATEBUNDLE_CONFIRM_V1}"
export NOTES="${NOTES:-Focused confirm of the reduced late-stack candidates from the minimality ablation against the current static champions.}"
export CONDITION_NAMES="${CONDITION_NAMES:-control,anchor_single_mlp_0p125_bridge_3,anchor_single_mlp_0p125_layermatched_low_bridge_3,anchor_layermatched_low_bridge_3,anchor_drop_L25_vproj_bridge_3,anchor_late_only_bridge_3}"

exec bash "$REPO_ROOT/scripts/runpod_mistral_anchor_layermatched_protocol_confirm_v1_queue.sh"
