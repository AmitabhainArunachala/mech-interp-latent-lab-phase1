#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OUTPUT_STEM="${OUTPUT_STEM:-induced_persistence_anchor_layermatched_long_v1}"
export QUEUE_GROUP="${QUEUE_GROUP:-mistral_induced_persistence_anchor_layermatched_long_v1}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-induced_persistence_anchor_layermatched_long_v1}"
export CLAIM_ID="${CLAIM_ID:-INDUCED_PERSISTENCE_ANCHOR_LAYERMATCHED_LONG_V1}"
export SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/anchor_layermatched_protocol_confirm_v1/20260316_092017}"
export SOURCE_CONDITIONS="${SOURCE_CONDITIONS:-control,anchor_single_mlp_0p125_bridge_3,anchor_layermatched_low_bridge_3}"
export BASELINE_GROUPS="${BASELINE_GROUPS:-baseline}"
export TOP_K_PER_GROUP="${TOP_K_PER_GROUP:-8}"
export SELECTION_STRATEGY="${SELECTION_STRATEGY:-median}"
export MAX_TURNS="${MAX_TURNS:-24}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export REP_PENALTY="${REP_PENALTY:-1.35}"
export NOTES="${NOTES:-Long-horizon persistence stress test for the confirmed bridge-3 winner family.}"
export AMIROS_SESSION="${AMIROS_SESSION:-mistral_induced_persistence_anchor_layermatched_long_v1}"

exec bash "$REPO_ROOT/scripts/runpod_mistral_induced_persistence_followup_queue.sh"
