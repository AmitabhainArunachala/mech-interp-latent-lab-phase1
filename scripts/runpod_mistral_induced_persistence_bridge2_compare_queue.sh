#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OUTPUT_STEM="${OUTPUT_STEM:-induced_persistence_bridge2_compare_v1}"
export QUEUE_GROUP="${QUEUE_GROUP:-mistral_induced_persistence_bridge2_compare_v1}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-induced_persistence_bridge2_compare_v1}"
export CLAIM_ID="${CLAIM_ID:-INDUCED_PERSISTENCE_BRIDGE2_COMPARE_V1}"
export SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/phase1_mechanism/runs/20260315_005925_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v6_bridge2_bridge3_compare}"
export SOURCE_CONDITIONS="${SOURCE_CONDITIONS:-control,bridge_only_2,bridge_only_3,anchor_bridge_2,anchor_bridge_3,anchor_early_mlp_0p125_bridge_2,anchor_early_mlp_0p125_bridge_3}"
export BASELINE_GROUPS="${BASELINE_GROUPS:-baseline_math,baseline_factual,baseline_creative}"
export TOP_K_PER_GROUP="${TOP_K_PER_GROUP:-4}"
export SELECTION_STRATEGY="${SELECTION_STRATEGY:-median}"
export MAX_TURNS="${MAX_TURNS:-24}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export REP_PENALTY="${REP_PENALTY:-1.35}"
export NOTES="${NOTES:-Median-seed persistence gate comparing bridge=2 against bridge=3 anchor bundles on ordinary baselines.}"
export AMIROS_SESSION="${AMIROS_SESSION:-mistral_induced_persistence_bridge2_compare_v1}"

exec bash "$REPO_ROOT/scripts/runpod_mistral_induced_persistence_followup_queue.sh"
