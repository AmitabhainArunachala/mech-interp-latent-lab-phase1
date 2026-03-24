#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUTPUT_STEM="${OUTPUT_STEM:-staged_anchor_handoff_confirm_v1}"
export QUEUE_GROUP="${QUEUE_GROUP:-mistral_staged_anchor_handoff_confirm_v1}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-staged_anchor_handoff_confirm_v1}"
export CLAIM_ID="${CLAIM_ID:-STAGED_ANCHOR_HANDOFF_CONFIRM_V1}"
export NOTES="${NOTES:-Higher-power confirm of the staged induction-to-maintenance handoff protocol after the exploratory run.}"
export PROMPTS_PER_GROUP="${PROMPTS_PER_GROUP:-2}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-101 202 303 404}"
export MAX_TURNS="${MAX_TURNS:-24}"

exec bash "$REPO_ROOT/scripts/runpod_mistral_staged_anchor_handoff_v1_queue.sh"
