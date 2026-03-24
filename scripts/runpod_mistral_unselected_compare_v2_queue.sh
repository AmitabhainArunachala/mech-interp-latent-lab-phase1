#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log_step() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

log_step "1/2 higher-power unselected-seed robustness on reduced late-only maintainer"
OUTPUT_STEM=induced_persistence_unselected_reduced_late_only_v2 \
QUEUE_GROUP=mistral_induced_persistence_unselected_reduced_late_only_v2 \
EXPERIMENT_ID=induced_persistence_unselected_reduced_late_only_v2 \
CLAIM_ID=INDUCED_PERSISTENCE_UNSELECTED_REDUCED_LATE_ONLY_V2 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_late_only_bridge_3 \
BASELINE_GROUPS=baseline \
SESSIONS_PER_ARM=48 \
MAX_TURNS=15 \
bash scripts/runpod_mistral_induced_persistence_unselected_seed_v1_queue.sh

log_step "2/2 higher-power unselected-seed robustness on reduced drop-L25 maintainer"
OUTPUT_STEM=induced_persistence_unselected_reduced_drop_l25_v2 \
QUEUE_GROUP=mistral_induced_persistence_unselected_reduced_drop_l25_v2 \
EXPERIMENT_ID=induced_persistence_unselected_reduced_drop_l25_v2 \
CLAIM_ID=INDUCED_PERSISTENCE_UNSELECTED_REDUCED_DROP_L25_V2 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_drop_L25_vproj_bridge_3 \
BASELINE_GROUPS=baseline \
SESSIONS_PER_ARM=48 \
MAX_TURNS=15 \
bash scripts/runpod_mistral_induced_persistence_unselected_seed_v1_queue.sh

log_step "unselected compare v2 complete"
