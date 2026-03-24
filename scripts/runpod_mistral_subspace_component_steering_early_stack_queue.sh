#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_GROUP="mistral_subspace_component_steering_l3_v1" \
EXPERIMENT_ID="subspace_component_steering_l3_v1" \
CLAIM_ID="SUBSPACE_COMPONENT_STEERING_L3_V1" \
OUTPUT_STEM="subspace_component_steering_l3_v1" \
LAYER="3" \
ALPHAS="${ALPHAS:-2.0,3.0,4.0}" \
TRAIN_PER_GROUP="${TRAIN_PER_GROUP:-6}" \
TEST_PER_GROUP="${TEST_PER_GROUP:-6}" \
GENERATION_SEEDS="${GENERATION_SEEDS:-101,202,303}" \
NOTES="Map early subspace steering at L3 to complete the L4/L5/L25/L27 transformation pattern." \
bash scripts/runpod_mistral_subspace_component_steering_queue.sh

QUEUE_GROUP="mistral_subspace_component_steering_l2_v1" \
EXPERIMENT_ID="subspace_component_steering_l2_v1" \
CLAIM_ID="SUBSPACE_COMPONENT_STEERING_L2_V1" \
OUTPUT_STEM="subspace_component_steering_l2_v1" \
LAYER="2" \
ALPHAS="${ALPHAS:-2.0,3.0,4.0}" \
TRAIN_PER_GROUP="${TRAIN_PER_GROUP:-6}" \
TEST_PER_GROUP="${TEST_PER_GROUP:-6}" \
GENERATION_SEEDS="${GENERATION_SEEDS:-101,202,303}" \
NOTES="Map earliest stable subspace steering at L2 to test whether the steering object simplifies further upstream." \
bash scripts/runpod_mistral_subspace_component_steering_queue.sh
