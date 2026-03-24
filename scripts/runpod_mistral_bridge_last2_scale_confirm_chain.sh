#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CONDITION_NAME="${CONDITION_NAME:-anti_bridge_only}"
export TOKEN_WINDOW="${TOKEN_WINDOW:-2}"
export TRAIN_PER_GROUP="${TRAIN_PER_GROUP:-6}"
export TEST_PER_GROUP="${TEST_PER_GROUP:-4}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-101,202,303,404,505,606,707,808,909,1001,1102,1203,1304,1405,1506,1607,1708,1809,1910,2011,2112,2213,2314,2415}"

export SCALE="${SCALE_FIRST:-1.0}"
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh

export SCALE="${SCALE_SECOND:-1.25}"
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh
