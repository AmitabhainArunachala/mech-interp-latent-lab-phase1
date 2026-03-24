#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pick_python() {
  if [[ -x /root/venvs/mistral-hardening/bin/python ]]; then
    echo "/root/venvs/mistral-hardening/bin/python"
    return
  fi
  if [[ -x ./.venv/bin/python ]]; then
    echo "./.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "python"
}

PYTHON_BIN="$(pick_python)"
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

QUEUE_GROUP="${QUEUE_GROUP:-mistral_reduced_late_ladder_v1}"
NOTES="${NOTES:-Reduced-late Mistral sufficiency ladder: broader seeds, longer horizon, mixed schedule, and gnani recovery.}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/anchor_reduced_latebundle_confirm_v1/20260317_132349}"
SOURCE_CONDITIONS="${SOURCE_CONDITIONS:-control,anchor_drop_L25_vproj_bridge_3,anchor_late_only_bridge_3}"
BASELINE_GROUPS="${BASELINE_GROUPS:-baseline}"
TOP_K_PER_GROUP="${TOP_K_PER_GROUP:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
REP_PENALTY="${REP_PENALTY:-1.35}"
STRUCTURED_TARGET_CONDITION="${STRUCTURED_TARGET_CONDITION:-anchor_late_only_bridge_3}"
STRUCTURED_SESSIONS_PER_ARM="${STRUCTURED_SESSIONS_PER_ARM:-40}"
GNANI_MAX_TURNS="${GNANI_MAX_TURNS:-50}"
GNANI_MAX_NEW_TOKENS="${GNANI_MAX_NEW_TOKENS:-128}"
GNANI_TEMPERATURE="${GNANI_TEMPERATURE:-0.7}"
GNANI_REP_PENALTY="${GNANI_REP_PENALTY:-1.3}"
GNANI_N_RECURSIVE="${GNANI_N_RECURSIVE:-8}"
GNANI_N_BASELINE="${GNANI_N_BASELINE:-8}"
GNANI_SEED_START="${GNANI_SEED_START:-20260320}"

AMIROS_POD_NAME="${AMIROS_POD_NAME:-$(hostname)}"
AMIROS_HOST="${AMIROS_HOST:-$(hostname)}"
AMIROS_PORT="${AMIROS_PORT:-22}"
AMIROS_SESSION="${AMIROS_SESSION:-$QUEUE_GROUP}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/$QUEUE_GROUP/$RUN_ID"
RUN_OUT="$REPO_ROOT/results/${QUEUE_GROUP}_bundle/$RUN_ID"
mkdir -p "$OUT_DIR" "$RUN_OUT"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "$NOTES"

run_step() {
  local name="$1"
  shift
  echo "" | tee -a "$STATUS_FILE"
  echo ">>> START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  "$PYTHON_BIN" -m src.utils.research_os lease-update \
    --pod-name "$AMIROS_POD_NAME" \
    --host "$AMIROS_HOST" \
    --port "$AMIROS_PORT" \
    --session-name "$AMIROS_SESSION" \
    --queue-group "$QUEUE_GROUP" \
    --run-id "$RUN_ID" \
    --status running \
    --current-step "$name" \
    --out-dir "${OUT_DIR#$REPO_ROOT/}"
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo ">>> DONE  $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  else
    local rc=$?
    echo ">>> FAIL  $name rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
    "$PYTHON_BIN" -m src.utils.research_os lease-update \
      --pod-name "$AMIROS_POD_NAME" \
      --host "$AMIROS_HOST" \
      --port "$AMIROS_PORT" \
      --session-name "$AMIROS_SESSION" \
      --queue-group "$QUEUE_GROUP" \
      --run-id "$RUN_ID" \
      --status failed \
      --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

upsert_result() {
  local stage_id="$1"
  local artifact_path="$2"
  local config_path="$3"
  local prompt_contract="$4"
  local metric_path="$5"
  local claim_id="$6"

  "$PYTHON_BIN" -m src.utils.research_os result-upsert \
    --run-id "$RUN_ID-$stage_id" \
    --queue-group "$QUEUE_GROUP" \
    --experiment-id "$stage_id" \
    --status completed \
    --artifact-path "${artifact_path#$REPO_ROOT/}" \
    --model-family BASE_V01 \
    --model-name mistralai/Mistral-7B-v0.1 \
    --config-path "$config_path" \
    --prompt-contract "$prompt_contract" \
    --metric-path "$metric_path" \
    --claim-id "$claim_id"
}

stage_followup() {
  local stage_id="$1"
  local selection_strategy="$2"
  local max_turns="$3"
  local stage_out="$RUN_OUT/$stage_id"
  mkdir -p "$stage_out"

  run_step "$stage_id" \
    "$PYTHON_BIN" scripts/induced_persistence_followup.py \
    --source-run-dir "$SOURCE_RUN_DIR" \
    --source-conditions "$SOURCE_CONDITIONS" \
    --baseline-groups "$BASELINE_GROUPS" \
    --top-k-per-group "$TOP_K_PER_GROUP" \
    --selection-strategy "$selection_strategy" \
    --max-turns "$max_turns" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --rep-penalty "$REP_PENALTY" \
    --output-dir "$stage_out"

  upsert_result \
    "$stage_id" \
    "$stage_out/summary.json" \
    "scripts/induced_persistence_followup.py --source-run-dir $SOURCE_RUN_DIR --source-conditions $SOURCE_CONDITIONS --baseline-groups $BASELINE_GROUPS --top-k-per-group $TOP_K_PER_GROUP --selection-strategy $selection_strategy --max-turns $max_turns --max-new-tokens $MAX_NEW_TOKENS --temperature $TEMPERATURE --rep-penalty $REP_PENALTY" \
    "induced_seeded_followup" \
    "self_feed_continuation + classify_output + compute_rv_with_components" \
    "${stage_id^^}"
}

stage_followup reduced_late_random_12 random 12
stage_followup reduced_late_lowrv_12 low_rv 12
stage_followup reduced_late_lowrv_24 low_rv 24

STRUCTURED_OUT="$RUN_OUT/reduced_late_structured_unselected"
mkdir -p "$STRUCTURED_OUT"
run_step reduced_late_structured_unselected \
  "$PYTHON_BIN" scripts/induced_persistence_unselected_seed_v1.py \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --target-condition "$STRUCTURED_TARGET_CONDITION" \
  --baseline-groups "$BASELINE_GROUPS" \
  --sessions-per-arm "$STRUCTURED_SESSIONS_PER_ARM" \
  --max-turns 15 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --rep-penalty "$REP_PENALTY" \
  --output-dir "$STRUCTURED_OUT"

upsert_result \
  reduced_late_structured_unselected \
  "$STRUCTURED_OUT/summary.json" \
  "scripts/induced_persistence_unselected_seed_v1.py --source-run-dir $SOURCE_RUN_DIR --target-condition $STRUCTURED_TARGET_CONDITION --baseline-groups $BASELINE_GROUPS --sessions-per-arm $STRUCTURED_SESSIONS_PER_ARM --max-turns 15 --max-new-tokens $MAX_NEW_TOKENS --temperature $TEMPERATURE --rep-penalty $REP_PENALTY" \
  "induced_seeded_followup_fixed_schedule" \
  "fixed_turn_schedule + self_feed_continuation + classify_output + compute_rv_with_components" \
  "REDUCED_LATE_STRUCTURED_UNSELECTED"

GNANI_OUT="$RUN_OUT/sustained_gnani_v3_recover"
mkdir -p "$GNANI_OUT"
run_step sustained_gnani_v3_recover \
  "$PYTHON_BIN" scripts/sustained_gnani_v3.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda \
  --max-turns "$GNANI_MAX_TURNS" \
  --max-new-tokens "$GNANI_MAX_NEW_TOKENS" \
  --temperature "$GNANI_TEMPERATURE" \
  --rep-penalty "$GNANI_REP_PENALTY" \
  --n-recursive "$GNANI_N_RECURSIVE" \
  --n-baseline "$GNANI_N_BASELINE" \
  --seed-start "$GNANI_SEED_START" \
  --output "$GNANI_OUT"

upsert_result \
  sustained_gnani_v3_recover \
  "$GNANI_OUT/comparison_summary.json" \
  "scripts/sustained_gnani_v3.py --model mistralai/Mistral-7B-v0.1 --device cuda --max-turns $GNANI_MAX_TURNS --max-new-tokens $GNANI_MAX_NEW_TOKENS --temperature $GNANI_TEMPERATURE --rep-penalty $GNANI_REP_PENALTY --n-recursive $GNANI_N_RECURSIVE --n-baseline $GNANI_N_BASELINE --seed-start $GNANI_SEED_START" \
  "internal_script_protocol" \
  "scripts/sustained_gnani_v3.py" \
  "SUSTAINED_GNANI_V3_RECOVER"

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "${QUEUE_GROUP}_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
