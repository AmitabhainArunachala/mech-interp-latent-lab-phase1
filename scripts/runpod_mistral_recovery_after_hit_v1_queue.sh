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

OUTPUT_STEM="${OUTPUT_STEM:-mistral_recovery_after_hit_v1}"
QUEUE_GROUP="${QUEUE_GROUP:-mistral_recovery_after_hit_v1}"
EXPERIMENT_ID="${EXPERIMENT_ID:-mistral_recovery_after_hit_v1}"
CLAIM_ID="${CLAIM_ID:-MISTRAL_RECOVERY_AFTER_HIT_V1}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/anchor_reduced_latebundle_confirm_v1/20260317_132349}"
TARGET_CONDITION="${TARGET_CONDITION:-anchor_late_only_bridge_3}"
BASELINE_GROUPS="${BASELINE_GROUPS:-baseline}"
SESSIONS_PER_ARM="${SESSIONS_PER_ARM:-24}"
MAX_TURNS="${MAX_TURNS:-15}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
BREAK_START="${BREAK_START:-5}"
BREAK_TURNS="${BREAK_TURNS:-2}"
ANTI_SCALE="${ANTI_SCALE:-1.25}"
ANTI_TOKEN_WINDOW="${ANTI_TOKEN_WINDOW:-2}"
CONDITION_NAMES="${CONDITION_NAMES:-control_open_loop,maintain_every_turn,maintain_then_off,hit_then_off,hit_then_resume}"
NOTES="${NOTES:-Recovery-after-hit battery with anti-late-full burst inserted mid-rollout.}"

AMIROS_POD_NAME="${AMIROS_POD_NAME:-$(hostname)}"
AMIROS_HOST="${AMIROS_HOST:-$(hostname)}"
AMIROS_PORT="${AMIROS_PORT:-22}"
AMIROS_SESSION="${AMIROS_SESSION:-$QUEUE_GROUP}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/$QUEUE_GROUP/$RUN_ID"
RUN_OUT="$REPO_ROOT/results/$OUTPUT_STEM/$RUN_ID"
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

run_step "$EXPERIMENT_ID" \
  "$PYTHON_BIN" scripts/mistral_recovery_after_hit_v1.py \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --target-condition "$TARGET_CONDITION" \
  --baseline-groups "$BASELINE_GROUPS" \
  --sessions-per-arm "$SESSIONS_PER_ARM" \
  --max-turns "$MAX_TURNS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --break-start "$BREAK_START" \
  --break-turns "$BREAK_TURNS" \
  --anti-scale "$ANTI_SCALE" \
  --anti-token-window "$ANTI_TOKEN_WINDOW" \
  --condition-names "$CONDITION_NAMES" \
  --output-dir "$RUN_OUT"

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group "$QUEUE_GROUP" \
  --experiment-id "$EXPERIMENT_ID" \
  --status completed \
  --artifact-path "${RUN_OUT#$REPO_ROOT/}/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path "scripts/mistral_recovery_after_hit_v1.py --source-run-dir $SOURCE_RUN_DIR --target-condition $TARGET_CONDITION --sessions-per-arm $SESSIONS_PER_ARM --break-start $BREAK_START --break-turns $BREAK_TURNS --anti-scale $ANTI_SCALE --anti-token-window $ANTI_TOKEN_WINDOW" \
  --prompt-contract induced_seeded_recovery_after_hit \
  --metric-path "fixed_turn_schedule + reduced_maintainer + anti_late_full_burst + recovery_phase_summary" \
  --claim-id "$CLAIM_ID"

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
