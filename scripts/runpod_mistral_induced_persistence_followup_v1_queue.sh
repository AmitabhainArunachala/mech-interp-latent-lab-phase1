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

AMIROS_POD_NAME="${AMIROS_POD_NAME:-$(hostname)}"
AMIROS_HOST="${AMIROS_HOST:-$(hostname)}"
AMIROS_PORT="${AMIROS_PORT:-22}"
AMIROS_SESSION="${AMIROS_SESSION:-mistral_induced_persistence_followup_v1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_induced_persistence_followup_v1/$RUN_ID"
RUN_OUT="$REPO_ROOT/results/induced_persistence_followup_v1/$RUN_ID"
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
  --queue-group mistral_induced_persistence_followup_v1 \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Test whether anchor-induced ordinary-baseline generations persist after intervention removal"

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
    --queue-group mistral_induced_persistence_followup_v1 \
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
      --queue-group mistral_induced_persistence_followup_v1 \
      --run-id "$RUN_ID" \
      --status failed \
      --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

run_step induced_persistence_followup_v1 \
  "$PYTHON_BIN" scripts/induced_persistence_followup.py \
  --source-run-dir results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory \
  --source-conditions control,bridge_only_3,anchor_bridge_3,anchor_early_mlp_0p125_bridge_3 \
  --baseline-groups baseline_math,baseline_factual,baseline_creative \
  --top-k-per-group 2 \
  --max-turns 12 \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --rep-penalty 1.35 \
  --output-dir "$RUN_OUT"

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group mistral_induced_persistence_followup_v1 \
  --experiment-id induced_persistence_followup_v1 \
  --status completed \
  --artifact-path "${RUN_OUT#$REPO_ROOT/}/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path "scripts/induced_persistence_followup.py --source-run-dir results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory --source-conditions control,bridge_only_3,anchor_bridge_3,anchor_early_mlp_0p125_bridge_3 --baseline-groups baseline_math,baseline_factual,baseline_creative --top-k-per-group 2 --max-turns 12" \
  --prompt-contract induced_seeded_followup_from_anchor_bundle_v5 \
  --metric-path "self_feed_continuation + classify_output + compute_rv_with_components" \
  --claim-id INDUCED_PERSISTENCE_FOLLOWUP_V1

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_induced_persistence_followup_v1_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group mistral_induced_persistence_followup_v1 \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
