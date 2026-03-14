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
AMIROS_SESSION="${AMIROS_SESSION:-mistral_scaffold_ablation_confirm_v1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_scaffold_ablation_confirm_v1/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group mistral_scaffold_ablation_confirm_v1 \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Higher-n confirmatory scaffold ladder focused on minimal anchor effect"

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
    --queue-group mistral_scaffold_ablation_confirm_v1 \
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
      --queue-group mistral_scaffold_ablation_confirm_v1 \
      --run-id "$RUN_ID" \
      --status failed \
      --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

capture_latest_artifact() {
  local glob_pattern="$1"
  local label="$2"
  "$PYTHON_BIN" - "$glob_pattern" "$label" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

glob_pattern = sys.argv[1]
label = sys.argv[2]
out_dir = Path(sys.argv[3])
matches = sorted(Path(".").glob(glob_pattern), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit(f"no artifact found for {glob_pattern}")
artifact = matches[-1]
(out_dir / f"{label}_artifact.txt").write_text(str(artifact), encoding="utf-8")
print(artifact)
PY
}

SCAFFOLD_OUT="results/self_feeding_scaffold_ablation_v2/$RUN_ID"
run_step self_feeding_scaffold_ladder_v2 \
  "$PYTHON_BIN" scripts/self_feeding_loop.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda \
  --condition-set scaffold_ladder \
  --max-turns "${SELF_FEED_MAX_TURNS:-50}" \
  --max-new-tokens "${SELF_FEED_MAX_NEW_TOKENS:-160}" \
  --temperature "${SELF_FEED_TEMPERATURE:-0.7}" \
  --rep-penalty "${SELF_FEED_REP_PENALTY:-1.35}" \
  --n-sessions "${SELF_FEED_N_SESSIONS:-16}" \
  --seed-start "${SELF_FEED_SEED_START:-20260314}" \
  --output "$SCAFFOLD_OUT"

capture_latest_artifact "$SCAFFOLD_OUT/self_feeding_summary_*.json" scaffold_ablation_summary_v2 | tee "$OUT_DIR/scaffold_ablation_summary_v2_artifact.log"
SCAFFOLD_ARTIFACT="$(cat "$OUT_DIR/scaffold_ablation_summary_v2_artifact.txt")"

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group mistral_scaffold_ablation_confirm_v1 \
  --experiment-id scaffold_ablation_ladder_v2 \
  --status completed \
  --artifact-path "$SCAFFOLD_ARTIFACT" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --prompt-contract internal_script_protocol \
  --metric-path scripts/self_feeding_loop.py \
  --claim-id SCAFFOLD_ABLATION_V2

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_scaffold_ablation_confirm_v1_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group mistral_scaffold_ablation_confirm_v1 \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
