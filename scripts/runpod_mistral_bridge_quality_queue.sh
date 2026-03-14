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
AMIROS_SESSION="${AMIROS_SESSION:-mistral_bridge_quality}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_bridge_quality/$RUN_ID"
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
  --queue-group bridge_quality \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Quality-focused bridge queue"

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
    --queue-group bridge_quality \
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
      --queue-group bridge_quality \
      --run-id "$RUN_ID" \
      --status failed \
      --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

capture_run_dir() {
  local run_name="$1"
  local label="$2"
  "$PYTHON_BIN" - "$run_name" "$label" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

run_name = sys.argv[1]
label = sys.argv[2]
out_dir = Path(sys.argv[3])
root = Path("results/phase1_cross_architecture/runs")
matches = sorted(
    [p for p in root.glob(f"*{run_name}*") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
)
if not matches:
    raise SystemExit(f"no run dir found for {run_name}")
run_dir = matches[-1]
(out_dir / f"{label}_run_dir.txt").write_text(str(run_dir), encoding="utf-8")
print(run_dir)
PY
}

run_step bridge_low_trunc_quality_n24 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/multi_token_bridge_mistral_low_trunc_quality_n24.json
BRIDGE_LOW_RUN_DIR="$(capture_run_dir mistral_7b_bridge_low_trunc_quality_n24 bridge_low_trunc_quality_n24)"
printf '%s\n' "$BRIDGE_LOW_RUN_DIR" | tee "$OUT_DIR/bridge_low_trunc_quality_n24_run_dir.log"
"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group bridge_quality \
  --experiment-id bridge_low_trunc_quality_n24 \
  --status completed \
  --artifact-path "$BRIDGE_LOW_RUN_DIR/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path configs/canonical/multi_token_bridge_mistral_low_trunc_quality_n24.json \
  --prompt-contract heldout_causal_slice \
  --metric-path "src/metrics/behavioral_bridge.py + src/pipelines/canonical/multi_token_bridge.py" \
  --claim-id BRIDGE_QUALITY_LOW_TRUNC

run_step bridge_true_longgen_quality_n18 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/multi_token_bridge_mistral_true_longgen_quality_n18.json
BRIDGE_LONGGEN_RUN_DIR="$(capture_run_dir mistral_7b_bridge_true_longgen_quality_n18 bridge_true_longgen_quality_n18)"
printf '%s\n' "$BRIDGE_LONGGEN_RUN_DIR" | tee "$OUT_DIR/bridge_true_longgen_quality_n18_run_dir.log"
"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group bridge_quality \
  --experiment-id bridge_true_longgen_quality_n18 \
  --status completed \
  --artifact-path "$BRIDGE_LONGGEN_RUN_DIR/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path configs/canonical/multi_token_bridge_mistral_true_longgen_quality_n18.json \
  --prompt-contract heldout_causal_slice \
  --metric-path "src/metrics/behavioral_bridge.py + src/pipelines/canonical/multi_token_bridge.py" \
  --claim-id BRIDGE_QUALITY_LONGGEN

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_bridge_quality_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group bridge_quality \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
