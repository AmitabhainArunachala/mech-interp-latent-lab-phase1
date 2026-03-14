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
AMIROS_SESSION="${AMIROS_SESSION:-mistral_anchor_bundle_v1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_anchor_bundle_v1/$RUN_ID"
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
  --queue-group mistral_anchor_bundle_v1 \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Minimal anchor plus L25 bridge sufficiency lane"

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
    --queue-group mistral_anchor_bundle_v1 \
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
      --queue-group mistral_anchor_bundle_v1 \
      --run-id "$RUN_ID" \
      --status failed \
      --current-step "$name" \
      --out-dir "${OUT_DIR#$REPO_ROOT/}"
    exit "$rc"
  fi
}

capture_run_dir() {
  local run_name="$1"
  "$PYTHON_BIN" - "$run_name" <<'PY'
import sys
from pathlib import Path

run_name = sys.argv[1]
runs_root = Path("results/phase1_mechanism/runs")
matches = sorted([p for p in runs_root.glob(f"*{run_name}*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit(f"no run dir found for {run_name}")
print(matches[-1])
PY
}

run_step anchor_bundle_v1 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v1.json

RUN_DIR="$(capture_run_dir mistral_anchor_bundle_v1)"
printf '%s\n' "$RUN_DIR" | tee "$OUT_DIR/run_dir.log"

run_step summarize_anchor_bundle_v1 \
  "$PYTHON_BIN" - "$RUN_DIR" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

by_condition = summary.get("by_condition", {})
rows = []
for name, payload in by_condition.items():
    baseline = (payload.get("by_prompt_mode") or {}).get("baseline") or {}
    recursive = (payload.get("by_prompt_mode") or {}).get("recursive") or {}
    rows.append({
        "condition": name,
        "baseline_bt_art_rate": baseline.get("bt_art_rate"),
        "recursive_bt_art_rate": recursive.get("bt_art_rate"),
        "baseline_mean_output_rv": baseline.get("mean_output_rv"),
        "recursive_mean_output_rv": recursive.get("mean_output_rv"),
        "baseline_malformed_rate": baseline.get("malformed_rate"),
        "baseline_repetitive_rate": baseline.get("repetitive_rate"),
        "prompt_suffix_applied": any(
            item.get("name") == name and item.get("prompt_suffix_by_mode")
            for item in summary.get("multisite_interventions", [])
        ),
    })

def score(row):
    baseline = row.get("baseline_bt_art_rate") or 0.0
    recursive = row.get("recursive_bt_art_rate") or 0.0
    rv = row.get("baseline_mean_output_rv")
    rv_term = 0.0 if rv is None else -(rv)
    return (baseline, recursive, rv_term)

rows_sorted = sorted(rows, key=score, reverse=True)
payload = {
    "verdict": summary.get("verdict"),
    "top_by_baseline_induction": rows_sorted[:8],
    "control_baseline_bt_art_rate": ((by_condition.get("control") or {}).get("by_prompt_mode") or {}).get("baseline", {}).get("bt_art_rate"),
    "control_recursive_bt_art_rate": ((by_condition.get("control") or {}).get("by_prompt_mode") or {}).get("recursive", {}).get("bt_art_rate"),
}
(out_dir / "anchor_bundle_ranking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group mistral_anchor_bundle_v1 \
  --experiment-id anchor_bundle_v1 \
  --status completed \
  --artifact-path "$RUN_DIR/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path configs/canonical/causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v1.json \
  --prompt-contract heldout_causal_slice_plus_anchor_suffix \
  --metric-path "causal_state_benchmark_v4_multisite + anchor_bundle_ranking" \
  --claim-id ANCHOR_BUNDLE_V1

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_anchor_bundle_v1_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group mistral_anchor_bundle_v1 \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"

