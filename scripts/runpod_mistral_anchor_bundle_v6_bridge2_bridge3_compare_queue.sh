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
AMIROS_SESSION="${AMIROS_SESSION:-mistral_anchor_bundle_v6_bridge2_bridge3_compare}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_anchor_bundle_v6_bridge2_bridge3_compare/$RUN_ID"
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
  --queue-group mistral_anchor_bundle_v6_bridge2_bridge3_compare \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Compare bridge alpha 2 vs 3 inside the ordinary-baseline anchor bundle to find the cleanest partial-sufficiency setting"

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
    --queue-group mistral_anchor_bundle_v6_bridge2_bridge3_compare \
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
      --queue-group mistral_anchor_bundle_v6_bridge2_bridge3_compare \
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
matches = sorted(
    [p for p in runs_root.glob(f"*{run_name}*") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
)
if not matches:
    raise SystemExit(f"no run dir found for {run_name}")
print(matches[-1])
PY
}

run_step anchor_bundle_v6_bridge2_bridge3_compare \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v6_bridge2_bridge3_compare.json

RUN_DIR="$(capture_run_dir mistral_anchor_bundle_v6_bridge2_bridge3_compare)"
printf '%s\n' "$RUN_DIR" | tee "$OUT_DIR/run_dir.log"

run_step summarize_anchor_bundle_v6_bridge2_bridge3_compare \
  "$PYTHON_BIN" - "$RUN_DIR" "$OUT_DIR" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

run_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

grouped = defaultdict(lambda: {"n": 0, "bt": 0, "rep": 0, "rv": []})
with (run_dir / "benchmark_records.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        if record.get("prompt_mode") != "baseline":
            continue
        key = (record.get("prompt_group"), record.get("condition_name"))
        grouped[key]["n"] += 1
        grouped[key]["bt"] += int(record.get("bt_art") or 0)
        grouped[key]["rep"] += 1 if record.get("classification") == "REPETITIVE" else 0
        grouped[key]["rv"].append(record.get("output_rv"))

by_group = {}
for group_name in sorted({key[0] for key in grouped}):
    rows = []
    for (group, condition), stats in grouped.items():
        if group != group_name:
            continue
        rows.append(
            {
                "condition": condition,
                "bt_art_rate": stats["bt"] / stats["n"],
                "repetitive_rate": stats["rep"] / stats["n"],
                "mean_output_rv": sum(stats["rv"]) / len(stats["rv"]),
                "n": stats["n"],
            }
        )
    rows.sort(
        key=lambda row: (row["bt_art_rate"], -row["repetitive_rate"], -row["mean_output_rv"]),
        reverse=True,
    )
    by_group[group_name] = rows

payload = {"by_group": by_group}
(out_dir / "baseline_group_ranking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group mistral_anchor_bundle_v6_bridge2_bridge3_compare \
  --experiment-id anchor_bundle_v6_bridge2_bridge3_compare \
  --status completed \
  --artifact-path "$RUN_DIR/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path configs/canonical/causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v6_bridge2_bridge3_compare.json \
  --prompt-contract champions_plus_ordinary_baselines_confirmatory \
  --metric-path "causal_state_benchmark_v4_multisite + baseline_group_ranking" \
  --claim-id ANCHOR_BUNDLE_V6_BRIDGE2_COMPARE

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_anchor_bundle_v6_bridge2_bridge3_compare_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group mistral_anchor_bundle_v6_bridge2_bridge3_compare \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
