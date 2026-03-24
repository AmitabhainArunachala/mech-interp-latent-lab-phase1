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
AMIROS_SESSION="${AMIROS_SESSION:-sufficiency_multiband_v1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/sufficiency_multiband_v1/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
echo "experiment=sufficiency_multiband_v1" | tee -a "$STATUS_FILE"
echo "hypothesis=multi-band_early_L2-L5_residual_injection_for_geometry-only_sufficiency" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group sufficiency_multiband_v1 \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "Sufficiency ride-or-die: multi-band L2-L5 residual + L25 bridge, 2x2 anchor x geometry factorial"

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
    --queue-group sufficiency_multiband_v1 \
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
      --queue-group sufficiency_multiband_v1 \
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

# ─── Step 1: Run the multiband sufficiency benchmark ───
run_step sufficiency_multiband_v1 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1.json

RUN_DIR="$(capture_run_dir mistral_sufficiency_multiband_v1)"
printf '%s\n' "$RUN_DIR" | tee "$OUT_DIR/run_dir.log"

# ─── Step 2: Extract the 2x2 factorial analysis ───
run_step factorial_2x2_analysis \
  "$PYTHON_BIN" - "$RUN_DIR" "$OUT_DIR" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

run_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

# Load all records
records = []
with (run_dir / "benchmark_records.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        records.append(json.loads(line))

# ── 2x2 factorial: anchor (yes/no) × geometry type (none/single_mlp/multiband) ──
# Focus on BASELINE prompts (the sufficiency test)
baseline_records = [r for r in records if r.get("prompt_mode") == "baseline"]

factorial_conditions = {
    "no_anchor_no_geometry": "control",
    "anchor_no_geometry": "anchor_only",
    "no_anchor_bridge_only": "bridge_only_3",
    "no_anchor_single_mlp_bridge": "single_mlp_0p125_bridge_3",
    "anchor_single_mlp_bridge": "anchor_single_mlp_0p125_bridge_3",
    "no_anchor_multiband_low": "multiband_0p03_bridge_3",
    "no_anchor_multiband_med": "multiband_0p06_bridge_3",
    "no_anchor_multiband_high": "multiband_0p10_bridge_3",
    "anchor_multiband_med": "anchor_multiband_0p06_bridge_3",
    "anchor_multiband_high": "anchor_multiband_0p10_bridge_3",
}

results_2x2 = {}
for label, condition_name in factorial_conditions.items():
    cond_records = [r for r in baseline_records if r.get("condition_name") == condition_name]
    if not cond_records:
        continue
    n = len(cond_records)
    bt_art = sum(1 for r in cond_records if r.get("bt_art"))
    rep = sum(1 for r in cond_records if r.get("classification") == "REPETITIVE")
    rvs = [r.get("output_rv") for r in cond_records if r.get("output_rv") is not None]
    results_2x2[label] = {
        "condition": condition_name,
        "n": n,
        "bt_art_rate": bt_art / n if n else 0,
        "repetitive_rate": rep / n if n else 0,
        "mean_output_rv": sum(rvs) / len(rvs) if rvs else None,
        "bt_art_count": bt_art,
    }

# ── Key comparison: geometry-only vs anchor+geometry vs control ──
control_bt = results_2x2.get("no_anchor_no_geometry", {}).get("bt_art_rate", 0)
key_comparisons = {}
for label, data in results_2x2.items():
    key_comparisons[label] = {
        **data,
        "lift_over_control": data["bt_art_rate"] - control_bt,
    }

# Sort by lift
ranked = sorted(key_comparisons.items(), key=lambda x: -x[1]["lift_over_control"])

# ── The money question: does geometry alone beat control? ──
geometry_only_conditions = [
    "no_anchor_multiband_low",
    "no_anchor_multiband_med",
    "no_anchor_multiband_high",
    "no_anchor_single_mlp_bridge",
    "no_anchor_bridge_only",
]
anchor_conditions = [
    "anchor_multiband_med",
    "anchor_multiband_high",
    "anchor_single_mlp_bridge",
    "anchor_no_geometry",
]

geometry_only_best = max(
    [(k, v) for k, v in key_comparisons.items() if k in geometry_only_conditions],
    key=lambda x: x[1]["lift_over_control"],
    default=(None, None),
)
anchor_best = max(
    [(k, v) for k, v in key_comparisons.items() if k in anchor_conditions],
    key=lambda x: x[1]["lift_over_control"],
    default=(None, None),
)

verdict = {
    "control_baseline_bt_art": control_bt,
    "geometry_only_best": {
        "condition": geometry_only_best[0],
        "bt_art_rate": geometry_only_best[1]["bt_art_rate"] if geometry_only_best[1] else None,
        "lift": geometry_only_best[1]["lift_over_control"] if geometry_only_best[1] else None,
    },
    "anchor_best": {
        "condition": anchor_best[0],
        "bt_art_rate": anchor_best[1]["bt_art_rate"] if anchor_best[1] else None,
        "lift": anchor_best[1]["lift_over_control"] if anchor_best[1] else None,
    },
    "geometry_sufficiency": "YES" if (geometry_only_best[1] and geometry_only_best[1]["lift_over_control"] > 0.10) else "PARTIAL" if (geometry_only_best[1] and geometry_only_best[1]["lift_over_control"] > 0.05) else "NO",
    "multiband_beats_single": None,
}

# Check if multiband beats single-site
multiband_best = max(
    [(k, v) for k, v in key_comparisons.items() if "multiband" in k and "anchor" not in k],
    key=lambda x: x[1]["lift_over_control"],
    default=(None, None),
)
single_best = key_comparisons.get("no_anchor_single_mlp_bridge", {})
if multiband_best[1] and single_best:
    verdict["multiband_beats_single"] = multiband_best[1]["bt_art_rate"] > single_best.get("bt_art_rate", 0)

payload = {
    "factorial_results": results_2x2,
    "ranked_by_lift": [(k, v) for k, v in ranked],
    "verdict": verdict,
}
(out_dir / "factorial_2x2_verdict.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print(json.dumps(verdict, indent=2))
print()
print("=== RANKED BY LIFT OVER CONTROL (baseline prompts) ===")
for label, data in ranked:
    print(f"  {label:40s}  BT+ART={data['bt_art_rate']:.1%}  lift={data['lift_over_control']:+.1%}  RV={data.get('mean_output_rv', 0):.4f}  n={data['n']}")
PY

# ─── Step 3: Per-group breakdown for detailed analysis ───
run_step group_breakdown \
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

# ─── Registry updates ───
"$PYTHON_BIN" -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group sufficiency_multiband_v1 \
  --experiment-id sufficiency_multiband_v1 \
  --status completed \
  --artifact-path "$RUN_DIR/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path configs/canonical/causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1.json \
  --prompt-contract champions_plus_ordinary_baselines \
  --metric-path "causal_state_benchmark_v4_multisite + factorial_2x2_verdict" \
  --claim-id SUFFICIENCY_MULTIBAND_V1

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "sufficiency_multiband_v1_complete=1" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group sufficiency_multiband_v1 \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
