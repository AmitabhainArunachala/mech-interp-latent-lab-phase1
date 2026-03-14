#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f /root/venvs/mistral-hardening/bin/activate ]]; then
  echo "Missing Runpod venv at /root/venvs/mistral-hardening" >&2
  exit 1
fi

source /root/venvs/mistral-hardening/bin/activate

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/causal_state_gate_bridge_scan/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID" | tee "$OUT_DIR/STATUS.txt"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT_DIR/STATUS.txt"

run_step() {
  local name="$1"
  shift
  echo "" | tee -a "$OUT_DIR/STATUS.txt"
  echo ">>> START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT_DIR/STATUS.txt"
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo ">>> DONE  $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT_DIR/STATUS.txt"
  else
    local rc=$?
    echo ">>> FAIL  $name rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT_DIR/STATUS.txt"
    exit $rc
  fi
}

capture_run_dir() {
  local run_name="$1"
  local label="$2"
  python - "$run_name" "$label" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_name = sys.argv[1]
label = sys.argv[2]
out_dir = Path(sys.argv[3])
runs_root = Path("results/phase1_mechanism/runs")
matches = sorted(
    [p for p in runs_root.glob(f"*_{run_name}") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
)
if not matches:
    raise SystemExit(f"no run dir found for {run_name}")
run_dir = matches[-1]
(out_dir / f"{label}_run_dir.txt").write_text(str(run_dir), encoding="utf-8")
print(run_dir)
PY
}

score_gate_runs() {
  python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
labels = ["L5", "L3", "L7"]
scores = {}

def recursive_score(summary: dict) -> tuple[float, float]:
    dose = (summary.get("dose_response") or {}).get("recursive") or {}
    bt = dose.get("bt_art_rate_by_condition") or {}
    rv = dose.get("mean_output_rv_by_condition") or {}
    control = float(bt.get("none") or 0.0)
    toward = None
    for key in ("toward_alpha_3", "toward_alpha_2", "toward_alpha_1"):
        if key in bt:
            toward = key
            break
    if toward is None:
        return (float("-inf"), 0.0)
    bt_delta = float(bt[toward]) - control
    rv_delta = 0.0
    if "none" in rv and toward in rv:
        rv_delta = float(rv["none"]) - float(rv[toward])
    return (bt_delta, rv_delta)

for label in labels:
    run_dir = Path((out_dir / f"{label}_run_dir.txt").read_text(encoding="utf-8").strip())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    bt_delta, rv_delta = recursive_score(summary)
    scores[label] = {
        "run_dir": str(run_dir),
        "bt_delta": bt_delta,
        "rv_contraction_gain": rv_delta,
    }

best = max(scores.items(), key=lambda item: (item[1]["bt_delta"], item[1]["rv_contraction_gain"]))
payload = {
    "best_gate_label": best[0],
    "best_gate_layer": int(best[0][1:]),
    "scores": scores,
}
(out_dir / "gate_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
}

build_multisite_config() {
  python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
selection = json.loads((out_dir / "gate_selection.json").read_text(encoding="utf-8"))
best_layer = int(selection["best_gate_layer"])
cfg_path = Path("configs/canonical/causal_state_benchmark_v4_multisite_L5_L25.json")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
cfg["run_name"] = f"mistral_multisite_gate_L{best_layer}_bridge_L25"
cfg["params"]["source_layers"]["gate"]["layer"] = best_layer
tmp_cfg = out_dir / f"causal_state_benchmark_v4_multisite_L{best_layer}_L25.generated.json"
tmp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(tmp_cfg)
PY
}

run_step gate_L5_v2 \
  python -m src.pipelines.run --config configs/canonical/causal_state_benchmark_v2_gate_L5.json
capture_run_dir mistral_gate_discovery_L5 L5 | tee "$OUT_DIR/gate_L5_run_dir.log"

run_step gate_L3_v2 \
  python -m src.pipelines.run --config configs/canonical/causal_state_benchmark_v2_gate_L3.json
capture_run_dir mistral_gate_discovery_L3 L3 | tee "$OUT_DIR/gate_L3_run_dir.log"

run_step gate_L7_v2 \
  python -m src.pipelines.run --config configs/canonical/causal_state_benchmark_v2_gate_L7.json
capture_run_dir mistral_gate_discovery_L7 L7 | tee "$OUT_DIR/gate_L7_run_dir.log"

echo "" | tee -a "$OUT_DIR/STATUS.txt"
echo ">>> SELECT_GATE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT_DIR/STATUS.txt"
score_gate_runs | tee "$OUT_DIR/gate_selection.log"
MULTISITE_CFG="$(build_multisite_config)"
echo "multisite_config=$MULTISITE_CFG" | tee -a "$OUT_DIR/STATUS.txt"

run_step multisite_v4 \
  python -m src.pipelines.run --config "$MULTISITE_CFG"

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT_DIR/STATUS.txt"
echo "causal_state_gate_bridge_scan_complete=1" | tee -a "$OUT_DIR/STATUS.txt"
