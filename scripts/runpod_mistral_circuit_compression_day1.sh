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

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/mistral_circuit_compression/$RUN_ID"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/STATUS.txt"
echo "run_id=$RUN_ID" | tee "$STATUS_FILE"
echo "python_bin=$PYTHON_BIN" | tee -a "$STATUS_FILE"
date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"

run_step() {
  local name="$1"
  shift
  echo "" | tee -a "$STATUS_FILE"
  echo ">>> START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo ">>> DONE  $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
  else
    local rc=$?
    echo ">>> FAIL  $name rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
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

write_bundle_manifest() {
  "$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
payload = {"out_dir": str(out_dir), "artifacts": {}}
for path in sorted(out_dir.glob("*_run_dir.txt")):
    payload["artifacts"][path.stem] = path.read_text(encoding="utf-8").strip()
for path in sorted(out_dir.glob("*_artifact.txt")):
    payload["artifacts"][path.stem] = path.read_text(encoding="utf-8").strip()
(out_dir / "bundle_manifest.json").write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
PY
}

run_step multisite_v4 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_gate_bridge_confirmatory.json
capture_run_dir mistral_multisite_gate_l5_bridge_l25_confirmatory multisite_v4 | tee "$OUT_DIR/multisite_v4_run_dir.log"

run_step path_patching \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/discovery/path_patching_mechanism_mistral_circuit_compression.json
capture_run_dir mistral_l25_circuit_compression_path_patch path_patching | tee "$OUT_DIR/path_patching_run_dir.log"

run_step head_ablation \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/mistral_7b_v0_1/head_ablation_l27_h5_readout_base.json
capture_run_dir mistral_l27_h5_readout_ablation_base head_ablation | tee "$OUT_DIR/head_ablation_run_dir.log"

run_step full_head_sweep \
  "$PYTHON_BIN" scripts/full_head_sweep.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda \
  --n-prompts 30 \
  --batch-layers 4
capture_latest_artifact "results/full_head_sweep/full_head_sweep_*.json" full_head_sweep | tee "$OUT_DIR/full_head_sweep_artifact.log"

run_step circuit_tracing \
  "$PYTHON_BIN" scripts/circuit_tracing_analysis.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda \
  --n-prompts 30
capture_latest_artifact "results/circuit_tracing/circuit_trace_*.json" circuit_tracing | tee "$OUT_DIR/circuit_tracing_artifact.log"

echo "" | tee -a "$STATUS_FILE"
echo ">>> BUNDLE_MANIFEST $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
write_bundle_manifest | tee "$OUT_DIR/bundle_manifest.log"

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_circuit_compression_day1_complete=1" | tee -a "$STATUS_FILE"
