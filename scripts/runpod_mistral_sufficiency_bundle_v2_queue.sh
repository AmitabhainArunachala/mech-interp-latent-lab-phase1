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
OUT_DIR="$REPO_ROOT/results/mistral_sufficiency_bundle_v2/$RUN_ID"
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
roots = [
    Path("results/phase1_mechanism/runs"),
    Path("results/phase1_cross_architecture/runs"),
]
matches = []
for root in roots:
    if root.exists():
        matches.extend([p for p in root.glob(f"*{run_name}*") if p.is_dir()])
matches = sorted(matches, key=lambda p: p.stat().st_mtime)
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

run_step bridge_low_trunc_confirmatory_n24 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/multi_token_bridge_mistral_low_trunc_confirmatory_n24.json
capture_run_dir mistral_7b_bridge_low_trunc_confirmatory_n24 bridge_low_trunc_confirmatory_n24 | tee "$OUT_DIR/bridge_low_trunc_confirmatory_n24_run_dir.log"

run_step bridge_true_longgen_n18 \
  "$PYTHON_BIN" -m src.pipelines.run \
  --config configs/canonical/multi_token_bridge_mistral_true_longgen_n18.json
capture_run_dir mistral_7b_bridge_true_longgen_n18 bridge_true_longgen_n18 | tee "$OUT_DIR/bridge_true_longgen_n18_run_dir.log"

SELF_FEED_OUT="results/self_feeding_loop_bundle_v2/$RUN_ID"
run_step self_feeding_loop_v2 \
  "$PYTHON_BIN" scripts/self_feeding_loop.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda \
  --max-turns "${SELF_FEED_MAX_TURNS:-50}" \
  --max-new-tokens "${SELF_FEED_MAX_NEW_TOKENS:-128}" \
  --temperature "${SELF_FEED_TEMPERATURE:-0.7}" \
  --rep-penalty "${SELF_FEED_REP_PENALTY:-1.3}" \
  --n-sessions "${SELF_FEED_N_SESSIONS:-8}" \
  --seed-start "${SELF_FEED_SEED_START:-20260313}" \
  --output "$SELF_FEED_OUT"
capture_latest_artifact "$SELF_FEED_OUT/self_feeding_summary_*.json" self_feeding_summary_v2 | tee "$OUT_DIR/self_feeding_summary_v2_artifact.log"

GNANI_OUT="results/sustained_gnani_v3_bundle_v2/$RUN_ID"
run_step sustained_gnani_v3_v2 \
  "$PYTHON_BIN" scripts/sustained_gnani_v3.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda \
  --max-turns "${GNANI_MAX_TURNS:-50}" \
  --max-new-tokens "${GNANI_MAX_NEW_TOKENS:-128}" \
  --temperature "${GNANI_TEMPERATURE:-0.7}" \
  --rep-penalty "${GNANI_REP_PENALTY:-1.3}" \
  --n-recursive "${GNANI_N_RECURSIVE:-8}" \
  --n-baseline "${GNANI_N_BASELINE:-8}" \
  --seed-start "${GNANI_SEED_START:-20260313}" \
  --output "$GNANI_OUT"
capture_latest_artifact "$GNANI_OUT/comparison_summary.json" sustained_gnani_summary_v2 | tee "$OUT_DIR/sustained_gnani_summary_v2_artifact.log"

echo "" | tee -a "$STATUS_FILE"
echo ">>> BUNDLE_MANIFEST $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS_FILE"
write_bundle_manifest | tee "$OUT_DIR/bundle_manifest.log"

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$STATUS_FILE"
echo "mistral_sufficiency_bundle_v2_complete=1" | tee -a "$STATUS_FILE"
