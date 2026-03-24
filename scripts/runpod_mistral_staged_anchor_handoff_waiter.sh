#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/staged_anchor_handoff_v1/20260318_002144}"
SOURCE_SUMMARY="${SOURCE_SUMMARY:-$SOURCE_RUN_DIR/summary.json}"
STATUS_FILE="${STATUS_FILE:-results/mistral_staged_anchor_handoff_v1/$(basename "$SOURCE_RUN_DIR")/STATUS.txt}"
POLL_SECONDS="${POLL_SECONDS:-120}"
HANDOFF_MIN="${HANDOFF_MIN:-0.15}"
HANDOFF_MARGIN="${HANDOFF_MARGIN:-0.03}"

echo "waiting_for_summary=$SOURCE_SUMMARY"
while [[ ! -f "$SOURCE_SUMMARY" ]]; do
  if [[ -f "$STATUS_FILE" ]] && grep -q '>>> FAIL' "$STATUS_FILE"; then
    echo "No auto-launch: source run failed according to $STATUS_FILE"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

DECISION="$(
  python3 - "$SOURCE_SUMMARY" "$HANDOFF_MIN" "$HANDOFF_MARGIN" <<'PY'
import json
import sys

summary_path = sys.argv[1]
handoff_min = float(sys.argv[2])
handoff_margin = float(sys.argv[3])

with open(summary_path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

conds = payload["conditions"]
handoff_best = max(
    conds[name]["bt_art_rate"]
    for name in ("handoff_drop_to_late_4", "handoff_drop_to_late_8")
)
seed_best = max(
    conds[name]["bt_art_rate"]
    for name in ("seed_drop_l25_only", "seed_late_only")
)
control = conds["control_open_loop"]["bt_art_rate"]

promote = (
    handoff_best >= handoff_min
    and handoff_best >= seed_best + handoff_margin
    and handoff_best >= control + handoff_margin
)

print("PROMOTE=1" if promote else "PROMOTE=0")
print(f"HANDOFF_BEST={handoff_best:.6f}")
print(f"SEED_BEST={seed_best:.6f}")
print(f"CONTROL={control:.6f}")
PY
)"

echo "$DECISION"

if grep -q '^PROMOTE=1$' <<<"$DECISION"; then
  exec bash "$REPO_ROOT/scripts/runpod_mistral_staged_anchor_handoff_confirm_v1_queue.sh"
fi

echo "No auto-launch: exploratory handoff did not clear the promotion threshold."
