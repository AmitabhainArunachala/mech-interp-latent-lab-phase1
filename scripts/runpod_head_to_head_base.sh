#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -f /root/venvs/mistral-hardening/bin/activate ]]; then
  source /root/venvs/mistral-hardening/bin/activate
  if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
  fi
elif [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "No usable Python interpreter found on this pod." >&2
    exit 1
  fi
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/results/head_circuit_runs/$RUN_ID"
mkdir -p "$OUT_DIR"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
DEVICE="${DEVICE:-cuda}"
HEAD_SWEEP_FILE="${HEAD_SWEEP_FILE:-results/full_head_sweep/full_head_sweep_20260312_052013.json}"
N_PROMPTS="${N_PROMPTS:-30}"
TOP_K="${TOP_K:-24}"
MAX_PAIRS="${MAX_PAIRS:-20}"
PAIR_SOURCE="${PAIR_SOURCE:-top_effect}"
PAIR_POOL_SIZE="${PAIR_POOL_SIZE:-8}"
PAIR_PROMPT_LIMIT="${PAIR_PROMPT_LIMIT:-20}"
RANKING_METRIC="${RANKING_METRIC:-entropy_d}"
SINGLE_HEAD_THRESHOLD="${SINGLE_HEAD_THRESHOLD:-0.3}"
MANUAL_HEADS="${MANUAL_HEADS:-}"
CANONICAL_GIT_COMMIT="${CANONICAL_GIT_COMMIT:-unknown}"

if [[ "$MODEL" == *"Instruct"* && "${ALLOW_INSTRUCT:-0}" != "1" ]]; then
  echo "Refusing to run head-to-head on Instruct while base v0.1 is canonical." >&2
  exit 1
fi

if [[ ! -f "$HEAD_SWEEP_FILE" ]]; then
  echo "Missing head sweep file: $HEAD_SWEEP_FILE" >&2
  exit 1
fi

echo "run_id=$RUN_ID" | tee "$OUT_DIR/STATUS.txt"
echo "model=$MODEL" | tee -a "$OUT_DIR/STATUS.txt"
echo "device=$DEVICE" | tee -a "$OUT_DIR/STATUS.txt"
echo "python_bin=$PYTHON_BIN" | tee -a "$OUT_DIR/STATUS.txt"
echo "head_sweep_file=$HEAD_SWEEP_FILE" | tee -a "$OUT_DIR/STATUS.txt"
echo "n_prompts=$N_PROMPTS" | tee -a "$OUT_DIR/STATUS.txt"
echo "top_k=$TOP_K" | tee -a "$OUT_DIR/STATUS.txt"
echo "max_pairs=$MAX_PAIRS" | tee -a "$OUT_DIR/STATUS.txt"
echo "pair_source=$PAIR_SOURCE" | tee -a "$OUT_DIR/STATUS.txt"
echo "pair_pool_size=$PAIR_POOL_SIZE" | tee -a "$OUT_DIR/STATUS.txt"
echo "pair_prompt_limit=$PAIR_PROMPT_LIMIT" | tee -a "$OUT_DIR/STATUS.txt"
echo "ranking_metric=$RANKING_METRIC" | tee -a "$OUT_DIR/STATUS.txt"
echo "single_head_threshold=$SINGLE_HEAD_THRESHOLD" | tee -a "$OUT_DIR/STATUS.txt"
echo "manual_heads=${MANUAL_HEADS:-<none>}" | tee -a "$OUT_DIR/STATUS.txt"
echo "git_commit=$CANONICAL_GIT_COMMIT" | tee -a "$OUT_DIR/STATUS.txt"
echo "hf_hub_offline=$HF_HUB_OFFLINE" | tee -a "$OUT_DIR/STATUS.txt"
echo "transformers_offline=$TRANSFORMERS_OFFLINE" | tee -a "$OUT_DIR/STATUS.txt"
echo "provenance_note=base_v01_only_launcher_for_head_to_head_path_patching" | tee -a "$OUT_DIR/STATUS.txt"
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

CMD=(
  "$PYTHON_BIN" -u scripts/head_to_head_patching.py
  --model "$MODEL"
  --device "$DEVICE"
  --head-sweep-file "$HEAD_SWEEP_FILE"
  --n-prompts "$N_PROMPTS"
  --top-k "$TOP_K"
  --max-pairs "$MAX_PAIRS"
  --pair-source "$PAIR_SOURCE"
  --pair-pool-size "$PAIR_POOL_SIZE"
  --pair-prompt-limit "$PAIR_PROMPT_LIMIT"
  --ranking-metric "$RANKING_METRIC"
  --single-head-threshold "$SINGLE_HEAD_THRESHOLD"
)

if [[ -n "$MANUAL_HEADS" ]]; then
  CMD+=(--manual-heads "$MANUAL_HEADS")
fi

export CANONICAL_GIT_COMMIT
run_step head_to_head_base "${CMD[@]}"

date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT_DIR/STATUS.txt"
echo "head_to_head_base_complete=1" | tee -a "$OUT_DIR/STATUS.txt"
