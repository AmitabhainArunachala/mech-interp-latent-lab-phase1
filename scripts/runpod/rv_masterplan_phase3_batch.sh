#!/bin/bash
# ============================================================
# R_V MASTER PLAN — PHASE 3 GPU BATCH
# ============================================================
# Runs the three remaining GPU experiments in priority order:
#   1. SAE Feature Analysis (E3.1 + E3.4) — ~3h on Gemma-2-2B
#   2. Circuit Tracing (E3.2) — ~2h on Gemma-2-2B
#   3. DII Intervention (E2.4) — ~2h on Mistral-7B
#
# Usage:
#   bash scripts/runpod/rv_masterplan_phase3_batch.sh
#
# Prerequisites:
#   pip install sae-lens transformer-lens  (for SAE analysis)
#   pip install scipy torch transformers
# ============================================================

set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="results/rv_masterplan/phase3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "R_V MASTER PLAN — PHASE 3 GPU BATCH"
echo "Started: $(date)"
echo "Repo: $REPO_ROOT"
echo "Logs: $LOG_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# ── Install dependencies ──
echo ""
echo "[SETUP] Installing dependencies..."
pip install -q sae-lens transformer-lens 2>&1 | tail -5
pip install -q scipy transformers accelerate 2>&1 | tail -5

# ── Experiment 1: SAE Feature Analysis (highest priority) ──
echo ""
echo "============================================================"
echo "[E3.1] SAE FEATURE ANALYSIS — Gemma-2-2B"
echo "Started: $(date)"
echo "============================================================"
python3 scripts/sae_feature_analysis.py \
    --model google/gemma-2-2b \
    --device cuda \
    --n-prompts 20 \
    2>&1 | tee "$LOG_DIR/e3.1_sae_features.log"
SAE_EXIT=$?
echo "[E3.1] Exit code: $SAE_EXIT — $(date)"

# ── Experiment 2: Circuit Tracing ──
echo ""
echo "============================================================"
echo "[E3.2] CIRCUIT TRACING — Gemma-2-2B"
echo "Started: $(date)"
echo "============================================================"
python3 scripts/circuit_tracing_analysis.py \
    --model google/gemma-2-2b \
    --device cuda \
    --n-prompts 20 \
    2>&1 | tee "$LOG_DIR/e3.2_circuit_tracing.log"
CT_EXIT=$?
echo "[E3.2] Exit code: $CT_EXIT — $(date)"

# Clear GPU memory before loading Mistral
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null

# ── Experiment 3: DII Intervention ──
echo ""
echo "============================================================"
echo "[E2.4] DII INTERVENTION — Mistral-7B"
echo "Started: $(date)"
echo "============================================================"
python3 scripts/dii_intervention.py \
    --model mistralai/Mistral-7B-v0.1 \
    --device cuda \
    --n-prompts 20 \
    2>&1 | tee "$LOG_DIR/e2.4_dii_intervention.log"
DII_EXIT=$?
echo "[E2.4] Exit code: $DII_EXIT — $(date)"

# ── Summary ──
echo ""
echo "============================================================"
echo "PHASE 3 BATCH COMPLETE — $(date)"
echo "============================================================"
echo "  E3.1 SAE Features:     exit=$SAE_EXIT"
echo "  E3.2 Circuit Tracing:  exit=$CT_EXIT"
echo "  E2.4 DII Intervention: exit=$DII_EXIT"
echo ""
echo "Results:"
ls -la results/sae_features/ 2>/dev/null
ls -la results/circuit_tracing/ 2>/dev/null
ls -la results/dii_intervention/ 2>/dev/null
echo ""
echo "Logs: $LOG_DIR"
echo "============================================================"
