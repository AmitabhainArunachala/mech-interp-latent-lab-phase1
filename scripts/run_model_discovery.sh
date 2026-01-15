#!/bin/bash
# Multi-Model R_V Circuit Discovery Runner
# Usage: ./scripts/run_model_discovery.sh llama3_8b [phase]
#
# Phases:
#   1 = Characterization (CPU)
#   2 = Baseline R_V
#   3 = Source hunt (MLP ablation sweep)
#   4 = Transfer hunt (MLP steering sweep)
#   5 = Readout validation (4-way controls)
#   6 = Head identification
#   all = Run all phases

set -e

MODEL_SHORT=${1:-"llama3_8b"}
PHASE=${2:-"all"}
RESULTS_DIR="results/phase2_generalization/${MODEL_SHORT}"

mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "R_V Circuit Discovery: ${MODEL_SHORT}"
echo "Phase: ${PHASE}"
echo "Results: ${RESULTS_DIR}"
echo "========================================"

run_phase() {
    local phase_num=$1
    local config=$2
    local desc=$3

    echo ""
    echo "--- Phase ${phase_num}: ${desc} ---"

    if [ -f "${config}" ]; then
        python -m src.pipelines.run --config "${config}" --output-dir "${RESULTS_DIR}/phase${phase_num}"
        echo "Phase ${phase_num} complete."
    else
        echo "WARNING: Config not found: ${config}"
        echo "Skipping phase ${phase_num}."
    fi
}

# Phase 1: Characterization (no config needed)
if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
    echo ""
    echo "--- Phase 1: Architecture Characterization ---"
    python -c "
from src.core.models import load_model
import json

model_map = {
    'llama3_8b': 'meta-llama/Meta-Llama-3-8B',
    'gemma2_9b': 'google/gemma-2-9b',
    'phi3_medium': 'microsoft/Phi-3-medium-4k-instruct',
    'qwen2_7b': 'Qwen/Qwen2-7B',
    'mistral_7b': 'mistralai/Mistral-7B-v0.1',
}

model_name = model_map.get('${MODEL_SHORT}', '${MODEL_SHORT}')
print(f'Loading {model_name}...')

try:
    model, tokenizer = load_model(model_name)
    info = {
        'model_name': model_name,
        'num_layers': model.config.num_hidden_layers,
        'num_heads': model.config.num_attention_heads,
        'hidden_size': model.config.hidden_size,
        'late_layer_candidate': model.config.num_hidden_layers - 5,
        'depth_84_pct': int(model.config.num_hidden_layers * 0.84),
    }
    print(json.dumps(info, indent=2))

    with open('${RESULTS_DIR}/00_characterization.json', 'w') as f:
        json.dump(info, f, indent=2)
    print('Saved to ${RESULTS_DIR}/00_characterization.json')
except Exception as e:
    print(f'Error: {e}')
    print('Model may require authentication (HF_TOKEN) or different name.')
"
fi

# Phase 2: Baseline R_V
if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
    run_phase 2 "configs/discovery/${MODEL_SHORT}/01_baseline_rv.json" "Baseline R_V Separation"
fi

# Phase 3: Source Hunt
if [ "$PHASE" = "3" ] || [ "$PHASE" = "all" ]; then
    run_phase 3 "configs/discovery/${MODEL_SHORT}/02_source_hunt_sweep.json" "Source Layer Hunt (MLP Ablation)"
fi

# Phase 4: Transfer Hunt
if [ "$PHASE" = "4" ] || [ "$PHASE" = "all" ]; then
    run_phase 4 "configs/discovery/${MODEL_SHORT}/03_transfer_hunt_sweep.json" "Transfer Sweet Spot Hunt (MLP Steering)"
fi

# Phase 5: Readout Validation
if [ "$PHASE" = "5" ] || [ "$PHASE" = "all" ]; then
    run_phase 5 "configs/canonical/${MODEL_SHORT}/rv_causal_validation.json" "Readout Layer Validation (4-Way Controls)"
fi

# Phase 6: Head Identification
if [ "$PHASE" = "6" ] || [ "$PHASE" = "all" ]; then
    run_phase 6 "configs/canonical/${MODEL_SHORT}/head_ablation.json" "Critical Head Identification"
fi

echo ""
echo "========================================"
echo "Discovery complete for ${MODEL_SHORT}"
echo "Results in: ${RESULTS_DIR}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review results in ${RESULTS_DIR}/"
echo "2. Create CIRCUIT_MAP.md with findings"
echo "3. Compare to Mistral baseline"
