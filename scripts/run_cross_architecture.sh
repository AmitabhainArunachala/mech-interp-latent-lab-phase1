#!/bin/bash
# Run cross-architecture validation experiments

set -e

echo "=== Cross-Architecture Validation ==="
echo ""

# Run Mistral-7B first
echo "1. Running Mistral-7B validation..."
python3 -m src.pipelines.run --config configs/cross_architecture_mistral.json

# Run Llama-3-8B (if available)
echo ""
echo "2. Running Llama-3-8B validation..."
python3 -m src.pipelines.run --config configs/cross_architecture_llama.json || {
    echo "Warning: Llama-3-8B may not be available. Skipping..."
}

echo ""
echo "✅ Cross-architecture validation complete!"
echo ""
echo "Results saved to:"
echo "  - results/phase2_generalization/runs/*_cross_architecture_validation/"
