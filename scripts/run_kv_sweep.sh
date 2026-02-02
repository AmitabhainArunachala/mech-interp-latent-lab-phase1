#!/bin/bash
# Run KV layer sweep experiments

cd /root/mech-interp-latent-lab-phase1

echo "=== KV Layer Sweep Experiments ==="
echo ""

for config in configs/kv_sweep_l0_l8.json configs/kv_sweep_l8_l16.json configs/kv_sweep_l16_l24.json configs/kv_sweep_l24_l32.json; do
    layer_range=$(basename $config .json | sed 's/kv_sweep_//')
    echo "Running: $layer_range"
    
    python3 -c "
import sys
sys.path.insert(0, '.')
from src.pipelines.kv_mechanism import run_kv_mechanism_from_config
from pathlib import Path
import json
from datetime import datetime

cfg = json.load(open('$config'))
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = Path('results/phase1_mechanism/runs') / f'{timestamp}_kv_sweep_${layer_range}'
run_dir.mkdir(parents=True, exist_ok=True)

print(f'Layer range: {cfg[\"params\"][\"kv_layer_range\"]}')
result = run_kv_mechanism_from_config(cfg, run_dir)

print(f'Transfer efficiency: {result.summary.get(\"transfer_efficiency\", 0):.1f}%')
print(f'Results: {run_dir}')
print('')
" 2>&1 | tee /tmp/kv_sweep_${layer_range}.log
    
    echo "✅ $layer_range complete"
    echo ""
done

echo "=== All KV Sweep Experiments Complete ==="
