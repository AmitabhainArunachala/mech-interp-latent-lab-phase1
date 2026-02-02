#!/usr/bin/env python3
"""Stage 1 Smoke Test: L0 Necessity (Ablation)"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.mlp_ablation_necessity import run_mlp_ablation_necessity_from_config

# Load config
config_path = Path(__file__).parent.parent / "configs" / "mlp_ablation_necessity_l0.json"
with open(config_path) as f:
    cfg = json.load(f)

# Override n_pairs for smoke test
cfg['params']['n_pairs'] = 5  # Small smoke test

# Create run directory
run_name = 'l0_necessity_smoke_test'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = Path(f'results/phase1_mechanism/runs/{timestamp}_{run_name}')
run_dir.mkdir(parents=True, exist_ok=True)

print('=' * 60)
print('STAGE 1 SMOKE TEST: L0 Necessity (Ablation)')
print('=' * 60)
print(f'Pairs: 5 (smoke test)')
print(f'Run directory: {run_dir}')
print('=' * 60)

# Run experiment
result = run_mlp_ablation_necessity_from_config(cfg, run_dir)

print('=' * 60)
print('✅ Smoke test 1/2 completed!')
print('=' * 60)


