#!/usr/bin/env python3
"""Run logit lens analysis experiment."""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.logit_lens_analysis import run_logit_lens_analysis_from_config
from src.pipelines.registry import ExperimentResult

if __name__ == "__main__":
    cfg_path = Path(__file__).parent.parent / "configs" / "logit_lens_analysis.json"
    cfg = json.load(open(cfg_path))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results/phase1_mechanism/runs") / f"{timestamp}_logit_lens_analysis"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Config: {cfg_path}")
    print(f"n_pairs: {cfg['params']['n_pairs']}")
    print("")
    print("Starting logit lens analysis...")
    print("="*60)
    
    result = run_logit_lens_analysis_from_config(cfg, run_dir)
    
    print("")
    print("="*60)
    print("✅ Experiment complete!")
    print(f"Results: {run_dir}")
