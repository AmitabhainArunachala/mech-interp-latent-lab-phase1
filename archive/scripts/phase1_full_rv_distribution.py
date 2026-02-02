"""
phase1_full_rv_distribution.py

Task: Measure R_V distribution across the entire prompt bank (370 prompts).
Goal: Understand the natural variance of R_V across different categories (pillars/groups).
This will help us see if "Questioning" prompts naturally have low R_V, or if it's specific to Recursion.

Categories in Bank:
- dose_response (Recursive, Levels 1-5)
- baselines (Factual, Creative, Math, Impossible, etc.)
- control (Repetitive, Long, Pseudo-Recursive)
- generality (Philosophical, Abstract)
- kill_switch (Safety commands)

Metrics:
- R_V (PR_late / PR_early)
- PR_early, PR_late
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv, participation_ratio
from src.core.hooks import capture_v_projection
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 16

def run_distribution_test():
    print("Initializing Phase 1: Full R_V Distribution Test...")
    set_seed(42)
    
    # Load Model
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # Get ALL prompts
    # loader.prompts is a dict keyed by ID
    all_prompts = loader.prompts
    print(f"Loaded {len(all_prompts)} prompts from bank.")
    
    results = []
    
    for pid, pdata in tqdm(all_prompts.items(), desc="Processing Prompts"):
        text = pdata["text"]
        pillar = pdata.get("pillar", "unknown")
        group = pdata.get("group", "unknown")
        ptype = pdata.get("type", "unknown")
        level = pdata.get("level", 0)
        
        # Compute R_V
        # detailed metrics
        enc = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
        
        v_early = None
        v_late = None
        
        # Layers 5 and 27
        with capture_v_projection(model, 5) as s5:
            with torch.no_grad():
                model(**enc)
            v_early = s5.get("v")
            
        with capture_v_projection(model, 27) as s27:
            with torch.no_grad():
                model(**enc)
            v_late = s27.get("v")
            
        pr_early = participation_ratio(v_early, WINDOW_SIZE)
        pr_late = participation_ratio(v_late, WINDOW_SIZE)
        rv = pr_late / pr_early if pr_early > 0 else float('nan')
        
        results.append({
            "id": pid,
            "pillar": pillar,
            "group": group,
            "type": ptype,
            "level": level,
            "text": text[:50] + "...",
            "pr_early": pr_early,
            "pr_late": pr_late,
            "rv": rv
        })
        
    # Save Results
    os.makedirs("results/dec11_evening", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/dec11_evening/phase1_full_distribution.csv", index=False)
    
    # Analysis
    print("\n--- Analysis by Pillar ---")
    print(df.groupby("pillar")["rv"].agg(['mean', 'std', 'count']))
    
    print("\n--- Analysis by Group (Recursive) ---")
    rec_df = df[df["pillar"] == "dose_response"]
    print(rec_df.groupby("group")["rv"].agg(['mean', 'std', 'count']))
    
    print("\n--- Analysis by Group (Baselines) ---")
    base_df = df[df["pillar"] == "baselines"]
    print(base_df.groupby("group")["rv"].agg(['mean', 'std', 'count']))

    print("\n--- Analysis by Group (Control) ---")
    ctrl_df = df[df["pillar"] == "control"]
    print(ctrl_df.groupby("group")["rv"].agg(['mean', 'std', 'count']))

if __name__ == "__main__":
    run_distribution_test()

