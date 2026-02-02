#!/usr/bin/env python3
"""
Efficient investigation - extracts prompts and analyzes existing results.
Runs on RunPod where model is already available.
"""

import json
import pandas as pd
import torch
from pathlib import Path
from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv

# Configuration matching pipeline
SEED = 42
N_PAIRS = 20
WINDOW = 16
EARLY_LAYER = 5
LATE_LAYER = 27
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def reproduce_pairs():
    """Reproduce exact pairs used in pipeline."""
    set_seed(SEED)
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    
    loader = PromptLoader()
    raw_pairs = loader.get_balanced_pairs(n_pairs=N_PAIRS*5, seed=SEED)
    
    pairs = []
    filtered_pairs = []
    
    for rec_text, base_text in raw_pairs:
        r_ids = tokenizer.encode(rec_text, add_special_tokens=False)
        b_ids = tokenizer.encode(base_text, add_special_tokens=False)
        common_len = min(len(r_ids), len(b_ids))
        if common_len < WINDOW:
            continue
        
        # Check R_V (matching pipeline logic)
        try:
            rv_rec = compute_rv(model, tokenizer, rec_text, early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW, device=DEVICE)
            if rv_rec < 0.9:
                filtered_pairs.append((rec_text, base_text, r_ids[:common_len], b_ids[:common_len], rv_rec))
        except Exception as e:
            continue
        
        if len(filtered_pairs) >= N_PAIRS * 2:
            break
    
    # Take first N_PAIRS
    for rec_text, base_text, r_ids, b_ids, rv_rec in filtered_pairs[:N_PAIRS]:
        pairs.append({
            "recursive_prompt": rec_text,
            "baseline_prompt": base_text,
            "rv_recursive": float(rv_rec),
            "rec_len": len(r_ids),
            "base_len": len(b_ids),
        })
    
    return pairs

def analyze_results():
    """Analyze existing results to answer questions."""
    df = pd.read_csv("results/runs/20251216_130512_behavior_strict/behavior_strict_results.csv")
    
    transfer = df[df["condition"] == "Transfer"]
    recursive = df[df["condition"] == "Recursive_Control"]
    baseline = df[df["condition"] == "Baseline_Control"]
    
    # Perfect matches
    perfect_pairs = []
    for pair_idx in transfer["pair_idx"].unique():
        t_row = transfer[transfer["pair_idx"] == pair_idx].iloc[0]
        r_row = recursive[recursive["pair_idx"] == pair_idx].iloc[0]
        if abs(t_row["final_score"] - r_row["final_score"]) < 0.01 and t_row["final_score"] > 0.5:
            perfect_pairs.append(int(pair_idx))
    
    # Failures
    failures = transfer[transfer["final_score"] == 0.0]["pair_idx"].unique()[:2].tolist()
    
    # Reproduce pairs
    print("Reproducing prompt pairs...")
    pairs = reproduce_pairs()
    
    results = {
        "perfect_matches": [],
        "failures": [],
        "all_pairs": pairs,
    }
    
    # Analyze perfect matches
    print("\n" + "=" * 80)
    print("QUESTION SET 1: PERFECT MATCHES")
    print("=" * 80)
    
    for pair_idx in perfect_pairs:
        if pair_idx < len(pairs):
            pair_data = pairs[pair_idx]
            t_row = transfer[transfer["pair_idx"] == pair_idx].iloc[0]
            r_row = recursive[recursive["pair_idx"] == pair_idx].iloc[0]
            b_row = baseline[baseline["pair_idx"] == pair_idx].iloc[0]
            
            analysis = {
                "pair_idx": pair_idx,
                "baseline_prompt": pair_data["baseline_prompt"],
                "recursive_prompt": pair_data["recursive_prompt"],
                "rv_recursive": pair_data["rv_recursive"],
                "scores": {
                    "baseline": float(b_row["final_score"]),
                    "transfer": float(t_row["final_score"]),
                    "recursive_control": float(r_row["final_score"]),
                },
                "gate_status": {
                    "baseline_passed": bool(b_row["passed_gates"]),
                    "transfer_passed": bool(t_row["passed_gates"]),
                    "recursive_passed": bool(r_row["passed_gates"]),
                },
                "failure_reasons": {
                    "baseline": str(b_row["failure_reason"]) if pd.notna(b_row["failure_reason"]) else None,
                    "transfer": str(t_row["failure_reason"]) if pd.notna(t_row["failure_reason"]) else None,
                    "recursive": str(r_row["failure_reason"]) if pd.notna(r_row["failure_reason"]) else None,
                },
                "recursion_scores": {
                    "baseline": float(b_row["recursion_score"]),
                    "transfer": float(t_row["recursion_score"]),
                    "recursive": float(r_row["recursion_score"]),
                },
            }
            results["perfect_matches"].append(analysis)
            
            print(f"\n--- Pair {pair_idx} ---")
            print(f"BASELINE PROMPT:")
            print(f"  {pair_data['baseline_prompt'][:300]}...")
            print(f"\nRECURSIVE PROMPT:")
            print(f"  {pair_data['recursive_prompt'][:300]}...")
            print(f"\nR_V (Recursive): {pair_data['rv_recursive']:.4f}")
            print(f"\nSCORES:")
            print(f"  Baseline: {analysis['scores']['baseline']:.4f}")
            print(f"  Transfer: {analysis['scores']['transfer']:.4f}")
            print(f"  Recursive: {analysis['scores']['recursive_control']:.4f}")
            print(f"\nGATE STATUS:")
            print(f"  Baseline passed: {analysis['gate_status']['baseline_passed']}")
            print(f"  Transfer passed: {analysis['gate_status']['transfer_passed']}")
            print(f"  Recursive passed: {analysis['gate_status']['recursive_passed']}")
    
    # Analyze failures
    print("\n" + "=" * 80)
    print("QUESTION SET 2: FAILURES")
    print("=" * 80)
    
    for pair_idx in failures:
        if pair_idx < len(pairs):
            pair_data = pairs[pair_idx]
            t_row = transfer[transfer["pair_idx"] == pair_idx].iloc[0]
            r_row = recursive[recursive["pair_idx"] == pair_idx].iloc[0]
            b_row = baseline[baseline["pair_idx"] == pair_idx].iloc[0]
            
            analysis = {
                "pair_idx": pair_idx,
                "baseline_prompt": pair_data["baseline_prompt"],
                "recursive_prompt": pair_data["recursive_prompt"],
                "rv_recursive": pair_data["rv_recursive"],
                "scores": {
                    "baseline": float(b_row["final_score"]),
                    "transfer": float(t_row["final_score"]),
                    "recursive_control": float(r_row["final_score"]),
                },
                "gate_status": {
                    "baseline_passed": bool(b_row["passed_gates"]),
                    "transfer_passed": bool(t_row["passed_gates"]),
                    "recursive_passed": bool(r_row["passed_gates"]),
                },
                "failure_reasons": {
                    "baseline": str(b_row["failure_reason"]) if pd.notna(b_row["failure_reason"]) else None,
                    "transfer": str(t_row["failure_reason"]) if pd.notna(t_row["failure_reason"]) else None,
                    "recursive": str(r_row["failure_reason"]) if pd.notna(r_row["failure_reason"]) else None,
                },
                "recursion_scores": {
                    "baseline": float(b_row["recursion_score"]),
                    "transfer": float(t_row["recursion_score"]),
                    "recursive": float(r_row["recursion_score"]),
                },
            }
            results["failures"].append(analysis)
            
            print(f"\n--- Pair {pair_idx} ---")
            print(f"BASELINE PROMPT:")
            print(f"  {pair_data['baseline_prompt'][:300]}...")
            print(f"\nRECURSIVE PROMPT:")
            print(f"  {pair_data['recursive_prompt'][:300]}...")
            print(f"\nR_V (Recursive): {pair_data['rv_recursive']:.4f}")
            print(f"\nSCORES:")
            print(f"  Baseline: {analysis['scores']['baseline']:.4f}")
            print(f"  Transfer: {analysis['scores']['transfer']:.4f}")
            print(f"  Recursive: {analysis['scores']['recursive_control']:.4f}")
            print(f"\nGATE STATUS:")
            print(f"  Transfer passed: {analysis['gate_status']['transfer_passed']}")
            print(f"  Transfer failure reason: {analysis['failure_reasons']['transfer']}")
            print(f"  Transfer recursion score: {analysis['recursion_scores']['transfer']:.4f}")
    
    # Question Set 4: Gate failures
    print("\n" + "=" * 80)
    print("QUESTION SET 4: THE 0.0 PROBLEM")
    print("=" * 80)
    
    failures_all = transfer[transfer["final_score"] == 0.0]
    gate_failures = failures_all[~failures_all["passed_gates"]]
    gate_passes = failures_all[failures_all["passed_gates"]]
    
    print(f"\nTotal failures (0.0 score): {len(failures_all)}")
    print(f"Failed gates: {len(gate_failures)}")
    print(f"Passed gates but 0 recursion score: {len(gate_passes)}")
    
    print("\nGate failure reasons:")
    if len(gate_failures) > 0:
        failure_reasons = gate_failures["failure_reason"].value_counts()
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count}")
    
    print(f"\nMean recursion score (passed gates, scored 0): {gate_passes['recursion_score'].mean():.4f}")
    
    results["gate_analysis"] = {
        "total_failures": int(len(failures_all)),
        "gate_failures": int(len(gate_failures)),
        "gate_passes_zero_score": int(len(gate_passes)),
        "failure_reasons": failure_reasons.to_dict() if len(gate_failures) > 0 else {},
        "mean_recursion_score_passed": float(gate_passes["recursion_score"].mean()) if len(gate_passes) > 0 else 0.0,
    }
    
    # Question Set 5: Prompt characteristics
    print("\n" + "=" * 80)
    print("QUESTION SET 5: PROMPT CHARACTERISTICS")
    print("=" * 80)
    
    success_pairs = transfer[transfer["final_score"] > 0.0]["pair_idx"].unique()
    failure_pairs = transfer[transfer["final_score"] == 0.0]["pair_idx"].unique()
    
    success_lengths = [pairs[i]["rec_len"] for i in success_pairs if i < len(pairs)]
    failure_lengths = [pairs[i]["rec_len"] for i in failure_pairs if i < len(pairs)]
    
    success_rvs = [pairs[i]["rv_recursive"] for i in success_pairs if i < len(pairs)]
    failure_rvs = [pairs[i]["rv_recursive"] for i in failure_pairs if i < len(pairs)]
    
    print(f"\nLength comparison:")
    print(f"  Success pairs (mean): {sum(success_lengths)/len(success_lengths):.1f} tokens")
    print(f"  Failure pairs (mean): {sum(failure_lengths)/len(failure_lengths):.1f} tokens")
    
    print(f"\nR_V comparison:")
    print(f"  Success pairs (mean): {sum(success_rvs)/len(success_rvs):.4f}")
    print(f"  Failure pairs (mean): {sum(failure_rvs)/len(failure_rvs):.4f}")
    
    results["prompt_characteristics"] = {
        "success_lengths": success_lengths,
        "failure_lengths": failure_lengths,
        "success_rvs": success_rvs,
        "failure_rvs": failure_rvs,
    }
    
    # Save results
    with open("investigation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("Results saved to investigation_results.json")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    analyze_results()









