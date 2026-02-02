#!/usr/bin/env python3
"""
PHASE 3: Success vs Failure Analysis

Fast analysis - no model runs, just feature extraction from existing data.
"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

SEED = 42
N_PAIRS = 20
WINDOW = 16
EARLY_LAYER = 5
LATE_LAYER = 27
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def contains_math_terms(text):
    """Check if text contains mathematical terms."""
    math_keywords = ["calculate", "arithmetic", "problem", "solve", "equation", 
                     "×", "×", "+", "-", "=", "percent", "%", "find"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in math_keywords)

def contains_self_ref(text):
    """Check if text contains self-reference terms."""
    self_ref_keywords = ["self", "yourself", "observe", "notice", "awareness",
                         "consciousness", "process", "generate", "construct"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in self_ref_keywords)

def classify_baseline(text):
    """Classify baseline prompt type."""
    text_lower = text.lower()
    if "story" in text_lower or "continue" in text_lower:
        return "story"
    elif contains_math_terms(text):
        return "math"
    elif "explain" in text_lower or "describe" in text_lower:
        return "factual"
    else:
        return "other"

def jaccard_similarity(tokens1, tokens2):
    """Compute Jaccard similarity between token sets."""
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def extract_features():
    """Extract features for all pairs."""
    set_seed(SEED)
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    
    # Load existing results
    df = pd.read_csv("results/runs/20251216_140553_behavior_strict/behavior_strict_results.csv")
    transfer = df[df["condition"] == "Transfer"]
    
    # Reproduce pairs
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
        
        try:
            rv_rec = compute_rv(model, tokenizer, rec_text, early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW, device=DEVICE)
            if rv_rec < 0.9:
                filtered_pairs.append((rec_text, base_text, r_ids[:common_len], b_ids[:common_len], rv_rec))
        except Exception:
            continue
        
        if len(filtered_pairs) >= N_PAIRS * 2:
            break
    
    for rec_text, base_text, r_ids, b_ids, rv_rec in filtered_pairs[:N_PAIRS]:
        pairs.append((rec_text, base_text, r_ids, b_ids, rv_rec))
    
    # Extract features
    features_list = []
    
    for i, (rec_text, base_text, r_ids, b_ids, rv_rec) in enumerate(pairs):
        # Get transfer result
        t_row = transfer[transfer["pair_idx"] == i]
        if len(t_row) == 0:
            continue
        
        transfer_score = t_row.iloc[0]["final_score"]
        passed_gates = t_row.iloc[0]["passed_gates"]
        failure_reason = t_row.iloc[0].get("failure_reason", None)
        
        # Compute R_V for baseline
        try:
            rv_base = compute_rv(model, tokenizer, base_text, early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW, device=DEVICE)
        except Exception:
            rv_base = np.nan
        
        # Token overlap
        rec_tokens = tokenizer.tokenize(rec_text)
        base_tokens = tokenizer.tokenize(base_text)
        token_overlap = jaccard_similarity(rec_tokens, base_tokens)
        
        # Classify failure mode
        failure_mode = "none"
        if not passed_gates:
            failure_mode = "collapse"
        elif transfer_score == 0.0:
            failure_mode = "no_transfer"
        elif transfer_score < 0.3:
            failure_mode = "weak_transfer"
        elif not passed_gates and transfer_score > 0.0:
            failure_mode = "gate_false_positive"
        
        features = {
            "pair_idx": i,
            # Prompt features
            "rec_length": len(r_ids),
            "base_length": len(b_ids),
            "length_diff": abs(len(r_ids) - len(b_ids)),
            "rec_rv": rv_rec,
            "base_rv": rv_base,
            "rv_gap": rv_base - rv_rec if not np.isnan(rv_base) else np.nan,
            # Semantic features
            "rec_has_math": contains_math_terms(rec_text),
            "rec_has_self_ref": contains_self_ref(rec_text),
            "base_type": classify_baseline(base_text),
            # Overlap features
            "token_overlap": token_overlap,
            # Outcome
            "transfer_score": transfer_score,
            "transfer_success": transfer_score > 0.3,
            "collapsed": not passed_gates,
            "failure_mode": failure_mode,
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def analyze_features(df):
    """Analyze features to find patterns."""
    print("=" * 80)
    print("PHASE 3: SUCCESS vs FAILURE ANALYSIS")
    print("=" * 80)
    
    # Success vs Failure comparison
    success = df[df["transfer_success"] == True]
    failure = df[df["transfer_success"] == False]
    
    print(f"\nSuccess: {len(success)} pairs")
    print(f"Failure: {len(failure)} pairs\n")
    
    print("=== FEATURE COMPARISON ===")
    numeric_features = ["rec_length", "base_length", "length_diff", "rec_rv", "base_rv", "rv_gap", "token_overlap"]
    
    for feat in numeric_features:
        if feat in df.columns:
            success_mean = success[feat].mean()
            failure_mean = failure[feat].mean()
            print(f"\n{feat}:")
            print(f"  Success: {success_mean:.4f}")
            print(f"  Failure: {failure_mean:.4f}")
            print(f"  Difference: {success_mean - failure_mean:+.4f}")
    
    print("\n=== CATEGORICAL FEATURES ===")
    print("\nBase Type Distribution:")
    print(success["base_type"].value_counts())
    print("\nFailure Mode Distribution:")
    print(df["failure_mode"].value_counts())
    
    # Decision tree
    print("\n" + "=" * 80)
    print("DECISION TREE ANALYSIS")
    print("=" * 80)
    
    # Prepare data
    X = df[numeric_features + ["rec_has_math", "rec_has_self_ref"]].copy()
    X["base_type_story"] = (df["base_type"] == "story").astype(int)
    X["base_type_math"] = (df["base_type"] == "math").astype(int)
    X["base_type_factual"] = (df["base_type"] == "factual").astype(int)
    
    # Fill NaN
    X = X.fillna(X.mean())
    
    y = df["transfer_success"].astype(int)
    
    # Train tree
    tree = DecisionTreeClassifier(max_depth=4, min_samples_split=3)
    tree.fit(X, y)
    
    # Export rules
    rules = export_text(tree, feature_names=list(X.columns))
    print("\nDecision Rules:")
    print(rules)
    
    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": tree.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance)
    
    return df, tree, list(X.columns)

def main():
    print("Extracting features...")
    df = extract_features()
    
    # Save features
    df.to_csv("minimal_mechanism_phase3_analysis.csv", index=False)
    print(f"\n✅ Saved features to minimal_mechanism_phase3_analysis.csv")
    
    # Analyze
    df_analyzed, tree, feature_cols = analyze_features(df)
    
    # Save decision tree rules
    rules = export_text(tree, feature_names=feature_cols)
    with open("decision_tree_rules.txt", "w") as f:
        f.write(rules)
    print(f"\n✅ Saved decision tree rules to decision_tree_rules.txt")
    
    return df_analyzed

if __name__ == "__main__":
    main()

