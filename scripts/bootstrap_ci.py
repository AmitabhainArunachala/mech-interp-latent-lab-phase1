#!/usr/bin/env python3
"""
Bootstrap confidence intervals for all key R_V treatment effects.
P0 COLM 2026 submission requirement.

Computes BCa (bias-corrected and accelerated) bootstrap CIs for Cohen's d
across all core effects: mode atlas, cross-architecture, causal validation.

Usage:
    python scripts/bootstrap_ci.py [--n-bootstrap 10000] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

# ============================================================
# Bootstrap utilities
# ============================================================

def cohens_d(group1, group2):
    """Compute Cohen's d (pooled SD). Ignores NaN values."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (m1 - m2) / pooled_sd


def bootstrap_cohens_d_bca(group1, group2, n_bootstrap=10000, alpha=0.05, seed=42):
    """
    BCa bootstrap confidence interval for Cohen's d.
    
    Returns dict with:
        d_observed, ci_lower, ci_upper, ci_method, n_bootstrap,
        bootstrap_mean, bootstrap_std, bootstrap_distribution (list)
    """
    rng = np.random.RandomState(seed)
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    
    d_obs = cohens_d(g1, g2)
    
    # Bootstrap distribution
    d_boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx1 = rng.randint(0, n1, size=n1)
        idx2 = rng.randint(0, n2, size=n2)
        d_boot[i] = cohens_d(g1[idx1], g2[idx2])
    
    # BCa correction
    # Bias correction factor z0
    prop_below = np.mean(d_boot < d_obs)
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)  # avoid ±inf
    z0 = stats.norm.ppf(prop_below)
    
    # Acceleration factor a (jackknife)
    d_jack = np.empty(n1 + n2)
    for i in range(n1):
        g1_j = np.delete(g1, i)
        d_jack[i] = cohens_d(g1_j, g2)
    for i in range(n2):
        g2_j = np.delete(g2, i)
        d_jack[n1 + i] = cohens_d(g1, g2_j)
    
    d_jack_mean = np.mean(d_jack)
    num = np.sum((d_jack_mean - d_jack) ** 3)
    den = 6 * (np.sum((d_jack_mean - d_jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    
    # BCa percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)
    
    def bca_quantile(z0, a, z_val):
        numer = z0 + z_val
        denom = 1 - a * numer
        if abs(denom) < 1e-10:
            return 0.5  # fallback
        return stats.norm.cdf(z0 + numer / denom)
    
    p_lower = bca_quantile(z0, a, z_alpha)
    p_upper = bca_quantile(z0, a, z_1_alpha)
    
    # Clamp to valid [0, 1] range; handle NaN from edge cases
    if np.isnan(p_lower) or np.isnan(p_upper):
        # Fall back to simple percentile CI
        p_lower = alpha / 2
        p_upper = 1 - alpha / 2
    p_lower = float(np.clip(p_lower, 0.001, 0.999))
    p_upper = float(np.clip(p_upper, 0.001, 0.999))
    
    ci_lower = float(np.percentile(d_boot, p_lower * 100))
    ci_upper = float(np.percentile(d_boot, p_upper * 100))
    
    return {
        "d_observed": float(d_obs),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_method": "BCa",
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "bootstrap_mean": float(np.mean(d_boot)),
        "bootstrap_std": float(np.std(d_boot)),
        "bootstrap_bias": float(np.mean(d_boot) - d_obs),
        "z0": float(z0),
        "acceleration": float(a),
        "n1": n1,
        "n2": n2,
    }


# ============================================================
# Data loading
# ============================================================

def load_mode_atlas(base_path):
    """Load per-sample R_V values from mode atlas."""
    path = base_path / "results" / "mode_atlas" / "atlas_summary_20260227_075328.json"
    with open(path) as f:
        data = json.load(f)
    
    modes = {}
    for mode_name, samples in data["all_results"].items():
        modes[mode_name] = [s["rv"] for s in samples]
    return modes


def load_cross_architecture(base_path):
    """Load per-sample R_V values from cross-architecture experiments."""
    hardening_path = base_path / "results" / "statistical_hardening" / "hardening_summary_20260227_075339.json"
    
    # The hardening file has summary stats but not raw samples.
    # We need to load per-sample data from the source experiments.
    cross_arch = {}
    
    # Check power_up results for per-sample data
    power_up_dir = base_path / "results" / "power_up"
    if power_up_dir.exists():
        for f in sorted(power_up_dir.iterdir()):
            if f.suffix == ".json" and "power_up" in f.name:
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    if "per_sample" in d:
                        model = d.get("model", f.stem)
                        cross_arch[model] = d["per_sample"]
                    elif "results" in d and isinstance(d["results"], list):
                        model = d.get("model", f.stem)
                        rec = [r["rv"] for r in d["results"] if r.get("condition") == "recursive"]
                        base = [r["rv"] for r in d["results"] if r.get("condition") == "baseline"]
                        if rec and base:
                            cross_arch[model] = {"recursive": rec, "baseline": base}
                except Exception as e:
                    print(f"  Warning: could not load {f.name}: {e}")
    
    # Also check gap_experiments / scaling_gap
    for subdir_name in ["gap_experiments", "scaling_gap"]:
        subdir = base_path / "results" / subdir_name
        if subdir.exists():
            for f in sorted(subdir.iterdir()):
                if f.suffix == ".json":
                    try:
                        with open(f) as fh:
                            d = json.load(fh)
                        if isinstance(d, dict) and "recursive_rv" in d and "baseline_rv" in d:
                            model = d.get("model", f.stem)
                            if isinstance(d["recursive_rv"], list) and isinstance(d["baseline_rv"], list):
                                cross_arch[model] = {"recursive": d["recursive_rv"], "baseline": d["baseline_rv"]}
                        elif isinstance(d, dict) and "results" in d:
                            results = d["results"]
                            if isinstance(results, list) and len(results) > 0:
                                rec = [r.get("rv") or r.get("recursive_rv") for r in results if r.get("condition") == "recursive" and (r.get("rv") or r.get("recursive_rv"))]
                                base = [r.get("rv") or r.get("baseline_rv") for r in results if r.get("condition") == "baseline" and (r.get("rv") or r.get("baseline_rv"))]
                                if rec and base:
                                    model = d.get("model", f.stem)
                                    cross_arch[model] = {"recursive": rec, "baseline": base}
                    except Exception:
                        pass
    
    return cross_arch


def load_per_sample_csv(base_path):
    """Load the master per_sample.csv which has R_V values per prompt."""
    csv_path = base_path / "results" / "per_sample.csv"
    if not csv_path.exists():
        return None
    
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for R_V effects")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-path", type=str, default=".")
    args = parser.parse_args()
    
    base = Path(args.base_path)
    n_boot = args.n_bootstrap
    seed = args.seed
    
    print(f"Bootstrap CIs: n_bootstrap={n_boot}, seed={seed}")
    print(f"Base path: {base.resolve()}")
    print()
    
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_bootstrap": n_boot,
        "seed": seed,
        "effects": []
    }
    
    # ---- 1. Mode Atlas: self-referential vs each other mode ----
    print("=" * 60)
    print("1. MODE ATLAS — Self-referential vs other modes")
    print("=" * 60)
    
    modes = load_mode_atlas(base)
    sr_rv = modes.get("self_referential", [])
    
    if sr_rv:
        # Self-ref vs all other modes pooled
        other_rv = []
        for mode, vals in modes.items():
            if mode != "self_referential":
                other_rv.extend(vals)
        
        print(f"  Self-ref n={len(sr_rv)}, Other pooled n={len(other_rv)}")
        ci = bootstrap_cohens_d_bca(sr_rv, other_rv, n_boot, seed=seed)
        ci["name"] = "Self-ref vs all others (pooled)"
        print(f"  d={ci['d_observed']:.3f}, BCa 95% CI=[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
        results["effects"].append(ci)
        
        # Self-ref vs each mode individually
        for mode, vals in sorted(modes.items()):
            if mode == "self_referential":
                continue
            ci = bootstrap_cohens_d_bca(sr_rv, vals, n_boot, seed=seed)
            ci["name"] = f"Self-ref vs {mode}"
            print(f"  vs {mode}: d={ci['d_observed']:.3f}, CI=[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
            results["effects"].append(ci)
    else:
        print("  WARNING: No mode atlas data found")
    
    # ---- 2. Cross-architecture ----
    print()
    print("=" * 60)
    print("2. CROSS-ARCHITECTURE — per-model bootstrap CIs")
    print("=" * 60)
    
    cross_arch = load_cross_architecture(base)
    
    if cross_arch:
        for model, data in sorted(cross_arch.items()):
            if isinstance(data, dict) and "recursive" in data and "baseline" in data:
                rec = data["recursive"]
                bas = data["baseline"]
                print(f"  {model}: n_rec={len(rec)}, n_bas={len(bas)}")
                ci = bootstrap_cohens_d_bca(rec, bas, n_boot, seed=seed)
                ci["name"] = f"Cross-arch: {model}"
                print(f"    d={ci['d_observed']:.3f}, CI=[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
                results["effects"].append(ci)
    else:
        print("  WARNING: No cross-architecture per-sample data found")
        print("  Will attempt to compute from per_sample.csv or live model inference...")
    
    # ---- 3. Per-sample CSV as fallback ----
    print()
    print("=" * 60)
    print("3. PER-SAMPLE CSV ANALYSIS")
    print("=" * 60)
    
    csv_path = base / "results" / "per_sample.csv"
    if csv_path.exists():
        import csv
        rv_rec_list = []
        rv_base_list = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rv_rec_list.append(float(row["rv_rec"]))
                    rv_base_list.append(float(row["rv_base"]))
                except (ValueError, KeyError):
                    pass
        
        print(f"  per_sample.csv: {len(rv_rec_list)} pairs")
        print(f"  Recursive mean={np.nanmean(rv_rec_list):.4f}, Baseline mean={np.nanmean(rv_base_list):.4f}")
        
        if rv_rec_list and rv_base_list:
            ci = bootstrap_cohens_d_bca(rv_rec_list, rv_base_list, n_boot, seed=seed)
            ci["name"] = "Canonical causal (per_sample.csv, Mistral L5/L27)"
            print(f"  d={ci['d_observed']:.3f}, BCa 95% CI=[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
            results["effects"].append(ci)
    else:
        print("  No per_sample.csv found")
    
    # ---- 4. Live model inference for key effects needing bootstrap ----
    # This is the main GPU-dependent section
    print()
    print("=" * 60)
    print("4. LIVE MODEL BOOTSTRAP (GPU)")
    print("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("  No CUDA available, skipping live inference")
        else:
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Use GeometricProbe for correct V-projection R_V computation
            sys.path.insert(0, str(base))
            from geometric_lens.probe import GeometricProbe
            
            # Load prompts from mode atlas
            atlas_path = base / "results" / "mode_atlas" / "atlas_summary_20260227_075328.json"
            with open(atlas_path) as f:
                atlas = json.load(f)
            recursive_prompts = [s["text"] for s in atlas["all_results"]["self_referential"]]
            baseline_prompts = [s["text"] for s in atlas["all_results"]["factual_recall"]]
            
            print(f"  Recursive prompts: {len(recursive_prompts)}")
            print(f"  Baseline prompts: {len(baseline_prompts)}")
            
            # Models to bootstrap — cross-architecture
            models_to_test = [
                "mistralai/Mistral-7B-v0.1",
                "facebook/opt-6.7b",
                "openai-community/gpt2-xl",
                "Qwen/Qwen2.5-7B",
                "EleutherAI/pythia-1.4b",
            ]
            
            for model_name in models_to_test:
                print(f"\n  === {model_name} ===")
                t0 = time.time()
                
                try:
                    probe = GeometricProbe(
                        model_name=model_name,
                        device="cuda",
                    )
                    print(f"  Loaded in {time.time()-t0:.1f}s, layers: early={probe.early_layer}, late={probe.late_layer}")
                    
                    # Compute R_V for all prompts
                    rec_rvs = []
                    for prompt in recursive_prompts:
                        result = probe.measure(prompt, metrics=["rv"])
                        rec_rvs.append(result.rv)
                    
                    bas_rvs = []
                    for prompt in baseline_prompts:
                        result = probe.measure(prompt, metrics=["rv"])
                        bas_rvs.append(result.rv)
                    
                    rec_rvs_clean = [v for v in rec_rvs if not np.isnan(v)]
                    bas_rvs_clean = [v for v in bas_rvs if not np.isnan(v)]
                    
                    print(f"  Recursive: mean={np.mean(rec_rvs_clean):.4f}, std={np.std(rec_rvs_clean):.4f}, n={len(rec_rvs_clean)}")
                    print(f"  Baseline:  mean={np.mean(bas_rvs_clean):.4f}, std={np.std(bas_rvs_clean):.4f}, n={len(bas_rvs_clean)}")
                    
                    # Bootstrap
                    print(f"  Running {n_boot} bootstrap iterations...")
                    t1 = time.time()
                    ci = bootstrap_cohens_d_bca(rec_rvs_clean, bas_rvs_clean, n_boot, seed=seed)
                    ci["name"] = f"Cross-arch: {model_name}"
                    ci["model"] = model_name
                    ci["early_layer"] = probe.early_layer
                    ci["late_layer"] = probe.late_layer
                    ci["recursive_rv_values"] = rec_rvs_clean
                    ci["baseline_rv_values"] = bas_rvs_clean
                    print(f"  d={ci['d_observed']:.3f}, BCa 95% CI=[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
                    results["effects"].append(ci)
                    
                    # Clean up
                    del probe
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                
    except ImportError as e:
        print(f"  Cannot import required libraries: {e}")
    except Exception as e:
        print(f"  Error during live inference: {e}")
        import traceback
        traceback.print_exc()
    
    # ---- Save results ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for eff in results["effects"]:
        name = eff["name"]
        d = eff["d_observed"]
        lo = eff["ci_lower"]
        hi = eff["ci_upper"]
        print(f"  {name}: d={d:.3f}, BCa 95% CI=[{lo:.3f}, {hi:.3f}]")
    
    # Save
    out_dir = base / "results" / "bootstrap_ci"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bootstrap_ci_{results['timestamp']}.json"
    
    # Remove large arrays for the save file
    save_results = json.loads(json.dumps(results, default=str))
    for eff in save_results["effects"]:
        eff.pop("recursive_rv_values", None)
        eff.pop("baseline_rv_values", None)
    
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
