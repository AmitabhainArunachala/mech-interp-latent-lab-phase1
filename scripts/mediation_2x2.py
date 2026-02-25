#!/usr/bin/env python3
"""
MEDIATION 2×2: Does L0 MLP mediate the L27 V-proj contraction effect?

Design:
  For each prompt PAIR (recursive donor, baseline target):
    1. Extract recursive V-proj at L27 (the "donor activation")
    2. Measure R_V on the BASELINE prompt under 4 conditions:
       A: No intervention           (baseline reference)
       B: L27 V-proj patched only   (known to cause contraction from bridge test)
       C: L0 MLP ablated only       (control for ablation effect on baselines)
       D: L0 MLP ablated + L27 patched  (THE MEDIATION TEST)
    3. Also measure R_V on the recursive prompt (reference)

Critical comparison:
  - If D ≈ C (patch effect blocked by L0 ablation) → L0 mediates L27 effect
  - If D ≈ B (patch effect survives L0 ablation) → L27 is independent of L0

Uses existing infrastructure:
  - MLPAblationHook from mlp_ablation_necessity_prompt_pass.py
  - PersistentVPatcher + extract_v_activation from src/core/patching.py
  - compute_rv_with_components from src/metrics/rv.py
  - PromptLoader from prompts/loader.py
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.rv import compute_rv_with_components


# ── MLP Ablation Hook (from canonical pipeline) ─────────────────────────────

class MLPAblationHook:
    """Zero out MLP output at specified layer."""

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.handle = None

    def register(self):
        mlp = self.model.model.layers[self.layer_idx].mlp

        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                return (torch.zeros_like(out[0]),) + out[1:]
            return torch.zeros_like(out)

        self.handle = mlp.register_forward_hook(hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mediation 2×2: L0 ablation × L27 patching")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-pairs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--ablation-layer", type=int, default=0)
    parser.add_argument("--patch-layer", type=int, default=27)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--output", default="results/mediation")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)
    model.eval()

    loader = PromptLoader()
    pairs = loader.get_balanced_pairs_with_ids(
        n_pairs=args.n_pairs, seed=args.seed,
    )
    print(f"Got {len(pairs)} prompt pairs")

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  MEDIATION 2×2: L{args.ablation_layer} ablation × L{args.patch_layer} V-proj patching")
    print(f"  {len(pairs)} prompt pairs, early=L{args.early_layer}, late=L{args.late_layer}")
    print(f"{sep}\n")

    results = []

    for idx, (rec_id, bas_id, rec_text, bas_text) in enumerate(tqdm(pairs, desc="Pairs")):
        t0 = time.time()

        # Step 1: Extract recursive V-proj at patch layer (the donor)
        try:
            v_donor = extract_v_activation(
                model, tokenizer, rec_text,
                layer_idx=args.patch_layer, device=args.device,
            )
        except Exception as e:
            print(f"  Pair {idx}: failed to extract donor V-proj: {e}")
            continue

        # Step 2: Recursive reference (no intervention)
        rv_rec, pr_e_rec, pr_l_rec = compute_rv_with_components(
            model, tokenizer, rec_text,
            early=args.early_layer, late=args.late_layer,
            window=args.window, device=args.device,
        )

        # Step 3: Condition A — baseline, no intervention
        rv_A, pr_e_A, pr_l_A = compute_rv_with_components(
            model, tokenizer, bas_text,
            early=args.early_layer, late=args.late_layer,
            window=args.window, device=args.device,
        )

        # Step 4: Condition B — baseline, L27 patched only
        patcher = PersistentVPatcher(model, v_donor)
        patcher.register(layer_idx=args.patch_layer)
        try:
            rv_B, pr_e_B, pr_l_B = compute_rv_with_components(
                model, tokenizer, bas_text,
                early=args.early_layer, late=args.late_layer,
                window=args.window, device=args.device,
            )
        finally:
            patcher.remove()

        # Step 5: Condition C — baseline, L0 ablated only
        ablator = MLPAblationHook(model, args.ablation_layer)
        with ablator:
            rv_C, pr_e_C, pr_l_C = compute_rv_with_components(
                model, tokenizer, bas_text,
                early=args.early_layer, late=args.late_layer,
                window=args.window, device=args.device,
            )

        # Step 6: Condition D — baseline, L0 ablated + L27 patched (THE MEDIATION TEST)
        ablator2 = MLPAblationHook(model, args.ablation_layer)
        patcher2 = PersistentVPatcher(model, v_donor)
        ablator2.register()
        patcher2.register(layer_idx=args.patch_layer)
        try:
            rv_D, pr_e_D, pr_l_D = compute_rv_with_components(
                model, tokenizer, bas_text,
                early=args.early_layer, late=args.late_layer,
                window=args.window, device=args.device,
            )
        finally:
            patcher2.remove()
            ablator2.remove()

        elapsed = time.time() - t0

        # Patch effect = B - A (how much does L27 patching alone change R_V?)
        patch_effect = rv_B - rv_A if not (np.isnan(rv_B) or np.isnan(rv_A)) else float("nan")
        # Mediated patch effect = D - C (does L27 patching work when L0 is ablated?)
        mediated_effect = rv_D - rv_C if not (np.isnan(rv_D) or np.isnan(rv_C)) else float("nan")

        row = {
            "pair_idx": idx,
            "rec_id": rec_id,
            "bas_id": bas_id,
            # Recursive reference
            "rv_recursive": rv_rec,
            # 2x2 conditions on baseline prompt
            "rv_A_baseline": rv_A,
            "rv_B_patch_only": rv_B,
            "rv_C_ablate_only": rv_C,
            "rv_D_ablate_and_patch": rv_D,
            # PR components
            "pr_e_A": pr_e_A, "pr_l_A": pr_l_A,
            "pr_e_B": pr_e_B, "pr_l_B": pr_l_B,
            "pr_e_C": pr_e_C, "pr_l_C": pr_l_C,
            "pr_e_D": pr_e_D, "pr_l_D": pr_l_D,
            # Derived
            "patch_effect_B_minus_A": patch_effect,
            "mediated_effect_D_minus_C": mediated_effect,
            "elapsed": elapsed,
        }
        results.append(row)

        if idx % 10 == 0:
            print(f"  [{idx:3d}] A={rv_A:.3f} B={rv_B:.3f} C={rv_C:.3f} D={rv_D:.3f} "
                  f"patch={patch_effect:+.3f} mediated={mediated_effect:+.3f} "
                  f"rec={rv_rec:.3f} ({elapsed:.1f}s)")

    # ── Analysis ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  MEDIATION 2×2 RESULTS")
    print(f"{sep}\n")

    # Collect clean values
    rvs = {k: [] for k in ["A", "B", "C", "D", "rec", "patch_effect", "mediated_effect"]}
    for r in results:
        for cond, key in [("A", "rv_A_baseline"), ("B", "rv_B_patch_only"),
                          ("C", "rv_C_ablate_only"), ("D", "rv_D_ablate_and_patch"),
                          ("rec", "rv_recursive")]:
            v = r[key]
            if not np.isnan(v):
                rvs[cond].append(v)
        pe = r["patch_effect_B_minus_A"]
        me = r["mediated_effect_D_minus_C"]
        if not np.isnan(pe):
            rvs["patch_effect"].append(pe)
        if not np.isnan(me):
            rvs["mediated_effect"].append(me)

    print("Condition means (R_V on baseline prompts):")
    for cond in ["A", "B", "C", "D"]:
        vals = rvs[cond]
        if vals:
            print(f"  {cond}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")
    print(f"  Recursive ref: {np.mean(rvs['rec']):.4f} ± {np.std(rvs['rec']):.4f}")

    print(f"\nPatch effect (B - A): {np.mean(rvs['patch_effect']):+.4f} ± {np.std(rvs['patch_effect']):.4f}")
    print(f"Mediated effect (D - C): {np.mean(rvs['mediated_effect']):+.4f} ± {np.std(rvs['mediated_effect']):.4f}")

    # Critical tests
    print(f"\n--- Critical Comparisons ---")

    # Test 1: Does patching work? (B vs A)
    if len(rvs["A"]) >= 2 and len(rvs["B"]) >= 2:
        t, p = stats.ttest_rel(rvs["B"][:len(rvs["A"])], rvs["A"][:len(rvs["B"])])
        pooled = np.sqrt((np.std(rvs["A"])**2 + np.std(rvs["B"])**2) / 2)
        d = (np.mean(rvs["B"]) - np.mean(rvs["A"])) / pooled if pooled > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  B vs A (patch effect):     d={d:+.3f}, p={p:.2e} {sig}")

    # Test 2: Does ablation disrupt baseline? (C vs A)
    if len(rvs["A"]) >= 2 and len(rvs["C"]) >= 2:
        t, p = stats.ttest_rel(rvs["C"][:len(rvs["A"])], rvs["A"][:len(rvs["C"])])
        pooled = np.sqrt((np.std(rvs["A"])**2 + np.std(rvs["C"])**2) / 2)
        d = (np.mean(rvs["C"]) - np.mean(rvs["A"])) / pooled if pooled > 0 else 0
        print(f"  C vs A (ablation effect):  d={d:+.3f}, p={p:.2e}")

    # Test 3: THE MEDIATION TEST — does patching work when L0 is ablated? (D vs C)
    if len(rvs["C"]) >= 2 and len(rvs["D"]) >= 2:
        t, p = stats.ttest_rel(rvs["D"][:len(rvs["C"])], rvs["C"][:len(rvs["D"])])
        pooled = np.sqrt((np.std(rvs["C"])**2 + np.std(rvs["D"])**2) / 2)
        d = (np.mean(rvs["D"]) - np.mean(rvs["C"])) / pooled if pooled > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  D vs C (mediated patch):   d={d:+.3f}, p={p:.2e} {sig}")

    # Test 4: Mediation ratio — what fraction of patch effect survives ablation?
    pe_mean = np.mean(rvs["patch_effect"]) if rvs["patch_effect"] else 0
    me_mean = np.mean(rvs["mediated_effect"]) if rvs["mediated_effect"] else 0
    if abs(pe_mean) > 1e-6:
        mediation_ratio = me_mean / pe_mean
        print(f"\n  Mediation ratio (mediated/total): {mediation_ratio:.3f}")
        print(f"    = 1.0 → L0 does NOT mediate (L27 independent)")
        print(f"    = 0.0 → L0 fully mediates (L27 blocked)")
        print(f"    < 1.0 → partial mediation")

    # Test 5: Interaction test — does ablation change the patch effect?
    if len(rvs["patch_effect"]) >= 2 and len(rvs["mediated_effect"]) >= 2:
        n = min(len(rvs["patch_effect"]), len(rvs["mediated_effect"]))
        t, p = stats.ttest_rel(
            rvs["mediated_effect"][:n],
            rvs["patch_effect"][:n],
        )
        print(f"\n  Interaction test (mediated vs total patch effect):")
        print(f"    t={t:.3f}, p={p:.2e}")
        print(f"    If p < 0.05: L0 ablation significantly changes the patch effect → mediation")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "n_pairs": len(results),
        "ablation_layer": args.ablation_layer,
        "patch_layer": args.patch_layer,
        "early_layer": args.early_layer,
        "late_layer": args.late_layer,
        "condition_means": {
            cond: {"mean": float(np.mean(rvs[cond])), "std": float(np.std(rvs[cond])), "n": len(rvs[cond])}
            for cond in ["A", "B", "C", "D", "rec"]
            if rvs[cond]
        },
        "patch_effect_mean": float(pe_mean),
        "mediated_effect_mean": float(me_mean),
        "mediation_ratio": float(me_mean / pe_mean) if abs(pe_mean) > 1e-6 else None,
        "results": results,
    }

    with open(out_dir / f"mediation_2x2_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
