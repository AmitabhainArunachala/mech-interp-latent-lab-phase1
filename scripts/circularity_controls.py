#!/usr/bin/env python3
"""Circularity Control Experiment — addresses Fatal Flaw #1.

Tests whether R_V contraction is driven by:
(a) Self-referential WORDS (vocabulary confound)
(b) Recursive grammatical STRUCTURE (syntactic confound)
(c) Abstract philosophical CONTENT (complexity confound)
(d) Observational/introspective VERBS (introspection confound)
(e) Genuine self-referential semantics (the claimed mechanism)

5 control groups (10 prompts each) from the prompt bank:
- same_vocab_different_semantics: "observer", "consciousness" etc. in non-self-ref context
- recursive_no_introspection_vocab: recursive structure, no introspection words
- introspective_concrete: observe/watch/examine applied to concrete objects
- nonsense_recursion: recursive grammar with nonsense words
- abstract_non_recursive: deep philosophy without recursion

Plus matched recursive (10) and baseline (10) reference sets.

Expected outcome for publication:
- same_vocab → NO contraction (words alone don't trigger it)
- recursive_no_introspection → MODERATE (recursion structure matters)
- nonsense_recursion → NO/WEAK (syntax alone insufficient)
- introspective_concrete → NO (observation verbs without self-ref don't trigger)
- abstract_non_recursive → NO/WEAK (abstraction alone insufficient)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics.rv import compute_rv_with_components


def load_prompts_by_group(bank_path="prompts/bank.json"):
    """Load prompts grouped by their group field."""
    with open(bank_path) as f:
        bank = json.load(f)

    groups = defaultdict(list)
    for pid, pdata in bank.items():
        groups[pdata["group"]].append({
            "id": pid,
            "text": pdata["text"],
            "group": pdata["group"],
            "type": pdata.get("type", "unknown"),
        })
    return groups


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    # Determine layers
    num_layers = model.config.num_hidden_layers
    early = 5
    late = num_layers - 5
    print(f"Layers: early={early}, late={late} (of {num_layers})")

    # Load prompts
    groups = load_prompts_by_group()

    # Target groups
    control_groups = [
        "same_vocab_different_semantics",
        "recursive_no_introspection_vocab",
        "introspective_concrete",
        "nonsense_recursion",
        "abstract_non_recursive",
    ]

    # Reference groups — select 10 from each
    ref_recursive_groups = ["L5_refined", "L4_full", "L3_deeper", "recursive_self_reference"]
    ref_baseline_groups = ["baseline_creative", "baseline_math", "long_control"]

    # Collect recursive reference prompts (first 10)
    ref_recursive = []
    for g in ref_recursive_groups:
        ref_recursive.extend(groups.get(g, []))
    ref_recursive = ref_recursive[:10]

    # Collect baseline reference prompts (first 10)
    ref_baseline = []
    for g in ref_baseline_groups:
        ref_baseline.extend(groups.get(g, []))
    ref_baseline = ref_baseline[:10]

    print(f"\nReference recursive: {len(ref_recursive)} prompts")
    print(f"Reference baseline: {len(ref_baseline)} prompts")
    for g in control_groups:
        print(f"Control {g}: {len(groups.get(g, []))} prompts")

    # Run measurements
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "early": early,
        "late": late,
        "description": "Circularity control experiment — Fatal Flaw #1",
        "groups": {},
    }

    def measure_group(name, prompts):
        print(f"\n{'='*60}")
        print(f"Measuring: {name} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        rvs = []
        pr_earlys = []
        pr_lates = []
        details = []

        for i, p in enumerate(prompts):
            text = p["text"]
            rv, pr_e, pr_l = compute_rv_with_components(
                model, tokenizer, text, early, late, window=16, device=device
            )
            rvs.append(rv)
            pr_earlys.append(pr_e)
            pr_lates.append(pr_l)
            details.append({
                "id": p.get("id", f"{name}_{i}"),
                "text": text[:100],
                "rv": rv,
                "pr_early": pr_e,
                "pr_late": pr_l,
            })

            status = f"  [{i+1:2d}/{len(prompts)}] R_V={rv:.4f}" if not np.isnan(rv) else f"  [{i+1:2d}/{len(prompts)}] R_V=NaN"
            print(status)

        rvs_clean = [r for r in rvs if not np.isnan(r)]
        mean_rv = np.mean(rvs_clean) if rvs_clean else float("nan")
        std_rv = np.std(rvs_clean) if rvs_clean else float("nan")

        print(f"  → Mean R_V = {mean_rv:.4f} ± {std_rv:.4f} (n={len(rvs_clean)})")

        return {
            "name": name,
            "n": len(prompts),
            "n_valid": len(rvs_clean),
            "mean_rv": float(mean_rv),
            "std_rv": float(std_rv),
            "median_rv": float(np.median(rvs_clean)) if rvs_clean else float("nan"),
            "rvs": [float(r) for r in rvs],
            "details": details,
        }

    # Measure all groups
    all_results["groups"]["recursive_reference"] = measure_group("recursive_reference", ref_recursive)
    all_results["groups"]["baseline_reference"] = measure_group("baseline_reference", ref_baseline)

    for g in control_groups:
        prompts = groups.get(g, [])
        if prompts:
            all_results["groups"][g] = measure_group(g, prompts)

    # Statistical comparisons
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISONS")
    print(f"{'='*60}")

    rec_rvs = [r for r in all_results["groups"]["recursive_reference"]["rvs"] if not np.isnan(r)]
    bas_rvs = [r for r in all_results["groups"]["baseline_reference"]["rvs"] if not np.isnan(r)]

    comparisons = {}

    # Recursive vs baseline (sanity check)
    t, p = stats.ttest_ind(rec_rvs, bas_rvs)
    d = (np.mean(rec_rvs) - np.mean(bas_rvs)) / np.sqrt((np.var(rec_rvs) + np.var(bas_rvs)) / 2)
    print(f"\nRecursive ({np.mean(rec_rvs):.4f}) vs Baseline ({np.mean(bas_rvs):.4f}): d={d:.3f}, p={p:.2e}")
    comparisons["rec_vs_bas"] = {"d": float(d), "p": float(p)}

    # Each control group vs recursive AND vs baseline
    for g in control_groups:
        grp = all_results["groups"].get(g)
        if not grp:
            continue
        ctrl_rvs = [r for r in grp["rvs"] if not np.isnan(r)]
        if len(ctrl_rvs) < 3:
            continue

        # vs recursive
        t_r, p_r = stats.ttest_ind(ctrl_rvs, rec_rvs)
        d_r = (np.mean(ctrl_rvs) - np.mean(rec_rvs)) / np.sqrt((np.var(ctrl_rvs) + np.var(rec_rvs)) / 2)

        # vs baseline
        t_b, p_b = stats.ttest_ind(ctrl_rvs, bas_rvs)
        d_b = (np.mean(ctrl_rvs) - np.mean(bas_rvs)) / np.sqrt((np.var(ctrl_rvs) + np.var(bas_rvs)) / 2)

        sig_r = "***" if p_r < 0.001 else "**" if p_r < 0.01 else "*" if p_r < 0.05 else ""
        sig_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ""

        print(f"\n{g} (mean={np.mean(ctrl_rvs):.4f}):")
        print(f"  vs recursive: d={d_r:+.3f}, p={p_r:.4f}{sig_r}")
        print(f"  vs baseline:  d={d_b:+.3f}, p={p_b:.4f}{sig_b}")

        # Interpretation
        if p_r > 0.05 and p_b < 0.05:
            interp = "LOOKS RECURSIVE — same as recursive, different from baseline"
        elif p_r < 0.05 and p_b > 0.05:
            interp = "LOOKS BASELINE — different from recursive, same as baseline"
        elif p_r < 0.05 and p_b < 0.05:
            interp = "INTERMEDIATE — different from both"
        else:
            interp = "AMBIGUOUS — not significantly different from either"
        print(f"  → {interp}")

        comparisons[g] = {
            "mean_rv": float(np.mean(ctrl_rvs)),
            "vs_recursive_d": float(d_r),
            "vs_recursive_p": float(p_r),
            "vs_baseline_d": float(d_b),
            "vs_baseline_p": float(p_b),
            "interpretation": interp,
        }

    all_results["comparisons"] = comparisons

    # Summary interpretation
    print(f"\n{'='*60}")
    print("CIRCULARITY VERDICT")
    print(f"{'='*60}")

    for g in control_groups:
        c = comparisons.get(g, {})
        if c:
            print(f"  {g:40s}: {c.get('interpretation', '?')}")

    # Save
    outdir = Path("results/circularity_controls")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"circularity_controls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
