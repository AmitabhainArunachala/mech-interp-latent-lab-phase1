#!/usr/bin/env python3
"""Circularity + Perplexity Confound Control v2.

Combined experiment addressing:
1. Fatal Flaw #1 (circularity) — with padded prompts, all groups n=10
2. Perplexity confound — measures prompt perplexity alongside R_V
3. Partial correlation — R_V ~ group after controlling for perplexity

Groups:
- recursive_reference (30 prompts from L5_refined + L4_full + L3_deeper)
- baseline_reference (30 prompts from baseline_creative + baseline_math + long_control)
- same_vocab_different_semantics (10)
- recursive_no_introspection_vocab (10)
- introspective_concrete (10)
- nonsense_recursion (10)
- abstract_non_recursive (10)
"""

import sys
import json
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
    with open(bank_path) as f:
        bank = json.load(f)
    groups = defaultdict(list)
    for pid, pdata in bank.items():
        groups[pdata["group"]].append({
            "id": pid, "text": pdata["text"],
            "group": pdata["group"], "type": pdata.get("type", "unknown"),
        })
    return groups


def compute_prompt_perplexity(model, tokenizer, text, device="cuda"):
    """Compute perplexity of a prompt (how surprised is the model by this text?)."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = enc["input_ids"][0]
    if len(input_ids) < 2:
        return float("nan")
    with torch.no_grad():
        outputs = model(**enc)
    logits = outputs.logits[0]
    shift_logits = logits[:-1]
    shift_labels = input_ids[1:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(-token_log_probs.mean()).cpu())


def partial_correlation(x, y, z):
    """Partial correlation of x and y controlling for z."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        return np.nan, np.nan, int(len(x))
    # Regress x on z, y on z, correlate residuals
    from numpy.polynomial.polynomial import polyfit
    x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
    y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p), int(len(x))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    num_layers = model.config.num_hidden_layers
    early = 5
    late = num_layers - 5
    print(f"Layers: early={early}, late={late}")

    groups = load_prompts_by_group()

    # Build measurement groups
    control_groups = [
        "same_vocab_different_semantics",
        "recursive_no_introspection_vocab",
        "introspective_concrete",
        "nonsense_recursion",
        "abstract_non_recursive",
    ]

    # Reference groups — up to 30 each for better N
    ref_recursive_groups = ["L5_refined", "L4_full", "L3_deeper", "recursive_self_reference"]
    ref_baseline_groups = ["baseline_creative", "baseline_math", "long_control"]

    ref_recursive = []
    for g in ref_recursive_groups:
        ref_recursive.extend(groups.get(g, []))
    ref_recursive = ref_recursive[:30]

    ref_baseline = []
    for g in ref_baseline_groups:
        ref_baseline.extend(groups.get(g, []))
    ref_baseline = ref_baseline[:30]

    measure_groups = {
        "recursive_reference": ref_recursive,
        "baseline_reference": ref_baseline,
    }
    for g in control_groups:
        measure_groups[g] = groups.get(g, [])

    print(f"\nGroups to measure:")
    for name, prompts in measure_groups.items():
        print(f"  {name}: {len(prompts)} prompts")

    # Measure everything
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "early": early, "late": late,
        "description": "Circularity + perplexity confound v2 (padded prompts)",
        "groups": {},
    }

    # Collect all data for partial correlation
    all_rvs = []
    all_ppls = []
    all_is_recursive = []  # 1 for recursive_reference, 0 for everything else
    all_group_labels = []

    for name, prompts in measure_groups.items():
        print(f"\n{'='*60}")
        print(f"Measuring: {name} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        rvs, ppls, details = [], [], []

        for i, p in enumerate(prompts):
            text = p["text"]

            # R_V
            rv, pr_e, pr_l = compute_rv_with_components(
                model, tokenizer, text, early, late, window=16, device=device
            )
            # Prompt perplexity
            ppl = compute_prompt_perplexity(model, tokenizer, text, device=device)

            rvs.append(rv)
            ppls.append(ppl)
            details.append({
                "id": p.get("id", f"{name}_{i}"),
                "text": text[:80],
                "rv": float(rv) if not np.isnan(rv) else None,
                "ppl": float(ppl) if not np.isnan(ppl) else None,
                "pr_early": float(pr_e) if not np.isnan(pr_e) else None,
                "pr_late": float(pr_l) if not np.isnan(pr_l) else None,
            })

            rv_str = f"{rv:.4f}" if not np.isnan(rv) else "NaN"
            ppl_str = f"{ppl:.1f}" if not np.isnan(ppl) else "NaN"
            print(f"  [{i+1:2d}/{len(prompts)}] R_V={rv_str} ppl={ppl_str}")

            # Collect for partial correlation
            all_rvs.append(rv)
            all_ppls.append(ppl)
            all_is_recursive.append(1 if name == "recursive_reference" else 0)
            all_group_labels.append(name)

        rvs_clean = [r for r in rvs if not np.isnan(r)]
        ppls_clean = [p for p in ppls if not np.isnan(p)]

        grp_result = {
            "n": len(prompts),
            "n_valid_rv": len(rvs_clean),
            "n_valid_ppl": len(ppls_clean),
            "mean_rv": float(np.mean(rvs_clean)) if rvs_clean else None,
            "std_rv": float(np.std(rvs_clean)) if rvs_clean else None,
            "mean_ppl": float(np.mean(ppls_clean)) if ppls_clean else None,
            "std_ppl": float(np.std(ppls_clean)) if ppls_clean else None,
            "details": details,
        }
        all_results["groups"][name] = grp_result
        print(f"  → R_V={grp_result['mean_rv']:.4f}±{grp_result['std_rv']:.4f}, PPL={grp_result['mean_ppl']:.1f}±{grp_result['std_ppl']:.1f} (n={len(rvs_clean)})")

    # ═══════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")

    rec = all_results["groups"]["recursive_reference"]
    bas = all_results["groups"]["baseline_reference"]
    rec_rvs = [d["rv"] for d in rec["details"] if d["rv"] is not None]
    bas_rvs = [d["rv"] for d in bas["details"] if d["rv"] is not None]
    rec_ppls = [d["ppl"] for d in rec["details"] if d["ppl"] is not None]
    bas_ppls = [d["ppl"] for d in bas["details"] if d["ppl"] is not None]

    t, p = stats.ttest_ind(rec_rvs, bas_rvs)
    d_rb = (np.mean(rec_rvs) - np.mean(bas_rvs)) / np.sqrt((np.var(rec_rvs) + np.var(bas_rvs)) / 2)
    print(f"\nRecursive R_V={np.mean(rec_rvs):.4f} (ppl={np.mean(rec_ppls):.1f}) vs Baseline R_V={np.mean(bas_rvs):.4f} (ppl={np.mean(bas_ppls):.1f})")
    print(f"  R_V: d={d_rb:.3f}, p={p:.2e}")

    comparisons = {"rec_vs_bas": {"d": float(d_rb), "p": float(p)}}

    # Per-control-group comparisons
    for g in control_groups:
        grp = all_results["groups"].get(g)
        if not grp:
            continue
        ctrl_rvs = [d["rv"] for d in grp["details"] if d["rv"] is not None]
        ctrl_ppls = [d["ppl"] for d in grp["details"] if d["ppl"] is not None]
        if len(ctrl_rvs) < 3:
            continue

        t_r, p_r = stats.ttest_ind(ctrl_rvs, rec_rvs)
        d_r = (np.mean(ctrl_rvs) - np.mean(rec_rvs)) / np.sqrt((np.var(ctrl_rvs) + np.var(rec_rvs)) / 2)
        t_b, p_b = stats.ttest_ind(ctrl_rvs, bas_rvs)
        d_b = (np.mean(ctrl_rvs) - np.mean(bas_rvs)) / np.sqrt((np.var(ctrl_rvs) + np.var(bas_rvs)) / 2)

        sig_r = "***" if p_r < 0.001 else "**" if p_r < 0.01 else "*" if p_r < 0.05 else ""
        sig_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ""

        if p_r > 0.05 and p_b < 0.05:
            interp = "LOOKS RECURSIVE"
        elif p_r < 0.05 and p_b > 0.05:
            interp = "LOOKS BASELINE"
        elif p_r < 0.05 and p_b < 0.05:
            interp = "DIFFERENT FROM BOTH (no contraction)"
        else:
            interp = "AMBIGUOUS"

        print(f"\n{g} (R_V={np.mean(ctrl_rvs):.4f}, ppl={np.mean(ctrl_ppls):.1f}, n={len(ctrl_rvs)}):")
        print(f"  vs recursive: d={d_r:+.3f} p={p_r:.4f}{sig_r}")
        print(f"  vs baseline:  d={d_b:+.3f} p={p_b:.4f}{sig_b}")
        print(f"  → {interp}")

        comparisons[g] = {
            "mean_rv": float(np.mean(ctrl_rvs)),
            "mean_ppl": float(np.mean(ctrl_ppls)),
            "vs_recursive_d": float(d_r), "vs_recursive_p": float(p_r),
            "vs_baseline_d": float(d_b), "vs_baseline_p": float(p_b),
            "interpretation": interp,
        }

    # ═══════════════════════════════════════════════════════════════
    # PERPLEXITY CONFOUND ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PERPLEXITY CONFOUND ANALYSIS")
    print(f"{'='*70}")

    # 1. Raw correlation: R_V vs perplexity
    rvs_arr = np.array(all_rvs)
    ppls_arr = np.array(all_ppls)
    valid = np.isfinite(rvs_arr) & np.isfinite(ppls_arr)
    r_raw, p_raw = stats.spearmanr(rvs_arr[valid], ppls_arr[valid])
    print(f"\n1. R_V vs Perplexity (all prompts): rho={r_raw:.3f}, p={p_raw:.4f}, n={valid.sum()}")

    # 2. Perplexity difference between recursive and baseline
    t_ppl, p_ppl = stats.ttest_ind(rec_ppls, bas_ppls)
    d_ppl = (np.mean(rec_ppls) - np.mean(bas_ppls)) / np.sqrt((np.var(rec_ppls) + np.var(bas_ppls)) / 2)
    print(f"2. Perplexity: recursive={np.mean(rec_ppls):.1f} vs baseline={np.mean(bas_ppls):.1f}, d={d_ppl:.3f}, p={p_ppl:.4f}")

    # 3. Partial correlation: R_V vs is_recursive, controlling for perplexity
    is_rec_arr = np.array(all_is_recursive, dtype=float)
    r_partial, p_partial, n_partial = partial_correlation(rvs_arr, is_rec_arr, ppls_arr)
    print(f"3. Partial correlation (R_V vs recursive | perplexity): r={r_partial:.3f}, p={p_partial:.4f}, n={n_partial}")

    # 4. Within-perplexity-matched comparison
    # Find baseline prompts with perplexity >= median recursive perplexity
    rec_ppl_median = np.median(rec_ppls)
    high_ppl_non_rec = []
    low_ppl_rec = []
    for i, (rv, ppl, is_rec, label) in enumerate(zip(all_rvs, all_ppls, all_is_recursive, all_group_labels)):
        if np.isnan(rv) or np.isnan(ppl):
            continue
        if not is_rec and ppl >= rec_ppl_median:
            high_ppl_non_rec.append(rv)
        if is_rec and ppl <= np.median(bas_ppls):
            low_ppl_rec.append(rv)

    if high_ppl_non_rec and low_ppl_rec:
        t_m, p_m = stats.ttest_ind(low_ppl_rec, high_ppl_non_rec)
        d_m = (np.mean(low_ppl_rec) - np.mean(high_ppl_non_rec)) / np.sqrt((np.var(low_ppl_rec) + np.var(high_ppl_non_rec)) / 2)
        print(f"4. Perplexity-matched: low-ppl recursive R_V={np.mean(low_ppl_rec):.4f} (n={len(low_ppl_rec)}) vs high-ppl non-recursive R_V={np.mean(high_ppl_non_rec):.4f} (n={len(high_ppl_non_rec)}), d={d_m:.3f}, p={p_m:.4f}")

    perplexity_analysis = {
        "rv_vs_ppl_rho": float(r_raw), "rv_vs_ppl_p": float(p_raw),
        "ppl_recursive_mean": float(np.mean(rec_ppls)),
        "ppl_baseline_mean": float(np.mean(bas_ppls)),
        "ppl_group_diff_d": float(d_ppl), "ppl_group_diff_p": float(p_ppl),
        "partial_r": float(r_partial), "partial_p": float(p_partial), "partial_n": n_partial,
    }

    all_results["comparisons"] = comparisons
    all_results["perplexity_analysis"] = perplexity_analysis

    # Summary
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    print("\nCIRCULARITY:")
    for g in control_groups:
        c = comparisons.get(g, {})
        if c:
            print(f"  {g:42s}: R_V={c.get('mean_rv',0):.3f} ppl={c.get('mean_ppl',0):.0f} → {c.get('interpretation','?')}")

    print(f"\nPERPLEXITY CONFOUND:")
    if p_partial < 0.05:
        print(f"  R_V contraction SURVIVES perplexity control (partial r={r_partial:.3f}, p={p_partial:.4f})")
    else:
        print(f"  WARNING: R_V contraction may be confounded with perplexity (partial r={r_partial:.3f}, p={p_partial:.4f})")

    # Save
    outdir = Path("results/circularity_controls")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"circularity_perplexity_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
