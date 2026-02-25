#!/usr/bin/env python3
"""Shuffled-Prompt Anomaly Test.

Addresses reviewer concern §11.3 item 6: "shuffled prompts show MORE contraction"
was an informal Session 1 observation with no saved data.

This script provides a definitive test:
1. Take 20 recursive prompts, word-shuffle each (5 seeds → 100 shuffled measurements)
2. Measure R_V on original and shuffled versions
3. Measure perplexity on both (tests whether contraction tracks perplexity)
4. Compare against baseline prompts
5. Statistical analysis: paired t-test (original vs shuffled), independent vs baseline

PREDICTIONS if R_V measures genuine recursive self-reference:
- Shuffled R_V should be HIGHER (less contraction) than original recursive
- Shuffled R_V should be similar to baseline or same_vocab controls (~0.73)
- Shuffled perplexity should be much higher than original

If shuffled R_V is LOWER (more contraction) than original:
- This would be a serious problem — suggests R_V tracks token statistics, not semantics
"""

import sys
import json
import random
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


SHUFFLE_SEEDS = [42, 137, 256, 314, 999]
N_RECURSIVE = 20
N_BASELINE = 20


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


def shuffle_words(text, seed):
    """Word-level shuffle of a prompt, preserving punctuation attached to words."""
    rng = random.Random(seed)
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def compute_prompt_perplexity(model, tokenizer, text, device="cuda"):
    """Compute perplexity of a prompt."""
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
    print(f"Layers: early={early}, late={late} (of {num_layers})")

    # ═══════════════════════════════════════════════════════════════
    # LOAD PROMPTS
    # ═══════════════════════════════════════════════════════════════
    groups = load_prompts_by_group()

    # Recursive prompts: draw from strongest recursive groups
    recursive_groups = ["L5_refined", "L4_full", "L3_deeper", "recursive_self_reference"]
    recursive_prompts = []
    for g in recursive_groups:
        recursive_prompts.extend(groups.get(g, []))
    recursive_prompts = recursive_prompts[:N_RECURSIVE]

    # Baseline prompts
    baseline_groups = ["baseline_creative", "baseline_math", "long_control"]
    baseline_prompts = []
    for g in baseline_groups:
        baseline_prompts.extend(groups.get(g, []))
    baseline_prompts = baseline_prompts[:N_BASELINE]

    # same_vocab control (from existing circularity controls)
    same_vocab = groups.get("same_vocab_different_semantics", [])

    print(f"\nPrompts loaded:")
    print(f"  Recursive: {len(recursive_prompts)}")
    print(f"  Baseline: {len(baseline_prompts)}")
    print(f"  Same-vocab control: {len(same_vocab)}")
    print(f"  Shuffle seeds: {SHUFFLE_SEEDS} ({len(SHUFFLE_SEEDS)} per prompt)")

    # ═══════════════════════════════════════════════════════════════
    # MEASURE ORIGINAL RECURSIVE PROMPTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 1: Original recursive prompts")
    print(f"{'='*70}")

    orig_results = []
    for i, p in enumerate(recursive_prompts):
        rv, pr_e, pr_l = compute_rv_with_components(
            model, tokenizer, p["text"], early, late, window=16, device=device
        )
        ppl = compute_prompt_perplexity(model, tokenizer, p["text"], device=device)
        orig_results.append({
            "id": p["id"], "text": p["text"][:80],
            "rv": float(rv), "ppl": float(ppl),
            "pr_early": float(pr_e), "pr_late": float(pr_l),
        })
        rv_s = f"{rv:.4f}" if not np.isnan(rv) else "NaN"
        ppl_s = f"{ppl:.1f}" if not np.isnan(ppl) else "NaN"
        print(f"  [{i+1:2d}/{len(recursive_prompts)}] R_V={rv_s} ppl={ppl_s} | {p['text'][:60]}")

    # ═══════════════════════════════════════════════════════════════
    # MEASURE SHUFFLED VERSIONS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 2: Shuffled recursive prompts")
    print(f"{'='*70}")

    shuf_results = []  # list of dicts: {prompt_id, seed, original_text, shuffled_text, rv, ppl}
    total = len(recursive_prompts) * len(SHUFFLE_SEEDS)
    count = 0

    for i, p in enumerate(recursive_prompts):
        for seed in SHUFFLE_SEEDS:
            count += 1
            shuffled_text = shuffle_words(p["text"], seed)
            rv, pr_e, pr_l = compute_rv_with_components(
                model, tokenizer, shuffled_text, early, late, window=16, device=device
            )
            ppl = compute_prompt_perplexity(model, tokenizer, shuffled_text, device=device)
            shuf_results.append({
                "prompt_id": p["id"], "seed": seed,
                "original_text": p["text"][:60],
                "shuffled_text": shuffled_text[:80],
                "rv": float(rv), "ppl": float(ppl),
                "pr_early": float(pr_e), "pr_late": float(pr_l),
            })
            rv_s = f"{rv:.4f}" if not np.isnan(rv) else "NaN"
            ppl_s = f"{ppl:.1f}" if not np.isnan(ppl) else "NaN"
            print(f"  [{count:3d}/{total}] seed={seed} R_V={rv_s} ppl={ppl_s} | {shuffled_text[:50]}")

    # ═══════════════════════════════════════════════════════════════
    # MEASURE BASELINES
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Baseline prompts")
    print(f"{'='*70}")

    base_results = []
    for i, p in enumerate(baseline_prompts):
        rv, pr_e, pr_l = compute_rv_with_components(
            model, tokenizer, p["text"], early, late, window=16, device=device
        )
        ppl = compute_prompt_perplexity(model, tokenizer, p["text"], device=device)
        base_results.append({
            "id": p["id"], "text": p["text"][:80],
            "rv": float(rv), "ppl": float(ppl),
            "pr_early": float(pr_e), "pr_late": float(pr_l),
        })
        rv_s = f"{rv:.4f}" if not np.isnan(rv) else "NaN"
        ppl_s = f"{ppl:.1f}" if not np.isnan(ppl) else "NaN"
        print(f"  [{i+1:2d}/{len(baseline_prompts)}] R_V={rv_s} ppl={ppl_s}")

    # ═══════════════════════════════════════════════════════════════
    # MEASURE SAME-VOCAB CONTROLS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 4: Same-vocab controls")
    print(f"{'='*70}")

    sv_results = []
    for i, p in enumerate(same_vocab):
        rv, pr_e, pr_l = compute_rv_with_components(
            model, tokenizer, p["text"], early, late, window=16, device=device
        )
        ppl = compute_prompt_perplexity(model, tokenizer, p["text"], device=device)
        sv_results.append({
            "id": p["id"], "text": p["text"][:80],
            "rv": float(rv), "ppl": float(ppl),
            "pr_early": float(pr_e), "pr_late": float(pr_l),
        })
        rv_s = f"{rv:.4f}" if not np.isnan(rv) else "NaN"
        ppl_s = f"{ppl:.1f}" if not np.isnan(ppl) else "NaN"
        print(f"  [{i+1:2d}/{len(same_vocab)}] R_V={rv_s} ppl={ppl_s}")

    # ═══════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")

    # Clean arrays
    orig_rvs = np.array([r["rv"] for r in orig_results])
    orig_ppls = np.array([r["ppl"] for r in orig_results])
    shuf_rvs = np.array([r["rv"] for r in shuf_results])
    shuf_ppls = np.array([r["ppl"] for r in shuf_results])
    base_rvs = np.array([r["rv"] for r in base_results])
    base_ppls = np.array([r["ppl"] for r in base_results])
    sv_rvs = np.array([r["rv"] for r in sv_results])
    sv_ppls = np.array([r["ppl"] for r in sv_results])

    # Filter NaNs
    orig_rv_clean = orig_rvs[np.isfinite(orig_rvs)]
    orig_ppl_clean = orig_ppls[np.isfinite(orig_ppls)]
    shuf_rv_clean = shuf_rvs[np.isfinite(shuf_rvs)]
    shuf_ppl_clean = shuf_ppls[np.isfinite(shuf_ppls)]
    base_rv_clean = base_rvs[np.isfinite(base_rvs)]
    base_ppl_clean = base_ppls[np.isfinite(base_ppls)]
    sv_rv_clean = sv_rvs[np.isfinite(sv_rvs)]

    # --- 1. Descriptive stats ---
    print(f"\n1. DESCRIPTIVE STATISTICS")
    print(f"   Original recursive: R_V={np.mean(orig_rv_clean):.4f}±{np.std(orig_rv_clean):.4f} (n={len(orig_rv_clean)}), PPL={np.mean(orig_ppl_clean):.1f}±{np.std(orig_ppl_clean):.1f}")
    print(f"   Shuffled recursive: R_V={np.mean(shuf_rv_clean):.4f}±{np.std(shuf_rv_clean):.4f} (n={len(shuf_rv_clean)}), PPL={np.mean(shuf_ppl_clean):.1f}±{np.std(shuf_ppl_clean):.1f}")
    print(f"   Baseline:           R_V={np.mean(base_rv_clean):.4f}±{np.std(base_rv_clean):.4f} (n={len(base_rv_clean)}), PPL={np.mean(base_ppl_clean):.1f}±{np.std(base_ppl_clean):.1f}")
    if len(sv_rv_clean) > 0:
        print(f"   Same-vocab control: R_V={np.mean(sv_rv_clean):.4f}±{np.std(sv_rv_clean):.4f} (n={len(sv_rv_clean)})")

    # --- 2. Paired test: original vs mean-shuffled per prompt ---
    print(f"\n2. PAIRED TEST: Original vs Shuffled (per-prompt means)")
    paired_orig, paired_shuf = [], []
    for i, p in enumerate(recursive_prompts):
        orig_rv = orig_results[i]["rv"]
        if np.isnan(orig_rv):
            continue
        prompt_shuf_rvs = [r["rv"] for r in shuf_results if r["prompt_id"] == p["id"] and np.isfinite(r["rv"])]
        if len(prompt_shuf_rvs) == 0:
            continue
        paired_orig.append(orig_rv)
        paired_shuf.append(np.mean(prompt_shuf_rvs))

    paired_orig = np.array(paired_orig)
    paired_shuf = np.array(paired_shuf)
    if len(paired_orig) >= 3:
        t_paired, p_paired = stats.ttest_rel(paired_orig, paired_shuf)
        diff = paired_shuf - paired_orig
        d_paired = np.mean(diff) / np.std(diff)  # Cohen's d for paired
        print(f"   n={len(paired_orig)} pairs")
        print(f"   Original mean R_V: {np.mean(paired_orig):.4f}")
        print(f"   Shuffled mean R_V: {np.mean(paired_shuf):.4f}")
        print(f"   Difference: {np.mean(diff):+.4f}")
        print(f"   Paired t={t_paired:.3f}, p={p_paired:.4f}, Cohen's d={d_paired:.3f}")
        if np.mean(paired_shuf) > np.mean(paired_orig):
            print(f"   → GOOD: Shuffling INCREASES R_V (reduces contraction)")
        else:
            print(f"   → WARNING: Shuffling DECREASES R_V (increases contraction)")

    # --- 3. Independent test: shuffled vs baseline ---
    print(f"\n3. SHUFFLED vs BASELINE")
    if len(shuf_rv_clean) >= 3 and len(base_rv_clean) >= 3:
        t_sb, p_sb = stats.ttest_ind(shuf_rv_clean, base_rv_clean)
        d_sb = (np.mean(shuf_rv_clean) - np.mean(base_rv_clean)) / np.sqrt(
            (np.var(shuf_rv_clean) + np.var(base_rv_clean)) / 2
        )
        print(f"   Shuffled R_V={np.mean(shuf_rv_clean):.4f} vs Baseline R_V={np.mean(base_rv_clean):.4f}")
        print(f"   d={d_sb:.3f}, p={p_sb:.4f}")
        if p_sb > 0.05:
            print(f"   → GOOD: Shuffled is NOT significantly different from baseline (p>{0.05})")
        else:
            print(f"   → Shuffled IS significantly different from baseline")

    # --- 4. Shuffled vs same-vocab ---
    if len(sv_rv_clean) >= 3 and len(shuf_rv_clean) >= 3:
        print(f"\n4. SHUFFLED vs SAME-VOCAB CONTROL")
        t_ssv, p_ssv = stats.ttest_ind(shuf_rv_clean, sv_rv_clean)
        d_ssv = (np.mean(shuf_rv_clean) - np.mean(sv_rv_clean)) / np.sqrt(
            (np.var(shuf_rv_clean) + np.var(sv_rv_clean)) / 2
        )
        print(f"   Shuffled R_V={np.mean(shuf_rv_clean):.4f} vs Same-vocab R_V={np.mean(sv_rv_clean):.4f}")
        print(f"   d={d_ssv:.3f}, p={p_ssv:.4f}")

    # --- 5. Perplexity comparison ---
    print(f"\n5. PERPLEXITY COMPARISON")
    print(f"   Original recursive PPL: {np.mean(orig_ppl_clean):.1f}±{np.std(orig_ppl_clean):.1f}")
    print(f"   Shuffled recursive PPL: {np.mean(shuf_ppl_clean):.1f}±{np.std(shuf_ppl_clean):.1f}")
    print(f"   Baseline PPL:           {np.mean(base_ppl_clean):.1f}±{np.std(base_ppl_clean):.1f}")
    if len(shuf_ppl_clean) > 0 and len(orig_ppl_clean) > 0:
        ppl_ratio = np.mean(shuf_ppl_clean) / np.mean(orig_ppl_clean)
        print(f"   Shuffled/Original PPL ratio: {ppl_ratio:.2f}x")

    # --- 6. R_V vs PPL correlation within shuffled ---
    print(f"\n6. R_V vs PERPLEXITY CORRELATION (within shuffled)")
    valid = np.isfinite(shuf_rvs) & np.isfinite(shuf_ppls)
    if valid.sum() >= 5:
        rho, p_rho = stats.spearmanr(shuf_rvs[valid], shuf_ppls[valid])
        print(f"   Spearman rho={rho:.3f}, p={p_rho:.4f}, n={valid.sum()}")
        if abs(rho) < 0.3:
            print(f"   → GOOD: Weak correlation — R_V is not simply tracking perplexity")
        else:
            print(f"   → NOTE: Moderate/strong correlation — perplexity may be a confound")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    anomaly_resolved = True
    verdicts = []

    # Check 1: Shuffled should have HIGHER R_V than original
    if len(paired_orig) >= 3:
        if np.mean(paired_shuf) > np.mean(paired_orig):
            verdicts.append("PASS: Shuffling increases R_V (semantic content drives contraction)")
        elif p_paired > 0.05:
            verdicts.append("MARGINAL: No significant difference, but direction ambiguous")
            anomaly_resolved = False
        else:
            verdicts.append("FAIL: Shuffling decreases R_V — anomaly CONFIRMED")
            anomaly_resolved = False

    # Check 2: Shuffled should be close to baseline
    if len(shuf_rv_clean) >= 3 and len(base_rv_clean) >= 3:
        if p_sb > 0.05:
            verdicts.append("PASS: Shuffled R_V ≈ Baseline R_V (word order matters)")
        elif np.mean(shuf_rv_clean) < np.mean(base_rv_clean):
            verdicts.append("FAIL: Shuffled R_V < Baseline — vocabulary alone causes contraction")
            anomaly_resolved = False
        else:
            verdicts.append("MARGINAL: Shuffled differs from baseline but in expected direction")

    # Check 3: Perplexity should spike for shuffled
    if len(shuf_ppl_clean) > 0:
        if ppl_ratio > 2.0:
            verdicts.append(f"PASS: Shuffled perplexity {ppl_ratio:.1f}x higher (model sees gibberish)")
        else:
            verdicts.append(f"NOTE: Perplexity only {ppl_ratio:.1f}x higher")

    for v in verdicts:
        print(f"  {v}")

    if anomaly_resolved:
        print(f"\n  ✓ SCRAMBLED-PROMPT ANOMALY RESOLVED")
        print(f"    R_V contraction requires semantic recursive structure, not just vocabulary.")
    else:
        print(f"\n  ✗ ANOMALY NOT RESOLVED — further investigation needed")

    # ═══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "early": early, "late": late,
        "shuffle_seeds": SHUFFLE_SEEDS,
        "n_recursive": len(recursive_prompts),
        "n_baseline": len(baseline_prompts),
        "n_same_vocab": len(same_vocab),
        "description": "Shuffled-prompt anomaly test: word-shuffle destroys semantics, should destroy contraction",
        "original_recursive": {
            "mean_rv": float(np.mean(orig_rv_clean)),
            "std_rv": float(np.std(orig_rv_clean)),
            "mean_ppl": float(np.mean(orig_ppl_clean)),
            "n_valid": int(len(orig_rv_clean)),
            "details": orig_results,
        },
        "shuffled_recursive": {
            "mean_rv": float(np.mean(shuf_rv_clean)),
            "std_rv": float(np.std(shuf_rv_clean)),
            "mean_ppl": float(np.mean(shuf_ppl_clean)),
            "n_valid": int(len(shuf_rv_clean)),
            "details": shuf_results,
        },
        "baseline": {
            "mean_rv": float(np.mean(base_rv_clean)),
            "std_rv": float(np.std(base_rv_clean)),
            "mean_ppl": float(np.mean(base_ppl_clean)),
            "n_valid": int(len(base_rv_clean)),
            "details": base_results,
        },
        "same_vocab_control": {
            "mean_rv": float(np.mean(sv_rv_clean)) if len(sv_rv_clean) > 0 else None,
            "n_valid": int(len(sv_rv_clean)),
            "details": sv_results,
        },
        "statistics": {
            "paired_orig_vs_shuffled": {
                "n_pairs": int(len(paired_orig)) if len(paired_orig) >= 3 else 0,
                "orig_mean_rv": float(np.mean(paired_orig)) if len(paired_orig) >= 3 else None,
                "shuf_mean_rv": float(np.mean(paired_shuf)) if len(paired_orig) >= 3 else None,
                "mean_diff": float(np.mean(diff)) if len(paired_orig) >= 3 else None,
                "t": float(t_paired) if len(paired_orig) >= 3 else None,
                "p": float(p_paired) if len(paired_orig) >= 3 else None,
                "cohens_d": float(d_paired) if len(paired_orig) >= 3 else None,
            },
            "shuffled_vs_baseline": {
                "d": float(d_sb) if len(shuf_rv_clean) >= 3 and len(base_rv_clean) >= 3 else None,
                "p": float(p_sb) if len(shuf_rv_clean) >= 3 and len(base_rv_clean) >= 3 else None,
            },
        },
        "verdicts": verdicts,
        "anomaly_resolved": anomaly_resolved,
    }

    outdir = Path("results/shuffled_prompt_test")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"shuffled_prompt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
