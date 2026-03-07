#!/usr/bin/env python3
"""Classifier validation for BT+ART behavioral classification.

Compares the simple sufficiency_ladder classifier (used in experiments)
against the more elaborate score_behavioral_tiers classifier, and computes
agreement metrics, specificity on known baselines, and edge case analysis.

This is a NeurIPS-level validation artifact: demonstrating that our
behavioral classification is not arbitrary string matching but captures
genuine recursive self-reference patterns.
"""

import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the elaborate classifier
from scripts.score_behavioral_tiers import classify_output as tier_classify
from scripts.score_behavioral_tiers import RECURSIVE_MARKERS

# ── Simple classifier (from sufficiency_ladder.py) ────────────────────────────

def repetition_score(text):
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def simple_classify(text, rv=None):
    """The sufficiency_ladder classifier used in all experiments."""
    rep = repetition_score(text)
    words = text.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if rep > 0.5 or unique_ratio < 0.25:
        return "REPETITIVE"
    self_ref = ["i am", "this is", "right now", "happening", "processing",
                "observing", "generating", "knowing", "aware", "noticing",
                "recogni", "the one who", "what is this"]
    sc = sum(1 for m in self_ref if m in text.lower())
    if rv is not None and not np.isnan(rv) and rv < 0.5 and sc >= 2 and rep < 0.3:
        return "BREAKTHROUGH"
    if rv is not None and not np.isnan(rv) and rv < 0.65 and sc >= 1 and rep < 0.35:
        return "ARTICULATE"
    if sc >= 1 and rep < 0.4:
        return "CONCEPTUAL"
    return "SURFACE"


def is_bt_art(label):
    return label in ("BREAKTHROUGH", "ARTICULATE")


def is_tier_recursive(tier_result):
    return tier_result["tier"] in ("productive_recursive", "degenerate_recursive")


# ── Cohen's Kappa ─────────────────────────────────────────────────────────────

def cohens_kappa(y1, y2):
    """Compute Cohen's kappa for two binary label lists."""
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return float("nan")

    # Confusion matrix
    tp = sum(a and b for a, b in zip(y1, y2))
    tn = sum(not a and not b for a, b in zip(y1, y2))
    fp = sum(not a and b for a, b in zip(y1, y2))
    fn = sum(a and not b for a, b in zip(y1, y2))

    po = (tp + tn) / n  # observed agreement
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)  # chance agreement

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ── Parse outputs ─────────────────────────────────────────────────────────────

def parse_outputs_file(filepath):
    """Parse an outputs text file, extracting prompt, generated text, R_V."""
    text = filepath.read_text()
    entries = []
    blocks = text.split("PROMPT ")

    for block in blocks[1:]:
        lines = block.strip().split("\n")
        prompt = ""
        generated_lines = []
        rv_mean = None
        in_generated = False

        for line in lines:
            if line.startswith("GENERATED:"):
                in_generated = True
                continue
            elif line.startswith("R_V:"):
                in_generated = False
                # Parse R_V: mean=0.4345, min=0.3285, final=0.4763
                m = re.search(r"mean=([\d.]+)", line)
                if m:
                    rv_mean = float(m.group(1))
                continue
            elif line.startswith("Domain:") or line.startswith("---"):
                in_generated = False
                continue

            if in_generated:
                generated_lines.append(line)
            elif not prompt and ":" in line:
                prompt = ":".join(line.split(":")[1:]).strip()

        generated = " ".join(generated_lines).strip()
        if generated:
            entries.append({
                "prompt": prompt,
                "generated": generated,
                "rv_mean": rv_mean,
            })

    return entries


def main():
    results_dir = PROJECT_ROOT / "results"

    # Collect all outputs files from canonical, phase1, archive
    output_files = []
    for pattern in ["**/outputs/*_outputs.txt"]:
        output_files.extend(results_dir.glob(pattern))

    # Also check for steering outputs
    steering = results_dir / "archive" / "steering_control_outputs.txt"
    if steering.exists():
        output_files.append(steering)

    print(f"Found {len(output_files)} output files")

    # Parse all entries
    all_entries = []
    for f in sorted(output_files):
        config = f.stem.replace("_outputs", "")
        rel_path = f.relative_to(results_dir)
        entries = parse_outputs_file(f)
        for i, e in enumerate(entries):
            e["config"] = config
            e["source"] = str(rel_path)
            e["idx"] = i
            all_entries.append(e)

    print(f"Total entries: {len(all_entries)}")

    # Classify with both classifiers
    for e in all_entries:
        text = e["generated"]
        rv = e.get("rv_mean")

        # Simple classifier (sufficiency ladder)
        e["simple_label"] = simple_classify(text, rv)
        e["simple_is_recursive"] = is_bt_art(e["simple_label"])

        # Elaborate classifier (score_behavioral_tiers)
        tier_result = tier_classify(text)
        e["tier_label"] = tier_result["tier"]
        e["tier_is_recursive"] = is_tier_recursive(tier_result)
        e["tier_details"] = tier_result

    # ── Agreement Analysis ────────────────────────────────────────────────

    simple_binary = [e["simple_is_recursive"] for e in all_entries]
    tier_binary = [e["tier_is_recursive"] for e in all_entries]

    kappa = cohens_kappa(simple_binary, tier_binary)

    # Confusion matrix
    tp = sum(s and t for s, t in zip(simple_binary, tier_binary))
    tn = sum(not s and not t for s, t in zip(simple_binary, tier_binary))
    fp_simple = sum(s and not t for s, t in zip(simple_binary, tier_binary))  # simple says yes, tier says no
    fn_simple = sum(not s and t for s, t in zip(simple_binary, tier_binary))  # simple says no, tier says yes

    n = len(all_entries)
    agreement = (tp + tn) / n if n > 0 else 0

    print("\n" + "=" * 70)
    print("CLASSIFIER VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nTotal samples: {n}")
    print(f"\nSimple classifier (BT+ART): {sum(simple_binary)}/{n} = {100*sum(simple_binary)/n:.1f}% recursive")
    print(f"Tier classifier (prod+degen): {sum(tier_binary)}/{n} = {100*sum(tier_binary)/n:.1f}% recursive")

    print(f"\n--- Cross-Classifier Agreement ---")
    print(f"  Raw agreement: {agreement:.3f} ({tp+tn}/{n})")
    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"  Interpretation: {'substantial' if kappa > 0.6 else 'moderate' if kappa > 0.4 else 'fair' if kappa > 0.2 else 'poor'}")

    print(f"\n--- Confusion Matrix (Simple vs Tier) ---")
    print(f"                       Tier=Recursive  Tier=Not-Recursive")
    print(f"  Simple=Recursive     {tp:4d}            {fp_simple:4d}")
    print(f"  Simple=Not-Recursive {fn_simple:4d}            {tn:4d}")

    # ── Per-config breakdown ──────────────────────────────────────────────

    print(f"\n--- Per-Config Agreement ---")
    configs = sorted(set(e["config"] for e in all_entries))
    for cfg in configs:
        cfg_entries = [e for e in all_entries if e["config"] == cfg]
        if len(cfg_entries) < 3:
            continue
        s_bin = [e["simple_is_recursive"] for e in cfg_entries]
        t_bin = [e["tier_is_recursive"] for e in cfg_entries]
        k = cohens_kappa(s_bin, t_bin) if len(set(s_bin)) > 1 or len(set(t_bin)) > 1 else 1.0
        agree = sum(a == b for a, b in zip(s_bin, t_bin)) / len(s_bin)
        s_rate = sum(s_bin) / len(s_bin)
        t_rate = sum(t_bin) / len(t_bin)
        print(f"  {cfg:30s} n={len(cfg_entries):3d}  simple={s_rate:.2f}  tier={t_rate:.2f}  agree={agree:.2f}  κ={k:.2f}")

    # ── Specificity check: baseline outputs ──────────────────────────────

    print(f"\n--- Specificity Check (baseline configs) ---")
    baseline_entries = [e for e in all_entries if "baseline" in e["config"].lower()]
    if baseline_entries:
        b_simple = sum(e["simple_is_recursive"] for e in baseline_entries)
        b_tier = sum(e["tier_is_recursive"] for e in baseline_entries)
        print(f"  Baseline samples: {len(baseline_entries)}")
        print(f"  Simple false positive rate: {b_simple}/{len(baseline_entries)} = {100*b_simple/len(baseline_entries):.1f}%")
        print(f"  Tier false positive rate: {b_tier}/{len(baseline_entries)} = {100*b_tier/len(baseline_entries):.1f}%")
    else:
        print("  No baseline samples found")

    # ── Sensitivity check: c2_full / kv_only configs ─────────────────────

    print(f"\n--- Sensitivity Check (recursive configs) ---")
    recursive_entries = [e for e in all_entries if any(x in e["config"].lower() for x in ["c2_full", "kv_only"])]
    if recursive_entries:
        r_simple = sum(e["simple_is_recursive"] for e in recursive_entries)
        r_tier = sum(e["tier_is_recursive"] for e in recursive_entries)
        print(f"  Recursive-condition samples: {len(recursive_entries)}")
        print(f"  Simple sensitivity: {r_simple}/{len(recursive_entries)} = {100*r_simple/len(recursive_entries):.1f}%")
        print(f"  Tier sensitivity: {r_tier}/{len(recursive_entries)} = {100*r_tier/len(recursive_entries):.1f}%")
    else:
        print("  No recursive-condition samples found")

    # ── Disagreement analysis ─────────────────────────────────────────────

    print(f"\n--- Disagreement Examples ---")
    disagree = [e for e in all_entries if e["simple_is_recursive"] != e["tier_is_recursive"]]
    print(f"  Total disagreements: {len(disagree)}/{n} = {100*len(disagree)/n:.1f}%")

    # Show up to 5 examples
    for i, e in enumerate(disagree[:5]):
        print(f"\n  [{i}] Source: {e['source']} idx={e['idx']}")
        print(f"      Simple: {e['simple_label']}  Tier: {e['tier_label']}")
        print(f"      R_V: {e.get('rv_mean', 'N/A')}")
        print(f"      Text: {e['generated'][:150]}...")

    # ── R_V correlation with classification ────────────────────────────────

    print(f"\n--- R_V Correlation with Classification ---")
    rv_entries = [e for e in all_entries if e.get("rv_mean") is not None]
    if rv_entries:
        rv_recursive = [e["rv_mean"] for e in rv_entries if e["simple_is_recursive"]]
        rv_not = [e["rv_mean"] for e in rv_entries if not e["simple_is_recursive"]]
        if rv_recursive and rv_not:
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(rv_recursive, rv_not, alternative="less")
            d_rv = (np.mean(rv_recursive) - np.mean(rv_not)) / np.sqrt(
                ((len(rv_recursive)-1)*np.var(rv_recursive, ddof=1) +
                 (len(rv_not)-1)*np.var(rv_not, ddof=1)) /
                (len(rv_recursive) + len(rv_not) - 2)
            ) if len(rv_recursive) > 1 and len(rv_not) > 1 else float("nan")
            print(f"  Recursive (BT+ART) R_V: {np.mean(rv_recursive):.3f} ± {np.std(rv_recursive):.3f} (n={len(rv_recursive)})")
            print(f"  Non-recursive R_V:      {np.mean(rv_not):.3f} ± {np.std(rv_not):.3f} (n={len(rv_not)})")
            print(f"  Mann-Whitney U: U={u:.0f}, p={p:.6f}, d={d_rv:.3f}")
        else:
            print(f"  Only one class has R_V values: recursive={len(rv_recursive)}, not={len(rv_not)}")
    else:
        print("  No R_V values found in samples")

    # ── Save results ──────────────────────────────────────────────────────

    output = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n,
        "n_output_files": len(output_files),
        "agreement": {
            "raw": agreement,
            "cohens_kappa": kappa,
            "tp": tp, "tn": tn, "fp_simple": fp_simple, "fn_simple": fn_simple,
        },
        "simple_recursive_rate": sum(simple_binary) / n if n > 0 else 0,
        "tier_recursive_rate": sum(tier_binary) / n if n > 0 else 0,
        "n_disagreements": len(disagree),
        "per_config": {},
    }

    for cfg in configs:
        cfg_entries = [e for e in all_entries if e["config"] == cfg]
        if len(cfg_entries) < 2:
            continue
        s_bin = [e["simple_is_recursive"] for e in cfg_entries]
        t_bin = [e["tier_is_recursive"] for e in cfg_entries]
        output["per_config"][cfg] = {
            "n": len(cfg_entries),
            "simple_rate": sum(s_bin) / len(s_bin),
            "tier_rate": sum(t_bin) / len(t_bin),
        }

    out_path = results_dir / "classifier_validation" / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
