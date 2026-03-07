#!/usr/bin/env python3
"""Per-token R_V trajectory analysis.

Analyzes existing per-token R_V data for:
1. Phase transition / crystallization detection
2. Recursive vs baseline trajectory divergence timing
3. Rate of contraction in early vs late generation
4. Token-position-linked R_V dynamics
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent


def detect_changepoint(values, min_segment=5):
    """Simple changepoint detection using max cumulative sum.

    Returns the index where the mean shift is largest.
    """
    n = len(values)
    if n < 2 * min_segment:
        return None, 0.0

    arr = np.array(values, dtype=float)
    best_idx = None
    best_score = 0

    for i in range(min_segment, n - min_segment):
        left_mean = np.mean(arr[:i])
        right_mean = np.mean(arr[i:])
        left_std = np.std(arr[:i], ddof=1) if i > 1 else 1e-6
        right_std = np.std(arr[i:], ddof=1) if (n - i) > 1 else 1e-6
        pooled_std = np.sqrt(((i - 1) * left_std**2 + (n - i - 1) * right_std**2) / (n - 2))
        if pooled_std < 1e-10:
            continue
        score = abs(left_mean - right_mean) / pooled_std
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


def exponential_fit(values, max_points=100):
    """Fit R_V(t) = a * exp(-b * t) + c to early trajectory.

    Returns (a, b, c, r_squared).
    """
    y = np.array(values[:max_points], dtype=float)
    x = np.arange(len(y), dtype=float)

    # Remove NaN
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None

    # Simple approach: fit log(y - y_min) ~ -b*x + log(a)
    y_min = np.min(y)
    y_shifted = y - y_min + 0.001  # avoid log(0)

    try:
        log_y = np.log(y_shifted)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)
        b = -slope
        a = np.exp(intercept)
        c = y_min - 0.001
        r_sq = r_value ** 2
        return {"a": float(a), "b": float(b), "c": float(c), "r_squared": float(r_sq), "half_life": float(np.log(2) / b) if b > 0 else float("inf")}
    except Exception:
        return None


def compute_divergence_timing(recursive_mean, baseline_mean, threshold_d=0.3):
    """Find the first token position where recursive and baseline trajectories
    diverge by more than threshold_d effect size."""
    n = min(len(recursive_mean), len(baseline_mean))
    if n < 10:
        return None

    # Use a sliding window comparison
    window = 5
    for i in range(window, n - window):
        rec_window = recursive_mean[i:i+window]
        base_window = baseline_mean[i:i+window]
        rec_vals = [v for v in rec_window if not np.isnan(v)]
        base_vals = [v for v in base_window if not np.isnan(v)]
        if len(rec_vals) < 3 or len(base_vals) < 3:
            continue
        d = (np.mean(rec_vals) - np.mean(base_vals))
        pooled = np.sqrt((np.var(rec_vals) + np.var(base_vals)) / 2)
        if pooled > 0 and abs(d / pooled) > threshold_d:
            return i
    return None


def main():
    # Load per-token data
    data_path = PROJECT_ROOT / "results" / "batch_per_token_rv" / "batch_per_token_rv_20260220_161603.json"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    with open(data_path) as f:
        data = json.load(f)

    summary = data["summary"]
    print("=" * 70)
    print("PER-TOKEN R_V TRAJECTORY ANALYSIS")
    print("=" * 70)
    print(f"\nModel: {summary['model']}")
    print(f"Recursive prompts: {summary['n_recursive']}, Baseline: {summary['n_baseline']}")
    print(f"Max tokens: {summary['max_new_tokens']}")

    # Extract mean trajectories
    rec_mean = summary.get("trajectory_recursive", {}).get("mean", [])
    base_mean = summary.get("trajectory_baseline", {}).get("mean", [])

    # Clean NaN at start
    rec_clean = [v if not np.isnan(v) else None for v in rec_mean]
    base_clean = [v if not np.isnan(v) else None for v in base_mean]

    # Convert to float arrays, dropping initial NaN
    rec_arr = np.array([v for v in rec_mean if not np.isnan(v)])
    base_arr = np.array([v for v in base_mean if not np.isnan(v)])

    print(f"\nRecursive trajectory: {len(rec_arr)} valid points")
    print(f"Baseline trajectory: {len(base_arr)} valid points")

    # ── Phase Analysis ────────────────────────────────────────────────────

    print(f"\n--- Phase Transition Detection ---")

    # Recursive trajectory
    cp_rec, cp_score_rec = detect_changepoint(rec_arr)
    if cp_rec:
        print(f"  Recursive changepoint: token {cp_rec} (score={cp_score_rec:.2f})")
        print(f"    Before: R_V={np.mean(rec_arr[:cp_rec]):.4f}")
        print(f"    After:  R_V={np.mean(rec_arr[cp_rec:]):.4f}")
        print(f"    Drop:   {np.mean(rec_arr[:cp_rec]) - np.mean(rec_arr[cp_rec:]):.4f}")
    else:
        print("  No significant changepoint in recursive trajectory")

    # Baseline trajectory
    cp_base, cp_score_base = detect_changepoint(base_arr)
    if cp_base:
        print(f"  Baseline changepoint: token {cp_base} (score={cp_score_base:.2f})")
        print(f"    Before: R_V={np.mean(base_arr[:cp_base]):.4f}")
        print(f"    After:  R_V={np.mean(base_arr[cp_base:]):.4f}")
    else:
        print("  No significant changepoint in baseline trajectory")

    # ── Exponential Decay Fit ─────────────────────────────────────────────

    print(f"\n--- Exponential Decay Fit (R_V(t) = a*exp(-b*t) + c) ---")

    rec_fit = exponential_fit(rec_arr)
    if rec_fit:
        print(f"  Recursive: a={rec_fit['a']:.4f}, b={rec_fit['b']:.6f}, c={rec_fit['c']:.4f}")
        print(f"    R²={rec_fit['r_squared']:.4f}, half-life={rec_fit['half_life']:.1f} tokens")
    else:
        print("  Recursive: fit failed")

    base_fit = exponential_fit(base_arr)
    if base_fit:
        print(f"  Baseline:  a={base_fit['a']:.4f}, b={base_fit['b']:.6f}, c={base_fit['c']:.4f}")
        print(f"    R²={base_fit['r_squared']:.4f}, half-life={base_fit['half_life']:.1f} tokens")
    else:
        print("  Baseline: fit failed")

    # ── Divergence Timing ─────────────────────────────────────────────────

    print(f"\n--- Trajectory Divergence Timing ---")
    rec_mean_clean = [v if not np.isnan(v) else 0.7 for v in rec_mean]
    base_mean_clean = [v if not np.isnan(v) else 0.7 for v in base_mean]

    div_token = compute_divergence_timing(rec_mean_clean, base_mean_clean, threshold_d=0.2)
    if div_token is not None:
        print(f"  Divergence begins at token ~{div_token} (d>0.2)")
    else:
        print(f"  No significant divergence detected (d>0.2 threshold)")

    div_token_strong = compute_divergence_timing(rec_mean_clean, base_mean_clean, threshold_d=0.5)
    if div_token_strong is not None:
        print(f"  Strong divergence at token ~{div_token_strong} (d>0.5)")

    # ── Windowed Statistics ───────────────────────────────────────────────

    print(f"\n--- Windowed R_V Statistics ---")
    windows = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 256)]
    print(f"  {'Window':12s}  {'Rec Mean':>8s}  {'Base Mean':>9s}  {'Diff':>6s}  {'p-value':>8s}")

    for start, end in windows:
        rec_w = rec_arr[start:min(end, len(rec_arr))]
        base_w = base_arr[start:min(end, len(base_arr))]
        if len(rec_w) < 3 or len(base_w) < 3:
            continue
        _, p = stats.mannwhitneyu(rec_w, base_w, alternative="two-sided")
        diff = np.mean(rec_w) - np.mean(base_w)
        print(f"  [{start:3d}-{end:3d}]   {np.mean(rec_w):.4f}    {np.mean(base_w):.4f}  {diff:+.4f}  {p:.6f}")

    # ── Crystallization Analysis ──────────────────────────────────────────

    print(f"\n--- Crystallization Analysis ---")
    # Check if R_V variance decreases over time (crystallization = stable low R_V)
    if len(rec_arr) >= 100:
        early_var = np.var(rec_arr[5:50])
        late_var = np.var(rec_arr[100:])
        print(f"  Recursive R_V variance (early tokens 5-50):   {early_var:.6f}")
        print(f"  Recursive R_V variance (late tokens 100+):    {late_var:.6f}")
        print(f"  Variance ratio (late/early):                  {late_var/early_var:.3f}")
        print(f"  Interpretation: {'Crystallized (variance decreasing)' if late_var < early_var else 'Not crystallized'}")

    if len(base_arr) >= 100:
        early_var_b = np.var(base_arr[5:50])
        late_var_b = np.var(base_arr[100:])
        print(f"  Baseline R_V variance (early tokens 5-50):    {early_var_b:.6f}")
        print(f"  Baseline R_V variance (late tokens 100+):     {late_var_b:.6f}")

    # ── Overall Summary ───────────────────────────────────────────────────

    print(f"\n--- Summary ---")
    print(f"  Recursive mean R_V (generation): {summary.get('mean_generation_rv_recursive', 'N/A'):.4f}")
    print(f"  Baseline mean R_V (generation):  {summary.get('mean_generation_rv_baseline', 'N/A'):.4f}")
    print(f"  Mann-Whitney p: {summary.get('mannwhitney_p', 'N/A'):.6f}")
    print(f"  Cohen's d: {summary.get('cohens_d_recursive_vs_baseline', 'N/A'):.4f}")

    # ── Individual trajectory analysis ────────────────────────────────────

    # Check individual trajectories in the data
    trajectories = data.get("trajectories", [])
    if trajectories:
        print(f"\n--- Individual Trajectory Statistics ---")
        rec_trajs = [t for t in trajectories if t.get("condition") == "recursive"]
        base_trajs = [t for t in trajectories if t.get("condition") == "baseline"]

        rec_mins = [t.get("min_rv", float("nan")) for t in rec_trajs]
        base_mins = [t.get("min_rv", float("nan")) for t in base_trajs]
        rec_mins = [v for v in rec_mins if not np.isnan(v)]
        base_mins = [v for v in base_mins if not np.isnan(v)]

        if rec_mins and base_mins:
            print(f"  Recursive min R_V: {np.mean(rec_mins):.4f} ± {np.std(rec_mins):.4f}")
            print(f"  Baseline min R_V:  {np.mean(base_mins):.4f} ± {np.std(base_mins):.4f}")
            _, p = stats.mannwhitneyu(rec_mins, base_mins, alternative="less")
            print(f"  Mann-Whitney (min R_V): p={p:.6f}")

        # Check how many reach deep contraction
        n_deep_rec = sum(1 for v in rec_mins if v < 0.5)
        n_deep_base = sum(1 for v in base_mins if v < 0.5)
        print(f"  Deep contraction (min R_V < 0.5): recursive={n_deep_rec}/{len(rec_mins)}, baseline={n_deep_base}/{len(base_mins)}")

    # ── Save results ──────────────────────────────────────────────────────

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": summary["model"],
        "n_recursive": summary["n_recursive"],
        "n_baseline": summary["n_baseline"],
        "changepoint_recursive": {"token": cp_rec, "score": float(cp_score_rec)} if cp_rec else None,
        "changepoint_baseline": {"token": cp_base, "score": float(cp_score_base)} if cp_base else None,
        "exponential_fit_recursive": rec_fit,
        "exponential_fit_baseline": base_fit,
        "divergence_token_d02": div_token,
        "divergence_token_d05": div_token_strong,
        "overall_d": float(summary.get("cohens_d_recursive_vs_baseline", float("nan"))),
        "overall_p": float(summary.get("mannwhitney_p", float("nan"))),
    }

    out_path = PROJECT_ROOT / "results" / "per_token_rv_analysis" / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
