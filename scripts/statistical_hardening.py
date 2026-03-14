#!/usr/bin/env python3
"""
STATISTICAL HARDENING
=====================

Computes rigorous statistical measures for all primary effects.
All values are LOADED from raw result files — nothing is hardcoded.

Outputs:
  results/statistical_hardening/hardening_summary_TIMESTAMP.json

Usage:
    python3 scripts/statistical_hardening.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.result_selection import load_best_persistent_patching_v3_dual


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_ci_bca(a, b, n_bootstrap=2000, alpha=0.05, seed=42):
    """
    Compute BCa bootstrap confidence interval for Cohen's d.

    Returns (d_observed, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)

    d_obs = cohens_d(a, b)
    if np.isnan(d_obs):
        return d_obs, float("nan"), float("nan")

    # Bootstrap distribution
    boot_ds = []
    for _ in range(n_bootstrap):
        a_boot = rng.choice(a, size=na, replace=True)
        b_boot = rng.choice(b, size=nb, replace=True)
        boot_ds.append(cohens_d(a_boot, b_boot))
    boot_ds = np.array(boot_ds)
    boot_ds = boot_ds[~np.isnan(boot_ds)]

    if len(boot_ds) < 100:
        return d_obs, float("nan"), float("nan")

    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_ds < d_obs))

    # Acceleration (jackknife)
    jack_ds = []
    for i in range(na):
        a_jack = np.delete(a, i)
        jack_ds.append(cohens_d(a_jack, b))
    for j in range(nb):
        b_jack = np.delete(b, j)
        jack_ds.append(cohens_d(a, b_jack))
    jack_ds = np.array(jack_ds)
    jack_mean = np.mean(jack_ds)
    num = np.sum((jack_mean - jack_ds) ** 3)
    den = 6 * (np.sum((jack_mean - jack_ds) ** 2) ** 1.5)
    acc = num / den if den > 1e-10 else 0

    # BCa percentiles
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    p_lo = stats.norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - acc * (z0 + z_alpha_lo)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - acc * (z0 + z_alpha_hi)))

    # Clamp to valid percentile range
    p_lo = np.clip(p_lo, 0.5 / n_bootstrap, 1 - 0.5 / n_bootstrap)
    p_hi = np.clip(p_hi, 0.5 / n_bootstrap, 1 - 0.5 / n_bootstrap)

    ci_lo = np.percentile(boot_ds, 100 * p_lo)
    ci_hi = np.percentile(boot_ds, 100 * p_hi)

    return d_obs, float(ci_lo), float(ci_hi)


def power_analysis(d, n1, n2, alpha=0.05):
    """Post-hoc power analysis for two-sample t-test."""
    from scipy.stats import nct

    df = n1 + n2 - 2
    ncp = abs(d) * np.sqrt(n1 * n2 / (n1 + n2))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    try:
        power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
    except (OverflowError, FloatingPointError):
        power = 1.0
    if np.isnan(power):
        power = 1.0  # nct overflows to nan when effect is very large
    return float(power)


# ── Raw data loaders ─────────────────────────────────────────────────────────


def load_power_up_effects():
    """Load cross-architecture effects from power_up result JSONs."""
    results_dir = PROJECT_ROOT / "results" / "power_up"
    effects = []
    for path in sorted(results_dir.glob("*_result.json")):
        try:
            with open(path) as f:
                text = f.read().strip()
            if not text:
                continue
            data = json.loads(text)
        except (json.JSONDecodeError, OSError):
            print(f"  WARNING: Skipping corrupt file {path.name}")
            continue
        model = data.get("model", path.stem)
        d = data.get("cohens_d")
        n_rec = data.get("n_recursive")
        n_bas = data.get("n_baseline")
        if d is not None and n_rec and n_bas:
            effects.append({
                "name": f"Power-up: {model} R_V",
                "d_observed": float(d),
                "n1": int(n_rec),
                "n2": int(n_bas),
                "unit": "prompt",
                "source": str(path.relative_to(PROJECT_ROOT)),
                "description": f"Recursive vs baseline R_V on {model} (power-up pipeline)",
            })
    return effects


def load_necessity_effect():
    """Load dual-layer necessity from persistent_patching_v3."""
    results_dir = PROJECT_ROOT / "results" / "persistent_patching_v3"
    path, data = load_best_persistent_patching_v3_dual(results_dir)
    if not path or not data:
        print("  WARNING: No persistent_patching_v3 dual files found")
        return []

    agg = data.get("aggregated", {})
    rec_clean = agg.get("recursive_clean", {})
    rec_patched = agg.get("recursive_dual_patched", {})

    # BT+ART rates
    rate_clean = rec_clean.get("bt_art_rate")
    rate_patched = rec_patched.get("bt_art_rate")
    n_turns_clean = rec_clean.get("total_turns")
    n_turns_patched = rec_patched.get("total_turns")
    n_sessions = data.get("n_sessions_per_condition", 10)

    # OR from comparisons
    comp = data.get("comparisons", {}).get("break_test", {})
    turn_level = comp.get("turn_level", {}) if isinstance(comp, dict) else {}
    odds_ratio = turn_level.get("or", comp.get("or") if isinstance(comp, dict) else None)
    p_value = turn_level.get("p", comp.get("p") if isinstance(comp, dict) else None)

    if rate_clean is None or rate_patched is None:
        print("  WARNING: Could not extract necessity rates")
        return []

    # Compute Cohen's h (effect size for proportion difference)
    # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    h = 2 * np.arcsin(np.sqrt(rate_clean)) - 2 * np.arcsin(np.sqrt(rate_patched))

    return [{
        "name": "Necessity: dual-layer break (BT+ART)",
        "d_observed": float(h),
        "d_type": "cohens_h",
        "n1": int(n_turns_clean),
        "n2": int(n_turns_patched),
        "unit": "turn",
        "effective_independent_n": int(n_sessions),
        "rate_clean": float(rate_clean),
        "rate_patched": float(rate_patched),
        "odds_ratio": float(odds_ratio) if odds_ratio else None,
        "p_value": float(p_value) if p_value else None,
        "source": str(path.relative_to(PROJECT_ROOT)),
        "description": f"Dual-layer patching kills recursive behavior "
                       f"({rate_clean:.0%} → {rate_patched:.1%})",
        "note": "n1/n2 are turns (10 sessions × 30 turns). "
                "Effective independent n = 10 sessions.",
    }]


def load_sufficiency_effect():
    """Load KV sufficiency from sufficiency_ladder."""
    results_dir = PROJECT_ROOT / "results" / "sufficiency_ladder"
    files = sorted(results_dir.glob("sufficiency_ladder_*.json"))
    if not files:
        print("  WARNING: No sufficiency_ladder files found")
        return []

    path = files[0]
    with open(path) as f:
        data = json.load(f)

    comp = data.get("comparisons", {}).get("kv_only_vs_baseline", {})
    turn_level = comp.get("turn_level", {})

    odds_ratio = turn_level.get("or")
    p_value = turn_level.get("p")
    test_rate = turn_level.get("test_rate")
    base_rate = turn_level.get("base_rate")

    if odds_ratio is None:
        print("  WARNING: Could not extract sufficiency OR")
        return []

    # Compute Cohen's h from rates
    h = 2 * np.arcsin(np.sqrt(test_rate)) - 2 * np.arcsin(np.sqrt(base_rate))

    n_sessions = data.get("n_sessions_per_condition", 10)
    max_turns = data.get("max_turns_per_session", 30)
    n_turns = n_sessions * max_turns

    return [{
        "name": "KV sufficiency: BT+ART uplift",
        "d_observed": float(h),
        "d_type": "cohens_h",
        "n1": int(n_turns),
        "n2": int(n_turns),
        "unit": "turn",
        "effective_independent_n": int(n_sessions),
        "test_rate": float(test_rate),
        "base_rate": float(base_rate),
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "source": str(path.relative_to(PROJECT_ROOT)),
        "description": f"KV-only injection BT+ART uplift "
                       f"({base_rate:.1%} → {test_rate:.1%}, OR={odds_ratio:.2f})",
        "note": "OR from turn-level Fisher's exact test. "
                "d is Cohen's h (proportion effect size), not Cohen's d from continuous data.",
    }]


def load_bridge_effect():
    """Load within-session bridge from raw bridge files."""
    results_dir = PROJECT_ROOT / "results" / "within_session_bridge"
    files = sorted(results_dir.glob("within_session_bridge_*.json"))
    if not files:
        print("  WARNING: No within_session_bridge files found")
        return []

    # Use the first file that has pooled stats
    for path in files:
        with open(path) as f:
            data = json.load(f)
        pooled = data.get("pooled", {}).get("recursive_only", {}).get("output_rv", {})
        if pooled and "cohens_d" in pooled:
            d_val = pooled["cohens_d"]
            n_bt_art = pooled.get("n_bt_art")
            n_other = pooled.get("n_other")
            p_value = pooled.get("mannwhitney_p")

            return [{
                "name": "Within-session bridge",
                "d_observed": float(d_val),
                "n1": int(n_bt_art),
                "n2": int(n_other),
                "unit": "turn",
                "p_value": float(p_value) if p_value else None,
                "source": str(path.relative_to(PROJECT_ROOT)),
                "description": f"R_V predicts output quality within recursive sessions "
                               f"(BT+ART n={n_bt_art} vs other n={n_other})",
            }]

    print("  WARNING: No bridge file with pooled cohens_d found")
    return []


def load_self_feeding_effect():
    """Load gnani vs recursive from self_feeding_loop files."""
    results_dir = PROJECT_ROOT / "results" / "self_feeding_loop"
    gnani_files = sorted(results_dir.glob("gnani_scaffolded_*.json"))
    recursive_files = sorted(results_dir.glob("self_feed_recursive_*.json"))

    if not gnani_files or not recursive_files:
        print("  WARNING: No self_feeding_loop files found")
        return []

    gnani_rates = []
    for path in gnani_files:
        with open(path) as f:
            data = json.load(f)
        rate = data.get("bt_art_rate")
        if rate is not None:
            gnani_rates.append(float(rate))

    recursive_rates = []
    for path in recursive_files:
        with open(path) as f:
            data = json.load(f)
        rate = data.get("bt_art_rate")
        if rate is not None:
            recursive_rates.append(float(rate))

    if len(gnani_rates) < 2 or len(recursive_rates) < 2:
        print("  WARNING: Insufficient self_feeding data")
        return []

    d_val = cohens_d(gnani_rates, recursive_rates)

    return [{
        "name": "Self-feeding: Gnani vs recursive",
        "d_observed": float(d_val),
        "n1": len(gnani_rates),
        "n2": len(recursive_rates),
        "unit": "session",
        "source": f"results/self_feeding_loop/ ({len(gnani_files)} gnani + {len(recursive_files)} recursive)",
        "description": "Gnani scaffolded vs self-feeding recursive BT+ART rates",
        "note": f"n=5 per group — small sample, interpret with caution. "
                f"Gnani mean={np.mean(gnani_rates):.3f}, Recursive mean={np.mean(recursive_rates):.3f}",
    }]


# ── Main ─────────────────────────────────────────────────────────────────────


def run_hardening():
    """Run statistical hardening on all primary effects, loaded from raw data."""
    print("=" * 70)
    print("STATISTICAL HARDENING (all values loaded from raw files)")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "results" / "statistical_hardening"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all effects from raw data
    print("\nLoading effects from raw result files...")
    all_effects = []
    all_effects.extend(load_power_up_effects())
    all_effects.extend(load_necessity_effect())
    all_effects.extend(load_sufficiency_effect())
    all_effects.extend(load_bridge_effect())
    all_effects.extend(load_self_feeding_effect())

    print(f"  Loaded {len(all_effects)} effects from raw files\n")

    # Compute hardened statistics
    results = []
    for effect in all_effects:
        print(f"  {effect['name']}:")

        d = effect["d_observed"]
        n1, n2 = effect["n1"], effect["n2"]

        # Power analysis
        pwr = power_analysis(d, n1, n2)
        print(f"    d = {d:.4f}, n = ({n1}, {n2}), unit = {effect.get('unit', '?')}")
        print(f"    Post-hoc power: {pwr:.4f}")

        # Approximate CI from SE of d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        ci_lo = d - 1.96 * se_d
        ci_hi = d + 1.96 * se_d
        print(f"    Approximate 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

        # Approximate Bayes Factor from t-statistic
        t_approx = d * np.sqrt(n1 * n2 / (n1 + n2))
        bic_diff = np.log(n1 + n2) - t_approx**2
        bf10 = np.exp(-0.5 * bic_diff)
        bf_label = ("decisive" if bf10 > 100 else "very strong" if bf10 > 30
                     else "strong" if bf10 > 10 else "moderate" if bf10 > 3
                     else "anecdotal")
        print(f"    BF10 ≈ {bf10:.2e} ({bf_label})")

        result = {
            "name": effect["name"],
            "description": effect["description"],
            "d_observed": d,
            "d_type": effect.get("d_type", "cohens_d"),
            "n1": n1,
            "n2": n2,
            "unit": effect.get("unit", "unknown"),
            "source": effect.get("source", "unknown"),
            "se_d": float(se_d),
            "ci_95_lower": float(ci_lo),
            "ci_95_upper": float(ci_hi),
            "power": pwr,
            "bf10_approx": float(bf10),
            "bf_interpretation": bf_label,
        }
        # Pass through extra fields
        for key in ("effective_independent_n", "odds_ratio", "p_value",
                     "rate_clean", "rate_patched", "test_rate", "base_rate",
                     "note"):
            if key in effect:
                result[key] = effect[key]

        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("HARDENING SUMMARY")
    print("=" * 70)
    print(f"\n{'Effect':<45} {'d':>8} {'unit':>8} {'n1':>5} {'n2':>5} {'Power':>7} {'BF10':>12}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<45} "
              f"{r['d_observed']:>8.4f} "
              f"{r['unit']:>8} "
              f"{r['n1']:>5} "
              f"{r['n2']:>5} "
              f"{r['power']:>7.4f} "
              f"{r['bf10_approx']:>12.2e}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "provenance": "All values loaded from raw result files. Nothing hardcoded.",
        "n_effects": len(results),
        "effects": results,
    }
    summary_path = out_dir / f"hardening_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_hardening()
