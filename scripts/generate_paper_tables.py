#!/usr/bin/env python3
"""
GENERATE PAPER TABLES FROM RAW DATA
====================================

Produces LaTeX table fragments for the paper by loading ALL values
from raw result JSON files. No hardcoded statistics.

Output:
  R_V_PAPER/generated_table_effects.tex    — Table 1 (effect sizes)

Usage:
    python3 scripts/generate_paper_tables.py
"""

import sys
import json
import math
from pathlib import Path

import numpy as np

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


def se_d(d, n1, n2):
    """Standard error of Cohen's d."""
    return np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))


def fmt_d(d_val):
    """Format d for LaTeX: negative gets $-$, bold if |d| > 0.5."""
    if np.isnan(d_val):
        return "---"
    sign = "$-$" if d_val < 0 else ""
    abs_d = abs(d_val)
    s = f"{sign}{abs_d:.2f}"
    if abs_d >= 0.5:
        s = f"\\textbf{{{s}}}"
    return s


def fmt_ci(lo, hi):
    """Format CI for LaTeX."""
    if np.isnan(lo) or np.isnan(hi):
        return "---"
    lo_str = f"-{abs(lo):.2f}" if lo < 0 else f"{lo:.2f}"
    hi_str = f"-{abs(hi):.2f}" if hi < 0 else f"{hi:.2f}"
    return f"$[{lo_str}, {hi_str}]$"


def fmt_bf(bf10):
    """Format Bayes Factor for LaTeX."""
    if bf10 > 1e6:
        exp = int(np.log10(bf10))
        return f"$> 10^{{{exp}}}$"
    elif bf10 > 100:
        return f"${bf10:.0f}$"
    elif bf10 > 1:
        return f"${bf10:.1f}$"
    else:
        return f"${bf10:.2f}$"


def approx_bf10(d_val, n1, n2):
    """BIC-approximated Bayes Factor."""
    t_approx = d_val * np.sqrt(n1 * n2 / (n1 + n2))
    bic_diff = np.log(n1 + n2) - t_approx**2
    return np.exp(-0.5 * bic_diff)


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_power_up():
    """Load cross-architecture R_V from power_up results."""
    results_dir = PROJECT_ROOT / "results" / "power_up"
    rows = []
    for path in sorted(results_dir.glob("*_result.json")):
        try:
            text = path.read_text().strip()
            if not text:
                continue
            data = json.loads(text)
        except (json.JSONDecodeError, OSError):
            continue
        d_val = data.get("cohens_d")
        n1 = data.get("n_recursive")
        n2 = data.get("n_baseline")
        model = data.get("model", path.stem)
        if d_val is None or n1 is None or n2 is None:
            continue

        # Display names
        display = {
            "mistral-7b": "Mistral-7B",
            "qwen2.5-7b": "Qwen2.5-7B",
            "opt-6.7b": "OPT-6.7B",
            "gpt2-xl": "GPT-2 XL",
            "pythia-1.4b": "Pythia-1.4B",
        }.get(model, model)

        se = se_d(d_val, n1, n2)
        ci_lo = d_val - 1.96 * se
        ci_hi = d_val + 1.96 * se

        rows.append({
            "name": display,
            "n1": n1, "n2": n2,
            "d": d_val,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "bf10": approx_bf10(d_val, n1, n2),
            "source": str(path.relative_to(PROJECT_ROOT)),
        })
    return rows


def load_scaling_gap():
    """Load scaling gap results."""
    results_dir = PROJECT_ROOT / "results" / "scaling_gap"
    rows = []
    for path in sorted(results_dir.glob("*_result.json")):
        try:
            data = json.loads(path.read_text().strip())
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if "error" in data:
            continue
        d_val = data.get("cohens_d")
        n1 = data.get("n_recursive")
        n2 = data.get("n_baseline")
        model = data.get("model", path.stem)
        if d_val is None or n1 is None or n2 is None:
            continue

        display = {
            "qwen2.5-3b": "Qwen2.5-3B",
            "phi-3-mini-4k": "Phi-3-mini-4k",
            "pythia-6.9b": "Pythia-6.9B",
        }.get(model, model)

        # Prefer bootstrap CIs from raw data over normal approximation
        raw_ci_lo = data.get("ci_95_lo")
        raw_ci_hi = data.get("ci_95_hi")
        if raw_ci_lo is not None and raw_ci_hi is not None:
            ci_lo, ci_hi = raw_ci_lo, raw_ci_hi
        else:
            se = se_d(d_val, n1, n2)
            ci_lo = d_val - 1.96 * se
            ci_hi = d_val + 1.96 * se

        rows.append({
            "name": display,
            "n1": n1, "n2": n2,
            "d": d_val,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "bf10": approx_bf10(d_val, n1, n2),
            "source": str(path.relative_to(PROJECT_ROOT)),
        })
    return rows


def load_causal_structural():
    """Load necessity, sufficiency, bridge, self-feeding from raw data."""
    rows = []

    # 1. Necessity (dual-layer break)
    nec_path = PROJECT_ROOT / "results" / "persistent_patching_v3"
    best_nec_path, data = load_best_persistent_patching_v3_dual(nec_path)
    if best_nec_path and data:
        agg = data.get("aggregated", {})
        rate_clean = agg.get("recursive_clean", {}).get("bt_art_rate")
        rate_patched = agg.get("recursive_dual_patched", {}).get("bt_art_rate")
        n1 = agg.get("recursive_clean", {}).get("total_turns", 300)
        n2 = agg.get("recursive_dual_patched", {}).get("total_turns", 300)
        n_sessions = data.get("n_sessions_per_condition", 10)
        comp = data.get("comparisons", {}).get("break_test", {})
        turn_level = comp.get("turn_level", {}) if isinstance(comp, dict) else {}
        or_val = turn_level.get("or", comp.get("or") if isinstance(comp, dict) else None)
        if or_val is None:
            or_text = "N/A"
        elif math.isinf(or_val):
            or_text = "inf"
        else:
            or_text = f"{or_val:.1f}"

        if rate_clean is not None and rate_patched is not None:
            h = 2 * np.arcsin(np.sqrt(rate_clean)) - 2 * np.arcsin(np.sqrt(rate_patched))
            se = se_d(h, n1, n2)
            rows.append({
                "name": f"Necessity (dual-layer break)",
                "n1": n1, "n2": n2,
                "d": h,
                "d_type": "h",
                "ci_lo": h - 1.96 * se,
                "ci_hi": h + 1.96 * se,
                "bf10": approx_bf10(h, n1, n2),
                "source": str(best_nec_path.relative_to(PROJECT_ROOT)),
                "note": f"Cohen's h; {rate_clean:.0%} -> {rate_patched:.1%}; "
                        f"OR={or_text}; n={n_sessions} sessions x 30 turns",
            })

    # 2. KV sufficiency
    suf_path = PROJECT_ROOT / "results" / "sufficiency_ladder"
    suf_files = sorted(suf_path.glob("sufficiency_ladder_*.json"))
    if suf_files:
        data = json.loads(suf_files[0].read_text())
        comp = data.get("comparisons", {}).get("kv_only_vs_baseline", {})
        turn = comp.get("turn_level", {})
        or_val = turn.get("or")
        test_rate = turn.get("test_rate")
        base_rate = turn.get("base_rate")
        n_sessions = data.get("n_sessions_per_condition", 10)
        max_turns = data.get("max_turns_per_session", 30)
        n_turns = n_sessions * max_turns

        if test_rate is not None and base_rate is not None:
            h = 2 * np.arcsin(np.sqrt(test_rate)) - 2 * np.arcsin(np.sqrt(base_rate))
            se = se_d(h, n_turns, n_turns)
            rows.append({
                "name": "KV sufficiency",
                "n1": n_turns, "n2": n_turns,
                "d": h,
                "d_type": "h",
                "ci_lo": h - 1.96 * se,
                "ci_hi": h + 1.96 * se,
                "bf10": approx_bf10(h, n_turns, n_turns),
                "source": str(suf_files[0].relative_to(PROJECT_ROOT)),
                "note": f"Cohen's h; {base_rate:.1%} -> {test_rate:.1%}; OR={or_val:.2f}",
            })

    # 3. Within-session bridge
    bridge_path = PROJECT_ROOT / "results" / "within_session_bridge"
    bridge_files = sorted(bridge_path.glob("within_session_bridge_*.json"))
    for bf in bridge_files:
        data = json.loads(bf.read_text())
        pooled = data.get("pooled", {}).get("recursive_only", {}).get("output_rv", {})
        if pooled and "cohens_d" in pooled:
            d_val = pooled["cohens_d"]
            n1 = pooled["n_bt_art"]
            n2 = pooled["n_other"]
            se_val = se_d(d_val, n1, n2)
            rows.append({
                "name": "Within-session bridge",
                "n1": n1, "n2": n2,
                "d": d_val,
                "ci_lo": d_val - 1.96 * se_val,
                "ci_hi": d_val + 1.96 * se_val,
                "bf10": approx_bf10(d_val, n1, n2),
                "source": str(bf.relative_to(PROJECT_ROOT)),
            })
            break

    # 4. Self-feeding
    sf_path = PROJECT_ROOT / "results" / "self_feeding_loop"
    gnani = sorted(sf_path.glob("gnani_scaffolded_*.json"))
    recursive = sorted(sf_path.glob("self_feed_recursive_*.json"))
    if gnani and recursive:
        g_rates = []
        for gf in gnani:
            d = json.loads(gf.read_text())
            if "bt_art_rate" in d:
                g_rates.append(d["bt_art_rate"])
        r_rates = []
        for rf in recursive:
            d = json.loads(rf.read_text())
            if "bt_art_rate" in d:
                r_rates.append(d["bt_art_rate"])
        if len(g_rates) >= 2 and len(r_rates) >= 2:
            d_val = cohens_d(g_rates, r_rates)
            se_val = se_d(d_val, len(g_rates), len(r_rates))
            rows.append({
                "name": "Self-feeding (Gnani vs recursive)",
                "n1": len(g_rates), "n2": len(r_rates),
                "d": d_val,
                "ci_lo": d_val - 1.96 * se_val,
                "ci_hi": d_val + 1.96 * se_val,
                "bf10": approx_bf10(d_val, len(g_rates), len(r_rates)),
                "source": f"results/self_feeding_loop/ ({len(gnani)} + {len(recursive)} files)",
            })

    return rows


# ── Table generation ─────────────────────────────────────────────────────────


def generate_effects_table():
    """Generate Table 1 (comprehensive effect sizes) from raw data."""
    causal = load_causal_structural()
    cross_arch = load_power_up()
    scaling = load_scaling_gap()

    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py")
    lines.append("% DO NOT EDIT — regenerate from raw data instead.")
    lines.append(f"% Generated: {__import__('datetime').datetime.now().isoformat()}")
    lines.append("")
    lines.append("\\begin{tabular}{lrrrrl}")
    lines.append("\\toprule")
    lines.append("Comparison & $n_1$ & $n_2$ & $d$ & 95\\% CI & $\\text{BF}_{10}$ \\\\")
    lines.append("\\midrule")

    # Causal section
    lines.append("\\multicolumn{6}{l}{\\emph{Causal \\& structural effects}} \\\\")
    for row in causal:
        d_type_note = "$^h$" if row.get("d_type") == "h" else ""
        lines.append(
            f"{row['name']} & {row['n1']} & {row['n2']} & "
            f"{fmt_d(row['d'])}{d_type_note} & "
            f"{fmt_ci(row['ci_lo'], row['ci_hi'])} & "
            f"{fmt_bf(row['bf10'])} \\\\"
        )

    lines.append("\\addlinespace")
    lines.append("\\multicolumn{6}{l}{\\emph{Cross-architecture (power-up pipeline)}} \\\\")
    for row in cross_arch:
        lines.append(
            f"{row['name']} & {row['n1']} & {row['n2']} & "
            f"{fmt_d(row['d'])} & "
            f"{fmt_ci(row['ci_lo'], row['ci_hi'])} & "
            f"--- \\\\"
        )

    if scaling:
        lines.append("\\addlinespace")
        lines.append("\\multicolumn{6}{l}{\\emph{Scaling gap}} \\\\")
        for row in scaling:
            lines.append(
                f"{row['name']} & {row['n1']} & {row['n2']} & "
                f"{fmt_d(row['d'])} & "
                f"{fmt_ci(row['ci_lo'], row['ci_hi'])} & "
                f"--- \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Provenance footer
    lines.append("")
    lines.append("% SOURCE PROVENANCE:")
    for section_name, section_rows in [("Causal", causal),
                                         ("Cross-arch", cross_arch),
                                         ("Scaling", scaling)]:
        for row in section_rows:
            lines.append(f"%   {row['name']}: {row.get('source', 'unknown')}")
            if "note" in row:
                lines.append(f"%     NOTE: {row['note']}")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("GENERATING PAPER TABLES FROM RAW DATA")
    print("=" * 70)

    tex = generate_effects_table()

    out_path = PROJECT_ROOT / "R_V_PAPER" / "generated_table_effects.tex"
    out_path.write_text(tex)
    print(f"\n  Table 1 written to: {out_path}")

    # Also print for review
    print("\n" + tex)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
