#!/usr/bin/env python3
"""
POWER ANALYSIS & COMPREHENSIVE EFFECT SIZE TABLE

Computes achieved statistical power for every reported effect and produces
a LaTeX-ready summary table consolidating all statistical evidence.

Usage:
    python3 scripts/power_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats


RESULTS = Path(__file__).parent.parent / "results"


def load_json(path):
    with open(path) as f:
        content = f.read().strip()
        if not content:
            return None
        content = content.replace(": Infinity", ": 1e308")
        content = content.replace(": -Infinity", ": -1e308")
        content = content.replace(": NaN", ": null")
        return json.loads(content)


def achieved_power(d, n1, n2, alpha=0.05):
    """Compute achieved power for a two-sample t-test given observed d, n1, n2."""
    # Non-centrality parameter
    ncp = abs(d) * np.sqrt(n1 * n2 / (n1 + n2))
    df = n1 + n2 - 2
    # Critical value
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Power = P(|T| > t_crit | ncp)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return power


def required_n(d, power=0.80, alpha=0.05):
    """Compute required n per group for given d and power."""
    if abs(d) < 0.01:
        return float('inf')
    # Binary search
    for n in range(5, 10000):
        if achieved_power(d, n, n, alpha) >= power:
            return n
    return 10000


def main():
    print("=" * 80)
    print("POWER ANALYSIS & COMPREHENSIVE EFFECT SIZE SUMMARY")
    print("=" * 80)

    rows = []

    # ── 1. Statistical hardening effects ──
    hard_files = sorted((RESULTS / "statistical_hardening").glob("hardening_summary_*.json"))
    if hard_files:
        data = load_json(hard_files[-1])
        if data:
            for e in data["effects"]:
                d = e["d_observed"]
                n1, n2 = e["n1"], e["n2"]
                p = e.get("p_value", None)
                bf = e.get("bf10", None)
                bf_interp = e.get("bf_interpretation", "")
                ci_lo = e.get("ci_95_lower", None)
                ci_hi = e.get("ci_95_upper", None)

                pwr = achieved_power(d, n1, n2)
                n_req = required_n(d)

                rows.append({
                    "source": "hardening",
                    "name": e["name"],
                    "d": d,
                    "n1": n1,
                    "n2": n2,
                    "p": p,
                    "ci": (ci_lo, ci_hi),
                    "bf10": bf,
                    "bf_interp": bf_interp,
                    "power": pwr,
                    "n_required_80": n_req,
                    "adequately_powered": pwr >= 0.80,
                })

    # ── 2. Scaling gap models ──
    for model_file in sorted((RESULTS / "scaling_gap").glob("*_result.json")):
        data = load_json(model_file)
        if not data or "recursive_rv_values" not in data:
            continue

        model = data.get("model", model_file.stem)
        rec = [v for v in data["recursive_rv_values"] if v is not None]
        bas = [v for v in data["baseline_rv_values"] if v is not None]
        d = data.get("cohens_d", 0)
        p = data.get("p_value", None)

        if not rec or not bas:
            continue

        n1, n2 = len(rec), len(bas)
        pwr = achieved_power(d, n1, n2)
        se = np.sqrt(2 / n1 + d**2 / (2 * (n1 + n2)))

        rows.append({
            "source": "scaling_gap",
            "name": f"Scaling: {model}",
            "d": d,
            "n1": n1,
            "n2": n2,
            "p": p,
            "ci": (d - 1.96 * se, d + 1.96 * se),
            "bf10": None,
            "bf_interp": "",
            "power": pwr,
            "n_required_80": required_n(d),
            "adequately_powered": pwr >= 0.80,
        })

    # ── 3. FDR correction data ──
    fdr_files = sorted((RESULTS / "fdr_correction").glob("fdr_results_*.json"))
    fdr_map = {}
    if fdr_files:
        fdr_data = load_json(fdr_files[-1])
        if fdr_data and "tests" in fdr_data:
            for t in fdr_data["tests"]:
                fdr_map[t.get("test", t.get("name", ""))] = {
                    "p_fdr": t.get("q_value", t.get("p_corrected", None)),
                    "significant_fdr": t.get("significant_fdr", False),
                }

    # ── 4. Cluster-robust SEs ──
    cr_files = sorted((RESULTS / "cluster_robust_se").glob("cluster_robust_results_*.json"))
    cr_map = {}
    if cr_files:
        cr_data = load_json(cr_files[-1])
        if cr_data and "results" in cr_data:
            for r in cr_data["results"]:
                cr_map[r["comparison"]] = {
                    "ci_robust": r.get("ci_robust", r.get("ci_conservative", None)),
                    "sig_robust": r.get("significant_robust", r.get("significant_conservative", None)),
                }

    # ── Print summary ──
    print(f"\n{'Name':<45} {'d':>6} {'n1':>4} {'n2':>4} {'p':>10} {'Power':>6} {'Powered?':>8}")
    print("-" * 90)

    for r in rows:
        p_str = f"{r['p']:.2e}" if r['p'] is not None else "N/A"
        print(f"{r['name'][:44]:<45} {r['d']:>6.2f} {r['n1']:>4} {r['n2']:>4} {p_str:>10} "
              f"{r['power']:>6.3f} {'✓' if r['adequately_powered'] else '✗'}")

    n_powered = sum(1 for r in rows if r['adequately_powered'])
    n_total = len(rows)
    print(f"\nAdequately powered (≥0.80): {n_powered}/{n_total}")

    underpowered = [r for r in rows if not r['adequately_powered']]
    if underpowered:
        print("\n⚠ Underpowered effects:")
        for r in underpowered:
            print(f"  {r['name']}: power={r['power']:.3f}, need n≥{r['n_required_80']} per group")

    # ── Generate LaTeX table ──
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Comprehensive Effect Size Summary for All R\_V Comparisons}")
    latex.append(r"\label{tab:effect_sizes}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lrrrrrl}")
    latex.append(r"\toprule")
    latex.append(r"Comparison & $n_1$ & $n_2$ & $d$ & 95\% CI & Power & BF$_{10}$ \\")
    latex.append(r"\midrule")

    # Group by source
    current_source = None
    for r in rows:
        if r["source"] != current_source:
            if current_source is not None:
                latex.append(r"\addlinespace")
            current_source = r["source"]

        name = r["name"].replace("_", r"\_").replace("&", r"\&")
        if len(name) > 40:
            name = name[:37] + "..."

        ci_lo, ci_hi = r["ci"]
        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if ci_lo is not None else "---"

        bf_str = ""
        if r["bf10"] is not None:
            if r["bf10"] > 1000:
                bf_str = f">{1000:.0f}"
            elif r["bf10"] > 100:
                bf_str = f"{r['bf10']:.0f}"
            else:
                bf_str = f"{r['bf10']:.1f}"

        pwr_str = f"{r['power']:.2f}"

        # Bold if adequately powered
        if r["adequately_powered"]:
            d_str = f"\\textbf{{{r['d']:.2f}}}"
        else:
            d_str = f"{r['d']:.2f}"

        latex.append(f"{name} & {r['n1']} & {r['n2']} & {d_str} & {ci_str} & {pwr_str} & {bf_str} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\vspace{1mm}")
    latex.append(r"\raggedright\footnotesize")
    latex.append(r"Bold $d$ values indicate adequately powered effects ($1-\beta \geq 0.80$).")
    latex.append(r"BF$_{10}$: Bayes factor in favor of $H_1$. CI: frequentist 95\% confidence interval.")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)

    # ── Save ──
    out_dir = RESULTS / "power_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    out = {
        "timestamp": timestamp,
        "n_effects": n_total,
        "n_adequately_powered": n_powered,
        "effects": rows,
    }
    json_path = out_dir / f"power_analysis_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # LaTeX
    tex_path = out_dir / f"effect_size_table_{timestamp}.tex"
    with open(tex_path, "w") as f:
        f.write(latex_str)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
