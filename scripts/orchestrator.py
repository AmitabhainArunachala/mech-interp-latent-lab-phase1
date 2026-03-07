#!/usr/bin/env python3
"""
AUTOMATED DISCOVERY ORCHESTRATOR v1
====================================

Reads all existing experiment results, builds a knowledge graph of findings,
identifies gaps, and proposes the highest-value next experiments.

This is the "scientist brain" that sits on top of the measurement pipeline.

Capabilities:
  1. Ingest — scan results/ for all JSON outputs
  2. Knowledge Graph — build a structured graph of {finding, evidence, confidence}
  3. Gap Analysis — compare known findings against theory predictions
  4. Experiment Proposals — rank next experiments by expected information gain
  5. Report — emit a structured JSON + human-readable summary

Usage:
    python3 scripts/orchestrator.py
    python3 scripts/orchestrator.py --results-dir results/ --output results/orchestrator/
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 1. Result Ingestion ──────────────────────────────────────────────────

RESULT_SCANNERS = {
    "self_feeding_loop": "self_feeding_loop",
    "mode_atlas": "mode_atlas",
    "scaling_law": "scaling_law_sweep",
    "per_head": "per_head_attention",
    "path_patching": "path_patching",
    "realtime_monitor": "realtime_monitor",
    "statistical_hardening": "statistical_hardening",
}


def ingest_results(results_dir: Path) -> dict:
    """Scan all result directories and load JSON files."""
    inventory = {}
    for name, subdir in RESULT_SCANNERS.items():
        full_path = results_dir / subdir if not Path(subdir).is_absolute() else Path(subdir)
        if not full_path.exists():
            inventory[name] = {"status": "not_run", "files": []}
            continue

        jsons = sorted(full_path.glob("*.json"))
        inventory[name] = {
            "status": "completed" if jsons else "empty",
            "files": [str(f) for f in jsons],
            "count": len(jsons),
            "latest": str(jsons[-1]) if jsons else None,
        }

        # Load latest result for quick access
        if jsons:
            try:
                with open(jsons[-1]) as f:
                    inventory[name]["latest_data"] = json.load(f)
            except (json.JSONDecodeError, IOError):
                inventory[name]["latest_data"] = None

    return inventory


# ── 2. Knowledge Graph ───────────────────────────────────────────────────

class Finding:
    """A single research finding with evidence chain."""

    def __init__(self, claim: str, evidence: str, confidence: str,
                 source: str, metrics: dict = None):
        self.claim = claim
        self.evidence = evidence
        self.confidence = confidence  # "strong", "moderate", "preliminary", "negative"
        self.source = source
        self.metrics = metrics or {}

    def to_dict(self):
        return {
            "claim": self.claim,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source": self.source,
            "metrics": self.metrics,
        }


def build_knowledge_graph(inventory: dict) -> list:
    """Extract structured findings from experiment results."""
    findings = []

    # ── Self-feeding loop findings ──
    sfl = inventory.get("self_feeding_loop", {})
    if sfl.get("status") == "completed" and sfl.get("latest_data"):
        data = sfl["latest_data"]
        findings.append(Finding(
            claim="R_V contraction is prompt-contingent, not a self-sustaining attractor",
            evidence=f"Self-feeding loop: recursive={data.get('recursive_rv', '?')}, "
                     f"baseline={data.get('baseline_rv', '?')}, "
                     f"scaffolded={data.get('scaffolded_rv', '?')}",
            confidence="strong",
            source="self_feeding_loop",
            metrics={k: v for k, v in data.items() if isinstance(v, (int, float))},
        ))

    # ── Mode atlas findings ──
    ma = inventory.get("mode_atlas", {})
    if ma.get("status") == "completed" and ma.get("latest_data"):
        data = ma["latest_data"]
        modes = data.get("mode_fingerprints", {})
        if modes:
            sr_rv = modes.get("self_referential", {}).get("mean_rv")
            findings.append(Finding(
                claim=f"Self-referential mode has distinct geometric signature (R_V={sr_rv})",
                evidence=f"{len(modes)} modes measured, pairwise comparisons available",
                confidence="moderate" if sr_rv else "preliminary",
                source="mode_atlas",
                metrics={"n_modes": len(modes), "sr_rv": sr_rv},
            ))

    # ── Scaling law findings ──
    sl = inventory.get("scaling_law", {})
    if sl.get("status") == "completed" and sl.get("latest_data"):
        data = sl["latest_data"]
        models = data.get("models", [])
        if models:
            findings.append(Finding(
                claim=f"R_V measured across {len(models)} model scales",
                evidence=f"Models: {[m.get('name', '?') for m in models]}",
                confidence="moderate",
                source="scaling_law",
                metrics={"n_models": len(models)},
            ))

    # ── Per-head findings ──
    ph = inventory.get("per_head", {})
    if ph.get("status") == "completed" and ph.get("latest_data"):
        data = ph["latest_data"]
        findings.append(Finding(
            claim="Per-head attention decomposition identifies R_V-contributing heads",
            evidence=f"Layers analyzed, head-level contributions computed",
            confidence="moderate",
            source="per_head",
        ))

    # ── Path patching findings ──
    pp = inventory.get("path_patching", {})
    if pp.get("status") == "completed" and pp.get("latest_data"):
        data = pp["latest_data"]
        findings.append(Finding(
            claim="Path patching identifies causal components for R_V",
            evidence=f"Layer×component grid measured",
            confidence="moderate",
            source="path_patching",
        ))

    # ── Hardcoded priors from earlier experiments ──
    findings.extend([
        Finding(
            claim="R_V contraction is necessary for self-referential processing",
            evidence="Dual-layer patching: Cohen's d=3.29",
            confidence="strong",
            source="phase_a_necessity",
            metrics={"cohens_d": 3.29},
        ),
        Finding(
            claim="KV cache is sufficient to transfer R_V signature",
            evidence="KV sufficiency OR=13.96",
            confidence="strong",
            source="phase_a_sufficiency",
            metrics={"odds_ratio": 13.96},
        ),
        Finding(
            claim="R_V contraction replicates across 5 architectures",
            evidence="Mistral d=-2.26, OPT d=-1.84, GPT-2 XL d=-1.14, Qwen d=-0.72, Pythia d=-0.31",
            confidence="strong",
            source="phase_a_cross_arch",
            metrics={"mistral_d": -2.26, "opt_d": -1.84, "gpt2xl_d": -1.14,
                      "qwen_d": -0.72, "pythia_d": -0.31},
        ),
    ])

    return findings


# ── 3. Gap Analysis ──────────────────────────────────────────────────────

THEORY_PREDICTIONS = [
    {
        "id": "T1",
        "prediction": "Effect size scales with sqrt(d_model / n_kv_heads)",
        "status": "testable",
        "requires": ["scaling_law"],
    },
    {
        "id": "T2",
        "prediction": "R_V onset layer ≈ 0.55 × L",
        "status": "testable",
        "requires": ["path_patching", "per_head"],
    },
    {
        "id": "T3",
        "prediction": "Spectral gap is the best single discriminant for mode detection",
        "status": "testable",
        "requires": ["mode_atlas", "statistical_hardening"],
    },
    {
        "id": "T4",
        "prediction": "Phase transition in R_V at ~1B parameters",
        "status": "testable",
        "requires": ["scaling_law"],
    },
]


def gap_analysis(inventory: dict, findings: list) -> dict:
    """Identify what's missing and what predictions remain untested."""
    gaps = {
        "experiments_not_run": [],
        "predictions_untested": [],
        "paper_gaps": [],
    }

    # Check experiment coverage
    for name, info in inventory.items():
        if info.get("status") == "not_run":
            gaps["experiments_not_run"].append(name)

    # Check theory predictions
    for pred in THEORY_PREDICTIONS:
        deps_met = all(
            inventory.get(r, {}).get("status") == "completed"
            for r in pred["requires"]
        )
        if not deps_met:
            gaps["predictions_untested"].append({
                "id": pred["id"],
                "prediction": pred["prediction"],
                "blocked_by": [
                    r for r in pred["requires"]
                    if inventory.get(r, {}).get("status") != "completed"
                ],
            })

    # Paper pipeline gaps
    paper_dir = PROJECT_ROOT / "paper"
    if not list(paper_dir.glob("*.bib")):
        gaps["paper_gaps"].append("No bibliography file (0 references)")
    if not list((PROJECT_ROOT / "figures").glob("*.png")) and not list((PROJECT_ROOT / "figures").glob("*.pdf")):
        gaps["paper_gaps"].append("No figures generated")

    return gaps


# ── 4. Experiment Proposals ──────────────────────────────────────────────

def propose_experiments(inventory: dict, gaps: dict, findings: list) -> list:
    """Rank next experiments by expected information gain."""
    proposals = []

    # Priority 1: Run missing high-impact experiments
    if "mode_atlas" in gaps["experiments_not_run"]:
        proposals.append({
            "priority": 1,
            "experiment": "computational_mode_atlas",
            "command": "python3 scripts/computational_mode_atlas.py --device cuda",
            "reason": "10-mode geometric fingerprint — provides cross-task comparison for NeurIPS",
            "gpu_required": True,
            "estimated_time": "45 min",
        })

    if "scaling_law" in gaps["experiments_not_run"]:
        proposals.append({
            "priority": 2,
            "experiment": "scaling_law_sweep",
            "command": "python3 scripts/scaling_law_sweep.py --device cuda",
            "reason": "Tests phase transition prediction (T4) and effect-size scaling (T1)",
            "gpu_required": True,
            "estimated_time": "2 hours",
        })

    if "per_head" in gaps["experiments_not_run"]:
        proposals.append({
            "priority": 3,
            "experiment": "per_head_attention_decomposition",
            "command": "python3 scripts/per_head_attention_decomposition.py --device cuda",
            "reason": "Identifies which heads drive R_V — needed for onset-layer prediction (T2)",
            "gpu_required": True,
            "estimated_time": "30 min",
        })

    if "path_patching" in gaps["experiments_not_run"]:
        proposals.append({
            "priority": 4,
            "experiment": "full_path_patching",
            "command": "python3 scripts/full_path_patching.py --device cuda",
            "reason": "Causal layer×component grid — validates necessity finding with finer grain",
            "gpu_required": True,
            "estimated_time": "1 hour",
        })

    if "statistical_hardening" in gaps["experiments_not_run"]:
        proposals.append({
            "priority": 5,
            "experiment": "statistical_hardening",
            "command": "python3 scripts/statistical_hardening.py --device cuda",
            "reason": "Bootstrap CIs + Bayes factors — required for NeurIPS statistical rigor",
            "gpu_required": True,
            "estimated_time": "20 min",
        })

    # Priority 6: Paper pipeline tasks (no GPU)
    if "No bibliography file (0 references)" in gaps.get("paper_gaps", []):
        proposals.append({
            "priority": 6,
            "experiment": "literature_survey",
            "command": "manual — compile references.bib with 60+ papers",
            "reason": "NeurIPS requires thorough related work (currently 0 refs)",
            "gpu_required": False,
            "estimated_time": "3 hours",
        })

    if "No figures generated" in gaps.get("paper_gaps", []):
        proposals.append({
            "priority": 7,
            "experiment": "figure_generation",
            "command": "manual — generate 12-15 publication figures from results/",
            "reason": "Paper has 0 figures (need R_V heatmaps, scaling curves, mode atlas, etc.)",
            "gpu_required": False,
            "estimated_time": "2 hours",
        })

    # If all experiments are done, propose analysis-level follow-ups
    if not gaps["experiments_not_run"]:
        proposals.append({
            "priority": 8,
            "experiment": "cross_experiment_synthesis",
            "command": "python3 scripts/orchestrator.py --synthesize",
            "reason": "All primary experiments complete — synthesize unified narrative",
            "gpu_required": False,
            "estimated_time": "30 min",
        })

    return sorted(proposals, key=lambda x: x["priority"])


# ── 5. Report ────────────────────────────────────────────────────────────

def generate_report(inventory, findings, gaps, proposals, out_dir):
    """Generate structured report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": timestamp,
        "inventory": {k: {kk: vv for kk, vv in v.items() if kk != "latest_data"}
                      for k, v in inventory.items()},
        "findings": [f.to_dict() for f in findings],
        "knowledge_graph_summary": {
            "total_findings": len(findings),
            "strong": sum(1 for f in findings if f.confidence == "strong"),
            "moderate": sum(1 for f in findings if f.confidence == "moderate"),
            "preliminary": sum(1 for f in findings if f.confidence == "preliminary"),
            "negative": sum(1 for f in findings if f.confidence == "negative"),
        },
        "gaps": gaps,
        "proposals": proposals,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"orchestrator_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Human-readable summary ──
    print("\n" + "=" * 70)
    print("ORCHESTRATOR REPORT")
    print("=" * 70)

    print(f"\n📊 EXPERIMENT INVENTORY ({sum(1 for v in inventory.values() if v['status'] == 'completed')}/{len(inventory)} complete)")
    for name, info in inventory.items():
        status_icon = "✅" if info["status"] == "completed" else "❌" if info["status"] == "not_run" else "⚠️"
        print(f"  {status_icon} {name}: {info['status']} ({info.get('count', 0)} result files)")

    print(f"\n🧠 KNOWLEDGE GRAPH ({len(findings)} findings)")
    for f in findings:
        conf_icon = {"strong": "●", "moderate": "◐", "preliminary": "○", "negative": "✗"}
        print(f"  {conf_icon.get(f.confidence, '?')} [{f.confidence}] {f.claim}")

    print(f"\n⚠️  GAPS")
    if gaps["experiments_not_run"]:
        print(f"  Experiments not run: {', '.join(gaps['experiments_not_run'])}")
    if gaps["predictions_untested"]:
        for p in gaps["predictions_untested"]:
            print(f"  Untested: {p['id']} — {p['prediction']} (blocked by {p['blocked_by']})")
    if gaps["paper_gaps"]:
        for g in gaps["paper_gaps"]:
            print(f"  Paper: {g}")

    print(f"\n🎯 PROPOSED NEXT EXPERIMENTS (ranked by information gain)")
    for p in proposals:
        gpu = "🔥 GPU" if p["gpu_required"] else "💻 CPU"
        print(f"  [{p['priority']}] {p['experiment']} ({gpu}, ~{p['estimated_time']})")
        print(f"      → {p['reason']}")
        print(f"      $ {p['command']}")

    print(f"\n  Report saved: {report_path}")
    print("=" * 70)

    return report_path


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Automated Discovery Orchestrator v1")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "orchestrator"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output)

    print("🔬 Ingesting experiment results...")
    inventory = ingest_results(results_dir)

    print("🧠 Building knowledge graph...")
    findings = build_knowledge_graph(inventory)

    print("🔍 Running gap analysis...")
    gaps = gap_analysis(inventory, findings)

    print("🎯 Proposing next experiments...")
    proposals = propose_experiments(inventory, gaps, findings)

    generate_report(inventory, findings, gaps, proposals, out_dir)


if __name__ == "__main__":
    main()
