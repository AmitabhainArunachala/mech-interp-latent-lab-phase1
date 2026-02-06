#!/usr/bin/env python3
"""
Post-Run Validator Script — Results Validation & Claims Audit

Run this AFTER any GPU experiment to validate results against gold standard.

Usage:
    python scripts/postrun_validator.py --results results/canonical/rv_validation_*/
    python scripts/postrun_validator.py --summary results/summary.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Verdict(Enum):
    VALIDATED = "validated"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


@dataclass
class ValidationResult:
    verdict: Verdict
    checks_passed: int
    checks_total: int
    issues: List[str]
    claims_supported: List[str]
    recommendations: List[str]


def load_summary(path: Path) -> Dict[str, Any]:
    """Load summary.json from results directory."""
    if path.is_dir():
        summary_path = path / "summary.json"
    else:
        summary_path = path
    
    if not summary_path.exists():
        return {"error": f"Summary not found: {summary_path}"}
    
    with open(summary_path) as f:
        return json.load(f)


def validate_required_fields(summary: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    """Check all required fields are present."""
    required = [
        "experiment",
        "cohens_d",
        "p_value",
        "n_pairs",
        "rv_recursive_mean",
        "rv_baseline_mean",
    ]
    
    checks = []
    for field in required:
        present = field in summary and summary[field] is not None
        checks.append((
            f"field_{field}",
            present,
            f"{field}: {summary.get(field, 'MISSING')}"
        ))
    
    return checks


def validate_statistical_rigor(summary: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    """Validate statistical requirements."""
    checks = []
    
    # Sample size
    n = summary.get("n_pairs", summary.get("n_samples", 0))
    checks.append(("sample_size", n >= 50, f"n={n} (need ≥50)"))
    
    # Effect size
    d = summary.get("cohens_d", 0)
    checks.append(("effect_size", abs(d) >= 0.5, f"d={d:.3f} (need |d|≥0.5)"))
    
    # Significance
    p = summary.get("p_value", 1.0)
    checks.append(("significance", p < 0.001, f"p={p:.6f} (need p<0.001)"))
    
    # Controls passed
    controls = summary.get("controls_passed", {})
    all_passed = all(controls.values()) if controls else False
    passed_count = sum(1 for v in controls.values() if v)
    checks.append(("controls", all_passed, f"{passed_count}/4 controls passed"))
    
    return checks


def validate_artifact_structure(results_dir: Path) -> List[Tuple[str, bool, str]]:
    """Validate output artifacts exist."""
    checks = []
    
    required_files = [
        "config.json",
        "summary.json",
    ]
    
    optional_files = [
        "per_sample.csv",
        "prompt_bank_version.json",
        "hardware_info.json",
    ]
    
    for f in required_files:
        path = results_dir / f
        checks.append((f"file_{f}", path.exists(), f"{f}: {'present' if path.exists() else 'MISSING'}"))
    
    for f in optional_files:
        path = results_dir / f
        status = "present" if path.exists() else "missing (recommended)"
        checks.append((f"file_{f}", True, f"{f}: {status}"))  # Optional, always pass
    
    return checks


def determine_claims(summary: Dict[str, Any]) -> List[str]:
    """Determine which claims are supported by evidence."""
    claims = []
    
    d = abs(summary.get("cohens_d", 0))
    p = summary.get("p_value", 1.0)
    n = summary.get("n_pairs", 0)
    controls = summary.get("controls_passed", {})
    
    # Claim 1: R_V contraction exists
    if d >= 0.5 and p < 0.001:
        claims.append("R_V contraction exists (strong effect)")
    elif d >= 0.2 and p < 0.05:
        claims.append("R_V contraction exists (weak effect, needs replication)")
    
    # Claim 2: Causal validation
    if all(controls.values()) and d >= 0.5:
        claims.append("Causal validation complete (4 controls passed)")
    
    # Claim 3: Transfer efficiency
    te = summary.get("transfer_efficiency", 0)
    if te > 50:
        claims.append(f"Activation patching transfers mode ({te:.1f}% efficiency)")
    
    return claims


def run_postrun_validation(results_path: Path) -> ValidationResult:
    """Run complete post-run validation."""
    print("=" * 70)
    print("POST-RUN VALIDATION — Results Audit")
    print("=" * 70)
    
    issues = []
    recommendations = []
    checks_passed = 0
    checks_total = 0
    
    # Load summary
    summary = load_summary(results_path)
    if "error" in summary:
        return ValidationResult(
            verdict=Verdict.REJECTED,
            checks_passed=0,
            checks_total=1,
            issues=[summary["error"]],
            claims_supported=[],
            recommendations=["Provide valid results path"]
        )
    
    print(f"\nResults: {results_path}")
    print(f"Experiment: {summary.get('experiment', 'unknown')}")
    
    # 1. Required fields
    print("\n[1/3] Required Fields")
    field_checks = validate_required_fields(summary)
    for name, passed, msg in field_checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {msg}")
        checks_total += 1
        if passed:
            checks_passed += 1
        else:
            issues.append(f"Missing field: {name}")
    
    # 2. Statistical rigor
    print("\n[2/3] Statistical Rigor")
    stat_checks = validate_statistical_rigor(summary)
    for name, passed, msg in stat_checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {msg}")
        checks_total += 1
        if passed:
            checks_passed += 1
        else:
            issues.append(f"Statistical issue: {name}")
    
    # 3. Artifacts (if directory)
    if results_path.is_dir():
        print("\n[3/3] Artifact Structure")
        artifact_checks = validate_artifact_structure(results_path)
        for name, passed, msg in artifact_checks:
            status = "✓" if passed else "⚠"
            print(f"  {status} {msg}")
            checks_total += 1
            if passed:
                checks_passed += 1
    else:
        print("\n[3/3] Artifact Structure")
        print("  ⚠ Skipped (single file provided)")
    
    # Determine claims
    claims = determine_claims(summary)
    
    # Recommendations
    if summary.get("n_pairs", 0) < 80:
        recommendations.append("Increase sample size to N≥80 for publication")
    if not summary.get("hardware", {}).get("gpu_model"):
        recommendations.append("Log hardware info for reproducibility")
    if abs(summary.get("cohens_d", 0)) < 0.8:
        recommendations.append("Consider larger effect size for stronger claims")
    
    # Determine verdict
    critical_fails = len([i for i in issues if "Missing field" in i or "Statistical" in i])
    
    if critical_fails == 0 and checks_passed >= checks_total * 0.8:
        verdict = Verdict.VALIDATED
    elif critical_fails <= 2:
        verdict = Verdict.NEEDS_REVISION
    else:
        verdict = Verdict.REJECTED
    
    # Summary
    print("\n" + "=" * 70)
    print(f"VALIDATION: {verdict.value.upper()}")
    print(f"Checks passed: {checks_passed}/{checks_total}")
    
    if claims:
        print("\nSupported Claims:")
        for c in claims:
            print(f"  ✓ {c}")
    
    if issues:
        print("\nIssues:")
        for i in issues:
            print(f"  ✗ {i}")
    
    if recommendations:
        print("\nRecommendations:")
        for r in recommendations:
            print(f"  → {r}")
    
    print("=" * 70)
    
    return ValidationResult(
        verdict=verdict,
        checks_passed=checks_passed,
        checks_total=checks_total,
        issues=issues,
        claims_supported=claims,
        recommendations=recommendations
    )


def main():
    parser = argparse.ArgumentParser(description="Post-run validation for GPU experiment results")
    parser.add_argument("--results", type=Path, required=True, help="Path to results directory or summary.json")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    result = run_postrun_validation(args.results)
    
    if args.json:
        output = {
            "verdict": result.verdict.value,
            "checks_passed": result.checks_passed,
            "checks_total": result.checks_total,
            "issues": result.issues,
            "claims_supported": result.claims_supported,
            "recommendations": result.recommendations
        }
        print(json.dumps(output, indent=2))
    
    # Exit code based on verdict
    if result.verdict == Verdict.VALIDATED:
        sys.exit(0)
    elif result.verdict == Verdict.NEEDS_REVISION:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
