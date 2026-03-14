#!/usr/bin/env python3
"""
LINEAR PROBE FOR SELF-REFERENTIAL DIRECTION (E4.1 + E4.2)
===========================================================

1. Train linear probe on residual stream to classify recursive vs baseline
2. Extract the learned direction vector (the "self-referential direction")
3. Measure probe accuracy at each layer (where does self-ref become separable?)
4. Compute alignment between self-ref direction and top V-projection singular directions
5. Concept erasure: project out self-ref direction, re-measure R_V

Connects to Marks & Tegmark 2024 (geometry of truth).

Output: results/linear_probe/probe_analysis_<timestamp>.json

Usage:
    python3 scripts/linear_probe_selfref.py --device cuda
    python3 scripts/linear_probe_selfref.py --device cuda --n-per-group 10
    python3 scripts/linear_probe_selfref.py --device cuda --recursive-groups L3_deeper,L4_full,L5_refined --baseline-groups baseline_factual,baseline_math,baseline_creative
"""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.loader import PromptLoader
from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_hidden_states, capture_v_projection
from geometric_lens.metrics import participation_ratio


# ── Prompt bank (loaded from prompts/bank.json) ──────────────────────────────
_loader = PromptLoader()
DEFAULT_RECURSIVE_GROUPS = ["L3_deeper", "L4_full", "L5_refined"]
DEFAULT_BASELINE_GROUPS = ["baseline_factual", "baseline_math", "baseline_creative"]


def parse_groups(raw: str, default: list[str]) -> list[str]:
    groups = [item.strip() for item in raw.split(",") if item.strip()]
    return groups or list(default)


def collect_prompts_by_group(
    loader: PromptLoader,
    group_names: list[str],
    n_per_group: int,
) -> tuple[list[str], dict[str, int]]:
    texts: list[str] = []
    counts: dict[str, int] = {}
    for group_name in group_names:
        prompts = loader.get_by_group(group_name)
        take = prompts[:n_per_group]
        texts.extend(take)
        counts[group_name] = len(take)
    return texts, counts


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def extract_representations(model, tokenizer, texts, layer_idx, device, window=16):
    """Extract mean-pooled residual stream representations at a layer.

    Returns:
        (N, hidden_dim) numpy array
    """
    reps = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with capture_hidden_states(model, layer_idx) as sh:
            with torch.no_grad():
                model(**enc)
            hidden = sh.get("hidden")

        if hidden is None:
            reps.append(None)
            continue

        if hidden.dim() == 3:
            hidden = hidden[0]  # (seq, dim)

        # Mean pool last W tokens
        W = min(window, hidden.shape[0])
        rep = hidden[-W:, :].float().mean(dim=0).cpu().numpy()
        reps.append(rep)

    # Filter None
    valid = [r for r in reps if r is not None]
    return np.stack(valid) if valid else np.array([])


def train_linear_probe(X, y, n_folds=5):
    """Train a linear probe with cross-validation.

    Returns:
        (mean_accuracy, mean_auc, trained_weights, trained_bias)
    """
    if len(X) < 10:
        return float("nan"), float("nan"), None, None

    dim = X.shape[1]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accs, aucs = [], []
    all_weights = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = torch.tensor(X[train_idx], dtype=torch.float32), \
                          torch.tensor(X[test_idx], dtype=torch.float32)
        y_train, y_test = torch.tensor(y[train_idx], dtype=torch.float32), \
                          torch.tensor(y[test_idx], dtype=torch.float32)

        # Simple logistic regression
        probe = nn.Linear(dim, 1)
        optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Train
        probe.train()
        for epoch in range(200):
            optimizer.zero_grad()
            logits = probe(X_train).squeeze(-1)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            logits_test = probe(X_test).squeeze(-1)
            preds = (torch.sigmoid(logits_test) > 0.5).numpy().astype(int)
            probs = torch.sigmoid(logits_test).numpy()

        accs.append(accuracy_score(y_test.numpy(), preds))
        try:
            aucs.append(roc_auc_score(y_test.numpy(), probs))
        except ValueError:
            aucs.append(float("nan"))

        all_weights.append(probe.weight.detach().numpy().flatten())

    # Average direction across folds
    mean_weights = np.mean(all_weights, axis=0)
    mean_weights = mean_weights / (np.linalg.norm(mean_weights) + 1e-10)

    return float(np.mean(accs)), float(np.nanmean(aucs)), mean_weights, None


def compute_direction_alignment(self_ref_direction, v_tensor, window=16):
    """Compute cosine alignment between self-ref direction and top V-projection singular directions.

    Args:
        self_ref_direction: (hidden_dim,) numpy array — the learned probe direction
        v_tensor: (seq, v_dim) torch tensor — V-projection output

    Returns:
        Dict with alignment stats for top-5 singular directions.
    """
    if v_tensor is None or self_ref_direction is None:
        return {"alignment_top1": float("nan")}

    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]

    T, D = v_tensor.shape
    W = min(window, T)
    v_cpu = v_tensor[-W:, :].cpu().double()

    try:
        U, S, Vt = torch.linalg.svd(v_cpu.T, full_matrices=False)
        # U columns are right singular vectors in the hidden dim space
        # Each column is a principal direction
        U_np = U.numpy()

        # Self-ref direction may be larger than V dim — truncate
        d = min(len(self_ref_direction), D)
        sr_dir = self_ref_direction[:d]
        sr_dir = sr_dir / (np.linalg.norm(sr_dir) + 1e-10)

        alignments = {}
        for k in range(min(5, U_np.shape[1])):
            sv_dir = U_np[:d, k]
            sv_dir = sv_dir / (np.linalg.norm(sv_dir) + 1e-10)
            cos = float(np.abs(np.dot(sr_dir, sv_dir)))
            alignments[f"alignment_sv{k+1}"] = cos

        return alignments
    except Exception:
        return {"alignment_top1": float("nan")}


def run_concept_erasure(probe_obj, texts, self_ref_direction, layer_idx, device, window=16):
    """Measure R_V after projecting out the self-ref direction from V-projections.

    This tests whether the contraction is carried by a specific linear subspace.
    """
    model = probe_obj.model
    tokenizer = probe_obj.tokenizer

    rvs_before = []
    rvs_after = []

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        # Measure R_V before erasure
        result = probe_obj.measure(text, metrics=["rv"])
        rvs_before.append(result.rv)

        # Measure V-projection with direction erased
        with capture_v_projection(model, probe_obj.late_layer) as sl:
            with torch.no_grad():
                model(**enc)
            v_late = sl.get("v")

        with capture_v_projection(model, probe_obj.early_layer) as se:
            with torch.no_grad():
                model(**enc)
            v_early = se.get("v")

        if v_late is None or v_early is None:
            rvs_after.append(float("nan"))
            continue

        # Project out self-ref direction from V-projection
        if v_late.dim() == 3:
            v_late = v_late[0]
        if v_early.dim() == 3:
            v_early = v_early[0]

        D = v_late.shape[1]
        d = min(len(self_ref_direction), D)
        sr_dir = torch.tensor(self_ref_direction[:d], dtype=v_late.dtype, device=v_late.device)
        sr_dir = sr_dir / (sr_dir.norm() + 1e-10)

        # Projection matrix: I - d*d^T
        proj = torch.eye(d, device=v_late.device, dtype=v_late.dtype) - sr_dir.unsqueeze(1) @ sr_dir.unsqueeze(0)

        v_late_erased = v_late[:, :d] @ proj.T
        v_early_erased = v_early[:, :d] @ proj.T

        # Compute PR on erased projections
        from geometric_lens.metrics import compute_rv
        rv_erased = compute_rv(v_early_erased.unsqueeze(0), v_late_erased.unsqueeze(0), window)
        rvs_after.append(rv_erased)

    return rvs_before, rvs_after


def run_linear_probe(args):
    """Run linear probe experiment."""
    print("=" * 70)
    print("LINEAR PROBE FOR SELF-REFERENTIAL DIRECTION (E4.1 + E4.2)")
    print("=" * 70)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        attn_implementation="eager",
    )
    model = probe.model
    tokenizer = probe.tokenizer
    spec = probe.spec

    print(f"Layers={spec.num_layers}, early={probe.early_layer}, late={probe.late_layer}")

    recursive_groups = parse_groups(args.recursive_groups, DEFAULT_RECURSIVE_GROUPS)
    baseline_groups = parse_groups(args.baseline_groups, DEFAULT_BASELINE_GROUPS)
    rec_prompts, rec_counts = collect_prompts_by_group(
        _loader, recursive_groups, args.n_per_group
    )
    bas_prompts, bas_counts = collect_prompts_by_group(
        _loader, baseline_groups, args.n_per_group
    )
    all_texts = rec_prompts + bas_prompts
    labels = np.array([1] * len(rec_prompts) + [0] * len(bas_prompts))

    # ── Step 1: Layer-by-layer probe accuracy ──
    print("\n  Step 1: Probe accuracy at each layer...")

    # Sample layers (every 2nd layer for speed, or all if few layers)
    step_size = max(1, spec.num_layers // 16)
    probe_layers = list(range(0, spec.num_layers, step_size))
    if probe.late_layer not in probe_layers:
        probe_layers.append(probe.late_layer)
    if probe.early_layer not in probe_layers:
        probe_layers.append(probe.early_layer)
    probe_layers = sorted(set(probe_layers))

    layer_results = {}
    best_layer = None
    best_acc = 0
    best_direction = None

    for layer_idx in probe_layers:
        print(f"\n    Layer {layer_idx}...")
        X = extract_representations(model, tokenizer, all_texts, layer_idx, args.device)
        if X.shape[0] != len(labels):
            print(f"      Skipped (got {X.shape[0]} reps, expected {len(labels)})")
            continue

        acc, auc, direction, _ = train_linear_probe(X, labels)
        layer_results[layer_idx] = {
            "accuracy": acc,
            "auc": auc,
            "has_direction": direction is not None,
        }

        marker = " <<<" if acc > 0.85 else ""
        print(f"      Acc={acc:.3f}, AUC={auc:.3f}{marker}")

        if acc > best_acc:
            best_acc = acc
            best_layer = layer_idx
            best_direction = direction

    print(f"\n  Best probe: Layer {best_layer}, Acc={best_acc:.3f}")

    # ── Step 2: Direction alignment with V-projection ──
    print("\n  Step 2: Self-ref direction alignment with V-projection...")

    alignment_results = {"recursive": [], "baseline": []}
    for condition, prompts in [("recursive", rec_prompts), ("baseline", bas_prompts)]:
        for text in prompts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(args.device)
            with capture_v_projection(model, probe.late_layer) as sv:
                with torch.no_grad():
                    model(**enc)
                v_tensor = sv.get("v")

            alignments = compute_direction_alignment(best_direction, v_tensor)
            alignment_results[condition].append(alignments)

    # Average alignment per condition
    for cond in ["recursive", "baseline"]:
        vals = alignment_results[cond]
        for k in ["alignment_sv1", "alignment_sv2", "alignment_sv3"]:
            cond_vals = [v.get(k, float("nan")) for v in vals if not np.isnan(v.get(k, float("nan")))]
            if cond_vals:
                print(f"    {cond} {k}: {np.mean(cond_vals):.3f} ± {np.std(cond_vals):.3f}")

    # ── Step 3: Concept erasure (E4.2) ──
    print("\n  Step 3: Concept erasure...")

    rec_before, rec_after = run_concept_erasure(
        probe, rec_prompts, best_direction, best_layer, args.device
    )
    bas_before, bas_after = run_concept_erasure(
        probe, bas_prompts, best_direction, best_layer, args.device
    )

    rec_before_valid = [v for v in rec_before if not np.isnan(v)]
    rec_after_valid = [v for v in rec_after if not np.isnan(v)]
    bas_before_valid = [v for v in bas_before if not np.isnan(v)]
    bas_after_valid = [v for v in bas_after if not np.isnan(v)]

    d_before = cohens_d(rec_before_valid, bas_before_valid) if rec_before_valid and bas_before_valid else float("nan")
    d_after = cohens_d(rec_after_valid, bas_after_valid) if rec_after_valid and bas_after_valid else float("nan")

    print(f"\n    Before erasure: d={d_before:.3f}")
    print(f"    After erasure:  d={d_after:.3f}")
    if not np.isnan(d_before) and not np.isnan(d_after) and abs(d_before) > 0.1:
        reduction = 1 - abs(d_after) / abs(d_before)
        print(f"    Reduction: {reduction*100:.1f}%")
        if reduction > 0.5:
            print("    → Contraction is largely carried by a single linear direction")
        else:
            print("    → Contraction is distributed / nonlinear")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E4.1_E4.2_linear_probe",
        "model": args.model,
        "recursive_groups": recursive_groups,
        "baseline_groups": baseline_groups,
        "recursive_prompt_counts": rec_counts,
        "baseline_prompt_counts": bas_counts,
        "n_recursive": len(rec_prompts),
        "n_baseline": len(bas_prompts),
        "layer_probe_results": {str(k): v for k, v in layer_results.items()},
        "best_layer": best_layer,
        "best_accuracy": best_acc,
        "alignment_results": {
            cond: [
                {k: float(v) for k, v in a.items()}
                for a in alignment_results[cond]
            ]
            for cond in ["recursive", "baseline"]
        },
        "concept_erasure": {
            "d_before": d_before,
            "d_after": d_after,
            "rv_recursive_before_mean": float(np.mean(rec_before_valid)) if rec_before_valid else float("nan"),
            "rv_recursive_after_mean": float(np.mean(rec_after_valid)) if rec_after_valid else float("nan"),
            "rv_baseline_before_mean": float(np.mean(bas_before_valid)) if bas_before_valid else float("nan"),
            "rv_baseline_after_mean": float(np.mean(bas_after_valid)) if bas_after_valid else float("nan"),
        },
        "self_ref_direction": best_direction.tolist() if best_direction is not None else None,
    }

    summary_path = out_dir / f"probe_analysis_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Probe (E4.1 + E4.2)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--recursive-groups",
        default=",".join(DEFAULT_RECURSIVE_GROUPS),
        help="Comma-separated recursive prompt groups",
    )
    parser.add_argument(
        "--baseline-groups",
        default=",".join(DEFAULT_BASELINE_GROUPS),
        help="Comma-separated baseline prompt groups",
    )
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=10,
        help="Prompts to take from each listed group",
    )
    parser.add_argument(
        "--output-dir",
        default="results/linear_probe",
        help="Directory for probe summaries",
    )
    args = parser.parse_args()
    run_linear_probe(args)
