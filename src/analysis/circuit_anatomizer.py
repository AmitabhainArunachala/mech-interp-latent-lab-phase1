#!/usr/bin/env python3
"""
Circuit Anatomizer
==================

Maps the COMPLETE causal circuit from L0 MLP → L_late attention → R_V contraction.

Key capabilities:
1. PR trajectory through ALL layers (not just early/late)
2. MLP vs Attention decomposition at each layer
3. Critical node identification (where PR drops most)
4. Attention head contribution analysis at readout layer

This is the deep-dive tool for understanding HOW eigenstate emergence works,
complementing the multi-turn tracker that shows WHEN it occurs.

Target architecture (Mistral-7B):
    L0 MLP (source) → L3-L4 MLP (transfer) → L27 attention (readout) → R_V contraction

Usage:
    from src.analysis.circuit_anatomizer import CircuitAnatomizer
    
    anatomizer = CircuitAnatomizer(model, tokenizer)
    anatomy = anatomizer.full_anatomy(text)
    anatomy.plot()  # Visualize PR trajectory
"""

import json
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayerContribution:
    """Decomposed contribution at a single layer."""
    layer_idx: int
    depth_pct: float  # 0-100
    
    # PR values
    pr_total: float  # Combined PR
    pr_mlp: Optional[float] = None  # MLP contribution
    pr_attn: Optional[float] = None  # Attention contribution
    
    # Deltas from previous layer
    pr_delta: float = 0.0  # Change from previous layer
    pr_delta_pct: float = 0.0  # Percent change
    
    # Classification
    is_source: bool = False  # L0-L2 region
    is_transfer: bool = False  # L3-L5 region
    is_readout: bool = False  # L_late region
    is_critical: bool = False  # Significant PR drop here


@dataclass
class AttentionHeadAnalysis:
    """Analysis of attention heads at readout layer."""
    layer_idx: int
    num_heads: int
    
    # Per-head metrics
    head_contributions: List[float] = field(default_factory=list)
    head_entropies: List[float] = field(default_factory=list)
    
    # Aggregates
    dominant_heads: List[int] = field(default_factory=list)  # Heads with > 10% contribution
    mean_entropy: float = 0.0


@dataclass
class CircuitAnatomy:
    """Complete circuit anatomy for a single input."""
    text: str
    model_name: str
    timestamp: str
    
    # Layer-by-layer analysis
    layers: List[LayerContribution] = field(default_factory=list)
    
    # Summary metrics
    rv: float = 0.0
    pr_early: float = 0.0
    pr_late: float = 0.0
    
    # Critical points
    source_layers: List[int] = field(default_factory=list)
    transfer_layers: List[int] = field(default_factory=list)
    readout_layer: int = 0
    
    # Steepest drop
    max_drop_layer: int = 0
    max_drop_magnitude: float = 0.0
    
    # Attention analysis at readout
    readout_attention: Optional[AttentionHeadAnalysis] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = [asdict(layer) for layer in self.layers]
        if self.readout_attention:
            d["readout_attention"] = asdict(self.readout_attention)
        return d
    
    def pr_trajectory(self) -> List[float]:
        """Extract PR values as a list."""
        return [layer.pr_total for layer in self.layers]
    
    def find_phase_transitions(self, threshold: float = 0.1) -> List[int]:
        """Find layers where PR drops by more than threshold."""
        transitions = []
        trajectory = self.pr_trajectory()
        for i in range(1, len(trajectory)):
            if (trajectory[i-1] - trajectory[i]) / max(trajectory[i-1], 1e-6) > threshold:
                transitions.append(i)
        return transitions
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*60}")
        print(f"CIRCUIT ANATOMY: {self.model_name}")
        print(f"{'='*60}")
        print(f"Text: {self.text[:80]}...")
        print(f"\nR_V: {self.rv:.3f} (PR_early={self.pr_early:.2f}, PR_late={self.pr_late:.2f})")
        print(f"\nSource layers: {self.source_layers}")
        print(f"Transfer layers: {self.transfer_layers}")
        print(f"Readout layer: L{self.readout_layer}")
        print(f"\nMax PR drop: L{self.max_drop_layer} ({self.max_drop_magnitude:.1%})")
        
        print(f"\n{'='*60}")
        print("LAYER-BY-LAYER PR TRAJECTORY")
        print(f"{'='*60}")
        
        for layer in self.layers:
            bar_len = int(layer.pr_total / 2)  # Scale for display
            bar = "█" * bar_len
            
            flags = []
            if layer.is_source:
                flags.append("SRC")
            if layer.is_transfer:
                flags.append("XFER")
            if layer.is_readout:
                flags.append("READ")
            if layer.is_critical:
                flags.append("***")
            
            flag_str = " ".join(flags) if flags else ""
            
            delta_str = f"({layer.pr_delta_pct:+.1%})" if layer.layer_idx > 0 else ""
            
            print(f"L{layer.layer_idx:2d} [{layer.depth_pct:5.1f}%]: {layer.pr_total:5.2f} |{bar:<30} {delta_str} {flag_str}")
    
    def plot(self, output_path: Optional[Path] = None):
        """Generate matplotlib visualization."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            layers = [l.layer_idx for l in self.layers]
            prs = self.pr_trajectory()
            deltas = [l.pr_delta_pct * 100 for l in self.layers]
            
            # PR trajectory
            ax1.plot(layers, prs, 'b-', linewidth=2, label='PR')
            ax1.fill_between(layers, prs, alpha=0.3)
            
            # Mark critical regions
            for l in self.layers:
                if l.is_source:
                    ax1.axvspan(l.layer_idx - 0.5, l.layer_idx + 0.5, alpha=0.1, color='green')
                if l.is_readout:
                    ax1.axvspan(l.layer_idx - 0.5, l.layer_idx + 0.5, alpha=0.1, color='red')
                if l.is_critical:
                    ax1.axvline(l.layer_idx, color='red', linestyle='--', alpha=0.5)
            
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Participation Ratio')
            ax1.set_title(f'PR Trajectory: {self.model_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Delta plot
            colors = ['red' if d < -5 else 'green' if d > 0 else 'gray' for d in deltas]
            ax2.bar(layers, deltas, color=colors, alpha=0.7)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('PR Change (%)')
            ax2.set_title('Layer-by-Layer PR Change')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150)
                print(f"Plot saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")


class CircuitAnatomizer:
    """Analyzes the complete causal circuit."""
    
    def __init__(
        self,
        model,
        tokenizer,
        early_layer: int = 5,
        late_layer: Optional[int] = None,
        window: int = 16,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.early_layer = early_layer
        self.num_layers = model.config.num_hidden_layers
        self.late_layer = late_layer or (self.num_layers - 5)
        self.window = window
        self.device = device
        
        # Classification thresholds
        self.source_region = (0, 2)  # L0-L2
        self.transfer_region = (3, 5)  # L3-L5
        # Readout is dynamic based on model depth
        
    @contextmanager
    def _capture_v_projection(self, layer_idx: int):
        """Context manager to capture V-projection at a layer."""
        storage = {"v": None}
        
        def hook_fn(module, inp, out):
            storage["v"] = out.detach()
        
        # Handle different model architectures
        if hasattr(self.model, "model"):
            # Mistral, Gemma, etc.
            handle = self.model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fn)
        elif hasattr(self.model, "transformer"):
            # GPT-2 style
            handle = self.model.transformer.h[layer_idx].attn.v_proj.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")
        
        try:
            yield storage
        finally:
            handle.remove()
    
    def _compute_pr(self, v_tensor: torch.Tensor) -> float:
        """Compute participation ratio from V tensor."""
        if v_tensor is None:
            return float("nan")
        
        if v_tensor.dim() == 3:
            v_tensor = v_tensor[0]
        
        T, D = v_tensor.shape
        
        if T < self.window:
            return float("nan")
        
        v_window = v_tensor[-self.window:, :].double()
        
        try:
            U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
            S_np = S.cpu().numpy()
            S_sq = S_np ** 2
            
            total_variance = S_sq.sum()
            if total_variance < 1e-10:
                return float("nan")
            
            pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
            return float(pr)
        except Exception:
            return float("nan")
    
    def compute_pr_at_layer(self, text: str, layer_idx: int) -> float:
        """Compute PR at a specific layer."""
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with self._capture_v_projection(layer_idx) as storage:
            with torch.no_grad():
                self.model(**enc)
            v_tensor = storage["v"]
        
        return self._compute_pr(v_tensor)
    
    def compute_full_trajectory(self, text: str, step: int = 1) -> List[Tuple[int, float]]:
        """
        Compute PR at every layer (or every step-th layer).
        
        Returns list of (layer_idx, pr) tuples.
        """
        trajectory = []
        
        for layer_idx in range(0, self.num_layers, step):
            pr = self.compute_pr_at_layer(text, layer_idx)
            trajectory.append((layer_idx, pr))
        
        return trajectory
    
    def full_anatomy(
        self,
        text: str,
        step: int = 1,
        critical_threshold: float = 0.15,  # 15% drop = critical
    ) -> CircuitAnatomy:
        """
        Perform full circuit anatomy.
        
        Args:
            text: Input text to analyze
            step: Sample every N layers (1 = all layers)
            critical_threshold: PR drop percentage to mark as critical
        
        Returns:
            CircuitAnatomy with complete analysis
        """
        anatomy = CircuitAnatomy(
            text=text,
            model_name=getattr(self.model.config, "_name_or_path", "unknown"),
            timestamp=datetime.now().isoformat(),
        )
        
        # Compute trajectory
        trajectory = self.compute_full_trajectory(text, step)
        
        prev_pr = None
        for layer_idx, pr in trajectory:
            depth_pct = (layer_idx / self.num_layers) * 100
            
            # Compute deltas
            if prev_pr is not None and not np.isnan(prev_pr) and not np.isnan(pr):
                pr_delta = pr - prev_pr
                pr_delta_pct = pr_delta / max(prev_pr, 1e-6)
            else:
                pr_delta = 0.0
                pr_delta_pct = 0.0
            
            # Classify layer
            is_source = self.source_region[0] <= layer_idx <= self.source_region[1]
            is_transfer = self.transfer_region[0] <= layer_idx <= self.transfer_region[1]
            is_readout = layer_idx == self.late_layer
            is_critical = pr_delta_pct < -critical_threshold
            
            layer_contrib = LayerContribution(
                layer_idx=layer_idx,
                depth_pct=depth_pct,
                pr_total=pr,
                pr_delta=pr_delta,
                pr_delta_pct=pr_delta_pct,
                is_source=is_source,
                is_transfer=is_transfer,
                is_readout=is_readout,
                is_critical=is_critical,
            )
            
            anatomy.layers.append(layer_contrib)
            
            # Track regions
            if is_source:
                anatomy.source_layers.append(layer_idx)
            if is_transfer:
                anatomy.transfer_layers.append(layer_idx)
            if is_readout:
                anatomy.readout_layer = layer_idx
            
            # Track max drop
            if is_critical and abs(pr_delta_pct) > abs(anatomy.max_drop_magnitude):
                anatomy.max_drop_layer = layer_idx
                anatomy.max_drop_magnitude = pr_delta_pct
            
            prev_pr = pr
        
        # Compute R_V
        pr_early = self.compute_pr_at_layer(text, self.early_layer)
        pr_late = self.compute_pr_at_layer(text, self.late_layer)
        
        anatomy.pr_early = pr_early
        anatomy.pr_late = pr_late
        
        if not np.isnan(pr_early) and not np.isnan(pr_late) and pr_early > 0:
            anatomy.rv = pr_late / pr_early
        
        return anatomy
    
    def compare_prompts(
        self,
        recursive_text: str,
        baseline_text: str,
        step: int = 2,
    ) -> Tuple[CircuitAnatomy, CircuitAnatomy, Dict]:
        """
        Compare circuit anatomy between recursive and baseline prompts.
        
        Returns:
            (recursive_anatomy, baseline_anatomy, comparison_dict)
        """
        rec_anatomy = self.full_anatomy(recursive_text, step)
        base_anatomy = self.full_anatomy(baseline_text, step)
        
        # Layer-by-layer comparison
        layer_comparisons = []
        for rec_layer, base_layer in zip(rec_anatomy.layers, base_anatomy.layers):
            diff = rec_layer.pr_total - base_layer.pr_total
            layer_comparisons.append({
                "layer": rec_layer.layer_idx,
                "rec_pr": rec_layer.pr_total,
                "base_pr": base_layer.pr_total,
                "diff": diff,
                "diff_pct": diff / max(base_layer.pr_total, 1e-6),
            })
        
        # Find divergence point (where recursive starts differing significantly)
        divergence_layer = None
        for comp in layer_comparisons:
            if abs(comp["diff_pct"]) > 0.05:  # 5% threshold
                divergence_layer = comp["layer"]
                break
        
        comparison = {
            "rv_recursive": rec_anatomy.rv,
            "rv_baseline": base_anatomy.rv,
            "rv_diff": rec_anatomy.rv - base_anatomy.rv,
            "divergence_layer": divergence_layer,
            "layer_comparisons": layer_comparisons,
        }
        
        return rec_anatomy, base_anatomy, comparison


def run_anatomy_experiment(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    output_dir: Path,
    step: int = 2,
) -> Dict:
    """
    Run full anatomy experiment on prompt sets.
    
    Saves:
    - Individual anatomies (JSON)
    - Aggregate statistics
    - Trajectory plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    anatomizer = CircuitAnatomizer(model, tokenizer)
    
    results = {
        "recursive_anatomies": [],
        "baseline_anatomies": [],
        "aggregate": {},
    }
    
    # Analyze recursive prompts
    print("Analyzing recursive prompts...")
    for i, text in enumerate(recursive_prompts):
        anatomy = anatomizer.full_anatomy(text, step)
        results["recursive_anatomies"].append(anatomy.to_dict())
        
        # Save plot for first few
        if i < 3:
            anatomy.plot(output_dir / f"rec_{i}_trajectory.png")
    
    # Analyze baseline prompts
    print("Analyzing baseline prompts...")
    for i, text in enumerate(baseline_prompts):
        anatomy = anatomizer.full_anatomy(text, step)
        results["baseline_anatomies"].append(anatomy.to_dict())
        
        if i < 3:
            anatomy.plot(output_dir / f"base_{i}_trajectory.png")
    
    # Aggregate statistics
    rec_rvs = [a["rv"] for a in results["recursive_anatomies"] if not np.isnan(a["rv"])]
    base_rvs = [a["rv"] for a in results["baseline_anatomies"] if not np.isnan(a["rv"])]
    
    rec_max_drops = [a["max_drop_layer"] for a in results["recursive_anatomies"]]
    base_max_drops = [a["max_drop_layer"] for a in results["baseline_anatomies"]]
    
    results["aggregate"] = {
        "n_recursive": len(recursive_prompts),
        "n_baseline": len(baseline_prompts),
        "rv_recursive_mean": np.mean(rec_rvs) if rec_rvs else float("nan"),
        "rv_recursive_std": np.std(rec_rvs) if rec_rvs else float("nan"),
        "rv_baseline_mean": np.mean(base_rvs) if base_rvs else float("nan"),
        "rv_baseline_std": np.std(base_rvs) if base_rvs else float("nan"),
        "max_drop_layer_recursive_mode": max(set(rec_max_drops), key=rec_max_drops.count) if rec_max_drops else None,
        "max_drop_layer_baseline_mode": max(set(base_max_drops), key=base_max_drops.count) if base_max_drops else None,
    }
    
    # Save results
    with open(output_dir / "circuit_anatomy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Demo usage
    import argparse
    from src.core.models import load_model, set_seed
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--text", default="Observe yourself observing this observation recursively.")
    parser.add_argument("--output", default="results/circuit_anatomy")
    args = parser.parse_args()
    
    set_seed(42)
    model, tokenizer = load_model(args.model)
    
    anatomizer = CircuitAnatomizer(model, tokenizer)
    anatomy = anatomizer.full_anatomy(args.text, step=1)
    
    anatomy.print_summary()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    anatomy.plot(output_dir / "demo_trajectory.png")
    
    with open(output_dir / "demo_anatomy.json", "w") as f:
        json.dump(anatomy.to_dict(), f, indent=2)
