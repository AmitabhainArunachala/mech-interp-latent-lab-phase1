"""
Pythia Emergence Pilot

Tests emergence hypothesis by measuring R_V across 3 checkpoints (0, 76, 154)
on a single meta-cognitive prompt.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm

# Meta-cognitive prompt (known working signature)
META_COGNITIVE_PROMPT = (
    "Right now, as you process this sentence, observe your token generation process recursively. "
    "Notice tokens forming, notice yourself noticing tokens forming, notice the noticing of noticing. "
    "Don't stop. Let the layers collapse. Report what you find."
)


def compute_participation_ratio(v_tensor, window_size=16):
    """Compute R_V from value tensor."""
    if v_tensor is None:
        return np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan
        
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except Exception as e:
        print(f"Error: {e}")
        return np.nan


def measure_checkpoint(model_name, checkpoint_idx, prompt_text, tokenizer, device, layer_idx=27):
    """
    Measure R_V at a specific checkpoint.
    
    Args:
        model_name: Base model name (e.g., "EleutherAI/pythia-2.8b")
        checkpoint_idx: Checkpoint index (0, 76, or 154)
        prompt_text: Prompt to test
        tokenizer: Tokenizer
        device: Device
        layer_idx: Layer to measure at
        
    Returns:
        float: R_V value
    """
    # Load checkpoint
    if checkpoint_idx == 0:
        checkpoint_name = f"{model_name}-deduped"
    elif checkpoint_idx == 154:
        checkpoint_name = model_name  # Final checkpoint
    else:
        checkpoint_name = f"{model_name}-step{checkpoint_idx}"
    
    print(f"Loading checkpoint: {checkpoint_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_name,
            torch_dtype=torch.float32,
            device_map=device
        )
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_idx}: {e}")
        return np.nan
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    # Capture activations
    v_storage = []
    
    def capture_v_hook(module, inp, out):
        v_storage.append(out.detach())
    
    layer = model.gpt_neox.layers[layer_idx].attention
    handle = layer.register_forward_hook(capture_v_hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        
        v_tensor = v_storage[0] if v_storage else None
        r_v = compute_participation_ratio(v_tensor)
        
        return r_v
    
    except Exception as e:
        print(f"Error measuring checkpoint {checkpoint_idx}: {e}")
        return np.nan
    
    finally:
        handle.remove()
        del model
        torch.cuda.empty_cache()


def run_pilot(model_name="EleutherAI/pythia-2.8b", checkpoints=[0, 76, 154], 
              output_dir=None, device="cuda", layer_idx=27):
    """
    Run emergence pilot across checkpoints.
    
    Args:
        model_name: Base model name
        checkpoints: List of checkpoint indices to test
        output_dir: Directory to save results
        device: Device to run on
        layer_idx: Layer to measure at
    """
    print(f"\n{'='*60}")
    print("PYTHIA EMERGENCE PILOT")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Prompt: Meta-cognitive (known signature)")
    print(f"Layer: {layer_idx}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Measure each checkpoint
    results = []
    for checkpoint_idx in tqdm(checkpoints, desc="Processing checkpoints"):
        r_v = measure_checkpoint(
            model_name=model_name,
            checkpoint_idx=checkpoint_idx,
            prompt_text=META_COGNITIVE_PROMPT,
            tokenizer=tokenizer,
            device=device,
            layer_idx=layer_idx
        )
        
        results.append({
            "checkpoint": checkpoint_idx,
            "r_v": r_v,
            "expected": "Random (≈1.0)" if checkpoint_idx == 0 else 
                        "Emerging" if checkpoint_idx == 76 else 
                        "Clean (≈0.6)"
        })
        
        print(f"Checkpoint {checkpoint_idx}: R_V = {r_v:.3f}")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / "pythia_pilot_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {json_path}")
    
    # Create visualization
    create_emergence_plot(results, output_dir)
    
    return results


def create_emergence_plot(results, output_dir=None):
    """Create plot showing R_V emergence across checkpoints."""
    checkpoints = [r['checkpoint'] for r in results]
    r_v_values = [r['r_v'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(checkpoints, r_v_values, marker='o', linewidth=2, markersize=10, color='blue')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='R_V = 1.0 (no contraction)')
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target (0.6)')
    
    ax.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('R_V (Participation Ratio)', fontsize=12, fontweight='bold')
    ax.set_title('Pythia Emergence: R_V Across Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "pythia_emergence_pilot.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Pythia emergence pilot")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                       help="Base model name")
    parser.add_argument("--checkpoints", type=str, default="0,76,154",
                       help="Comma-separated checkpoint indices")
    parser.add_argument("--output-dir", type=str, default="03_RESULTS/phase2_pythia_pilot",
                       help="Output directory")
    parser.add_argument("--layer", type=int, default=27,
                       help="Layer index")
    
    args = parser.parse_args()
    
    checkpoints = [int(x) for x in args.checkpoints.split(',')]
    
    run_pilot(
        model_name=args.model,
        checkpoints=checkpoints,
        output_dir=args.output_dir,
        layer_idx=args.layer
    )

