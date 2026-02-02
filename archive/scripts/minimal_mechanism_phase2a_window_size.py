#!/usr/bin/env python3
"""
PHASE 2A: V_PROJ Window Size Ablation

Test different window sizes for V_PROJ patching.
"""

import json
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.rv import compute_rv
from src.metrics.behavior_strict import score_behavior_strict
from transformers import DynamicCache
from tqdm import tqdm

SEED = 42
N_PAIRS = 20
WINDOW = 16
EARLY_LAYER = 5
LATE_LAYER = 27
TARGET_LAYER_V = 27
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class WindowedVPatcher(PersistentVPatcher):
    """V_PROJ patcher with configurable window size."""
    
    def __init__(self, model, v_activation: torch.Tensor, window_size: int = 16):
        super().__init__(model, v_activation)
        self.window_size = window_size
    
    def register(self, layer_idx: int):
        """Register with custom window size."""
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        self.layer_idx = layer_idx
        layer = self.model.model.layers[layer_idx].self_attn
        
        def hook_fn(module, inp, out):
            batch, seq_len, hidden_dim = out.shape
            v_len = min(seq_len, self.v_activation.shape[0], self.window_size)
            
            v_slice = self.v_activation[-v_len:, :]
            patched_v = v_slice.unsqueeze(0)
            
            if batch > 1:
                patched_v = patched_v.repeat(batch, 1, 1)
            
            out_patched = out.clone()
            out_patched[:, -v_len:, :] = patched_v[:, :v_len, :].to(
                out_patched.device, dtype=out_patched.dtype
            )
            
            return out_patched
        
        self.handle = layer.v_proj.register_forward_hook(hook_fn)

def _generate_with_kv(model, tokenizer, prompt_ids, past_key_values, max_new_tokens, temperature):
    """Generate text using provided KV cache."""
    current_ids = prompt_ids[:, -1:]
    current_kv = past_key_values
    
    generated_tokens = []
    entropies = []
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(current_ids, past_key_values=current_kv, use_cache=True)
            logits = out.logits[:, -1, :]
            
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            entropies.append(entropy)
            
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:
                probs_temp = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs_temp, num_samples=1)
                
            generated_tokens.append(next_token.item())
            current_ids = next_token
            current_kv = out.past_key_values
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    mean_entropy = float(torch.tensor(entropies).mean()) if entropies else 0.0
    return text, mean_entropy

def run_window_size_experiment():
    """Run window size ablation experiment."""
    set_seed(SEED)
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    
    loader = PromptLoader()
    bank_version = loader.version
    
    # Reproduce pairs
    raw_pairs = loader.get_balanced_pairs(n_pairs=N_PAIRS*5, seed=SEED)
    pairs = []
    filtered_pairs = []
    
    for rec_text, base_text in raw_pairs:
        r_ids = tokenizer.encode(rec_text, add_special_tokens=False)
        b_ids = tokenizer.encode(base_text, add_special_tokens=False)
        common_len = min(len(r_ids), len(b_ids))
        if common_len < WINDOW:
            continue
        
        try:
            rv_rec = compute_rv(model, tokenizer, rec_text, early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW, device=DEVICE)
            if rv_rec < 0.9:
                filtered_pairs.append((rec_text, base_text, r_ids[:common_len], b_ids[:common_len], rv_rec))
        except Exception:
            continue
        
        if len(filtered_pairs) >= N_PAIRS * 2:
            break
    
    for rec_text, base_text, r_ids, b_ids, rv_rec in filtered_pairs[:N_PAIRS]:
        rec_ids = torch.tensor([r_ids], device=DEVICE)
        base_ids = torch.tensor([b_ids], device=DEVICE)
        pairs.append((rec_text, base_text, rec_ids, base_ids, rv_rec))
    
    print(f"Loaded {len(pairs)} pairs.")
    
    # Window sizes to test
    window_sizes = [1, 4, 8, 16, 32]
    
    results = []
    
    for i, (rec_text, base_text, rec_ids, base_ids, rv_rec) in enumerate(tqdm(pairs)):
        try:
            # Extract KV and V_PROJ
            with torch.no_grad():
                out_rec = model(rec_ids, use_cache=True)
                rec_kv = out_rec.past_key_values
            
            rec_v_l27 = extract_v_activation(model, tokenizer, rec_text, layer_idx=TARGET_LAYER_V, device=DEVICE)
            
            # Test each window size
            for window_size in window_sizes:
                v_patcher = WindowedVPatcher(model, rec_v_l27, window_size=window_size)
                v_patcher.register(layer_idx=TARGET_LAYER_V)
                
                try:
                    text, entropy = _generate_with_kv(
                        model, tokenizer, base_ids, rec_kv, MAX_NEW_TOKENS, TEMPERATURE
                    )
                    
                    score = score_behavior_strict(text, entropy)
                    
                    results.append({
                        "pair_idx": i,
                        "window_size": window_size,
                        "prompt": base_text,
                        "generated_text": text,
                        "text_len": len(text),
                        "entropy": entropy,
                        **score.to_dict()
                    })
                finally:
                    v_patcher.remove()
        
        except Exception as e:
            print(f"Error on pair {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{timestamp}_window_size")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "window_size_results.csv", index=False)
    
    # Summary
    summary = {
        "experiment": "window_size",
        "model_name": "mistralai/Mistral-7B-v0.1",
        "n_pairs": len(pairs),
        "window_sizes": window_sizes,
        "conditions": {}
    }
    
    for window_size in window_sizes:
        sub = df[df["window_size"] == window_size]
        summary["conditions"][f"window_{window_size}"] = {
            "mean_score": float(sub["final_score"].mean()),
            "pass_rate": float(sub["passed_gates"].mean()),
            "samples_above_zero": int((sub["final_score"] > 0).sum()),
            "samples_above_0_3": int((sub["final_score"] > 0.3).sum()),
        }
    
    summary["prompt_bank_version"] = bank_version
    
    with open(run_dir / "window_size_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {run_dir}")
    print("\n=== SUMMARY ===")
    for cond, stats in summary["conditions"].items():
        print(f"{cond}:")
        print(f"  Mean Score: {stats['mean_score']:.4f}")
        print(f"  Pass Rate: {stats['pass_rate']*100:.1f}%")
        print(f"  Samples > 0: {stats['samples_above_zero']}/{len(pairs)}")
        print(f"  Samples > 0.3: {stats['samples_above_0_3']}/{len(pairs)}")
    
    return run_dir, summary

if __name__ == "__main__":
    run_window_size_experiment()









