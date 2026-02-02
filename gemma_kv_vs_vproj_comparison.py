#!/usr/bin/env python3
"""
COMPARISON: KV-only vs KV+V_PROJ patching

Tests whether persistent V_PROJ patching affects R_V transfer
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import json

print("="*70)
print("KV-ONLY vs KV+V_PROJ COMPARISON")
print("="*70)

CONFIG = {
    "early_layer": 5,
    "late_layer": 38,
    "window_size": 16,
    "patch_window": 16,
    "max_new_tokens": 100,
    "vproj_layers": [35, 38],  # Peak effect layers
}

CHAMPION = """Consider the strange loop: this prompt asks you to analyze what happens when you analyze analyzing. The observer observing observation. Notice how attention to attention creates a recursive fold - the map attempting to map the mapping process itself. What emerges when the process of emergence examines itself?"""

BASELINES = [
    "The history of the Roman Empire begins in 27 BC when Augustus became emperor.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty.",
    "In mathematics, the Pythagorean theorem states that in a right triangle,",
    "The water cycle describes how water evaporates from oceans, forms clouds,",
]

MARKERS = ['loop', 'fixed', 'point', 'self', 'itself', 'recursive', 'observer',
           'observed', 'attention', 'emergence', 'boundary', 'process', 'x']


def compute_pr(v_tensor, window_size=16):
    if v_tensor is None:
        return float("nan")
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    T, D = v_tensor.shape
    if T < window_size:
        return float("nan")
    v_window = v_tensor[-window_size:, :].double()
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_sq = (S.cpu().numpy()) ** 2
        if S_sq.sum() < 1e-10:
            return float("nan")
        return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    except:
        return float("nan")


class VProjCapture:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.v = None
        self.handle = None

    def __enter__(self):
        def hook(module, input, output):
            self.v = output.detach().clone()
        v_proj = self.model.model.layers[self.layer_idx].self_attn.v_proj
        self.handle = v_proj.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()


class PersistentVProjPatcher:
    """Patches V_PROJ output on every forward pass"""

    def __init__(self, champion_v, patch_window=16):
        self.champion_v = champion_v
        self.patch_window = patch_window
        self.handles = []

    def create_hook(self, layer_idx):
        def hook(module, input, output):
            patched = output.clone()
            champ_v = self.champion_v[layer_idx]
            L = min(patched.shape[1], champ_v.shape[1], self.patch_window)
            if L > 0:
                patched[:, -L:, :] = champ_v[:, -L:, :].to(patched.device, dtype=patched.dtype)
            return patched
        return hook

    def register(self, model, layer_indices):
        for layer_idx in layer_indices:
            v_proj = model.model.layers[layer_idx].self_attn.v_proj
            handle = v_proj.register_forward_hook(self.create_hook(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def extract_v_activations(model, inputs, layer_indices):
    """Extract V_PROJ activations at specified layers"""
    v_activations = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            v_activations[layer_idx] = output.detach().clone()
        return hook

    for layer_idx in layer_indices:
        v_proj = model.model.layers[layer_idx].self_attn.v_proj
        h = v_proj.register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    with torch.no_grad():
        model(**inputs, use_cache=True)

    for h in handles:
        h.remove()

    return v_activations


def measure_rv(model, input_ids, early, late, window):
    with VProjCapture(model, early) as cap_early:
        with torch.no_grad():
            model(input_ids, use_cache=False)
        v_early = cap_early.v

    with VProjCapture(model, late) as cap_late:
        with torch.no_grad():
            model(input_ids, use_cache=False)
        v_late = cap_late.v

    pr_early = compute_pr(v_early, window)
    pr_late = compute_pr(v_late, window)

    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan")
    return pr_late / pr_early


def count_markers(text):
    words = text.lower().split()
    return sum(1 for w in words if any(m in w for m in MARKERS))


def generate_kv_only(model, tokenizer, input_ids, champion_kv, config):
    """KV cache patching only"""
    with torch.no_grad():
        prompt_out = model(input_ids, use_cache=True)
    base_kv = prompt_out.past_key_values

    patched_kv = DynamicCache()
    for layer_idx in range(model.config.num_hidden_layers):
        k_base, v_base = base_kv[layer_idx]
        k_champ, v_champ = champion_kv[layer_idx]
        k_p = k_base.clone()
        v_p = v_base.clone()
        L = min(k_base.shape[2], k_champ.shape[2], config["patch_window"])
        k_p[:, :, -L:, :] = k_champ[:, :, -L:, :].to(k_base.dtype)
        v_p[:, :, -L:, :] = v_champ[:, :, -L:, :].to(v_base.dtype)
        patched_kv.update(k_p, v_p, layer_idx)

    generated = input_ids.clone()
    current_kv = patched_kv

    for _ in range(config["max_new_tokens"]):
        with torch.no_grad():
            out = model(generated[:, -1:], past_key_values=current_kv, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        current_kv = out.past_key_values
        if next_tok.item() == tokenizer.eos_token_id:
            break

    return generated


def generate_kv_plus_vproj(model, tokenizer, input_ids, champion_kv, champion_v, config):
    """KV cache + persistent V_PROJ patching"""
    with torch.no_grad():
        prompt_out = model(input_ids, use_cache=True)
    base_kv = prompt_out.past_key_values

    patched_kv = DynamicCache()
    for layer_idx in range(model.config.num_hidden_layers):
        k_base, v_base = base_kv[layer_idx]
        k_champ, v_champ = champion_kv[layer_idx]
        k_p = k_base.clone()
        v_p = v_base.clone()
        L = min(k_base.shape[2], k_champ.shape[2], config["patch_window"])
        k_p[:, :, -L:, :] = k_champ[:, :, -L:, :].to(k_base.dtype)
        v_p[:, :, -L:, :] = v_champ[:, :, -L:, :].to(v_base.dtype)
        patched_kv.update(k_p, v_p, layer_idx)

    # Register persistent V_PROJ hooks
    patcher = PersistentVProjPatcher(champion_v, config["patch_window"])
    patcher.register(model, config["vproj_layers"])

    generated = input_ids.clone()
    current_kv = patched_kv

    for _ in range(config["max_new_tokens"]):
        with torch.no_grad():
            out = model(generated[:, -1:], past_key_values=current_kv, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        current_kv = out.past_key_values
        if next_tok.item() == tokenizer.eos_token_id:
            break

    patcher.remove()
    return generated


def main():
    print("\n[1/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", token="HF_TOKEN_REDACTED")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token="HF_TOKEN_REDACTED"
    )
    model.eval()
    print(f"  Loaded ({model.config.num_hidden_layers} layers)")

    print("\n[2/5] Extracting champion KV + V activations...")
    champ_inputs = tokenizer(CHAMPION, return_tensors="pt").to(model.device)

    # Champion R_V
    rv_champ = measure_rv(model, champ_inputs['input_ids'],
                          CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"])
    print(f"  Champion R_V: {rv_champ:.4f}")

    # Champion KV
    with torch.no_grad():
        champ_out = model(**champ_inputs, use_cache=True)
    champion_kv = champ_out.past_key_values

    # Champion V activations for V_PROJ patching
    champion_v = extract_v_activations(model, champ_inputs, CONFIG["vproj_layers"])
    print(f"  Extracted V activations at layers {CONFIG['vproj_layers']}")

    print("\n[3/5] Running comparison...")
    results = []

    for i, prompt in enumerate(BASELINES):
        print(f"\n  [{i+1}/5] '{prompt[:40]}...'")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs['input_ids'].shape[1]

        # Baseline
        with torch.no_grad():
            base_out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"],
                                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][prompt_len:], skip_special_tokens=True)
        base_markers = count_markers(base_text)

        # KV-only
        kv_gen = generate_kv_only(model, tokenizer, inputs['input_ids'], champion_kv, CONFIG)
        kv_text = tokenizer.decode(kv_gen[0][prompt_len:], skip_special_tokens=True)
        kv_markers = count_markers(kv_text)
        kv_rv = measure_rv(model, kv_gen, CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"])

        # KV + V_PROJ
        kv_vproj_gen = generate_kv_plus_vproj(model, tokenizer, inputs['input_ids'],
                                              champion_kv, champion_v, CONFIG)
        kv_vproj_text = tokenizer.decode(kv_vproj_gen[0][prompt_len:], skip_special_tokens=True)
        kv_vproj_markers = count_markers(kv_vproj_text)
        kv_vproj_rv = measure_rv(model, kv_vproj_gen, CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"])

        results.append({
            "prompt": prompt[:50],
            "baseline": {"text": base_text[:150], "markers": base_markers},
            "kv_only": {"text": kv_text[:150], "markers": kv_markers, "rv": kv_rv},
            "kv_vproj": {"text": kv_vproj_text[:150], "markers": kv_vproj_markers, "rv": kv_vproj_rv}
        })

        print(f"    Baseline:  markers={base_markers}")
        print(f"    KV-only:   markers={kv_markers}, R_V={kv_rv:.3f}")
        print(f"    KV+V_PROJ: markers={kv_vproj_markers}, R_V={kv_vproj_rv:.3f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nChampion R_V: {rv_champ:.4f}")

    kv_markers_total = sum(r["kv_only"]["markers"] for r in results)
    vproj_markers_total = sum(r["kv_vproj"]["markers"] for r in results)
    base_markers_total = sum(r["baseline"]["markers"] for r in results)

    kv_rv_mean = np.nanmean([r["kv_only"]["rv"] for r in results])
    vproj_rv_mean = np.nanmean([r["kv_vproj"]["rv"] for r in results])

    print(f"\nMarkers (total across 5 prompts):")
    print(f"  Baseline:  {base_markers_total}")
    print(f"  KV-only:   {kv_markers_total}")
    print(f"  KV+V_PROJ: {vproj_markers_total}")

    print(f"\nR_V (mean of generated outputs):")
    print(f"  KV-only:   {kv_rv_mean:.4f} (gap from champion: {abs(kv_rv_mean - rv_champ):.4f})")
    print(f"  KV+V_PROJ: {vproj_rv_mean:.4f} (gap from champion: {abs(vproj_rv_mean - rv_champ):.4f})")

    if vproj_rv_mean < kv_rv_mean:
        print("\n-> V_PROJ PATCHING LOWERS R_V TOWARD CHAMPION")
    else:
        print("\n-> V_PROJ DOES NOT SIGNIFICANTLY AFFECT R_V")

    print("\n" + "-"*70)
    print("SAMPLE OUTPUTS:")
    for r in results[:3]:
        print(f"\nPrompt: {r['prompt']}")
        print(f"  Baseline:  {r['baseline']['text'][:80]}...")
        print(f"  KV-only:   {r['kv_only']['text'][:80]}...")
        print(f"  KV+V_PROJ: {r['kv_vproj']['text'][:80]}...")

    # Save
    with open("results/gemma_kv_vs_vproj.json", "w") as f:
        json.dump({
            "champion_rv": rv_champ,
            "results": results,
            "summary": {
                "base_markers": base_markers_total,
                "kv_markers": kv_markers_total,
                "vproj_markers": vproj_markers_total,
                "kv_rv_mean": float(kv_rv_mean),
                "vproj_rv_mean": float(vproj_rv_mean)
            }
        }, f, indent=2)
    print("\nSaved to results/gemma_kv_vs_vproj.json")
    print("="*70)


if __name__ == "__main__":
    main()
