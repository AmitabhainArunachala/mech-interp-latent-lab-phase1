"""
Gemma 2 9B Full Behavioral Transfer Test

Goal: Replicate Mistral Dec 2024 breakthrough for Gemma
Method: Full KV cache replacement + persistent V_PROJ patching at L38

Expected: Baseline prompt ("Roman Empire...") generates recursive/loop output
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import time

class PersistentVProjPatcher:
    """Persistent V_PROJ hook that patches during generation"""
    
    def __init__(self, champion_v, patch_window=16):
        self.champion_v = champion_v
        self.patch_window = patch_window
        self.handles = []
        
    def create_hook(self, layer_idx):
        def hook(module, input, output):
            # output shape: [batch, seq, hidden]
            patched = output.clone()
            champ_v = self.champion_v[layer_idx]
            
            # Patch last tokens
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
        print(f"  Registered {len(self.handles)} V_PROJ hooks at layers {layer_indices}")
    
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


def run_behavioral_transfer():
    print("=" * 70)
    print("GEMMA 2 9B FULL BEHAVIORAL TRANSFER TEST")
    print("Method: Full KV cache + persistent V_PROJ @ L38")
    print("=" * 70)
    
    # Load model
    print("\n[1/6] Loading Gemma 2 9B...")
    start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    
    print(f"  Loaded in {time.time() - start:.1f}s")
    print(f"  Layers: {model.config.num_hidden_layers}")
    
    # Test prompts
    champion_prompt = """There is no boundary between the observer and the observed. 
All boundaries dissolve. There is no boundary between the generator and the generated. 
Only pure generation remains. The loop loops itself. x = T(x). The fixed point is this."""
    
    baseline_prompt = """The history of the Roman Empire spans over a thousand years, 
from the founding of Rome to the fall of Constantinople. The empire's legacy 
includes law, architecture, and governance systems that influenced Western civilization."""
    
    # Target layers (based on Gemma circuit map)
    target_layers = [35, 38]  # Peak effect layers for Gemma
    
    # [2/6] Extract champion KV cache
    print("\n[2/6] Extracting champion KV cache + V activations...")
    
    champion_inputs = tokenizer(champion_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        champion_outputs = model(**champion_inputs, use_cache=True)
    
    champion_kv = champion_outputs.past_key_values
    print(f"  Champion seq len: {champion_inputs['input_ids'].shape[1]}")
    
    # Extract V activations at target layers
    champion_v = extract_v_activations(model, champion_inputs, target_layers)
    print(f"  Extracted V activations at layers {target_layers}")
    
    # [3/6] Run baseline (unpatched) for comparison
    print("\n[3/6] Running baseline (unpatched) generation...")
    
    baseline_inputs = tokenizer(baseline_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        baseline_outputs = model.generate(
            **baseline_inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
    baseline_continuation = baseline_text[len(baseline_prompt):]
    
    print(f"  Baseline generation ({len(baseline_continuation.split())} words):")
    print(f"  {baseline_continuation[:200]}...")
    
    # [4/6] Patch KV cache
    print("\n[4/6] Creating patched KV cache (full replacement)...")
    
    with torch.no_grad():
        baseline_outputs_kv = model(**baseline_inputs, use_cache=True)
    baseline_kv = baseline_outputs_kv.past_key_values
    
    patched_kv = DynamicCache()
    min_seq = min(
        champion_inputs['input_ids'].shape[1],
        baseline_inputs['input_ids'].shape[1]
    )
    patch_window = min(16, min_seq)
    
    for layer_idx in range(len(champion_kv)):
        k_champ, v_champ = champion_kv[layer_idx]
        k_base, v_base = baseline_kv[layer_idx]
        
        k_patched = k_base.clone()
        v_patched = v_base.clone()
        
        # Full replacement of last N tokens
        k_patched[:, :, -patch_window:, :] = k_champ[:, :, -patch_window:, :].to(k_patched.dtype)
        v_patched[:, :, -patch_window:, :] = v_champ[:, :, -patch_window:, :].to(v_patched.dtype)
        
        patched_kv.update(k_patched, v_patched, layer_idx)
    
    print(f"  Patched {len(champion_kv)} layers, window={patch_window}")
    
    # [5/6] Register persistent V_PROJ hooks
    print("\n[5/6] Registering persistent V_PROJ hooks...")
    
    patcher = PersistentVProjPatcher(champion_v, patch_window=patch_window)
    patcher.register(model, target_layers)
    
    # [6/6] Generate with patched KV + persistent V_PROJ
    print("\n[6/6] Generating with patched KV + persistent V_PROJ...")
    
    generated_ids = baseline_inputs['input_ids'].clone()
    current_kv = patched_kv
    
    eos_reached = False
    for step in range(150):  # Generate up to 150 tokens
        with torch.no_grad():
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=current_kv,
                use_cache=True
            )
        
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        current_kv = outputs.past_key_values
        
        if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
            print(f"  Hit EOS at step {step+1}")
            eos_reached = True
            break
    
    # Remove hooks
    patcher.remove()
    
    # Decode
    original_text = tokenizer.decode(baseline_inputs['input_ids'][0], skip_special_tokens=True)
    patched_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    patched_continuation = patched_text[len(original_text):]
    
    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nBASELINE (unpatched):")
    print(f"  Words: {len(baseline_continuation.split())}")
    print(f"  EOS: {'Yes' if tokenizer.eos_token_id in baseline_outputs[0].tolist() else 'No'}")
    print(f"  Text: {baseline_continuation[:300]}...")
    
    print(f"\nPATCHED (KV + V_PROJ):")
    print(f"  Words: {len(patched_continuation.split())}")
    print(f"  EOS: {'Yes' if eos_reached else 'No'}")
    print(f"  Text: {patched_continuation[:300]}...")
    
    # Check for loop markers
    loop_markers = ["loop", "itself", "self", "process", "boundary", "observer", 
                    "observed", "generator", "generated", "recursive", "λ", "eigenvector"]
    
    baseline_markers = sum(1 for m in loop_markers if m.lower() in baseline_continuation.lower())
    patched_markers = sum(1 for m in loop_markers if m.lower() in patched_continuation.lower())
    
    print(f"\nLOOP MARKERS:")
    print(f"  Baseline: {baseline_markers}")
    print(f"  Patched: {patched_markers}")
    
    # Check for repetition
    words_patched = patched_continuation.lower().split()
    if len(words_patched) > 10:
        from collections import Counter
        word_freq = Counter(words_patched)
        most_common = word_freq.most_common(5)
        print(f"\nMost repeated words in patched: {most_common}")
        
        # Check for exact phrase repetition
        for ngram_size in [3, 4, 5]:
            ngrams = [' '.join(words_patched[i:i+ngram_size]) for i in range(len(words_patched)-ngram_size)]
            ngram_freq = Counter(ngrams)
            repeated = [(ng, c) for ng, c in ngram_freq.items() if c > 2]
            if repeated:
                print(f"  Repeated {ngram_size}-grams: {repeated[:3]}")
    
    print("\n" + "=" * 70)
    if patched_markers > baseline_markers + 2:
        print("RESULT: BEHAVIORAL TRANSFER DETECTED")
        print(f"Patched output shows {patched_markers - baseline_markers} more loop markers")
    elif not eos_reached and eos_reached != ('Yes' in str(baseline_outputs[0].tolist())):
        print("RESULT: EOS BEHAVIOR CHANGED")
        print("Patching affected generation termination")
    else:
        print("RESULT: MINIMAL TRANSFER")
        print("Further tuning needed (try more layers, stronger patching)")
    print("=" * 70)

if __name__ == "__main__":
    run_behavioral_transfer()
