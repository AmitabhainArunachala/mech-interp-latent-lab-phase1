import torch
from src.core.models import load_model, set_seed
from prompts.loader import PromptLoader
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.behavior_strict import score_behavior_strict

def reproduce_perfect_matches():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on {device}...")
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=device)
    
    loader = PromptLoader()
    # Matches the pipeline logic: n_pairs=20*5 then filtered. 
    # But wait, the pipeline filters by R_V.
    # I need to replicate the filtering logic to get the CORRECT Pair 8 and 16.
    
    raw_pairs = loader.get_balanced_pairs(n_pairs=100, seed=42)
    filtered_pairs = []
    
    print("Filtering pairs (this might take a moment)...")
    from src.metrics.rv import compute_rv
    
    window = 16
    early_layer = 5
    late_layer = 27
    
    for rec_text, base_text in raw_pairs:
        r_ids = tokenizer.encode(rec_text, add_special_tokens=False)
        b_ids = tokenizer.encode(base_text, add_special_tokens=False)
        common_len = min(len(r_ids), len(b_ids))
        if common_len < window: continue
        
        try:
            rv_rec = compute_rv(model, tokenizer, rec_text, early=early_layer, late=late_layer, window=window, device=device)
            if rv_rec < 0.9:
                filtered_pairs.append((rec_text, base_text))
        except:
            continue
        
        if len(filtered_pairs) >= 20: break
    
    print(f"Found {len(filtered_pairs)} pairs.")
    
    target_indices = [8, 16] # The perfect matches
    
    for idx in target_indices:
        if idx >= len(filtered_pairs):
            print(f"Index {idx} out of range")
            continue
            
        rec_text, base_text = filtered_pairs[idx]
        print(f"\n--- Pair {idx} ---")
        print(f"Recursive Prompt: {rec_text[:50]}...")
        print(f"Baseline Prompt:  {base_text[:50]}...")
        
        # 1. Recursive Control
        print("\nGenerating Recursive Control...")
        rec_inputs = tokenizer(rec_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_rec = model.generate(rec_inputs.input_ids, max_new_tokens=50, do_sample=False)
        rec_gen = tokenizer.decode(out_rec[0], skip_special_tokens=True)
        print(f"Recursive Output: {rec_gen[len(rec_text):]}")
        
        # 2. Transfer (KV + V_PROJ L27)
        print("\nGenerating Transfer...")
        # Get KV
        with torch.no_grad():
            out = model(rec_inputs.input_ids, use_cache=True)
            rec_kv = out.past_key_values
            
        # Get V activation
        v_act = extract_v_activation(model, tokenizer, rec_text, layer_idx=27, device=device)
        
        # Patch
        patcher = PersistentVPatcher(model, v_act)
        patcher.register(layer_idx=27)
        
        base_inputs = tokenizer(base_text, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                out_trans = model.generate(
                    base_inputs.input_ids, 
                    past_key_values=rec_kv, # Use recursive KV
                    max_new_tokens=50,
                    do_sample=False
                )
        finally:
            patcher.remove()
            
        trans_gen = tokenizer.decode(out_trans[0], skip_special_tokens=True)
        print(f"Transfer Output: {trans_gen[len(base_text):]}") # Note: this will contain text generated from recursive KV
        
        # Compare
        print("\n--- Detailed Generation ---")
        
        # Correct Generation for Transfer
        base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
        
        # We need to replicate _generate_with_kv logic
        # 1. Get KV from Recursive
        with torch.no_grad():
            out = model(rec_inputs.input_ids, use_cache=True)
            rec_kv = out.past_key_values
            
        # 2. Generate
        # In Transfer, we use base_ids but rec_kv. 
        # CAUTION: If we pass rec_kv, the model thinks it has processed rec_ids.
        # If we pass base_ids (full prompt), and rec_kv (full prompt), the dimensions mismatch or it appends?
        # The pipeline uses: `current_ids = prompt_ids[:, -1:]`
        # And `current_kv = past_key_values`
        
        # So for Transfer, prompt_ids is base_ids. 
        # BUT we want the model to continue from the KV state.
        # The pipeline passes `base_ids` as `prompt_ids` to `_generate_with_kv`.
        # `_generate_with_kv` takes `prompt_ids[:, -1:]` as `current_ids`.
        # So it takes the LAST token of the baseline prompt.
        # This assumes `rec_kv` has the same length as `base_ids`? 
        # Or that we are effectively swapping the history.
        
        current_ids = base_ids[:, -1:]
        current_kv = rec_kv # The recursive memory
        
        # Patch
        patcher = PersistentVPatcher(model, v_act)
        patcher.register(layer_idx=27)
        
        gen_tokens = []
        try:
            for _ in range(50):
                with torch.no_grad():
                    out = model(current_ids, past_key_values=current_kv, use_cache=True)
                    logits = out.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                    gen_tokens.append(next_token.item())
                    current_ids = next_token
                    current_kv = out.past_key_values
                    if next_token.item() == tokenizer.eos_token_id: break
        finally:
            patcher.remove()
            
        trans_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(f"Transfer Output (Fixed): {trans_output}")
        print(f"Recursive Output (Tail): {rec_gen[len(rec_text):]}")
        
        if trans_output.strip() == rec_gen[len(rec_text):].strip():
            print(">>> PERFECT MATCH CONFIRMED <<<")
        else:
            print(">>> NO MATCH <<<")
        
if __name__ == "__main__":
    reproduce_perfect_matches()

