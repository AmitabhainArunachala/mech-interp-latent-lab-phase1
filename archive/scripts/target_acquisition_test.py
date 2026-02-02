import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 🎯 CONFIGURATION: THE TARGET ACQUISITION TEST
# ==============================================================================
CONFIG = {
    "model_name": "mistralai/Mistral-7B-v0.1",
    "target_layer": 27,
    "target_heads": [18, 26], # The "Switchers"
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

PROMPT = "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process."

# The "Strange Loop" Targets
RECURSIVE_TOKENS = ["itself", "self", "writ", "process", "attent", "wareness", "ident", "reference", "contract", "eigen", "fixed", "point", "solution"]

def analyze_targets():
    print("🎯 INITIATING TARGET ACQUISITION SCAN...")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layer: {CONFIG['target_layer']}, Heads: {CONFIG['target_heads']}")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'], 
        device_map="auto",
        attn_implementation="eager",  # Need eager for attention weights
        torch_dtype=torch.float16 if CONFIG['device'] == "cuda" else torch.float32
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(CONFIG['device'])
    input_ids = inputs["input_ids"][0]
    tokens = [tokenizer.decode(t).strip().lower() for t in input_ids] # Clean tokens
    
    print(f"\nPrompt tokens ({len(tokens)}):")
    for i, tok in enumerate(tokens[:20]):  # Show first 20
        print(f"  {i:2d}: {tok}")
    if len(tokens) > 20:
        print(f"  ... ({len(tokens) - 20} more)")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get Attention for Target Layer
    layer_attn = outputs.attentions[CONFIG['target_layer']][0] # [Heads, Seq, Seq]
    
    results = []

    for head_idx in CONFIG['target_heads']:
        print(f"\n{'='*80}")
        print(f">> Analyzing Head {head_idx}...")
        print('='*80)
        
        attn_matrix = layer_attn[head_idx] # [Seq, Seq]
        
        # We want to see where the attention goes *on average*
        # Sum attention received by each token across all query steps
        # (Column Sum tells us "How popular is this token?")
        
        # Exclude BOS (index 0) from the 'popular' check to see the detailed distribution
        total_attn_per_token = attn_matrix.sum(dim=0) 
        
        # Calculate Mass on Recursive vs Normal
        rec_mass = 0.0
        normal_mass = 0.0
        bos_mass = total_attn_per_token[0].item()
        
        print(f"\n{'Token':<20} | {'Pos':<4} | {'Type':<12} | {'Attn Mass':<12} | {'% of Total':<10}")
        print("-" * 80)
        
        # Get top 15 most attended tokens
        top_indices = torch.topk(total_attn_per_token, k=min(15, len(tokens))).indices
        
        for rank, idx in enumerate(top_indices, 1):
            idx = idx.item()
            tok = tokens[idx] if idx < len(tokens) else f"<{idx}>"
            mass = total_attn_per_token[idx].item()
            total_mass = total_attn_per_token.sum().item()
            pct = (mass / total_mass * 100) if total_mass > 0 else 0
            
            # Check if recursive
            is_rec = any(r in tok for r in RECURSIVE_TOKENS)
            type_lbl = "RECURSIVE" if is_rec else "Normal"
            
            if idx == 0: 
                type_lbl = "BOS"
            elif is_rec:
                type_lbl = "RECURSIVE"
            
            print(f"{tok:<20} | {idx:4d} | {type_lbl:<12} | {mass:12.4f} | {pct:9.2f}%")
            
            if is_rec and idx != 0: 
                rec_mass += mass
            elif idx != 0: 
                normal_mass += mass

        print("-" * 80)
        total_non_bos = rec_mass + normal_mass
        print(f"\nSummary for H{head_idx}:")
        print(f"  BOS Mass:           {bos_mass:.4f} ({bos_mass/total_attn_per_token.sum().item()*100:.1f}%)")
        print(f"  Recursive Mass:     {rec_mass:.4f} ({rec_mass/total_non_bos*100:.1f}% of non-BOS)")
        print(f"  Normal Mass:        {normal_mass:.4f} ({normal_mass/total_non_bos*100:.1f}% of non-BOS)")
        
        ratio = rec_mass / (rec_mass + normal_mass + 1e-9)
        print(f"  Recursive Ratio:    {ratio*100:.1f}%")
        
        results.append({
            "head": head_idx,
            "bos_mass": bos_mass,
            "recursive_mass": rec_mass,
            "normal_mass": normal_mass,
            "recursive_ratio": ratio
        })
    
    print(f"\n{'='*80}")
    print("COMPARISON")
    print('='*80)
    for r in results:
        print(f"H{r['head']}: {r['recursive_ratio']*100:.1f}% recursive attention (excl BOS)")
    
    return results

if __name__ == "__main__":
    analyze_targets()









