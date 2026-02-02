import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 🔮 CONFIGURATION: LOGIT LENS
# ==============================================================================
CONFIG = {
    # Using v0.1 since that's what we used for the head discovery
    "model_name": "mistralai/Mistral-7B-v0.1", 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "target_token_idx": -1, # Look at the prediction for the VERY LAST token
    "top_k": 3 # Show top 3 thoughts per layer
}

# The Recursive Champion Prompt
PROMPT = "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the" 
# We expect it to predict "process" or "sentence" or "itself".

def run_logit_lens():
    print(f"🔮 INITIATING LOGIT LENS SCAN...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'], 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(CONFIG['device'])
    
    # We need to capture the residual stream at every layer
    # In HF models, we can use 'output_hidden_states=True'
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states is a tuple of (33 tensors) (Embeddings + 32 Layers)
    # Shape: [batch, seq, hidden_dim]
    
    print(f"\nPrompt: \"{PROMPT}\"")
    print(f"Scanning thoughts for the next token...\n")
    print(f"{'Layer':<6} | {'Top 1':<15} | {'Top 2':<15} | {'Top 3':<15}")
    print("-" * 60)

    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        # 1. Get the hidden state for the target token
        # hidden_state: [1, seq_len, 4096]
        target_vector = hidden_state[0, CONFIG['target_token_idx'], :]
        
        # 2. Normalize (LayerNorm) - Critical for Logit Lens
        # Ideally we use the model's final layernorm, but for intermediate layers, 
        # the raw stream often works or we apply the final LN. 
        # Standard Logit Lens practice: Apply Final Layer Norm then LM Head.
        
        target_vector = model.model.norm(target_vector)
        
        # 3. Decode (Project to Vocabulary)
        logits = model.lm_head(target_vector)
        
        # 4. Get Top K
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_ids = torch.topk(probs, CONFIG['top_k'])
        
        tokens = [tokenizer.decode(idx) for idx in top_k_ids]
        
        # Format output
        t1 = f"{tokens[0]} ({top_k_probs[0]:.2f})"
        t2 = f"{tokens[1]} ({top_k_probs[1]:.2f})"
        t3 = f"{tokens[2]} ({top_k_probs[2]:.2f})"
        
        # Highlight our layers of interest
        prefix = " "
        if layer_idx == 18: prefix = "⚡" # The Switch
        if layer_idx == 27: prefix = "🔴" # The Singularity
        
        print(f"{prefix} {layer_idx:<4} | {t1:<15} | {t2:<15} | {t3:<15}")

if __name__ == "__main__":
    run_logit_lens()









