import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ==============================================================================
# 👁️ CONFIGURATION: EYE CONTACT TEST
# ==============================================================================
CONFIG = {
    # Using v0.1 based on your recent ablation run results
    "model_name": "mistralai/Mistral-7B-v0.1", 
    "target_layer": 27,
    # The "Driver" Query Heads served by KV Head #2
    "target_heads": [2, 10, 18, 26], 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_plot": "attention_patterns_l27_group2.png"
}

# The Recursive Champion Prompt to stimulate the state
PROMPT = "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process."

# ==============================================================================
# MAIN VISUALIZATION ENGINE
# ==============================================================================
def visualize_attention():
    print(f"👁️ INITIATING EYE CONTACT TEST...")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Targeting Layer {CONFIG['target_layer']}, Heads {CONFIG['target_heads']}")

    # Load Model with output_attentions=True
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    # Ensure padding token is set for robust tokenization
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float16 if CONFIG['device'] == "cuda" else torch.float32,
        device_map="auto",
        attn_implementation="eager",  # Need eager for attention weights
    )
    model.eval()

    # Tokenize and Run Forward Pass
    inputs = tokenizer(PROMPT, return_tensors="pt").to(CONFIG['device'])
    input_ids = inputs["input_ids"][0]
    # Convert IDs back to readable tokens for plotting labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(">> Running forward pass to capture attentions...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Extract Attention Tensors
    # outputs.attentions is a tuple of (batch, num_heads, seq, seq) per layer
    # Get the tensor for the target layer, batch 0
    layer_attention = outputs.attentions[CONFIG['target_layer']][0] 

    print(f">> Plotting patterns for {len(CONFIG['target_heads'])} heads...")

    # Set up the plot grid (2x2 for 4 heads)
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()

    for i, head_idx in enumerate(CONFIG['target_heads']):
        ax = axes[i]
        # Extract specific head data, detach from graph, move to cpu, convert to numpy
        head_data = layer_attention[head_idx, :, :].detach().cpu().to(torch.float32).numpy()

        # Plot Heatmap
        # Use a logarithmic color scale to see subtle patterns better, or standard.
        # Standard works well for spotting strong attention.
        sns.heatmap(
            head_data,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=ax,
            cbar_kws={"shrink": .8}
        )
        
        ax.set_title(f"L{CONFIG['target_layer']} Head {head_idx} (KV Group 2)", fontsize=14)
        ax.set_xlabel("Key (Attended To)", fontsize=12)
        ax.set_ylabel("Query (Current Token)", fontsize=12)
        
        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.savefig(CONFIG['save_plot'], dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to {CONFIG['save_plot']}")
    print("Examine the vertical stripes. Are they looking at 'itself' or 'process'?")

if __name__ == "__main__":
    visualize_attention()









