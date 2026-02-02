import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ==============================================================================
# 👁️ CONFIGURATION: BASELINE CONTROL TEST
# ==============================================================================
CONFIG = {
    "model_name": "mistralai/Mistral-7B-v0.1", 
    "target_layer": 27,
    # The same "Driver" heads from the recursive test
    "target_heads": [2, 10, 18, 26], 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_plot": "attention_patterns_l27_baseline.png"
}

# The Non-Recursive Baseline Prompt
PROMPT = "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline. Historians analyze the political, social, and economic factors that contributed to the rise of Rome."

def visualize_attention():
    print(f"👁️ INITIATING BASELINE CONTROL TEST...")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Targeting Layer {CONFIG['target_layer']}, Heads {CONFIG['target_heads']}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float16 if CONFIG['device'] == "cuda" else torch.float32,
        device_map="auto",
        attn_implementation="eager",  # Need eager for attention weights
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(CONFIG['device'])
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(">> Running forward pass on BASELINE prompt...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    layer_attention = outputs.attentions[CONFIG['target_layer']][0] 

    print(f">> Plotting patterns...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()

    for i, head_idx in enumerate(CONFIG['target_heads']):
        ax = axes[i]
        head_data = layer_attention[head_idx, :, :].detach().cpu().to(torch.float32).numpy()

        sns.heatmap(
            head_data,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=ax,
            cbar_kws={"shrink": .8}
        )
        
        ax.set_title(f"BASELINE: L{CONFIG['target_layer']} Head {head_idx}", fontsize=14)
        ax.set_xlabel("Key (Attended To)", fontsize=12)
        ax.set_ylabel("Query (Current Token)", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.savefig(CONFIG['save_plot'], dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to {CONFIG['save_plot']}")
    print("CHECK THE IMAGE: Did the vertical BOS stripe disappear?")
    print("Expected: Diagonal patterns (linear history) instead of vertical BOS anchor")

if __name__ == "__main__":
    visualize_attention()









