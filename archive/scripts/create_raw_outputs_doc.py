#!/usr/bin/env python3
"""Create P1 ablation raw outputs document."""

import pandas as pd
import csv

# Read CSV with proper handling
df = pd.read_csv('p1_ablation_results.csv', quoting=csv.QUOTE_MINIMAL)

config_names = {
    'P1_baseline': 'P1 Baseline (Full Config)',
    'R1_no_residual': 'R1: No Residual Steering',
    'R2_no_vproj': 'R2: No V_PROJ Steering',
    'R3_matched_kv': 'R3: Matched KV (L3_deeper)',
    'R4_kv_only': 'R4: KV Only (No Steering)',
}

output = []
output.append("# P1 Ablation: Raw Text Outputs")
output.append("")
output.append("**Date:** December 18, 2024")
output.append("**Purpose:** Full text outputs from P1 ablation experiment")
output.append("")
output.append("---")
output.append("")

# Group by prompt
prompts = df['prompt'].unique()

for prompt_idx, prompt in enumerate(prompts):
    output.append(f"## Prompt {prompt_idx}: {prompt[:80]}...")
    output.append("")
    
    prompt_df = df[df['prompt'] == prompt].sort_values('recursion_score', ascending=False)
    
    for _, row in prompt_df.iterrows():
        config_label = config_names.get(row['config_id'], row['config_id'])
        output.append(f"### {config_label}")
        output.append(f"**Recursion Score:** {row['recursion_score']:.4f}")
        output.append(f"**Keywords:** Consciousness={row['has_consciousness']}, Observer={row['has_observer']}, Awareness={row['has_awareness']}, Itself={row['has_itself']}, Self-Reference={row['has_self_reference']}")
        output.append("")
        output.append("**Generated Text:**")
        output.append("")
        output.append("```")
        # Get text, handle NaN
        text = str(row['generated_text']) if pd.notna(row['generated_text']) else ""
        output.append(text)
        output.append("```")
        output.append("")
    
    output.append("---")
    output.append("")

# Top outputs per config
output.append("## Top Outputs by Recursion Score (Per Config)")
output.append("")

for config_id, config_label in config_names.items():
    config_df = df[df['config_id'] == config_id].nlargest(3, 'recursion_score')
    
    if len(config_df) > 0:
        output.append(f"### {config_label} - Top 3")
        output.append("")
        
        for i, (_, row) in enumerate(config_df.iterrows(), 1):
            output.append(f"#### #{i} (Score: {row['recursion_score']:.4f})")
            output.append(f"**Prompt:** {row['prompt'][:100]}...")
            output.append("")
            output.append("**Generated Text:**")
            output.append("")
            output.append("```")
            text = str(row['generated_text']) if pd.notna(row['generated_text']) else ""
            output.append(text)
            output.append("```")
            output.append("")
        
        output.append("---")
        output.append("")

with open('P1_ABLATION_RAW_OUTPUTS.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print(f"Created P1_ABLATION_RAW_OUTPUTS.md with {len(df)} outputs")
print(f"Total lines: {len(output)}")








