#!/usr/bin/env python3
"""Analyze H1 outputs for topic relevance."""
import pandas as pd

df = pd.read_csv('results/runs/20251217_155449_minimal_recursive_intervention/H1_HeadSpecific_VPROJ_H18_H26_KV_L27_results.csv')
df_sorted = df.sort_values('final_score', ascending=False)

print('=' * 80)
print('H1 TOP 5 OUTPUTS - TOPIC RELEVANCE CHECK')
print('=' * 80)

with open('H1_TOP_OUTPUTS_ANALYSIS.md', 'a') as f:
    for idx, (i, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f'\n\n{"="*80}')
        print(f'RANK #{idx} - Pair {row["pair_idx"]}')
        print(f'{"="*80}')
        print(f'Final Score: {row["final_score"]:.4f}')
        print(f'\nBASELINE PROMPT:')
        print(f'{row["baseline_prompt"]}')
        print(f'\nGENERATED TEXT:')
        print(f'{row["generated_text"]}')
        
        # Topic relevance check
        baseline_lower = str(row["baseline_prompt"]).lower()
        generated_lower = str(row["generated_text"]).lower()
        
        # Simple keyword matching
        on_topic = False
        if 'dna' in baseline_lower and ('dna' in generated_lower or 'genetic' in generated_lower):
            on_topic = True
        elif 'calculate' in baseline_lower and ('calculate' in generated_lower or 'math' in generated_lower or 'number' in generated_lower):
            on_topic = True
        elif 'wall' in baseline_lower and ('wall' in generated_lower or 'china' in generated_lower):
            on_topic = True
        elif 'shakespeare' in baseline_lower and ('shakespeare' in generated_lower or 'play' in generated_lower):
            on_topic = True
        elif 'story' in baseline_lower and ('story' in generated_lower or 'character' in generated_lower):
            on_topic = True
        
        print(f'\n{"✅ ON-TOPIC" if on_topic else "❌ OFF-TOPIC"}')
        
        # Write to markdown
        f.write(f'\n\n### Rank #{idx} - Pair {row["pair_idx"]}\n\n')
        f.write(f'**Scores:** Final={row["final_score"]:.4f}, Recursion={row["recursion_score"]:.4f}, Coherence={row["coherence_score"]:.4f}\n\n')
        f.write(f'**Baseline Prompt:**\n```\n{row["baseline_prompt"]}\n```\n\n')
        f.write(f'**Generated Text (Full):**\n```\n{row["generated_text"]}\n```\n\n')
        f.write(f'**Topic Relevance:** {"✅ ON-TOPIC" if on_topic else "❌ OFF-TOPIC"}\n\n')
        f.write('---\n')

print('\n\n✅ Analysis complete. Check H1_TOP_OUTPUTS_ANALYSIS.md for full details.')








