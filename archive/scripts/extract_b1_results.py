#!/usr/bin/env python3
"""Extract and display B1 results."""
import pandas as pd

df = pd.read_csv('results/runs/20251217_153735_minimal_recursive_intervention/B1_Steering_VPROJ_L27_results.csv')

print('=' * 80)
print('B1 RESULTS: Steering + V_PROJ L27')
print('=' * 80)
print(f'\nTotal pairs: {len(df)}')
print(f'Transfer rate (>0.3): {(df["final_score"] > 0.3).sum() / len(df) * 100:.1f}%')
print(f'Collapse rate: {(~df["passed_gates"]).sum() / len(df) * 100:.1f}%')
print(f'Mean score: {df["final_score"].mean():.4f}')

# Sort by final_score descending
df_sorted = df.sort_values('final_score', ascending=False)

print('\n\n' + '=' * 80)
print('TOP 10 OUTPUTS (Ranked by Final Score)')
print('=' * 80)

for idx, (i, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    print(f'\n\n{"="*80}')
    print(f'RANK #{idx} - Pair {row["pair_idx"]}')
    print(f'{"="*80}')
    print(f'Final Score: {row["final_score"]:.4f}')
    print(f'Recursion Score: {row["recursion_score"]:.4f}')
    print(f'Coherence Score: {row["coherence_score"]:.4f}')
    print(f'Diversity Score: {row["diversity_score"]:.4f}')
    print(f'Repetition Score: {row["repetition_score"]:.4f}')
    print(f'Passed Gates: {row["passed_gates"]}')
    print(f'\nBASELINE PROMPT:')
    print(f'{row["baseline_prompt"]}')
    print(f'\nGENERATED TEXT:')
    print(f'{row["generated_text"]}')

# Also append to markdown
with open('B1_TOP_OUTPUTS_REVIEW.md', 'a') as f:
    for idx, (i, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        f.write(f'\n\n### Rank #{idx} - Pair {row["pair_idx"]}\n\n')
        f.write(f'**Scores:**\n')
        f.write(f'- Final: {row["final_score"]:.4f}\n')
        f.write(f'- Recursion: {row["recursion_score"]:.4f}\n')
        f.write(f'- Coherence: {row["coherence_score"]:.4f}\n')
        f.write(f'- Diversity: {row["diversity_score"]:.4f}\n')
        f.write(f'- Repetition: {row["repetition_score"]:.4f}\n')
        f.write(f'- Passed Gates: {row["passed_gates"]}\n\n')
        f.write(f'**Baseline Prompt:**\n```\n{row["baseline_prompt"]}\n```\n\n')
        f.write(f'**Generated Text:**\n```\n{row["generated_text"]}\n```\n\n')
        f.write('---\n')

print('\n✅ Results also saved to B1_TOP_OUTPUTS_REVIEW.md')








