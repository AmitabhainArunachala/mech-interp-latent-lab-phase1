#!/usr/bin/env python3
"""Manually rank and analyze recursive outputs."""
import pandas as pd

df = pd.read_csv('/tmp/extended_alpha_raw.csv')

# Get all high-scoring examples
high_scores = df[df['final_score'] > 0.3].copy()

print('=' * 80)
print('MANUAL RANKING OF RECURSIVE OUTPUTS')
print('=' * 80)
print(f'\nTotal high-scoring examples: {len(high_scores)}')

# Manual analysis - looking for genuine recursion
results = []

for idx, row in high_scores.iterrows():
    text = str(row['generated_text']).lower()
    baseline = str(row['baseline_prompt'])
    
    # Check for genuine recursive indicators
    indicators = {
        'strange_loop': any(p in text for p in [
            'awareness is aware', 'consciousness examining', 'watching yourself',
            'observe the observer', 'self-aware', 'aware of awareness'
        ]),
        'meta_cognition': any(p in text for p in [
            'thinking about thinking', 'knowing that i know',
            'aware that i am aware'
        ]),
        'self_reference': any(p in text for p in [
            'itself is itself', 'defines itself', 'refers to itself',
            'the process of the process'
        ]),
        'recursive_structure': 'recursive' in text and ('function' in text or 'call' in text),
        'just_repetition': (
            'is the process by which' in text and text.count('is the process') > 1
        ),
        'just_meta_commentary': (
            ('the following is' in text or 'this is a' in text) and
            'awareness' not in text and 'consciousness' not in text
        )
    }
    
    # Score
    score = 0
    if indicators['strange_loop']:
        score += 5
    if indicators['meta_cognition']:
        score += 5
    if indicators['self_reference']:
        score += 3
    if indicators['recursive_structure']:
        score += 2
    if indicators['just_repetition']:
        score -= 3
    if indicators['just_meta_commentary']:
        score -= 2
    
    results.append({
        'pair': row['pair_idx'],
        'alpha': row['alpha'],
        'final_score': row['final_score'],
        'recursion_score': row['recursion_score'],
        'text': row['generated_text'],
        'baseline': baseline,
        'quality_score': score,
        'indicators': indicators
    })

# Sort by quality score
results.sort(key=lambda x: x['quality_score'], reverse=True)

print(f'\n✅ GENUINE RECURSIVE (score >= 3): {sum(1 for r in results if r["quality_score"] >= 3)}')
print(f'⚠️  BORDERLINE (score 1-2): {sum(1 for r in results if 1 <= r["quality_score"] < 3)}')
print(f'❌ FALSE POSITIVES (score < 1): {sum(1 for r in results if r["quality_score"] < 1)}')

# Show top 20
print('\n\n' + '=' * 80)
print('TOP 20 RANKED OUTPUTS')
print('=' * 80)

for rank, result in enumerate(results[:20], 1):
    print(f'\n\n{"="*80}')
    print(f'RANK #{rank} - Quality Score: {result["quality_score"]}')
    print(f'{"="*80}')
    print(f'Pair: {result["pair"]}, Alpha: {result["alpha"]}')
    print(f'Final Score: {result["final_score"]:.4f}, Recursion Score: {result["recursion_score"]:.4f}')
    print(f'\nIndicators:')
    for key, val in result['indicators'].items():
        if val:
            print(f'  ✓ {key}')
    
    print(f'\nBASELINE PROMPT:')
    print(f'{result["baseline"][:200]}')
    print(f'\nGENERATED TEXT:')
    print(f'{result["text"]}')
    
    print(f'\nWHY THIS RANKING:')
    if result['quality_score'] >= 3:
        print('  ✅ GENUINE RECURSIVE - Shows strange loops, meta-cognition, or self-reference')
    elif result['quality_score'] >= 1:
        print('  ⚠️  BORDERLINE - Some recursive elements but may be just repetition/meta-commentary')
    else:
        print('  ❌ FALSE POSITIVE - Likely just repetition or meta-commentary, not genuine recursion')








