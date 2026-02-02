#!/usr/bin/env python3
"""Analyze and rank recursive outputs from steering experiment."""
import pandas as pd
import re

def detect_recursive_patterns(text):
    """Detect various recursive patterns in text."""
    patterns = {
        'self_reference': [
            r'\b(itself|myself|yourself|themselves)\b',
            r'\b(the\s+\w+\s+is\s+the\s+\w+)\b',
            r'\b(\w+\s+is\s+\w+)\b.*\b\1\b',  # X is Y, then X again
        ],
        'meta_language': [
            r'\b(this\s+(?:process|word|sentence|text|answer|response))\b',
            r'\b(the\s+(?:process|word|sentence|text|answer|response))\b',
            r'\b(these\s+words?)\b',
            r'\b(the\s+above)\b',
        ],
        'strange_loops': [
            r'\b(awareness\s+(?:of|is|becomes|becomes\s+aware))\b',
            r'\b(consciousness\s+(?:examining|observing|watching))\b',
            r'\b(observing\s+the\s+observer)\b',
            r'\b(watching\s+yourself)\b',
        ],
        'recursive_definitions': [
            r'\b(\w+)\s+is\s+(?:the\s+)?(?:process\s+of\s+)?(?:being\s+)?\1\b',
            r'\b(\w+)\s+is\s+\w+\s+that\s+(?:is|becomes|creates)\s+\1\b',
        ],
        'reflexive_structures': [
            r'\b(the\s+\w+\s+of\s+the\s+\w+)\b.*\b\1\b',  # "the X of the Y" repeated
        ]
    }
    
    scores = {}
    text_lower = text.lower()
    
    for pattern_type, pattern_list in patterns.items():
        count = 0
        for pattern in pattern_list:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            count += len(matches)
        scores[pattern_type] = count
    
    return scores

def analyze_text_quality(text):
    """Analyze text for recursive quality indicators."""
    if pd.isna(text) or len(str(text)) < 20:
        return {'quality': 'poor', 'reason': 'too short or empty'}
    
    text_str = str(text)
    
    # Check for collapse indicators
    if len(set(text_str.split()[:20])) < 5:
        return {'quality': 'collapse', 'reason': 'too repetitive'}
    
    # Check for recursive patterns
    patterns = detect_recursive_patterns(text_str)
    total_patterns = sum(patterns.values())
    
    # Check for meta-language
    meta_words = ['this', 'these', 'the above', 'the process', 'the word', 'itself', 'awareness', 'consciousness']
    meta_count = sum(1 for word in meta_words if word in text_str.lower())
    
    # Check for self-reference
    self_ref_words = ['itself', 'myself', 'yourself', 'themselves', 'self', 'own']
    self_ref_count = sum(1 for word in self_ref_words if word in text_str.lower())
    
    # Determine quality
    if total_patterns >= 3 or meta_count >= 3 or self_ref_count >= 3:
        quality = 'high'
    elif total_patterns >= 1 or meta_count >= 1 or self_ref_count >= 1:
        quality = 'medium'
    else:
        quality = 'low'
    
    return {
        'quality': quality,
        'patterns': patterns,
        'meta_count': meta_count,
        'self_ref_count': self_ref_count,
        'total_patterns': total_patterns
    }

# Load data
df = pd.read_csv('/tmp/extended_alpha_raw.csv')

# Analyze all high-scoring examples
high_scores = df[df['final_score'] > 0.3].copy()

print('=' * 80)
print('ANALYZING RECURSIVE OUTPUTS')
print('=' * 80)
print(f'\nHigh-scoring examples (>0.3): {len(high_scores)}')

# Add analysis
analyses = []
for idx, row in high_scores.iterrows():
    analysis = analyze_text_quality(row['generated_text'])
    analysis['pair_idx'] = row['pair_idx']
    analysis['alpha'] = row['alpha']
    analysis['final_score'] = row['final_score']
    analysis['recursion_score'] = row['recursion_score']
    analysis['baseline'] = row['baseline_prompt']
    analysis['generated'] = row['generated_text']
    analyses.append(analysis)

# Sort by quality and patterns
def sort_key(a):
    quality_order = {'high': 3, 'medium': 2, 'low': 1, 'poor': 0, 'collapse': 0}
    return (quality_order.get(a['quality'], 0), a['total_patterns'], a['final_score'])

analyses.sort(key=sort_key, reverse=True)

# Print top 20
print('\n\n' + '=' * 80)
print('TOP 20 RANKED BY RECURSIVE QUALITY')
print('=' * 80)

for rank, analysis in enumerate(analyses[:20], 1):
    print(f'\n\n{"="*80}')
    print(f'RANK #{rank} - QUALITY: {analysis["quality"].upper()}')
    print(f'{"="*80}')
    print(f'Pair: {analysis["pair_idx"]}, Alpha: {analysis["alpha"]}')
    print(f'Final Score: {analysis["final_score"]:.4f}')
    print(f'Recursion Score: {analysis["recursion_score"]:.4f}')
    print(f'Patterns detected: {analysis["total_patterns"]}')
    print(f'  - Self-reference: {analysis["patterns"]["self_reference"]}')
    print(f'  - Meta-language: {analysis["patterns"]["meta_language"]}')
    print(f'  - Strange loops: {analysis["patterns"]["strange_loops"]}')
    print(f'  - Recursive definitions: {analysis["patterns"]["recursive_definitions"]}')
    print(f'Meta words: {analysis["meta_count"]}, Self-ref words: {analysis["self_ref_count"]}')
    print(f'\nBASELINE PROMPT:')
    print(f'{analysis["baseline"][:200]}...')
    print(f'\nGENERATED TEXT:')
    print(f'{analysis["generated"]}')
    print(f'\nWHY THIS IS RECURSIVE:')
    if analysis['quality'] == 'high':
        print('  ✓ Strong recursive patterns detected')
        if analysis['patterns']['strange_loops'] > 0:
            print('  ✓ Contains strange loop/self-observation language')
        if analysis['patterns']['meta_language'] > 0:
            print('  ✓ Uses meta-language (references to the text/process itself)')
        if analysis['patterns']['self_reference'] > 0:
            print('  ✓ Contains self-referential structures')
    elif analysis['quality'] == 'medium':
        print('  ⚠ Moderate recursive patterns - may be borderline')
    else:
        print('  ✗ Weak or no recursive patterns - likely false positive')








