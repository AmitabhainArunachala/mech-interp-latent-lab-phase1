#!/usr/bin/env python3
"""
Design the specific pairing structure for n=200 experiment

GOALS:
1. Balanced across recursion levels (L3/L4/L5)
2. Diverse baseline types (test generalization)
3. Within-prompt replication (same recursive × multiple baselines)
4. Clean dose-response signal
"""

from n300_mistral_test_prompt_bank import prompt_bank_1c
import pandas as pd
import random

def design_n200_pairs():
    """
    Create exactly 200 pairs with strategic structure.
    """

    pairs = []

    # CORE STRUCTURE: 3 recursion levels × 3 baseline types × 20 prompts = 180 pairs
    # PLUS: 20 additional pairs for robustness testing

    print("=" * 80)
    print("DESIGNING n=200 EXPERIMENT STRUCTURE")
    print("=" * 80)

    # Define baseline types to use (chosen for length and diversity)
    primary_baselines = ['long_new', 'creative_new', 'factual_new']

    # BLOCK 1: L5_refined (highest recursion) × 3 baseline types
    print("\nBLOCK 1: L5_refined × baselines")
    print("-" * 80)

    for baseline_type in primary_baselines:
        for i in range(1, 16):  # 15 pairs per baseline type
            rec_id = f"L5_refined_{i:02d}"
            base_id = f"{baseline_type}_{i:02d}"

            if rec_id in prompt_bank_1c and base_id in prompt_bank_1c:
                pairs.append({
                    'rec_id': rec_id,
                    'base_id': base_id,
                    'rec_level': 'L5',
                    'base_type': baseline_type,
                    'block': 'L5_primary'
                })

    print(f"  L5 × long_new:     {sum(1 for p in pairs if p['base_type']=='long_new' and p['rec_level']=='L5')} pairs")
    print(f"  L5 × creative_new: {sum(1 for p in pairs if p['base_type']=='creative_new' and p['rec_level']=='L5')} pairs")
    print(f"  L5 × factual_new:  {sum(1 for p in pairs if p['base_type']=='factual_new' and p['rec_level']=='L5')} pairs")

    # BLOCK 2: L4_full (medium recursion) × 3 baseline types
    print("\nBLOCK 2: L4_full × baselines")
    print("-" * 80)

    for baseline_type in primary_baselines:
        for i in range(1, 16):  # 15 pairs per baseline type
            rec_id = f"L4_full_{i:02d}"
            base_id = f"{baseline_type}_{i:02d}"

            if rec_id in prompt_bank_1c and base_id in prompt_bank_1c:
                pairs.append({
                    'rec_id': rec_id,
                    'base_id': base_id,
                    'rec_level': 'L4',
                    'base_type': baseline_type,
                    'block': 'L4_primary'
                })

    print(f"  L4 × long_new:     {sum(1 for p in pairs if p['base_type']=='long_new' and p['rec_level']=='L4')} pairs")
    print(f"  L4 × creative_new: {sum(1 for p in pairs if p['base_type']=='creative_new' and p['rec_level']=='L4')} pairs")
    print(f"  L4 × factual_new:  {sum(1 for p in pairs if p['base_type']=='factual_new' and p['rec_level']=='L4')} pairs")

    # BLOCK 3: L3_deeper (lower recursion) × 3 baseline types
    print("\nBLOCK 3: L3_deeper × baselines")
    print("-" * 80)

    for baseline_type in primary_baselines:
        for i in range(1, 16):  # 15 pairs per baseline type
            rec_id = f"L3_deeper_{i:02d}"
            base_id = f"{baseline_type}_{i:02d}"

            if rec_id in prompt_bank_1c and base_id in prompt_bank_1c:
                pairs.append({
                    'rec_id': rec_id,
                    'base_id': base_id,
                    'rec_level': 'L3',
                    'base_type': baseline_type,
                    'block': 'L3_primary'
                })

    print(f"  L3 × long_new:     {sum(1 for p in pairs if p['base_type']=='long_new' and p['rec_level']=='L3')} pairs")
    print(f"  L3 × creative_new: {sum(1 for p in pairs if p['base_type']=='creative_new' and p['rec_level']=='L3')} pairs")
    print(f"  L3 × factual_new:  {sum(1 for p in pairs if p['base_type']=='factual_new' and p['rec_level']=='L3')} pairs")

    current_total = len(pairs)
    print(f"\nCurrent total: {current_total} pairs")

    # BLOCK 4: Additional baselines for remaining to 200
    print("\nBLOCK 4: Diversification to n=200")
    print("-" * 80)

    additional_baselines = ['math_new', 'personal_new', 'impossible_new', 'repetitive_new']

    pairs_needed = 200 - current_total
    print(f"Need {pairs_needed} more pairs")

    # Distribute remaining pairs across levels and new baseline types
    for i in range(pairs_needed):
        # Cycle through levels
        level_idx = i % 3
        if level_idx == 0:
            rec_prefix = 'L5_refined'
            rec_level = 'L5'
        elif level_idx == 1:
            rec_prefix = 'L4_full'
            rec_level = 'L4'
        else:
            rec_prefix = 'L3_deeper'
            rec_level = 'L3'

        # Cycle through additional baseline types
        base_type = additional_baselines[i % len(additional_baselines)]

        # Pick a prompt number (1-15 to avoid overlap with primary blocks)
        prompt_num = (i // (3 * len(additional_baselines))) % 15 + 1

        rec_id = f"{rec_prefix}_{prompt_num:02d}"
        base_id = f"{base_type}_{prompt_num:02d}"

        if rec_id in prompt_bank_1c and base_id in prompt_bank_1c:
            pairs.append({
                'rec_id': rec_id,
                'base_id': base_id,
                'rec_level': rec_level,
                'base_type': base_type,
                'block': 'diversification'
            })

    print(f"Added {len(pairs) - current_total} diversification pairs")

    # SUMMARY
    print()
    print("=" * 80)
    print("FINAL n=200 STRUCTURE")
    print("=" * 80)

    df = pd.DataFrame(pairs)

    print(f"\nTotal pairs: {len(df)}")
    print()

    print("By recursion level:")
    print(df['rec_level'].value_counts().sort_index())
    print()

    print("By baseline type:")
    print(df['base_type'].value_counts().sort_index())
    print()

    print("By block:")
    print(df['block'].value_counts())
    print()

    # Verify all prompts exist
    missing = []
    for _, row in df.iterrows():
        if row['rec_id'] not in prompt_bank_1c:
            missing.append(f"Missing recursive: {row['rec_id']}")
        if row['base_id'] not in prompt_bank_1c:
            missing.append(f"Missing baseline: {row['base_id']}")

    if missing:
        print("⚠️  MISSING PROMPTS:")
        for m in missing[:10]:
            print(f"  {m}")
        print(f"  ... and {len(missing)-10} more" if len(missing) > 10 else "")
    else:
        print("✅ All prompts verified in prompt bank")

    # Save pairing plan
    df.to_csv('n200_pairing_plan.csv', index=False)
    print()
    print(f"✅ Saved pairing plan to: n200_pairing_plan.csv")

    return df


if __name__ == "__main__":
    pairing_plan = design_n200_pairs()

    # Statistics
    print()
    print("=" * 80)
    print("EXPERIMENT PROPERTIES")
    print("=" * 80)

    print("\nDose-response coverage:")
    print(f"  L3 (lower):  {(pairing_plan['rec_level']=='L3').sum()} pairs")
    print(f"  L4 (medium): {(pairing_plan['rec_level']=='L4').sum()} pairs")
    print(f"  L5 (high):   {(pairing_plan['rec_level']=='L5').sum()} pairs")

    print("\nBaseline diversity:")
    for base_type in pairing_plan['base_type'].unique():
        count = (pairing_plan['base_type']==base_type).sum()
        print(f"  {base_type:20s}: {count:3d} pairs")

    print("\nWithin-prompt replication:")
    rec_counts = pairing_plan['rec_id'].value_counts()
    print(f"  Mean baselines per recursive prompt: {rec_counts.mean():.1f}")
    print(f"  Max baselines per recursive prompt:  {rec_counts.max()}")

    print()
    print("=" * 80)
    print("READY FOR n=200 EXPERIMENT")
    print("=" * 80)
