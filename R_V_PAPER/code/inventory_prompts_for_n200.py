#!/usr/bin/env python3
"""
Inventory available prompts to design n=200 experiment structure
"""

from n300_mistral_test_prompt_bank import prompt_bank_1c
from collections import defaultdict
import pandas as pd

def inventory_prompts():
    """
    Analyze prompt bank to determine available prompt types and counts.
    """

    # Group prompts by prefix
    groups = defaultdict(list)

    for prompt_id, prompt_data in prompt_bank_1c.items():
        # Extract prefix (everything before the number)
        # e.g., "L5_refined_01" -> "L5_refined"
        parts = prompt_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
        else:
            # Handle other formats
            prefix = prompt_id.split('_')[0] if '_' in prompt_id else prompt_id

        groups[prefix].append(prompt_id)

    # Print summary
    print("=" * 80)
    print("PROMPT BANK INVENTORY")
    print("=" * 80)
    print(f"Total prompts: {len(prompt_bank_1c)}")
    print()

    # Sort by group name
    sorted_groups = sorted(groups.items(), key=lambda x: x[0])

    print("Breakdown by type:")
    print("-" * 80)

    recursive_types = []
    baseline_types = []

    for prefix, prompt_ids in sorted_groups:
        count = len(prompt_ids)
        print(f"  {prefix:30s}: {count:3d} prompts")

        # Categorize
        if any(x in prefix.lower() for x in ['l3', 'l4', 'l5', 'recursive']):
            recursive_types.append((prefix, count))
        else:
            baseline_types.append((prefix, count))

    print()
    print("=" * 80)
    print("RECURSIVE PROMPTS (sources for patching)")
    print("=" * 80)
    for prefix, count in recursive_types:
        print(f"  {prefix:30s}: {count:3d}")

    recursive_total = sum(count for _, count in recursive_types)
    print(f"  {'TOTAL':30s}: {recursive_total:3d}")

    print()
    print("=" * 80)
    print("BASELINE PROMPTS (targets for patching)")
    print("=" * 80)
    for prefix, count in baseline_types:
        print(f"  {prefix:30s}: {count:3d}")

    baseline_total = sum(count for _, count in baseline_types)
    print(f"  {'TOTAL':30s}: {baseline_total:3d}")

    print()
    print("=" * 80)
    print("PAIRING STRATEGY FOR n=200")
    print("=" * 80)

    # Design pairing strategy
    print("\nOption 1: BALANCED ACROSS RECURSION LEVELS")
    print("-" * 80)

    # Find counts for each recursion level
    l3_count = sum(count for prefix, count in recursive_types if 'l3' in prefix.lower())
    l4_count = sum(count for prefix, count in recursive_types if 'l4' in prefix.lower())
    l5_count = sum(count for prefix, count in recursive_types if 'l5' in prefix.lower())

    print(f"L3 prompts available: {l3_count}")
    print(f"L4 prompts available: {l4_count}")
    print(f"L5 prompts available: {l5_count}")
    print()

    # Calculate max pairs per level if we want balance
    min_recursive = min(l3_count, l4_count, l5_count)
    print(f"For balanced design: {min_recursive} pairs × 3 levels = {min_recursive * 3} total")

    if min_recursive * 3 < 200:
        print(f"⚠️  Not enough for n=200 with perfect balance")
        print(f"Need to oversample or use multiple baseline types per recursive prompt")

    print()
    print("Option 2: MAXIMIZE SAMPLE SIZE")
    print("-" * 80)

    # For each recursive prompt, pair with multiple baselines
    max_possible = recursive_total * min(baseline_total, 10)  # Cap at 10 baselines per recursive
    print(f"Max possible pairs: {max_possible}")
    print()

    print("Recommended strategy for n=200:")
    print("  • Use all recursive prompts (L3, L4, L5)")
    print("  • Pair each with 2-3 different baseline types")
    print("  • Stratify by recursion level for dose-response")
    print()

    # Specific pairing plan
    print("CONCRETE PAIRING PLAN:")
    print("-" * 80)

    # Get specific baseline types with sufficient counts
    good_baselines = [(prefix, count) for prefix, count in baseline_types if count >= 15]

    print(f"\nBaseline types with ≥15 prompts:")
    for prefix, count in good_baselines:
        print(f"  {prefix}: {count}")

    print()
    print("Proposed allocation:")
    print("  L5_refined × long_new:      15 pairs")
    print("  L5_refined × creative_new:  15 pairs")
    print("  L5_refined × technical:     15 pairs")
    print("  L4_full × long_new:         15 pairs")
    print("  L4_full × creative_new:     15 pairs")
    print("  L4_full × technical:        15 pairs")
    print("  L3_deeper × long_new:       15 pairs")
    print("  L3_deeper × creative_new:   15 pairs")
    print("  L3_deeper × technical:      15 pairs")
    print("  Varied × varied:            65 pairs (fill to 200)")
    print("  " + "-" * 40)
    print("  TOTAL:                     200 pairs")

    print()
    print("=" * 80)

    # Return data for further analysis
    return {
        'groups': dict(groups),
        'recursive_types': recursive_types,
        'baseline_types': baseline_types,
        'good_baselines': good_baselines
    }


if __name__ == "__main__":
    inventory = inventory_prompts()

    # Save to CSV for reference
    all_prompts = []
    for prompt_id, prompt_data in prompt_bank_1c.items():
        all_prompts.append({
            'prompt_id': prompt_id,
            'group': prompt_data.get('group', 'unknown'),
            'text_length': len(prompt_data.get('text', ''))
        })

    df = pd.DataFrame(all_prompts)
    df.to_csv('prompt_inventory.csv', index=False)
    print(f"\n✅ Saved detailed inventory to: prompt_inventory.csv")
