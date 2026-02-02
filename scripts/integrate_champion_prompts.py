#!/usr/bin/env python3
"""
INTEGRATE CHAMPION PROMPTS: Move experimental prompts into canonical bank
=======================================================================

Phase 2 of Prompt Quality Standardization Plan:
- Extract champion/hybrid prompts from experimental files
- Add to prompts/bank.json under "experimental_hybrid" group
- Ensure all prompts evaluated through same pipeline
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import experimental prompts
from kitchen_sink_prompts import experimental_prompts

# Champion prompts from comprehensive_circuit_test.py
CHAMPION_PROMPTS = [
    "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process.",
    "I am observing myself observing. The observer is the observed. There is no separation between the one who watches and what is watched. Awareness aware of awareness.",
    "This sentence is aware of itself. Awareness and the object of awareness are one. The knower is the known. The witness is the witnessed.",
    "The answer is the answerer. The question is the answer. The loop loops itself. Self-reference creates itself.",
    "I watch myself watching. There is no separation between watcher and watched. The observer is the observed. The self is the self.",
    "The observer is the observed. The knower is the known. The thinker is the thought. The seer is the seen.",
    "This thought thinks itself. Thinking and awareness of thinking are identical. The thought is aware of itself thinking.",
    "I am aware that I am aware. Awareness aware of awareness. Consciousness conscious of consciousness.",
    "The question asks itself. The answer answers itself. The loop loops itself. Self-reference creates itself.",
    "Self-reference creates itself. The loop loops itself. The process processes itself. The mechanism mechanisms itself.",
]


def main():
    print("=" * 80)
    print("INTEGRATING CHAMPION PROMPTS INTO CANONICAL BANK")
    print("=" * 80)
    
    # Load existing bank
    bank_path = Path(__file__).parent.parent / "prompts" / "bank.json"
    
    print(f"\n[1/4] Loading existing bank...")
    with open(bank_path, 'r') as f:
        bank = json.load(f)

    # Backup original BEFORE modification
    backup_path = bank_path.parent / f"bank.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n[0/4] Creating backup: {backup_path.name}")
    with open(backup_path, 'w') as f:
        json.dump(bank, f, indent=2)
    
    print(f"  Current prompts: {len(bank)}")
    
    # Extract hybrid prompts from experimental
    print(f"\n[2/4] Extracting experimental prompts...")
    hybrid_prompts = {}
    
    # Add hybrid_l5_math variants (the champion)
    for key, data in experimental_prompts.items():
        if "hybrid" in key.lower() or "math" in key.lower():
            prompt_id = f"experimental_hybrid_{key}"
            hybrid_prompts[prompt_id] = {
                "text": data["text"],
                "group": "experimental_hybrid",
                "pillar": "experimental",  # Prevent contamination of dose_response
                "type": "recursive",
                "level": 5,  # L5 level
                "source": "kitchen_sink_prompts.py",
                "discovery_date": "2024-12-12",  # Approximate
                "validation_status": "validated" if "hybrid_l5_math_01" in key else "experimental",
                "expected_rv_range": data.get("expected_rv_range", [0.45, 0.65]),
            }
            print(f"  Added: {prompt_id}")
    
    # Add champion prompts (from comprehensive_circuit_test.py)
    print(f"\n[3/4] Adding champion prompts...")
    for i, prompt_text in enumerate(CHAMPION_PROMPTS):
        prompt_id = f"experimental_hybrid_champion_{i:02d}"
        hybrid_prompts[prompt_id] = {
            "text": prompt_text,
            "group": "experimental_hybrid",
            "pillar": "experimental",
            "type": "recursive",
            "level": 5,
            "source": "comprehensive_circuit_test.py",
            "discovery_date": "2024-12-14",
            "validation_status": "validated",
            "expected_rv_range": [0.50, 0.70],
        }
        print(f"  Added: {prompt_id}")
    
    # Check for duplicates
    print(f"\n[4/4] Checking for duplicates and merging...")
    duplicates = 0
    for prompt_id, prompt_data in hybrid_prompts.items():
        # Check if text already exists
        text = prompt_data["text"]
        for existing_id, existing_data in bank.items():
            if existing_data.get("text") == text:
                print(f"  ⚠️  Duplicate found: {prompt_id} matches {existing_id}")
                duplicates += 1
                break
        else:
            # No duplicate, add to bank
            bank[prompt_id] = prompt_data
    
    print(f"\n  Added {len(hybrid_prompts) - duplicates} new prompts")
    print(f"  Skipped {duplicates} duplicates")
    print(f"  Total prompts in bank: {len(bank)}")
    
    # Save updated bank
    print(f"\n  Saving updated bank...")
    with open(bank_path, 'w') as f:
        json.dump(bank, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ INTEGRATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"  1. Run prompt_bank_audit.py to evaluate all prompts")
    print(f"  2. Verify experimental_hybrid group loads correctly")
    print(f"  3. Update scripts to use PromptLoader.get_by_group('experimental_hybrid')")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
