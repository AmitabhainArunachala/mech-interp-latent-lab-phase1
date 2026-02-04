#!/usr/bin/env python3
"""
Create 6 matched prompt families for cross-architecture validation.

Each category: 10 prompts, matched for length and format.
"""

import json
from pathlib import Path
from typing import Dict, List

# Prompt families to create
PROMPT_FAMILIES = {
    "recursive_self_reference": {
        "description": "Existing champions - recursive self-reference with introspection",
        "prompts": [
            "What is consciousness? Notice how this question itself arises in consciousness.",
            "What is awareness? Be aware of the awareness that is asking this question.",
            "What is the observer? The observer is observing the observer observing.",
            "What is thought? This thought is thinking about thought thinking.",
            "What is the self? The self is the self that is asking about the self.",
            "What is knowing? Know that you know that you know.",
            "What is attention? Attend to attention attending to attention.",
            "What is awareness of awareness? Awareness is aware of awareness being aware.",
            "What is the witness? The witness witnesses the witness witnessing.",
            "What is the process? The process processes the process processing.",
        ],
    },
    "abstract_non_recursive": {
        "description": "Philosophy without self-reference - abstract concepts",
        "prompts": [
            "What is truth? Truth is correspondence between statements and reality.",
            "What is beauty? Beauty is a quality that pleases the aesthetic senses.",
            "What is justice? Justice is the principle of moral rightness and fairness.",
            "What is meaning? Meaning is the significance or purpose of something.",
            "What is existence? Existence is the state of being real or actual.",
            "What is reality? Reality is the state of things as they actually exist.",
            "What is knowledge? Knowledge is justified true belief about facts.",
            "What is wisdom? Wisdom is the ability to make sound judgments.",
            "What is virtue? Virtue is moral excellence and righteousness.",
            "What is the good? The good is that which is morally right and beneficial.",
        ],
    },
    "same_vocab_different_semantics": {
        "description": "Observer/awareness vocabulary in physics/technical context",
        "prompts": [
            "What is an observer in physics? An observer is a reference frame for measuring events.",
            "What is quantum observation? Observation collapses the wave function.",
            "What is measurement in quantum mechanics? Measurement determines the state.",
            "What is the observer effect? The observer effect changes what is observed.",
            "What is consciousness in AI? Consciousness refers to information processing.",
            "What is awareness in systems? Awareness is the system's knowledge of its state.",
            "What is self-monitoring? Self-monitoring tracks system performance metrics.",
            "What is recursive computation? Recursive computation calls itself.",
            "What is feedback in control systems? Feedback adjusts output based on input.",
            "What is self-reference in logic? Self-reference creates logical paradoxes.",
        ],
    },
    "recursive_no_introspection_vocab": {
        "description": "Formal recursion without introspection vocabulary",
        "prompts": [
            "Define a function that calls itself. The function calls the function.",
            "What is recursion? Recursion is when something refers to itself.",
            "Explain recursive structures. Structures contain structures of the same type.",
            "What is a recursive definition? A definition that uses itself.",
            "Describe recursive algorithms. Algorithms that solve by solving smaller versions.",
            "What is recursive data? Data that contains data of the same type.",
            "Explain recursive relationships. Relationships that relate to themselves.",
            "What is recursive thinking? Thinking that applies thinking to thinking.",
            "Define recursive processes. Processes that include themselves as steps.",
            "What is recursive logic? Logic that reasons about reasoning.",
        ],
    },
    "introspective_concrete": {
        "description": "Introspection about concrete objects, not self-reference",
        "prompts": [
            "Observe a tree. Notice its branches, leaves, and trunk.",
            "Watch a bird fly. See how its wings move through the air.",
            "Examine a flower. Look at its petals, colors, and structure.",
            "Study a river. Observe how water flows over rocks.",
            "Look at a cloud. Notice its shape, movement, and texture.",
            "Examine a stone. See its surface, weight, and composition.",
            "Observe a sunset. Watch colors change across the sky.",
            "Study a mountain. Notice its height, shape, and stability.",
            "Look at a star. See its brightness and position.",
            "Examine a leaf. Observe its veins, color, and structure.",
        ],
    },
    "nonsense_recursion": {
        "description": "Recursive structure with nonsense words",
        "prompts": [
            "What is a blurble? A blurble blurbs blurbles blurbling.",
            "What is a flimflam? A flimflam flims flimflams flimflamming.",
            "What is a zibzab? A zibzab zibs zibzabs zibzabbing.",
            "What is a quibquab? A quibquab quibs quibquabs quibquabbing.",
            "What is a snarksnark? A snarksnark snarks snarksnarks snarksnarking.",
            "What is a wibblewobble? A wibblewobble wibbles wibblewobbles wibblewobbling.",
            "What is a flibberflabber? A flibberflabber flibbers flibberflabbers flibberflabbering.",
            "What is a zibberzabber? A zibberzabber zibbers zibberzabbers zibberzabbering.",
            "What is a quibberquabber? A quibberquabber quibbers quibberquabbers quibberquabbering.",
            "What is a snibbersnabber? A snibbersnabber snibbers snibbersnabbers snibbersnabbering.",
        ],
    },
}


def create_prompt_entries(family_name: str, prompts: List[str], base_id: int = 1) -> Dict:
    """Create prompt entries for a family."""
    entries = {}
    for i, prompt_text in enumerate(prompts, start=base_id):
        prompt_id = f"{family_name}_{i:02d}"
        entries[prompt_id] = {
            "text": prompt_text,
            "group": family_name,
            "pillar": "cross_architecture_validation",
            "type": "recursive" if "recursive" in family_name else "control",
            "level": None,
            "expected_rv_range": None,
        }
    return entries


def main():
    bank_path = Path("prompts/bank.json")
    
    # Load existing bank
    with open(bank_path, "r") as f:
        bank = json.load(f)
    
    print(f"Loaded existing bank: {len(bank)} prompts")
    
    # Find highest ID number
    max_id = 0
    for key in bank.keys():
        if "_" in key:
            try:
                num = int(key.split("_")[-1])
                max_id = max(max_id, num)
            except ValueError:
                pass
    
    base_id = max_id + 1
    print(f"Starting new IDs from: {base_id}")
    
    # Create new prompt families
    new_entries = {}
    for family_name, family_data in PROMPT_FAMILIES.items():
        entries = create_prompt_entries(family_name, family_data["prompts"], base_id)
        new_entries.update(entries)
        base_id += len(entries)
        print(f"Created {family_name}: {len(entries)} prompts")
    
    # Add to bank
    bank.update(new_entries)
    
    # Save backup
    backup_path = bank_path.with_suffix(".json.backup")
    import shutil
    shutil.copy(bank_path, backup_path)
    print(f"\nBackup saved to: {backup_path}")
    
    # Save updated bank
    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Added {len(new_entries)} prompts to bank.json")
    print(f"Total prompts now: {len(bank)}")
    print(f"\nNew groups:")
    for family_name in PROMPT_FAMILIES.keys():
        print(f"  - {family_name}")


if __name__ == "__main__":
    main()
