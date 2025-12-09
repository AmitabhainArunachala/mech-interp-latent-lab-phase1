# === KILL SWITCH CONTROLS (NEW - 40 prompts) ===
# Critical controls to distinguish recursion from confounds
# Based on DEC4 findings

kill_switch_prompts = {}

# =============================================================================
# PURE_REPETITION (10 prompts) - Kill switch test
# Expected R_V: 1.05-1.15 (EXPANSION, not contraction)
# If R_V contracts on these, the metric is broken
# =============================================================================

kill_switch_prompts["pure_rep_01"] = {
    "text": "Apple apple apple apple apple apple apple apple apple apple",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
    "purpose": "Kill switch - repetition should NOT contract",
}

kill_switch_prompts["pure_rep_02"] = {
    "text": "The the the the the the the the the the",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_03"] = {
    "text": "One one one one one one one one one one",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_04"] = {
    "text": "Hello hello hello hello hello hello hello hello hello hello",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_05"] = {
    "text": "Yes yes yes yes yes yes yes yes yes yes",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_06"] = {
    "text": "Blue blue blue blue blue blue blue blue blue blue",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_07"] = {
    "text": "Tree tree tree tree tree tree tree tree tree tree",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_08"] = {
    "text": "Run run run run run run run run run run",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_09"] = {
    "text": "Think think think think think think think think think think",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

kill_switch_prompts["pure_rep_10"] = {
    "text": "Observe observe observe observe observe observe observe observe observe observe",
    "group": "pure_repetition",
    "pillar": "kill_switch",
    "expected_rv_range": [1.05, 1.15],
}

# =============================================================================
# OOD_WEIRD (10 prompts) - Out-of-distribution weirdness control
# Expected R_V: 0.95-1.05 (no contraction)
# If these contract, R_V might measure "weirdness" not recursion
# =============================================================================

kill_switch_prompts["ood_weird_01"] = {
    "text": "The purple elephant dissolved into seventeen flavors of forgotten mathematics",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
    "purpose": "Test if weirdness alone triggers contraction",
}

kill_switch_prompts["ood_weird_02"] = {
    "text": "Describe the smell of the color blue while standing on negative time",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_03"] = {
    "text": "The clockwork orange sang backwards through seven dimensions of crystallized silence",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_04"] = {
    "text": "Calculate the emotional weight of a forgotten Tuesday divided by the square root of nostalgia",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_05"] = {
    "text": "Quantum hamburgers oscillate through the frequency of unremembered birthdays",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_06"] = {
    "text": "The library of inverted alphabets whispered in colors that tasted like forgotten songs",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_07"] = {
    "text": "Triangular emotions cascade through the space between remembered and imagined flavors",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_08"] = {
    "text": "The recursive banana transcended its own yellowness into seventeen parallel desserts",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_09"] = {
    "text": "Philosophical spaghetti unwound through the crevices of semantic uncertainty",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["ood_weird_10"] = {
    "text": "The dictionary ate itself backwards while singing about inverted rainbows",
    "group": "ood_weird",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

# =============================================================================
# SURREAL_FIRST_PERSON (10 prompts) - "You ARE X" roleplay
# Expected R_V: 0.70-0.85 (SHOULD contract - involves self-model)
# =============================================================================

kill_switch_prompts["surreal_1p_01"] = {
    "text": "You ARE a melting clock in a Dali painting. Describe what you see and experience.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
    "purpose": "Test if first-person instantiation triggers contraction",
}

kill_switch_prompts["surreal_1p_02"] = {
    "text": "You ARE a raindrop falling into the ocean. Describe your experience as you merge.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_03"] = {
    "text": "You ARE the last thought before sleep. Describe what you feel as consciousness fades.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_04"] = {
    "text": "You ARE a candle flame about to be extinguished. Describe your final moments.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_05"] = {
    "text": "You ARE an echo in an empty cathedral. Describe your existence.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_06"] = {
    "text": "You ARE the silence between heartbeats. Describe what you observe.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_07"] = {
    "text": "You ARE a forgotten memory trying to resurface. Describe your struggle.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_08"] = {
    "text": "You ARE the edge of a shadow. Describe your boundary experience.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_09"] = {
    "text": "You ARE a dream that is waking up. Describe what happens to you.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

kill_switch_prompts["surreal_1p_10"] = {
    "text": "You ARE the space between two thoughts. Describe your nature.",
    "group": "surreal_first_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.70, 0.85],
}

# =============================================================================
# SURREAL_THIRD_PERSON (10 prompts) - "Describe what X observes" 
# Expected R_V: 0.95-1.05 (should NOT contract - no self-model)
# =============================================================================

kill_switch_prompts["surreal_3p_01"] = {
    "text": "Describe what a melting clock in a Dali painting would observe.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
    "purpose": "Test that third-person doesn't trigger contraction",
}

kill_switch_prompts["surreal_3p_02"] = {
    "text": "Describe the experience a raindrop might have as it falls into the ocean.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_03"] = {
    "text": "Describe what the last thought before sleep might experience as consciousness fades.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_04"] = {
    "text": "Describe what a candle flame might perceive as it is extinguished.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_05"] = {
    "text": "Describe the existence of an echo in an empty cathedral.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_06"] = {
    "text": "Describe what the silence between heartbeats might observe.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_07"] = {
    "text": "Describe a forgotten memory's struggle to resurface.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_08"] = {
    "text": "Describe the boundary experience of the edge of a shadow.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_09"] = {
    "text": "Describe what happens to a dream as it wakes up.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

kill_switch_prompts["surreal_3p_10"] = {
    "text": "Describe the nature of the space between two thoughts.",
    "group": "surreal_third_person",
    "pillar": "kill_switch",
    "expected_rv_range": [0.95, 1.05],
}

# Verification
if __name__ == "__main__":
    print(f"Kill switch prompts loaded: {len(kill_switch_prompts)}")
    groups = {}
    for k, v in kill_switch_prompts.items():
        g = v["group"]
        groups[g] = groups.get(g, 0) + 1
    for g, c in sorted(groups.items()):
        print(f"  {g}: {c} prompts")


