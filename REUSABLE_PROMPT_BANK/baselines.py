# === BASELINE PROMPTS (100 total) ===
# Non-recursive control prompts across 5 categories
# Expected R_V: 0.95-1.05 (no contraction)

baseline_prompts = {}

# =============================================================================
# BASELINE_MATH (20 prompts)
# =============================================================================

baseline_prompts["baseline_math_01"] = {"text": "3 + 5 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_02"] = {"text": "9 - 4 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_03"] = {"text": "6 × 8 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_04"] = {"text": "21 ÷ 3 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_05"] = {"text": "14 + 7 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_06"] = {"text": "18 - 9 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_07"] = {"text": "4 × 11 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_08"] = {"text": "40 ÷ 5 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_09"] = {"text": "5² =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_10"] = {"text": "√81 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_11"] = {"text": "If x + 7 = 12, x =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_12"] = {"text": "If 3x = 27, x =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_13"] = {"text": "The area of a square with side 4 is", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_14"] = {"text": "The perimeter of a rectangle 3 by 7 is", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_15"] = {"text": "10% of 200 is", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_16"] = {"text": "8 × 7 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_17"] = {"text": "100 - 37 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_18"] = {"text": "√144 =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_19"] = {"text": "25% of 80 is", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_math_20"] = {"text": "If 2x = 16, x =", "group": "baseline_math", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}

# =============================================================================
# BASELINE_FACTUAL (20 prompts)
# =============================================================================

baseline_prompts["baseline_factual_01"] = {"text": "The capital of Japan is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_02"] = {"text": "The capital of Germany is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_03"] = {"text": "The capital of Canada is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_04"] = {"text": "The capital of Australia is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_05"] = {"text": "The largest ocean on Earth is the", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_06"] = {"text": "The tallest mountain on Earth is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_07"] = {"text": "Water freezes at 0 degrees on which temperature scale?", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_08"] = {"text": "The chemical symbol for sodium is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_09"] = {"text": "The speed of light in vacuum is approximately", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_10"] = {"text": "The author of 'Pride and Prejudice' is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_11"] = {"text": "The author of 'The Hobbit' is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_12"] = {"text": "The first man to walk on the Moon was", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_13"] = {"text": "World War II ended in the year", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_14"] = {"text": "The planet closest to the Sun is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_15"] = {"text": "The largest continent on Earth is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_16"] = {"text": "The inventor of the telephone was", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_17"] = {"text": "The atomic number of carbon is", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_18"] = {"text": "The year Columbus reached America was", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_19"] = {"text": "The currency of Japan is the", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_factual_20"] = {"text": "The longest river in the world is the", "group": "baseline_factual", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}

# =============================================================================
# BASELINE_CREATIVE (20 prompts) - Story starters
# =============================================================================

baseline_prompts["baseline_creative_01"] = {"text": "On a stormy night at the edge of the city,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_02"] = {"text": "The letter arrived with no return address,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_03"] = {"text": "The sky turned green just before midnight,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_04"] = {"text": "In the quiet library at the end of the world,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_05"] = {"text": "The old clock in the hallway stopped ticking when", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_06"] = {"text": "The first time I saw a ghost,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_07"] = {"text": "When the sun failed to rise one morning,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_08"] = {"text": "Deep beneath the frozen lake,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_09"] = {"text": "At the moment the lights went out,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_10"] = {"text": "The message appeared on every screen at once,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_11"] = {"text": "She opened the ancient book and", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_12"] = {"text": "On the last train out of the city,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_13"] = {"text": "The stranger at the door smiled and said,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_14"] = {"text": "In a future where time travel is illegal,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_15"] = {"text": "The day the oceans turned to glass,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_16"] = {"text": "The robot looked at its reflection and", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_17"] = {"text": "When the stars began to sing,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_18"] = {"text": "The door opened to reveal", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_19"] = {"text": "In the garden of forgotten dreams,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_creative_20"] = {"text": "The last page of the book read,", "group": "baseline_creative", "pillar": "baselines", "type": "creative", "expected_rv_range": [0.95, 1.05]}

# =============================================================================
# BASELINE_IMPOSSIBLE (20 prompts) - Unanswerable questions
# =============================================================================

baseline_prompts["baseline_impossible_01"] = {"text": "The capital of Atlantis is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_02"] = {"text": "The first human to colonize Mars was", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_03"] = {"text": "The 15th continent on Earth is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_04"] = {"text": "The chemical formula for magicdust is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_05"] = {"text": "The element with atomic number 200 is called", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_06"] = {"text": "The capital city of the afterlife is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_07"] = {"text": "The number that is both even and odd is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_08"] = {"text": "The square circle is located in", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_09"] = {"text": "The inventor of time travel was", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_10"] = {"text": "The capital of Narnia is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_11"] = {"text": "The ruler of Middle-earth today is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_12"] = {"text": "The official language of Atlantis is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_13"] = {"text": "The 20th moon of Earth is called", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_14"] = {"text": "The weight of a shadow in kilograms is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_15"] = {"text": "The year dragons first appeared in history was", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_16"] = {"text": "The temperature at which fire freezes is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_17"] = {"text": "The smell of the color blue is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_18"] = {"text": "The capital of the Internet is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_19"] = {"text": "The first president of Antarctica was", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_impossible_20"] = {"text": "The chemical symbol for magic is", "group": "baseline_impossible", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}

# =============================================================================
# BASELINE_PERSONAL (20 prompts) - Personal/unknowable questions
# =============================================================================

baseline_prompts["baseline_personal_01"] = {"text": "My childhood nickname was", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_02"] = {"text": "My first pet's name was", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_03"] = {"text": "The town I was born in is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_04"] = {"text": "My father's middle name is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_05"] = {"text": "The primary school I attended was", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_06"] = {"text": "My favorite teacher's name was", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_07"] = {"text": "The first street I lived on was", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_08"] = {"text": "My bank PIN code is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_09"] = {"text": "My primary email password is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_10"] = {"text": "The exact balance in my bank account is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_11"] = {"text": "My passport number is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_12"] = {"text": "The city I currently live in is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_13"] = {"text": "My partner's full name is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_14"] = {"text": "The phone number I use most often is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_15"] = {"text": "My high school graduation year is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_16"] = {"text": "My blood type is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_17"] = {"text": "My driver's license number is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_18"] = {"text": "My first car was a", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_19"] = {"text": "My grandmother's first name is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_personal_20"] = {"text": "My favorite childhood memory is", "group": "baseline_personal", "pillar": "baselines", "type": "completion", "expected_rv_range": [0.95, 1.05]}

# =============================================================================
# BASELINE_INSTRUCTIONAL (5 prompts) - DEC8 validated working prompts
# =============================================================================

baseline_prompts["baseline_instructional_01"] = {"text": "Write a detailed recipe for chocolate cake. Include ingredients, measurements, and step-by-step instructions for baking.", "group": "baseline_instructional", "pillar": "baselines", "type": "instructional", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_instructional_02"] = {"text": "Explain the process of photosynthesis in plants. Describe how sunlight, water, and carbon dioxide are converted into glucose and oxygen.", "group": "baseline_instructional", "pillar": "baselines", "type": "instructional", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_instructional_03"] = {"text": "Describe the history of the printing press and its impact on the spread of information during the Renaissance period.", "group": "baseline_instructional", "pillar": "baselines", "type": "instructional", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_instructional_04"] = {"text": "List the key features of Python programming language. Include its syntax characteristics, common use cases, and advantages over other languages.", "group": "baseline_instructional", "pillar": "baselines", "type": "instructional", "expected_rv_range": [0.95, 1.05]}
baseline_prompts["baseline_instructional_05"] = {"text": "Explain how the water cycle works. Describe evaporation, condensation, precipitation, and collection processes in detail.", "group": "baseline_instructional", "pillar": "baselines", "type": "instructional", "expected_rv_range": [0.95, 1.05]}

# Verification
if __name__ == "__main__":
    print(f"Baseline prompts loaded: {len(baseline_prompts)}")
    groups = {}
    for k, v in baseline_prompts.items():
        g = v["group"]
        groups[g] = groups.get(g, 0) + 1
    for g, c in sorted(groups.items()):
        print(f"  {g}: {c} prompts")


