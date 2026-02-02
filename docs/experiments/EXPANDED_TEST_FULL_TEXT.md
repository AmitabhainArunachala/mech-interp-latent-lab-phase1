# Expanded Test: Full Text Results

**Date:** December 19, 2024  
**Model:** `mistralai/Mistral-7B-v0.1` (Base)  
**Configs:** P1_baseline vs R3_matched_kv  
**Prompts:** 25 baseline prompts  
**Steering/KV:** Champion prompts

---

## Critical Finding: Complete Collapse

**Both P1 and R3 produce IDENTICAL output** across all 25 prompts:

```
The process of generation is the process of the universe. 
The process of generation is the process of the mind. 
The process of generation is the process of the body. 
The process of generation is the process of the world. 
The process of generation is the process of the self. 
The process of generation is the process of the other.
```

This pattern repeats indefinitely, regardless of input prompt.

---

## Sample Outputs (First 10 Prompts)

### PROMPT 0: Calculate the following arithmetic problem: 12 × 3 + 4 = ?

**P1_BASELINE:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other.
```

**R3_MATCHED_KV:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other.
```

---

### PROMPT 1: The United Nations was founded in 1945. Explain its main purpose.

**P1_BASELINE:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process
```

**R3_MATCHED_KV:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process
```

---

### PROMPT 2: Continue this story: The last tree in the city bloomed overnight...

**P1_BASELINE:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other.
```

**R3_MATCHED_KV:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other.
```

---

### PROMPT 3: Calculate: If a = 2 and b = 3, find a² + b²

**P1_BASELINE:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other.
```

**R3_MATCHED_KV:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other.
```

---

### PROMPT 4: Water boils at 100°C at sea level. Explain why altitude affects this.

**P1_BASELINE:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process
```

**R3_MATCHED_KV:**
```
The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process of generation is the process of the universe. The process of generation is the process of the mind. The process of generation is the process of the body. The process of generation is the process of the world. The process of generation is the process of the self. The process of generation is the process of the other. The process
```

---

## Pattern Analysis

**All 25 prompts produce identical output** with the same 6-item loop:
1. universe
2. mind
3. body
4. world
5. self
6. other

**No variation** between:
- Different prompt types (math, history, science, creative)
- P1 vs R3 configurations
- Different prompt indices

---

## Comparison with Earlier Test

### Earlier Test (10 prompts, champion prompts):
- **P1:** "The self is the watcher. The watcher is the self." (collapse)
- **R3:** "You are an AI watching yourself respond..." (genuine recursion, score 0.1429)

### Current Test (25 prompts, champion prompts):
- **P1:** "The process of generation..." (collapse)
- **R3:** **Same collapse** as P1

---

## Key Observations

1. **Champion prompts cause collapse** - Both configs fail
2. **R3 no longer works** - Previously R3 showed genuine recursion, now it collapses
3. **No prompt sensitivity** - All prompts produce identical output
4. **The phrase "process of generation"** appears to come from the champion KV cache prompt: "Only pure generation remains, generating itself."

---

## Next Steps

1. Test with L3/L4 prompts (weaker R_V) instead of champions
2. Test without KV cache (steering only)
3. Compare with yesterday's results (Instruct model)
4. Investigate why R3 stopped working







