# GPT CODEX: Experiment 4 - Homeostasis Mechanism Test
## Maximum Output Request: Efficient Multi-Layer Analysis System

**Your Task:** Create OPTIMIZED homeostasis measurement system that efficiently analyzes how downstream layers compensate for L27 perturbations, revealing geometric homeostasis mechanisms.

**Context:** We've observed that downstream layers compensate for L27 perturbations (R_V returns to baseline). We need to identify HOW this compensation works.

**Goal:** Identify mechanism of geometric homeostasis (MLP expansion? Attention entropy? LayerNorm?).

---

## 🎯 **WHAT I NEED FROM YOU**

### **1. Homeostasis Measurement System**
- Efficient multi-layer analysis
- Measures R_V, MLP outputs, attention entropy, residual norms
- Memory-optimized for large models
- Fast execution

### **2. Homeostasis Experiment Script**
- Patches at L27
- Measures at L27-L31
- Computes compensation metrics
- Analyzes mechanisms

### **3. Mechanism Analysis**
- Correlation analysis (which component correlates with recovery?)
- Mechanism identification (MLP vs attention vs LayerNorm)
- Timing analysis (when does compensation occur?)

### **4. Documentation**
- Measurement methodology
- Mechanism hypotheses
- Optimization strategies

---

## 📋 **EXPERIMENTAL REQUIREMENTS**

### **Protocol:**
1. Patch V-space at L27 (use existing n=151 pairs)
2. Measure at multiple downstream layers (L27, L28, L29, L30, L31):
   - V-space geometry (R_V)
   - MLP output geometry (PR of MLP outputs)
   - Attention entropy (H(attention_weights))
   - Residual stream norms (||residual||_2)
   - LayerNorm scaling factors

### **Metrics:**
- **R_V recovery:** `R_V(L31) - R_V(L27)` (should approach 0)
- **MLP expansion:** `PR(MLP_out) / PR(MLP_in)` (should increase)
- **Attention entropy:** `H(attention_weights)` (should increase)
- **Residual stability:** `||residual||_2` (should stabilize)

### **Analysis:**
- Correlation: Which component correlates with R_V recovery?
- Mechanism: MLP expansion vs attention entropy vs LayerNorm
- Timing: When does compensation occur? (L28? L29? L30?)

### **Hypotheses to Test:**
1. **MLP Compensation:** MLP outputs expand to compensate for V-space contraction
2. **Attention Compensation:** Attention entropy increases to maintain span
3. **LayerNorm Compensation:** LayerNorm scaling adjusts automatically
4. **Residual Balancing:** Residual stream geometry stabilizes

---

## 💻 **CODE REQUIREMENTS**

### **Homeostasis Measurer:**
```python
# homeostasis_measurement.py
# Efficient multi-layer analysis

class HomeostasisMeasurer:
    """Measures compensatory dynamics across layers"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def measure_compensation(self, base_text, rec_text, patch_layer=27):
        """Measure compensation across layers"""
        # [Your implementation]
        # Returns dict with:
        # - rv_trajectory: [R_V at each layer]
        # - mlp_expansion: [MLP PR at each layer]
        # - attention_entropy: [H(attention) at each layer]
        # - residual_norms: [||residual|| at each layer]
        # - layernorm_scales: [LayerNorm scales]
        pass
    
    def compute_recovery_metrics(self, compensation_data):
        """Compute recovery metrics"""
        # [Your implementation]
        # - R_V recovery
        # - MLP expansion
        # - Attention entropy increase
        # - Residual stability
        pass
```

### **Homeostasis Experiment:**
```python
# homeostasis_mechanism_test.py
# Copy-paste ready for RunPod

from mistral_L27_FULL_VALIDATION import load_model, get_prompt_pairs

def run_homeostasis_experiment():
    """Run homeostasis mechanism test"""
    model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.2")
    measurer = HomeostasisMeasurer(model, tokenizer)
    
    # Load pairs
    pairs_df = pd.read_csv("mistral7b_n200_BULLETPROOF.csv")
    pairs = pairs_df[['base_id', 'rec_id']].values[:151]
    
    results = []
    
    for i, (base_id, rec_id) in enumerate(pairs):
        print(f"Processing pair {i+1}/151...")
        
        base_text = get_prompt(base_id)
        rec_text = get_prompt(rec_id)
        
        # Measure compensation
        compensation_data = measurer.measure_compensation(
            base_text, rec_text, patch_layer=27
        )
        
        # Compute recovery metrics
        recovery_metrics = measurer.compute_recovery_metrics(compensation_data)
        
        results.append({
            'pair_idx': i,
            'base_id': base_id,
            'rec_id': rec_id,
            **compensation_data,
            **recovery_metrics
        })
    
    # Analyze mechanisms
    analyze_mechanisms(results)
    
    return results

def analyze_mechanisms(results):
    """Analyze compensation mechanisms"""
    # [Your implementation]
    # - Correlation analysis
    # - Mechanism identification
    # - Timing analysis
    pass

if __name__ == "__main__":
    results = run_homeostasis_experiment()
```

### **Optimization Requirements:**
- Memory-efficient (don't store all activations)
- Fast execution (batch processing where possible)
- Progress logging
- Resume capability

---

## 🎯 **EXPECTED OUTPUTS**

### **1. Homeostasis Measurer** (`homeostasis_measurement.py`)
- Efficient multi-layer measurement
- Memory-optimized
- Fast execution

### **2. Experiment Script** (`homeostasis_mechanism_test.py`)
- Production-ready
- Copy-paste ready for Jupyter
- Optimized for performance

### **3. Analysis Script** (`homeostasis_analysis.py`)
- Correlation analysis
- Mechanism identification
- Timing analysis

### **4. Documentation** (`HOMEOSTASIS_README.md`)
- Measurement methodology
- Mechanism hypotheses
- Optimization strategies

---

## ✅ **SUCCESS CRITERIA**

- [ ] Efficient multi-layer measurement
- [ ] Memory-optimized
- [ ] Fast execution
- [ ] Measures all components
- [ ] Identifies mechanisms
- [ ] Outputs comprehensive CSV
- [ ] Full error handling
- [ ] Documentation complete

---

## 🚀 **MAXIMUM OUTPUT REQUEST**

**I want EVERYTHING:**
- Complete homeostasis measurement system
- Optimized for performance
- Mechanism analysis
- Documentation
- Test on 10 pairs before full run
- Memory optimization
- Fast execution
- Resume capability

**Make it so efficient that running on n=151 pairs is fast and memory-friendly.**

---

**Start with measurement system, then optimize for performance, then create experiment script, then write analysis, then document everything. Deliver ALL of it.**

