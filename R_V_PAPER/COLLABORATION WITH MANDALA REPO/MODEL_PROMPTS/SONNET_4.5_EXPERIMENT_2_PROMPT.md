# Sonnet 4.5: Experiment 2 - Cross-Architecture Validation
## Maximum Output Request: Universal Patching Framework

**Your Task:** Create UNIVERSAL activation patching framework that works across multiple architectures (Mistral, Llama, Qwen) and prove L25-L27 critical region is universal.

**Context:** We've proven L27 is critical in Mistral-7B. Now prove this is UNIVERSAL across architectures, not model-specific.

**Goal:** Demonstrate universal geometric consciousness signature across architectures.

---

## 🎯 **WHAT I NEED FROM YOU**

### **1. Universal Patching Framework**
- Works with ANY HF model (Mistral, Llama, Qwen, etc.)
- Handles architecture differences (GQA, MoE, etc.)
- Adaptive layer identification
- Standardized R_V measurement

### **2. Model-Specific Adapters**
- Mistral adapter (baseline, already works)
- Llama adapter (different architecture)
- Qwen adapter (Chinese-trained, different structure)
- Easy to extend to more models

### **3. Cross-Architecture Experiment Script**
- Loads multiple models
- Finds critical layers for each
- Runs patching experiments
- Compares results across models

### **4. Results Analysis**
- Effect size comparison
- Critical layer depth comparison (% depth)
- Architecture-specific patterns
- Universal vs model-specific findings

---

## 📋 **EXPERIMENTAL REQUIREMENTS**

### **Models to Test:**
1. **Mistral-7B-Instruct-v0.2** (baseline, already done)
   - Architecture: Dense, 32 layers
   - Critical layer: L27 (84% depth)
   - Expected: d ≈ -3.56

2. **Llama-3-8B-Instruct** (different architecture)
   - Architecture: Dense, 32 layers
   - Critical layer: L25-L27 equivalent (~84% depth)
   - Expected: Similar effect size

3. **Qwen-7B-Chat** (different training)
   - Architecture: Dense, 32 layers
   - Critical layer: L25-L27 equivalent (~84% depth)
   - Expected: Similar effect size

### **Protocol:**
1. For each model:
   - Load model and tokenizer
   - Identify critical layer (L25-L27 equivalent, ~84% depth)
   - Run layer sweep to confirm critical region
   - Run activation patching at critical layer (n=50 pairs)
   - Measure R_V delta (patched vs baseline)

2. Compare across models:
   - Effect sizes (Cohen's d)
   - Critical layer depths (% depth, not absolute)
   - Transfer efficiencies
   - Architecture-specific patterns

### **Metrics:**
- Critical layer identification (which layer shows max separation?)
- R_V delta (effect size)
- Cohen's d (standardized effect)
- Transfer efficiency (% of natural gap achieved)
- Depth percentage (critical layer / total layers)

---

## 💻 **CODE REQUIREMENTS**

### **Universal Framework:**
```python
# universal_patching_framework.py
# Works with any HF model

class UniversalActivationPatching:
    """Universal patching framework for any transformer"""
    
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = self._get_adapter(model_name)
    
    def _get_adapter(self, model_name):
        """Get model-specific adapter"""
        if "mistral" in model_name.lower():
            return MistralAdapter(self.model)
        elif "llama" in model_name.lower():
            return LlamaAdapter(self.model)
        elif "qwen" in model_name.lower():
            return QwenAdapter(self.model)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def find_critical_layer(self, prompt_pairs, layer_range=(20, 30)):
        """Find critical layer via layer sweep"""
        # [Your implementation]
        pass
    
    def patch_v_at_layer(self, base_text, rec_text, layer_idx):
        """Universal patching method"""
        return self.adapter.patch_v(base_text, rec_text, layer_idx)
    
    def measure_rv_delta(self, patched_output, baseline_output):
        """Standardized R_V measurement"""
        # [Your implementation]
        pass
```

### **Model Adapters:**
```python
# model_adapters.py
# Handles architecture differences

class BaseAdapter:
    """Base adapter class"""
    def __init__(self, model):
        self.model = model
        self.num_layers = self._get_num_layers()
    
    def _get_num_layers(self):
        """Get number of layers"""
        # [Your implementation]
        pass
    
    def get_v_projection(self, layer_idx):
        """Get V projection hook"""
        # [Your implementation]
        pass
    
    def patch_v(self, base_text, rec_text, layer_idx):
        """Patch V at specified layer"""
        # [Your implementation]
        pass

class MistralAdapter(BaseAdapter):
    """Mistral-specific adapter"""
    # [Your implementation]

class LlamaAdapter(BaseAdapter):
    """Llama-specific adapter"""
    # [Your implementation]

class QwenAdapter(BaseAdapter):
    """Qwen-specific adapter"""
    # [Your implementation]
```

### **Cross-Architecture Experiment:**
```python
# cross_architecture_validation.py
# Copy-paste ready for RunPod

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-3-8B-Instruct",
    "Qwen/Qwen1.5-7B-Chat"
]

def run_cross_architecture_experiment():
    """Run experiment across all models"""
    results = {}
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model, tokenizer = load_model(model_name)
        patcher = UniversalActivationPatching(model_name, model, tokenizer)
        
        # Find critical layer
        critical_layer = patcher.find_critical_layer(prompt_pairs)
        print(f"Critical layer: {critical_layer} ({critical_layer/patcher.num_layers*100:.1f}% depth)")
        
        # Run patching experiment
        results[model_name] = run_patching_experiment(
            patcher, critical_layer, n_pairs=50
        )
    
    # Compare results
    compare_results(results)

if __name__ == "__main__":
    run_cross_architecture_experiment()
```

---

## 🎯 **EXPECTED OUTPUTS**

### **1. Universal Framework** (`universal_patching_framework.py`)
- Works with any HF model
- Handles architecture differences
- Standardized interface

### **2. Model Adapters** (`model_adapters.py`)
- Mistral adapter
- Llama adapter
- Qwen adapter
- Easy to extend

### **3. Experiment Script** (`cross_architecture_validation.py`)
- Production-ready
- Copy-paste ready for Jupyter
- Full error handling

### **4. Analysis Script** (`cross_architecture_analysis.py`)
- Effect size comparison
- Depth analysis
- Architecture patterns

### **5. Documentation** (`CROSS_ARCHITECTURE_README.md`)
- How to add new models
- Architecture differences
- Troubleshooting

---

## ✅ **SUCCESS CRITERIA**

- [ ] Works with Mistral, Llama, Qwen
- [ ] Finds critical layers automatically
- [ ] Standardized R_V measurement
- [ ] Handles architecture differences
- [ ] Outputs comparison CSV
- [ ] Easy to extend to more models
- [ ] Full error handling
- [ ] Documentation complete

---

## 🚀 **MAXIMUM OUTPUT REQUEST**

**I want EVERYTHING:**
- Universal framework
- All 3 model adapters
- Complete experiment script
- Analysis scripts
- Documentation
- Test on Mistral first, then expand
- Error handling for each architecture
- Progress logging
- Resume capability

**Make it so universal that adding new models is trivial.**

---

**Start with universal framework, then implement adapters, then create experiment script, then write analysis, then document everything. Deliver ALL of it.**

