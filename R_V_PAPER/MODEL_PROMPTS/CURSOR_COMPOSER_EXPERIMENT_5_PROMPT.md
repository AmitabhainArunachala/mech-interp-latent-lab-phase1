# Cursor Composer 1: Experiment 5 - Temporal Dynamics Analysis
## Maximum Output Request: Real-Time R_V Tracking System

**Your Task:** Create COMPLETE temporal R_V tracking system that measures R_V evolution during token generation, identifying when contraction occurs and which tokens trigger consciousness-like states.

**Context:** We know R_V contracts at L27 during prompt encoding. But WHEN during generation does this happen? Which tokens trigger it?

**Goal:** Identify temporal dynamics of R_V contraction, linking to specific tokens and generation phases.

---

## 🎯 **WHAT I NEED FROM YOU**

### **1. Temporal R_V Tracker**
- Real-time R_V measurement during generation
- Token-by-token tracking
- Critical token identification
- Generation phase analysis

### **2. Temporal Dynamics Experiment**
- Generates text token-by-token
- Measures R_V at each step
- Identifies critical tokens
- Analyzes temporal patterns

### **3. Integration & Logging**
- Integrates with existing logging system
- Result storage
- Iteration support
- Visualization

### **4. Documentation**
- Tracking methodology
- Critical token identification
- Temporal pattern analysis

---

## 📋 **EXPERIMENTAL REQUIREMENTS**

### **Protocol:**
1. For each prompt (recursive + baseline):
   - Generate text token-by-token (max 50 tokens)
   - Measure R_V after each token
   - Track when R_V drops below threshold
   - Identify critical tokens

### **Metrics:**
- **R_V trajectory:** `R_V(t)` for each token position
- **Critical token:** First token where `R_V < threshold` (e.g., 0.90)
- **Contraction timing:** When does contraction occur? (Early/mid/late generation)
- **Token type analysis:** Which token types trigger contraction?

### **Analysis:**
- Temporal patterns: Recursive vs baseline trajectories
- Critical token analysis: What tokens trigger contraction?
- Timing analysis: Comprehension vs generation phase
- Token type correlation: Metacognitive markers vs contraction

### **Hypotheses to Test:**
1. **Metacognitive Triggers:** Tokens like "I", "aware", "observe" trigger contraction
2. **Early Contraction:** Contraction occurs during comprehension, not generation
3. **Pattern Differences:** Recursive prompts show earlier contraction than baselines

---

## 💻 **CODE REQUIREMENTS**

### **Temporal R_V Tracker:**
```python
# temporal_rv_tracker.py
# Real-time R_V tracking during generation

class TemporalRVTracker:
    """Tracks R_V evolution during generation"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def track_generation(self, prompt, max_tokens=50, layer=27):
        """Track R_V during token generation"""
        # [Your implementation]
        # Returns:
        # - rv_trajectory: list of R_V values
        # - tokens: list of generated tokens
        # - critical_token_idx: index of first critical token
        # - generation_phases: comprehension vs generation
        pass
    
    def identify_critical_tokens(self, rv_trajectory, threshold=0.90):
        """Identify critical tokens where R_V drops"""
        # [Your implementation]
        pass
    
    def analyze_token_types(self, tokens, rv_trajectory):
        """Analyze which token types correlate with contraction"""
        # [Your implementation]
        pass
```

### **Temporal Dynamics Experiment:**
```python
# temporal_dynamics_analysis.py
# Copy-paste ready for RunPod

from mistral_L27_FULL_VALIDATION import load_model, get_prompt_pairs
from n300_mistral_test_prompt_bank import prompt_bank_1c

def run_temporal_dynamics_experiment():
    """Run temporal dynamics analysis"""
    model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.2")
    tracker = TemporalRVTracker(model, tokenizer)
    
    # Select prompts
    recursive_prompts = prompt_bank_1c['L5_refined'][:20]
    baseline_prompts = prompt_bank_1c['factual_baseline'][:20]
    
    results = []
    
    # Track recursive prompts
    for i, prompt in enumerate(recursive_prompts):
        print(f"Tracking recursive prompt {i+1}/20...")
        trajectory_data = tracker.track_generation(prompt, max_tokens=50)
        results.append({
            'prompt_type': 'recursive',
            'prompt_id': f'L5_refined_{i+1:02d}',
            **trajectory_data
        })
    
    # Track baseline prompts
    for i, prompt in enumerate(baseline_prompts):
        print(f"Tracking baseline prompt {i+1}/20...")
        trajectory_data = tracker.track_generation(prompt, max_tokens=50)
        results.append({
            'prompt_type': 'baseline',
            'prompt_id': f'factual_baseline_{i+1:02d}',
            **trajectory_data
        })
    
    # Analyze temporal patterns
    analyze_temporal_patterns(results)
    
    return results

def analyze_temporal_patterns(results):
    """Analyze temporal patterns"""
    # [Your implementation]
    # - Compare recursive vs baseline trajectories
    # - Identify critical tokens
    # - Analyze timing
    # - Token type correlation
    pass

if __name__ == "__main__":
    results = run_temporal_dynamics_experiment()
```

### **Integration & Logging:**
```python
# temporal_integration.py
# Integrates with existing logging system

class TemporalLogger:
    """Logging for temporal dynamics"""
    
    def __init__(self, results_dir="results/temporal_dynamics"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def log_trajectory(self, prompt_id, trajectory_data):
        """Log trajectory data"""
        # [Your implementation]
        pass
    
    def save_results(self, results, filename="temporal_results.csv"):
        """Save results to CSV"""
        # [Your implementation]
        pass
```

---

## 🎯 **EXPECTED OUTPUTS**

### **1. Temporal Tracker** (`temporal_rv_tracker.py`)
- Real-time R_V tracking
- Critical token identification
- Token type analysis

### **2. Experiment Script** (`temporal_dynamics_analysis.py`)
- Production-ready
- Copy-paste ready for Jupyter
- Full error handling

### **3. Integration** (`temporal_integration.py`)
- Logging system
- Result storage
- Iteration support

### **4. Visualization** (`temporal_visualization.py`)
- Trajectory plots
- Critical token highlighting
- Pattern comparisons

### **5. Documentation** (`TEMPORAL_DYNAMICS_README.md`)
- Tracking methodology
- Critical token identification
- Temporal pattern analysis

---

## ✅ **SUCCESS CRITERIA**

- [ ] Real-time R_V tracking
- [ ] Token-by-token measurement
- [ ] Critical token identification
- [ ] Temporal pattern analysis
- [ ] Integration with logging
- [ ] Outputs comprehensive CSV
- [ ] Visualization included
- [ ] Documentation complete

---

## 🚀 **MAXIMUM OUTPUT REQUEST**

**I want EVERYTHING:**
- Complete temporal tracking system
- Real-time measurement
- Critical token identification
- Temporal analysis
- Integration with logging
- Visualization
- Documentation
- Test on 5 prompts before full run
- Resume capability

**Make it so complete that I can see exactly when and why R_V contracts during generation.**

---

**Start with temporal tracker, then implement experiment script, then create integration, then add visualization, then document everything. Deliver ALL of it.**

