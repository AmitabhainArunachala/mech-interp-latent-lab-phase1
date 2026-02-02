#!/usr/bin/env python3
"""
Mistral-7B-Instruct Load Test Script
Run this on the remote server: python3 remote_mistral_test.py
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("=" * 60)
    print("Mistral-7B-Instruct Load Test")
    print("=" * 60)
    
    # Environment check
    print("\n1. Environment Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model
    print("\n2. Loading Mistral-7B-Instruct-v0.2...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    try:
        print("   Loading tokenizer...", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓")
        
        print("   Loading model...", end=" ", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✓")
        
        # Test inference
        print("\n3. Running test inference...")
        test_prompt = "What is 2+2? Answer:"
        print(f"   Prompt: {test_prompt}")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Response: {response}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Mistral-7B-Instruct is working correctly.")
        print("=" * 60)
        return 0
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nPlease install required packages:")
        print("  pip install transformers torch accelerate")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

