print("Hello from python")
import torch
print(f"Torch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
import sys
import os
sys.path.insert(0, os.path.abspath("."))
try:
    from src.core.models import load_model
    print("Imported load_model")
except Exception as e:
    print(f"Failed to import load_model: {e}")

