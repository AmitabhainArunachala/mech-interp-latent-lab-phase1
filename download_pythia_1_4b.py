from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-1.4b"  # safe size for 18 GB RAM

print(f"Downloading {MODEL_NAME}...")
print("This will cache to ~/.cache/huggingface/")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="float32",   # CPU
    device_map="cpu"
)

print("\n✓ Model and tokenizer downloaded and cached locally.")
print(f"  Layers: {len(model.gpt_neox.layers)}")
print(f"  Hidden dim: {model.config.hidden_size}")
