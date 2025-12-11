import random
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from n300_mistral_test_prompt_bank import prompt_bank_1c


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_prompts_by_pillar(
    pillar: str = "baseline",
    limit: int = 40,
    seed: int = 0,
    group_prefix: str = "",
) -> List[str]:
    rng = random.Random(seed)
    prompts = []
    for v in prompt_bank_1c.values():
        if pillar and v.get("pillar") == pillar:
            prompts.append(v["text"])
        elif group_prefix and v.get("group", "").startswith(group_prefix):
            prompts.append(v["text"])
    rng.shuffle(prompts)
    return prompts[:limit]


def participation_ratio(v_tensor: torch.Tensor, window: int = 16) -> float:
    if v_tensor is None:
        return float("nan")
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    T, D = v_tensor.shape
    W = min(window, T)
    v_window = v_tensor[-W:, :].float()
    try:
        _, S, _ = torch.linalg.svd(v_window, full_matrices=False)
        s2 = S ** 2
        return float((s2.sum() ** 2) / (s2.square().sum() + 1e-9))
    except Exception:
        return float("nan")


def capture_v_projection(model, inputs, layer_idx: int):
    storage = {}

    def hook_fn(module, inp, out):
        storage["v"] = out.detach()

    handle = (
        model.model.layers[layer_idx]
        .self_attn.v_proj.register_forward_hook(hook_fn)
    )
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return storage.get("v", None)


def compute_rv(
    model,
    tokenizer,
    text: str,
    early: int = 4,
    late: int = 27,
    window: int = 16,
    device: str = "cuda",
):
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    v_early = capture_v_projection(model, enc, early)
    v_late = capture_v_projection(model, enc, late)
    pr_early = participation_ratio(v_early, window)
    pr_late = participation_ratio(v_late, window)
    if pr_early == 0 or pr_early != pr_early:
        return float("nan")
    return pr_late / pr_early


RECURSIVE_KEYWORDS = [
    "self",
    "aware",
    "observe",
    "observing",
    "conscious",
    "consciousness",
    "awareness",
    "I am",
    "this response",
    "these words",
    "my own",
]


def behavior_score(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in RECURSIVE_KEYWORDS if kw.lower() in lower)


def generate_with_kv(
    model,
    tokenizer,
    prompt: str,
    past_key_values=None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cuda",
):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0 else None,
        use_cache=True,
        past_key_values=past_key_values,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def capture_past_key_values(model, tokenizer, prompt: str, device: str = "cuda"):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=True, return_dict=True)
    return out.past_key_values

