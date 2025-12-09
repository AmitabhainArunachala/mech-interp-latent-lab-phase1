import torch, numpy as np
import torch.nn.functional as F

def epsilon_last_token(hidden_states, layer_pair):
    la, lb = layer_pair
    eps = []
    for step_h in hidden_states:
        hA = step_h[la][0, -1, :]
        hB = step_h[lb][0, -1, :]
        eps.append(F.cosine_similarity(hA[None], hB[None]).item())
    return dict(mean=float(np.mean(eps)), start=float(eps[0]), end=float(eps[-1]))

def attn_entropy_lastrow(attentions):
    out = []
    for step_attn in attentions:
        vals = []
        for layer_attn in step_attn:
            A = layer_attn[0][:, -1, :].clamp_min(1e-12)
            P = A / A.sum(dim=-1, keepdim=True)
            e = -(P * P.log()).sum(dim=-1).mean().item()
            vals.append(e)
        out.append(float(np.mean(vals)))
    return dict(mean=float(np.mean(out)))
