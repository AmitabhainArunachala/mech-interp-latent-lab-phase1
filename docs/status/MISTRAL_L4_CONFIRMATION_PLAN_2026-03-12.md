Goal: confirm that the new `L4 MLP` upstream assist is real, and test whether a shorter early token window reduces baseline leakage.

Why now:
- `L4 MLP micro-window8 + L25 bridge` beat `bridge_only_3` on recursive BT+ART.
- We need confirmation before treating it as a stable control result.
- Baseline contamination rose, so safety filtering is part of the acceptance criterion.

Queue:
1. `causal_state_benchmark_v4_multisite_mistral_l4_mlp_confirmation_window8.json`
2. `causal_state_benchmark_v4_multisite_mistral_l4_mlp_confirmation_window4.json`

Acceptance criteria:
- Beat `bridge_only_3` on recursive BT+ART.
- Do not materially worsen baseline BT+ART versus the window8 discovery run.
- Prefer the condition with lower mean recursive output `R_V` if behavior is tied.

Primary conditions under test:
- `early_mlp_0p125_bridge_2`
- `early_mlp_0p1875_bridge_3`
- `early_mlp_0p25_bridge_3`
- plus bridge-only and early-only controls.
