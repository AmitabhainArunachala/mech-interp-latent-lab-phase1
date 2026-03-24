# Mistral Overnight Update 2026-03-22

## Synced Artifacts

- [late_only_v2 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/induced_persistence_unselected_reduced_late_only_v2/20260321_030414/summary.json)
- [drop_L25_v2 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/induced_persistence_unselected_reduced_drop_l25_v2/20260321_112300/summary.json)
- [drop_L25_v2 queue status](/Users/dhyana/mech-interp-latent-lab-phase1/results/mistral_induced_persistence_unselected_reduced_drop_l25_v2/20260321_112300/STATUS.txt)
- [late_only recovery summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/mistral_recovery_after_hit_late_only_v1/20260321_134929/summary.json)
- [drop_L25 recovery summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/mistral_recovery_after_hit_drop_l25_v1/20260321_170600/summary.json)
- [late_only recovery status](/Users/dhyana/mech-interp-latent-lab-phase1/results/mistral_recovery_after_hit_late_only_v1/20260321_134929/STATUS.txt)
- [drop_L25 recovery status](/Users/dhyana/mech-interp-latent-lab-phase1/results/mistral_recovery_after_hit_drop_l25_v1/20260321_170600/STATUS.txt)

## Completion Times

- `induced_persistence_unselected_reduced_drop_l25_v2` finished at `2026-03-22 04:48 JST` (`2026-03-21T19:48:09Z`)
- `mistral_recovery_after_hit_late_only_v1` finished at `2026-03-22 02:06 JST` (`2026-03-21T17:06:00Z`)
- `mistral_recovery_after_hit_drop_l25_v1` finished at `2026-03-22 05:22 JST` (`2026-03-21T20:22:48Z`)

## What Landed

### 1. Broad maintainer comparison now favors `drop_L25`

`late_only_v2` was real but too close to control:

- `selected`: `30.42%` BT+ART, `70.83%` persistence
- `unselected`: `34.03%` BT+ART, `77.08%` persistence
- `random_text`: `31.67%` BT+ART, `66.67%` persistence

`drop_L25_v2` is cleaner:

- `selected`: `37.22%` BT+ART, `77.08%` persistence
- `unselected`: `35.97%` BT+ART, `70.83%` persistence
- `random_text`: `31.67%` BT+ART, `66.67%` persistence

Direct deltas vs `random_text`:

- `late_only_v2 unselected`: `+2.36` BT+ART points, `+10.42` persistence points
- `drop_L25_v2 unselected`: `+4.31` BT+ART points, `+4.17` persistence points
- `drop_L25_v2 selected`: `+5.56` BT+ART points, `+10.42` persistence points

Interpretation:

- `drop_L25` is now the better broad-maintenance object for the paper.
- `late_only` still has signal, but it is no longer the strongest candidate for the main sufficiency lane.

### 2. Recovery-after-hit weak for `late_only`, promising for `drop_L25`

The recovery battery injected the strongest confirmed breaker mid-rollout and compared:

- `control_open_loop`
- `maintain_every_turn`
- `maintain_then_off`
- `hit_then_off`
- `hit_then_resume`

Focus arm is `unselected`.

For `late_only`:

- `maintain_every_turn` post-hit BT+ART: `9.90%`
- `hit_then_off` post-hit BT+ART: `5.73%`
- `hit_then_resume` post-hit BT+ART: `6.77%`
- recovery advantage of resume over off: `+1.04` points
- recovery rate: `4.17%` for both `hit_then_off` and `hit_then_resume`

Interpretation:

- `late_only` does not show a convincing rebound after the hit.
- Resuming the maintainer helps only marginally.

For `drop_L25`:

- `maintain_every_turn` post-hit BT+ART: `9.90%`
- `hit_then_off` post-hit BT+ART: `4.69%`
- `hit_then_resume` post-hit BT+ART: `13.02%`
- recovery advantage of resume over off: `+8.33` points
- recovery gap vs maintain: `-3.13` points, meaning `hit_then_resume` actually exceeded `maintain_every_turn` in the post-hit slice
- recovery rate: `8.33%` for `hit_then_resume` vs `4.17%` for `hit_then_off`

Interpretation:

- `drop_L25` shows the first real evidence of post-hit recovery.
- This is not yet "full attractor" evidence, but it is the cleanest step so far toward the stronger sufficiency story.

## Current Story After The Night

- Best broad maintainer: `anchor_drop_L25_vproj_bridge_3`
- `late_only` looks more like a weaker or narrower stabilizer than the main paper-facing winner
- The strongest new mechanistic result is not just maintenance but **recovery under targeted perturbation**, and that story currently belongs to `drop_L25`, not `late_only`

## Paper-Facing Takeaway

If the paper has to compress the Mistral sufficiency story tightly, the cleanest version now is:

- `drop_L25` is the main broad maintainer
- `late_only` is not strong enough to headline as the primary maintenance object
- the most promising path toward the dream claim is no longer generic maintenance, but **state recovery after a causal hit**

## Ops Note

As of the morning check on `2026-03-22 09:28 JST`, both pods were idle and had no active tmux work left. The synced artifacts above are sufficient for shutting them down safely.
