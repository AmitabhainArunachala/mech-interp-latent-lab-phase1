# V-Projection Head Discovery - Status

**Started:** Running now  
**Method:** V-projection ablation (proven to work)  
**Log:** `v_proj_discovery_run.log`

---

## What's Running

**Pipeline:** `v_proj_head_discovery.py`

**Testing:**
- Layers: 8-27 (20 layers)
- Heads: 0-31 (32 heads per layer)
- Total: 640 head tests
- Prompts: 20 recursive prompts per head

**Expected runtime:** ~20-30 minutes

---

## Progress Tracking

**Check status:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && tail -20 v_proj_discovery_run.log"
```

**Check results:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && ls -lh results/head_discovery/v_proj_head_discovery_*.csv 2>/dev/null | tail -1"
```

**Monitor progress:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && tail -5 v_proj_discovery_run.log | grep -E '(Testing|Head ablation|Top|Layer)'"
```

---

## What to Expect

**Known heads to validate:**
- **L27H11:** Should show Δ ≈ +0.06 (6% impact)
- **L27H1:** Should show Δ ≈ +0.03 (3% impact)  
- **L27H22:** Should show Δ ≈ +0.02 (2.4% impact)

**New discoveries:**
- Important heads at layers 8-26
- Additional important heads at layer 27
- Heads that prevent contraction (positive delta)

---

**Pipeline is running! Check back in ~20-30 minutes for results.**









