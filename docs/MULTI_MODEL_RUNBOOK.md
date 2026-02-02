## Multi-Model Runbook (Automated)

### 1) Generate configs (Llama-3-8B example)
```bash
python3 - <<'PY'
from pathlib import Path
from src.utils.multi_model_discovery import generate_discovery_configs, validate_registry_compatibility, validate_config_fields

configs = generate_discovery_configs(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_config={"num_layers": 32, "num_heads": 32},
    out_dir=Path("configs/discovery"),
    model_short="llama3_8b",
    results_phase="phase2_generalization",
    seed=42,
    device="cuda",
    write_files=True,
)

print("generated", len(configs))
print("registry_missing", validate_registry_compatibility(configs))
print("field_errors", validate_config_fields(configs))
PY
```

Expected:
- `generated 23`
- `registry_missing []`
- `field_errors []`

### 2) Validate generated configs exist
```bash
ls -1 configs/discovery/llama3_8b
```
Expected files:
- `01_baseline_rv.json`
- `02_source_hunt_mlp_ablation_l0.json` ... `l8.json`
- `03_transfer_hunt_mlp_steer_l0.json` ... `l10.json`
- `04_readout_validation.json`
- `05_head_identification.json`

### 3) Registry compatibility (standalone)
```bash
python3 - <<'PY'
import json, re
from pathlib import Path

reg_text = Path("src/pipelines/registry.py").read_text()
keys = set(re.findall(r'"([^"]+)"\\s*:\\s*run_', reg_text))
bad = []
for p in Path("configs/discovery/llama3_8b").glob("*.json"):
    cfg = json.loads(p.read_text())
    exp = cfg.get("experiment")
    if exp not in keys:
        bad.append((p.name, exp))
print("missing", bad)
PY
```
Expected: `missing []`

### 4) Prompt bank version stamp
```bash
python3 - <<'PY'
import json
from pathlib import Path

missing = []
for p in Path("configs/discovery/llama3_8b").glob("*.json"):
    cfg = json.loads(p.read_text())
    if "prompt_bank_version" not in cfg:
        missing.append(p.name)
print("missing_prompt_bank_version", missing)
PY
```
Expected: `missing_prompt_bank_version []`

### 5) Run Phase 2 baseline (cross-arch)
```bash
python -m src.pipelines.run --config configs/discovery/llama3_8b/01_baseline_rv.json
```

### 6) Run Phase 3 source hunt (ablation sweep)
```bash
for l in 0 1 2 3 4 5 6 7 8; do
  python -m src.pipelines.run --config configs/discovery/llama3_8b/02_source_hunt_mlp_ablation_l${l}.json
done
```

### 7) Run Phase 4 transfer hunt (steering sweep)
```bash
for l in 0 1 2 3 4 5 6 7 8 9 10; do
  python -m src.pipelines.run --config configs/discovery/llama3_8b/03_transfer_hunt_mlp_steer_l${l}.json
done
```

### 8) Run Phase 5 readout validation + Phase 6 head identification
```bash
python -m src.pipelines.run --config configs/discovery/llama3_8b/04_readout_validation.json
python -m src.pipelines.run --config configs/discovery/llama3_8b/05_head_identification.json
```

### 9) Results location
```
results/phase2_generalization/llama3_8b/
```
Each phase writes into its own subdirectory with `runs/` beneath it.
