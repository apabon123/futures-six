"""Update meta.json with config for existing runs."""
import sys
import json
import yaml
from pathlib import Path

run_id = sys.argv[1]
run_dir = Path(f"reports/runs/{run_id}")

# Load meta.json
with open(run_dir / "meta.json", 'r', encoding='utf-8') as f:
    meta = json.load(f)

# Load configs
with open("configs/strategies.yaml", 'r', encoding='utf-8') as f:
    base_cfg = yaml.safe_load(f) or {}

with open("configs/canonical_frozen_stack_compute.yaml", 'r', encoding='utf-8') as f:
    compute_cfg = yaml.safe_load(f) or {}

# Merge configs (compute overrides base)
full_config = {**base_cfg, **compute_cfg}

# Update meta
meta['config'] = full_config

# Save
with open(run_dir / "meta.json", 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print(f"Updated meta.json for {run_id}")
