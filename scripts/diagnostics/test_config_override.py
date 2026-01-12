"""Test config override logic."""
import yaml
import tempfile
from pathlib import Path

# Load base config
base_config = Path('configs/strategies.yaml')
with open(base_config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Apply overrides for RT + Alloc-H
config_overrides = {
    'risk_targeting.enabled': True,
    'allocator_v1.enabled': True,
    'allocator_v1.mode': 'compute',
    'allocator_v1.profile': 'H',
}

for key, value in config_overrides.items():
    keys = key.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

# Check result
print('After overrides:')
print(f'risk_targeting.enabled: {config["risk_targeting"]["enabled"]}')
print(f'allocator_v1.enabled: {config["allocator_v1"]["enabled"]}')
print(f'allocator_v1.mode: {config["allocator_v1"]["mode"]}')
print(f'allocator_v1.profile: {config["allocator_v1"]["profile"]}')

# Write temp config
tmp_config = Path(tempfile.mkdtemp()) / 'strategies.yaml'
with open(tmp_config, 'w', encoding='utf-8') as f:
    yaml.dump(config, f)

print(f'\nTemp config written to: {tmp_config}')

# Read it back to verify
with open(tmp_config, 'r', encoding='utf-8') as f:
    reloaded = yaml.safe_load(f)
    
print('\nReloaded from temp file:')
print(f'risk_targeting.enabled: {reloaded["risk_targeting"]["enabled"]}')
print(f'allocator_v1.enabled: {reloaded["allocator_v1"]["enabled"]}')
print(f'allocator_v1.mode: {reloaded["allocator_v1"]["mode"]}')
print(f'allocator_v1.profile: {reloaded["allocator_v1"]["profile"]}')

