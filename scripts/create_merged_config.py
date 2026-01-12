"""Create merged config with strategy profiles + explicit allocator settings."""
import yaml
from pathlib import Path

# Load base config (has strategy profiles and all sleeves)
with open('configs/strategies.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Override allocator and RT settings
config['allocator_v1']['enabled'] = True
config['allocator_v1']['mode'] = 'compute'
config['allocator_v1']['profile'] = 'H'

config['risk_targeting']['enabled'] = True

# Write merged config
with open('configs/temp_phase1c_proof_merged.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Merged config created: configs/temp_phase1c_proof_merged.yaml')
print(f'allocator_v1.mode: {config["allocator_v1"]["mode"]}')
print(f'allocator_v1.enabled: {config["allocator_v1"]["enabled"]}')
print(f'allocator_v1.profile: {config["allocator_v1"]["profile"]}')

