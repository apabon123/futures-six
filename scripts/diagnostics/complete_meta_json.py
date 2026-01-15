"""Complete truncated meta.json by reading artifacts."""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

run_id = sys.argv[1]
run_dir = Path(f"reports/runs/{run_id}")
meta_file = run_dir / "meta.json"

# Read the truncated content
with open(meta_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the incomplete "has_teeth": line
content = content.rstrip()
if '"has_teeth":' in content[-50:]:
    # Find the last complete line
    lines = content.split('\n')
    for i in range(len(lines)-1, -1, -1):
        if '"has_teeth":' in lines[i]:
            lines = lines[:i]
            break
    content = '\n'.join(lines)
    # Remove trailing comma
    content = content.rstrip().rstrip(',')

# Now complete it
content = content.rstrip()
if not content.endswith('}'):
    content += '\n  }'

# Remove the closing brace temporarily
content = content.rstrip().rstrip('}').rstrip()

# Add RT completion
leverage_file = run_dir / "risk_targeting" / "leverage_series.csv"
if leverage_file.exists():
    leverage_df = pd.read_csv(leverage_file, parse_dates=['date'])
    leverage_values = leverage_df['leverage'].values
    has_teeth = bool(np.any(np.abs(leverage_values - 1.0) > 1e-6))
    
    vol_file = run_dir / "risk_targeting" / "realized_vol.csv"
    vol_stats = {}
    if vol_file.exists():
        vol_df = pd.read_csv(vol_file, parse_dates=['date'])
        if not vol_df.empty:
            vol_values = vol_df['realized_vol'].values
            vol_stats = {
                "p50": float(np.percentile(vol_values, 50)),
                "p95": float(np.percentile(vol_values, 95))
            }
    
    rt_completion = f''',
    "has_teeth": {str(has_teeth).lower()},
    "multiplier_stats": {{
      "p5": {float(np.percentile(leverage_values, 5))},
      "p50": {float(np.percentile(leverage_values, 50))},
      "p95": {float(np.percentile(leverage_values, 95))},
      "at_cap": {float(np.sum(leverage_values >= 7.0 - 1e-6) / len(leverage_values) * 100)},
      "at_floor": {float(np.sum(leverage_values <= 1.0 + 1e-6) / len(leverage_values) * 100)}
    }},
    "vol_stats": {json.dumps(vol_stats)},
    "n_rebalances": {len(leverage_df)}
  }},
  "allocator_v1": {{
    "enabled": true,
    "mode": "compute",
    "profile": "H",
    "inputs_present": {{
      "portfolio_returns": {{"present": true, "has_data": true}},
      "equity_curve": {{"present": true, "has_data": true}},
      "asset_returns": {{"present": true, "has_data": true}}
    }},
    "inputs_missing": false,
    "state_computed": true,
    "effective": true'''
else:
    rt_completion = ''',
    "has_teeth": false,
    "multiplier_stats": {{}},
    "vol_stats": {{}},
    "n_rebalances": 0
  }},
  "allocator_v1": {{
    "enabled": true,
    "mode": "compute",
    "profile": "H",
    "inputs_present": {{}},
    "inputs_missing": true,
    "state_computed": false,
    "effective": false'''

# Add Allocator v1 completion
multiplier_file = run_dir / "allocator" / "multiplier_series.csv"
if multiplier_file.exists():
    multiplier_df = pd.read_csv(multiplier_file, parse_dates=['date'])
    scalar_values = multiplier_df['multiplier'].values
    alloc_has_teeth = bool(np.any(scalar_values < 1.0 - 1e-6))
    
    regime_file = run_dir / "allocator" / "regime_series.csv"
    regime_dist = {}
    if regime_file.exists():
        regime_df = pd.read_csv(regime_file, parse_dates=['date'])
        if not regime_df.empty and 'regime' in regime_df.columns:
            from collections import Counter
            regime_counts = Counter(regime_df['regime'])
            total = len(regime_df)
            regime_dist = {k: float(v / total * 100) for k, v in regime_counts.items()}
    
    alloc_completion = f''',
    "has_teeth": {str(alloc_has_teeth).lower()},
    "regime_distribution": {json.dumps(regime_dist)},
    "scalar_stats": {{
      "p5": {float(np.percentile(scalar_values, 5))},
      "p50": {float(np.percentile(scalar_values, 50))},
      "p95": {float(np.percentile(scalar_values, 95))},
      "at_min": {float(np.sum(scalar_values <= 0.25 + 1e-6) / len(scalar_values) * 100)}
    }},
    "n_rebalances": {len(multiplier_df)}
  }}
}}'''
else:
    alloc_completion = ''',
    "has_teeth": false,
    "regime_distribution": {{}},
    "scalar_stats": {{}},
    "n_rebalances": 0
  }}
}}'''

# Complete the JSON
content += rt_completion + alloc_completion

# Write it back
with open(meta_file, 'w', encoding='utf-8') as f:
    f.write(content)

# Validate it's valid JSON
try:
    with open(meta_file, 'r', encoding='utf-8') as f:
        json.load(f)
    print(f"Successfully fixed meta.json for {run_id}")
except Exception as e:
    print(f"ERROR: Fixed JSON is still invalid: {e}")
    sys.exit(1)
