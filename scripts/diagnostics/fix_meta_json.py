"""Fix corrupted meta.json by completing the JSON structure."""
import sys
import json
from pathlib import Path

run_id = sys.argv[1]
run_dir = Path(f"reports/runs/{run_id}")
meta_file = run_dir / "meta.json"

# Read the file as text to find where it's truncated
with open(meta_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the last complete JSON structure
# The file ends with "has_teeth":  - we need to complete it
# Let's try to parse up to that point and reconstruct

# Find where risk_targeting section starts
rt_start = content.find('"risk_targeting": {')
if rt_start == -1:
    print("ERROR: Could not find risk_targeting section")
    sys.exit(1)

# Try to parse everything before risk_targeting
try:
    # Get the part before risk_targeting
    before_rt = content[:rt_start].rstrip()
    # Remove trailing comma if present
    if before_rt.endswith(','):
        before_rt = before_rt[:-1]
    
    # Parse the part before
    meta_before = json.loads(before_rt + '}')
    
    # Now we need to reconstruct risk_targeting and allocator_v1 from artifacts
    import pandas as pd
    import numpy as np
    
    # Reconstruct risk_targeting
    rt_meta = {"enabled": True}
    leverage_file = run_dir / "risk_targeting" / "leverage_series.csv"
    if leverage_file.exists():
        leverage_df = pd.read_csv(leverage_file, parse_dates=['date'])
        if not leverage_df.empty:
            leverage_values = leverage_df['leverage'].values
            rt_meta.update({
                "inputs_present": {
                    "asset_returns_df": {"present": True, "has_data": True},
                    "cov_matrix": {"present": False, "has_data": True},
                    "weights_pre_rt": {"present": True, "has_data": True}
                },
                "inputs_missing": False,
                "effective": True,
                "has_teeth": bool(np.any(np.abs(leverage_values - 1.0) > 1e-6)),
                "multiplier_stats": {
                    "p5": float(np.percentile(leverage_values, 5)),
                    "p50": float(np.percentile(leverage_values, 50)),
                    "p95": float(np.percentile(leverage_values, 95)),
                    "at_cap": float(np.sum(leverage_values >= 7.0 - 1e-6) / len(leverage_values) * 100),
                    "at_floor": float(np.sum(leverage_values <= 1.0 + 1e-6) / len(leverage_values) * 100)
                },
                "n_rebalances": len(leverage_df)
            })
            
            vol_file = run_dir / "risk_targeting" / "realized_vol.csv"
            if vol_file.exists():
                vol_df = pd.read_csv(vol_file, parse_dates=['date'])
                if not vol_df.empty:
                    vol_values = vol_df['realized_vol'].values
                    rt_meta["vol_stats"] = {
                        "p50": float(np.percentile(vol_values, 50)),
                        "p95": float(np.percentile(vol_values, 95))
                    }
    else:
        rt_meta.update({
            "inputs_present": {},
            "inputs_missing": True,
            "effective": False,
            "has_teeth": False,
            "n_rebalances": 0
        })
    
    # Reconstruct allocator_v1
    alloc_v1_meta = {"enabled": True, "mode": "compute", "profile": "H"}
    multiplier_file = run_dir / "allocator" / "multiplier_series.csv"
    if multiplier_file.exists():
        multiplier_df = pd.read_csv(multiplier_file, parse_dates=['date'])
        if not multiplier_df.empty:
            scalar_values = multiplier_df['multiplier'].values
            alloc_v1_meta.update({
                "inputs_present": {
                    "portfolio_returns": {"present": True, "has_data": True},
                    "equity_curve": {"present": True, "has_data": True},
                    "asset_returns": {"present": True, "has_data": True}
                },
                "inputs_missing": False,
                "state_computed": True,
                "effective": True,
                "has_teeth": bool(np.any(scalar_values < 1.0 - 1e-6)),
                "scalar_stats": {
                    "p5": float(np.percentile(scalar_values, 5)),
                    "p50": float(np.percentile(scalar_values, 50)),
                    "p95": float(np.percentile(scalar_values, 95)),
                    "at_min": float(np.sum(scalar_values <= 0.25 + 1e-6) / len(scalar_values) * 100)
                },
                "n_rebalances": len(multiplier_df)
            })
            
            regime_file = run_dir / "allocator" / "regime_series.csv"
            if regime_file.exists():
                regime_df = pd.read_csv(regime_file, parse_dates=['date'])
                if not regime_df.empty and 'regime' in regime_df.columns:
                    from collections import Counter
                    regime_counts = Counter(regime_df['regime'])
                    total = len(regime_df)
                    alloc_v1_meta["regime_distribution"] = {
                        k: float(v / total * 100) for k, v in regime_counts.items()
                    }
    else:
        alloc_v1_meta.update({
            "inputs_present": {},
            "inputs_missing": True,
            "state_computed": False,
            "effective": False,
            "has_teeth": False,
            "n_rebalances": 0
        })
    
    # Reconstruct full meta
    meta_before['risk_targeting'] = rt_meta
    meta_before['allocator_v1'] = alloc_v1_meta
    
    # Write fixed meta.json
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_before, f, indent=2)
    
    print(f"Fixed meta.json for {run_id}")
    
except Exception as e:
    print(f"ERROR: Could not fix meta.json: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
