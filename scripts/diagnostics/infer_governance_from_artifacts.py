"""Infer RT and Allocator governance from artifacts and update meta.json."""
import sys
import json
import pandas as pd
from pathlib import Path
import numpy as np

run_id = sys.argv[1]
run_dir = Path(f"reports/runs/{run_id}")

# Load meta.json
with open(run_dir / "meta.json", 'r', encoding='utf-8') as f:
    meta = json.load(f)

# Infer Risk Targeting governance from artifacts
rt_meta = meta.get('risk_targeting', {})
if rt_meta.get('enabled', False):
    leverage_file = run_dir / "risk_targeting" / "leverage_series.csv"
    vol_file = run_dir / "risk_targeting" / "realized_vol.csv"
    
    if leverage_file.exists():
        leverage_df = pd.read_csv(leverage_file, parse_dates=['date'])
        if not leverage_df.empty:
            leverage_values = leverage_df['leverage'].values
            rt_meta['n_rebalances'] = len(leverage_df)
            rt_meta['inputs_missing'] = False
            rt_meta['effective'] = True
            rt_meta['has_teeth'] = bool(np.any(np.abs(leverage_values - 1.0) > 1e-6))
            
            # Update inputs_present
            rt_meta['inputs_present'] = {
                'asset_returns_df': {'present': True, 'has_data': True},
                'cov_matrix': {'present': False, 'has_data': True},  # Computed from returns
                'weights_pre_rt': {'present': True, 'has_data': True}
            }
            
            # Compute stats
            rt_meta['multiplier_stats'] = {
                'p5': float(np.percentile(leverage_values, 5)),
                'p50': float(np.percentile(leverage_values, 50)),
                'p95': float(np.percentile(leverage_values, 95)),
                'at_cap': float(np.sum(leverage_values >= 7.0 - 1e-6) / len(leverage_values) * 100),
                'at_floor': float(np.sum(leverage_values <= 1.0 + 1e-6) / len(leverage_values) * 100)
            }
            
            if vol_file.exists():
                vol_df = pd.read_csv(vol_file, parse_dates=['date'])
                if not vol_df.empty:
                    vol_values = vol_df['realized_vol'].values
                    rt_meta['vol_stats'] = {
                        'p50': float(np.percentile(vol_values, 50)),
                        'p95': float(np.percentile(vol_values, 95))
                    }
    
    meta['risk_targeting'] = rt_meta

# Infer Allocator v1 governance from artifacts
alloc_v1_meta = meta.get('allocator_v1', {})
if alloc_v1_meta.get('enabled', False):
    regime_file = run_dir / "allocator" / "regime_series.csv"
    multiplier_file = run_dir / "allocator" / "multiplier_series.csv"
    
    if multiplier_file.exists():
        multiplier_df = pd.read_csv(multiplier_file, parse_dates=['date'])
        if not multiplier_df.empty:
            scalar_values = multiplier_df['multiplier'].values
            alloc_v1_meta['n_rebalances'] = len(multiplier_df)
            alloc_v1_meta['inputs_missing'] = False
            alloc_v1_meta['state_computed'] = True
            alloc_v1_meta['effective'] = True
            alloc_v1_meta['has_teeth'] = bool(np.any(scalar_values < 1.0 - 1e-6))
            
            # Update inputs_present
            alloc_v1_meta['inputs_present'] = {
                'portfolio_returns': {'present': True, 'has_data': True},
                'equity_curve': {'present': True, 'has_data': True},
                'asset_returns': {'present': True, 'has_data': True}
            }
            
            # Compute scalar stats
            alloc_v1_meta['scalar_stats'] = {
                'p5': float(np.percentile(scalar_values, 5)),
                'p50': float(np.percentile(scalar_values, 50)),
                'p95': float(np.percentile(scalar_values, 95)),
                'at_min': float(np.sum(scalar_values <= 0.25 + 1e-6) / len(scalar_values) * 100)
            }
            
            # Compute regime distribution
            if regime_file.exists():
                regime_df = pd.read_csv(regime_file, parse_dates=['date'])
                if not regime_df.empty and 'regime' in regime_df.columns:
                    from collections import Counter
                    regime_counts = Counter(regime_df['regime'])
                    total = len(regime_df)
                    alloc_v1_meta['regime_distribution'] = {
                        k: float(v / total * 100) for k, v in regime_counts.items()
                    }
    
    meta['allocator_v1'] = alloc_v1_meta

# Save updated meta.json
with open(run_dir / "meta.json", 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print(f"Updated meta.json for {run_id} with governance inferred from artifacts")
