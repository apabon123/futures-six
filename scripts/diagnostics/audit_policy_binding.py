"""
Policy Binding Audit Script

This script investigates why policy multipliers are all 1.0 (not binding).

It checks:
1. Feature values vs thresholds
2. Gate logic (inequality direction)
3. Units mismatch
4. Default-to-1 behavior masking NaNs
5. Rebalance alignment (daily state vs rebalance dates)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.canonical_window import load_canonical_window


def audit_policy_binding(run_id: str):
    """Audit policy binding for a given run."""
    run_dir = project_root / "reports" / "runs" / run_id
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    print("=" * 80)
    print(f"POLICY BINDING AUDIT: {run_id}")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print()
    
    # Load state file (daily)
    state_path = run_dir / "engine_policy_state_v1.csv"
    if not state_path.exists():
        print(f"⚠️  State file not found: {state_path}")
        print("   Policy state file should be written by compute_state() in compute mode.")
        return
    
    state_df = pd.read_csv(state_path)
    state_df['date'] = pd.to_datetime(state_df['date'])
    print(f"[OK] Loaded state file: {len(state_df)} rows")
    print(f"  Date range: {state_df['date'].min()} to {state_df['date'].max()}")
    print(f"  Engines: {sorted(state_df['engine'].unique())}")
    print()
    
    # Load applied multipliers (rebalance dates)
    applied_path = run_dir / "engine_policy_applied_v1.csv"
    if not applied_path.exists():
        print(f"[WARNING] Applied multipliers file not found: {applied_path}")
        return
    
    applied_df = pd.read_csv(applied_path)
    applied_df['rebalance_date'] = pd.to_datetime(applied_df['rebalance_date'])
    print(f"[OK] Loaded applied multipliers: {len(applied_df)} rows")
    print()
    
    # Load config to get thresholds
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"[OK] Loaded meta.json")
        print()
    
    # For each engine, analyze feature values and gate logic
    engines = sorted(state_df['engine'].unique())
    
    # Default thresholds (from DEFAULT_RULES in engine_policy_v1.py)
    default_thresholds = {
        'trend': 1,
        'vrp': 1,
    }
    
    for engine in engines:
        print("=" * 80)
        print(f"ENGINE: {engine.upper()}")
        print("=" * 80)
        
        engine_state = state_df[state_df['engine'] == engine].copy()
        engine_state = engine_state.sort_values('date')
        
        # Get feature values
        feature_values = engine_state['stress_value'].dropna()
        
        if len(feature_values) == 0:
            print(f"[ISSUE] NO FEATURE VALUES (all NaN)")
            print(f"   This engine has no feature data - policy defaults to ON (multiplier=1)")
            print(f"   This is the root cause: features are missing/not found")
            print()
            continue
        
        print(f"Feature values:")
        print(f"  Count: {len(feature_values)} / {len(engine_state)} ({len(feature_values)/len(engine_state)*100:.1f}% non-null)")
        print(f"  Min: {feature_values.min():.4f}")
        print(f"  Max: {feature_values.max():.4f}")
        print(f"  Mean: {feature_values.mean():.4f}")
        print(f"  Median: {feature_values.median():.4f}")
        print(f"  Std: {feature_values.std():.4f}")
        print()
        
        # Get threshold (from config or default)
        threshold = default_thresholds.get(engine, 1)
        print(f"Threshold: {threshold}")
        
        # Check gate logic (default: OFF when feature >= threshold, invert=False)
        # So: gate OFF (multiplier=0) when feature >= threshold
        gated_days = (feature_values >= threshold).sum()
        print(f"Days where feature >= threshold (should gate OFF): {gated_days} / {len(feature_values)} ({gated_days/len(feature_values)*100:.1f}%)")
        
        if gated_days > 0:
            first_gated = feature_values[feature_values >= threshold].index[0]
            first_gated_date = engine_state.loc[first_gated, 'date']
            print(f"  First gated date: {first_gated_date}")
        
        # Check policy state in state file
        gated_state_count = (engine_state['policy_multiplier'] == 0).sum()
        print(f"Policy multiplier = 0 in state file: {gated_state_count} / {len(engine_state)} ({gated_state_count/len(engine_state)*100:.1f}%)")
        print()
        
        # Check applied multipliers (rebalance dates)
        if engine == 'trend':
            applied_col = 'trend_multiplier'
        elif engine == 'vrp':
            applied_col = 'vrp_multiplier'
        else:
            continue
        
        if applied_col in applied_df.columns:
            applied_multipliers = applied_df[applied_col]
            gated_rebalances = (applied_multipliers < 0.999).sum()
            print(f"Applied multipliers (rebalance dates):")
            print(f"  Gated rebalances (< 0.999): {gated_rebalances} / {len(applied_multipliers)} ({gated_rebalances/len(applied_multipliers)*100:.1f}%)")
            print(f"  Min multiplier: {applied_multipliers.min():.4f}")
            print(f"  Max multiplier: {applied_multipliers.max():.4f}")
            
            if gated_rebalances > 0:
                gated_dates = applied_df[applied_multipliers < 0.999]['rebalance_date']
                print(f"  First 5 gated dates: {gated_dates.head(5).tolist()}")
            print()
        
        # Diagnostic: Check if feature values are reasonable
        print(f"Diagnostics:")
        if feature_values.max() < threshold:
            print(f"  [ISSUE] MAX FEATURE < THRESHOLD: Feature never exceeds threshold")
            print(f"      This suggests either:")
            print(f"      - Feature is wrong/unscaled")
            print(f"      - Threshold is too high")
            print(f"      - Units mismatch")
        elif gated_days == 0:
            print(f"  [ISSUE] NO DAYS GATED: Feature exceeds threshold but no days gated")
            print(f"      This suggests gate logic issue (wrong inequality direction?)")
        elif gated_days > 0 and gated_state_count == 0:
            print(f"  [ISSUE] STATE FILE NOT GATING: Feature exceeds threshold but state file shows multiplier=1")
            print(f"      This suggests compute_state() gate logic is broken")
        elif gated_state_count > 0 and gated_rebalances == 0:
            print(f"  [ISSUE] REBALANCE ALIGNMENT ISSUE: State file shows gating but rebalance multipliers=1")
            print(f"      This suggests lag_rebalances or rebalance alignment problem")
        else:
            print(f"  [OK] Policy binding appears correct")
        print()
    
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit policy binding for a run")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID to audit")
    
    args = parser.parse_args()
    
    audit_policy_binding(args.run_id)
