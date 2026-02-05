"""
Debug Allocator Calibration - Systematic Checks

Performs checks A-D to identify root cause of calibration issues.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_a_artifact_coverage(run_dir: Path):
    """Check A: Validate artifact coverage."""
    print("=" * 80)
    print("CHECK A: ARTIFACT COVERAGE")
    print("=" * 80)
    
    regime_file = run_dir / "allocator" / "regime_series.csv"
    if not regime_file.exists():
        print(f"ERROR: {regime_file} not found")
        return
    
    df = pd.read_csv(regime_file, parse_dates=['date'])
    
    print(f"Total rows: {len(df)}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    df['year'] = df['date'].dt.year
    print(f"\nBy year:")
    print(df['year'].value_counts().sort_index())
    
    print(f"\nExpected: ~263 rebalance dates (2020-01-06 to 2025-10-31, W-FRI)")
    print(f"Actual: {len(df)} dates")
    print(f"Missing: ~{263 - len(df)} dates")
    
    if len(df) < 250:
        print("\n[WARNING] Significantly fewer dates than expected!")
        print("   This suggests date alignment or warmup issues.")
    
    return df


def check_b_trigger_rates(run_dir: Path):
    """Check B: Compute trigger rates for each condition."""
    print("\n" + "=" * 80)
    print("CHECK B: TRIGGER RATES")
    print("=" * 80)
    
    state_file = run_dir / "allocator_state_v1.csv"
    if not state_file.exists():
        print(f"ERROR: {state_file} not found")
        return
    
    df = pd.read_csv(state_file, parse_dates=True, index_col=0)
    
    # Current thresholds (Run 2)
    thresholds_run2 = {
        'vol_accel_enter': 1.70,
        'corr_shock_enter': 0.20,
        'dd_enter': -0.14,
        'dd_stress_enter': -0.18,
        'dd_crisis_enter': -0.20,
        'dd_slope_enter': -0.10,
    }
    
    # Previous thresholds (Run 1)
    thresholds_run1 = {
        'vol_accel_enter': 1.50,
        'corr_shock_enter': 0.15,
        'dd_enter': -0.12,
        'dd_stress_enter': -0.15,
        'dd_crisis_enter': -0.20,
        'dd_slope_enter': -0.08,
    }
    
    print("\nTrigger rates (% of days where condition is true):")
    print("-" * 80)
    
    conditions = {
        'vol_accel >= threshold': ('vol_accel', 'vol_accel_enter'),
        'corr_shock >= threshold': ('corr_shock', 'corr_shock_enter'),
        'dd_level <= dd_enter': ('dd_level', 'dd_enter'),
        'dd_level <= dd_stress_enter': ('dd_level', 'dd_stress_enter'),
        'dd_level <= dd_crisis_enter': ('dd_level', 'dd_crisis_enter'),
        'dd_slope <= threshold': ('dd_slope_10d', 'dd_slope_enter'),
    }
    
    results = []
    for desc, (col, key) in conditions.items():
        if col not in df.columns:
            print(f"  {desc}: Column '{col}' not found")
            continue
        
        # Run 1 threshold
        thresh1 = thresholds_run1.get(key)
        trigger_rate1 = (df[col] >= thresh1).mean() * 100 if thresh1 is not None and thresh1 > 0 else (df[col] <= thresh1).mean() * 100
        
        # Run 2 threshold
        thresh2 = thresholds_run2.get(key)
        trigger_rate2 = (df[col] >= thresh2).mean() * 100 if thresh2 is not None and thresh2 > 0 else (df[col] <= thresh2).mean() * 100
        
        # Check monotonicity
        is_increasing = trigger_rate2 > trigger_rate1
        status = "[VIOLATED]" if is_increasing else "OK"
        
        results.append({
            'condition': desc,
            'run1_thresh': thresh1,
            'run1_rate': trigger_rate1,
            'run2_thresh': thresh2,
            'run2_rate': trigger_rate2,
            'change': trigger_rate2 - trigger_rate1,
            'status': status
        })
        
        print(f"  {desc}:")
        print(f"    Run 1: threshold={thresh1}, rate={trigger_rate1:.1f}%")
        print(f"    Run 2: threshold={thresh2}, rate={trigger_rate2:.1f}%")
        print(f"    Change: {trigger_rate2 - trigger_rate1:+.1f}%  {status}")
    
    return results


def check_c_crisis_logic():
    """Check C: Verify CRISIS assignment logic."""
    print("\n" + "=" * 80)
    print("CHECK C: CRISIS ASSIGNMENT LOGIC")
    print("=" * 80)
    
    # Read regime_v1.py to check logic
    regime_file = project_root / "src" / "allocator" / "regime_v1.py"
    with open(regime_file, 'r') as f:
        content = f.read()
    
    # Find CRISIS assignment
        if '_compute_enter_regime' in content:
            start_idx = content.find('_compute_enter_regime')
            end_idx = content.find('def _can_downgrade', start_idx)
            crisis_logic = content[start_idx:end_idx] if end_idx > start_idx else content[start_idx:start_idx+500]
            
            print("CRISIS assignment logic:")
            print("-" * 80)
            
            if 'dd_level <= dd_crisis_enter' in crisis_logic:
                print("  [OK] Found: dd_level <= dd_crisis_enter")
            if 'risk_score >= 3' in crisis_logic:
                print("  [OK] Found: risk_score >= 3")
            if 'and s_dd_worsening' in crisis_logic:
                print("  [OK] Found: (s_vol_fast and s_corr_spike and s_dd_worsening)")
        
        print("\n  Note: CRISIS triggers on ANY of these conditions (OR logic)")
        print("        If any single condition is common, CRISIS will be common too.")
    
    return True


def check_d_rt_gate_logic():
    """Check D: Verify RT gate doesn't affect regime classification."""
    print("\n" + "=" * 80)
    print("CHECK D: RT GATE LOGIC")
    print("=" * 80)
    
    # Read exec_sim.py to check gate logic
    exec_file = project_root / "src" / "agents" / "exec_sim.py"
    with open(exec_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find RT gate logic
    if 'RT leverage' in content or 'rt_gate' in content:
        print("RT gate logic found in exec_sim.py")
        print("-" * 80)
        
        # Check if gate affects regime or just scalar
        if 'current_regime' in content and 'rt_leverage' in content:
            print("  [OK] Gate uses both regime and RT leverage")
        
        # Check if it's OR or AND
        if '(Regime >= STRESS) OR (RT >= p75)' in content or 'stress_or_crisis or rt_high_enough' in content:
            print("  [OK] Gate uses OR logic: (Regime >= STRESS) OR (RT >= p75)")
        elif '(Regime >= STRESS) AND (RT >= p75)' in content:
            print("  [WARNING] Gate uses AND logic (should be OR)")
        
        # Check if gate affects regime classification
        if 'regime =' in content and 'rt_leverage' in content:
            print("  [WARNING] Gate may be affecting regime classification!")
        else:
            print("  [OK] Gate appears to only affect scalar, not regime")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    args = parser.parse_args()
    
    run_dir = project_root / "reports" / "runs" / args.run_id
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Run all checks
    df_regime = check_a_artifact_coverage(run_dir)
    trigger_rates = check_b_trigger_rates(run_dir)
    check_c_crisis_logic()
    check_d_rt_gate_logic()
    
    print("\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    print("\nReview the outputs above to identify the root cause.")
