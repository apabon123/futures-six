#!/usr/bin/env python3
"""
Phase 1C Artifact Validation Script

Validates that artifacts are correctly generated and meet acceptance criteria:
1. Risk Targeting artifacts exist and are correct
2. Allocator artifacts exist and show expected behavior
3. Sanity checks on weights (post ≈ pre * leverage)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def validate_risk_targeting_artifacts(run_dir: Path) -> Dict[str, bool]:
    """Validate Risk Targeting artifacts."""
    results = {}
    
    rt_dir = run_dir / "risk_targeting"
    
    # Check params.json exists
    params_file = rt_dir / "params.json"
    results['params_json_exists'] = params_file.exists()
    
    if params_file.exists():
        import json
        with open(params_file, 'r') as f:
            params = json.load(f)
        results['params_has_target_vol'] = 'target_vol' in params
        results['params_has_leverage_cap'] = 'leverage_cap' in params
        results['params_has_version_hash'] = 'version_hash' in params
    
    # Check leverage_series.csv
    leverage_file = rt_dir / "leverage_series.csv"
    results['leverage_series_exists'] = leverage_file.exists()
    
    if leverage_file.exists():
        leverage_df = pd.read_csv(leverage_file, parse_dates=['date'], index_col='date')
        results['leverage_has_data'] = len(leverage_df) > 0
        results['leverage_has_date_column'] = 'date' in leverage_df.columns or leverage_df.index.name == 'date'
        results['leverage_has_leverage_column'] = 'leverage' in leverage_df.columns
        if 'leverage' in leverage_df.columns:
            results['leverage_within_bounds'] = (
                leverage_df['leverage'].min() >= 0 and
                leverage_df['leverage'].max() <= 10.0  # Reasonable upper bound
            )
    
    # Check realized_vol.csv
    vol_file = rt_dir / "realized_vol.csv"
    results['realized_vol_exists'] = vol_file.exists()
    
    if vol_file.exists():
        vol_df = pd.read_csv(vol_file, parse_dates=['date'], index_col='date')
        results['realized_vol_has_data'] = len(vol_df) > 0
        results['realized_vol_has_vol_column'] = 'realized_vol' in vol_df.columns
    
    # Check weights files
    weights_pre_file = rt_dir / "weights_pre_risk_targeting.csv"
    weights_post_file = rt_dir / "weights_post_risk_targeting.csv"
    
    results['weights_pre_exists'] = weights_pre_file.exists()
    results['weights_post_exists'] = weights_post_file.exists()
    
    if weights_pre_file.exists() and weights_post_file.exists():
        weights_pre = pd.read_csv(weights_pre_file, parse_dates=['date'])
        weights_post = pd.read_csv(weights_post_file, parse_dates=['date'])
        
        results['weights_have_data'] = len(weights_pre) > 0 and len(weights_post) > 0
        
        # Sanity check: post_weights ≈ pre_weights * leverage_scalar(date)
        if results['weights_have_data'] and leverage_file.exists():
            leverage_df = pd.read_csv(leverage_file, parse_dates=['date'], index_col='date')
            
            # Sample a few dates for validation
            sample_dates = weights_pre['date'].unique()[:10]  # First 10 dates
            
            sanity_checks = []
            for date in sample_dates:
                date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                
                # Get leverage for this date
                if date_str in leverage_df.index.strftime('%Y-%m-%d').values:
                    leverage_idx = leverage_df.index.strftime('%Y-%m-%d') == date_str
                    if leverage_idx.any():
                        leverage_val = leverage_df.loc[leverage_idx, 'leverage'].iloc[0]
                    else:
                        continue
                else:
                    continue
                
                # Get weights for this date
                pre_weights = weights_pre[weights_pre['date'] == date].set_index('instrument')['weight']
                post_weights = weights_post[weights_post['date'] == date].set_index('instrument')['weight']
                
                # Align indices
                common = pre_weights.index.intersection(post_weights.index)
                if len(common) > 0:
                    pre_aligned = pre_weights.loc[common]
                    post_aligned = post_weights.loc[common]
                    
                    # Check: post ≈ pre * leverage (within tolerance)
                    expected_post = pre_aligned * leverage_val
                    diff = (post_aligned - expected_post).abs()
                    max_diff = diff.max()
                    relative_error = (diff / (pre_aligned.abs() + 1e-10)).max()
                    
                    sanity_checks.append({
                        'date': date_str,
                        'leverage': leverage_val,
                        'max_absolute_diff': max_diff,
                        'max_relative_error': relative_error,
                        'passes': max_diff < 0.01 or relative_error < 0.01  # 1% tolerance
                    })
            
            if sanity_checks:
                results['sanity_check_passes'] = all(c['passes'] for c in sanity_checks)
                results['sanity_check_details'] = sanity_checks
            else:
                results['sanity_check_passes'] = False
                results['sanity_check_details'] = []
    
    return results


def validate_allocator_artifacts(run_dir: Path) -> Dict[str, any]:
    """Validate Allocator artifacts."""
    results = {}
    
    alloc_dir = run_dir / "allocator"
    
    # Check regime_series.csv
    regime_file = alloc_dir / "regime_series.csv"
    results['regime_series_exists'] = regime_file.exists()
    
    if regime_file.exists():
        regime_df = pd.read_csv(regime_file, parse_dates=['date'], index_col='date')
        results['regime_has_data'] = len(regime_df) > 0
        results['regime_has_regime_column'] = 'regime' in regime_df.columns
        results['regime_has_profile_column'] = 'profile' in regime_df.columns
        
        if 'regime' in regime_df.columns:
            valid_regimes = ['NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']
            results['regime_values_valid'] = regime_df['regime'].isin(valid_regimes).all()
    
    # Check multiplier_series.csv
    multiplier_file = alloc_dir / "multiplier_series.csv"
    results['multiplier_series_exists'] = multiplier_file.exists()
    
    if multiplier_file.exists():
        multiplier_df = pd.read_csv(multiplier_file, parse_dates=['date'], index_col='date')
        results['multiplier_has_data'] = len(multiplier_df) > 0
        results['multiplier_has_multiplier_column'] = 'multiplier' in multiplier_df.columns
        results['multiplier_has_profile_column'] = 'profile' in multiplier_df.columns
        
        if 'multiplier' in multiplier_df.columns:
            results['multiplier_within_bounds'] = (
                multiplier_df['multiplier'].min() >= 0 and
                multiplier_df['multiplier'].max() <= 1.0
            )
            
            # Check for piecewise constant behavior (infrequent changes)
            changes = (multiplier_df['multiplier'].diff().abs() > 1e-6).sum()
            total_days = len(multiplier_df)
            pct_changes = (changes / total_days * 100) if total_days > 0 else 0
            results['multiplier_pct_days_with_changes'] = pct_changes
            
            # % days allocator active (multiplier != 1.0)
            active_days = (multiplier_df['multiplier'] != 1.0).sum()
            pct_active = (active_days / total_days * 100) if total_days > 0 else 0
            results['pct_days_allocator_active'] = pct_active
            results['total_days'] = total_days
            results['active_days'] = active_days
    
    return results


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Validate Phase 1C artifacts")
    ap.add_argument("--run_id", required=True, help="Run ID to validate")
    ap.add_argument("--run_dir", default="reports/runs", help="Base directory for runs")
    
    args = ap.parse_args()
    
    run_dir = Path(args.run_dir) / args.run_id
    
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return 1
    
    print(f"\n{'='*80}")
    print(f"VALIDATING ARTIFACTS FOR: {args.run_id}")
    print(f"{'='*80}\n")
    
    # Validate Risk Targeting artifacts
    print("Risk Targeting Artifacts:")
    print("-" * 80)
    rt_results = validate_risk_targeting_artifacts(run_dir)
    for key, value in rt_results.items():
        if key != 'sanity_check_details':
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
    
    if 'sanity_check_details' in rt_results and rt_results['sanity_check_details']:
        print("\n  Sanity Check Details (post ≈ pre * leverage):")
        for check in rt_results['sanity_check_details'][:5]:  # Show first 5
            status = "✓" if check['passes'] else "✗"
            print(f"    {status} {check['date']}: leverage={check['leverage']:.3f}, "
                  f"max_diff={check['max_absolute_diff']:.6f}, "
                  f"rel_error={check['max_relative_error']:.4f}")
    
    # Validate Allocator artifacts
    print("\nAllocator Artifacts:")
    print("-" * 80)
    alloc_results = validate_allocator_artifacts(run_dir)
    for key, value in alloc_results.items():
        status = "✓" if (isinstance(value, bool) and value) or (isinstance(value, (int, float)) and value >= 0) else "✗"
        if isinstance(value, float):
            print(f"  {status} {key}: {value:.2f}")
        else:
            print(f"  {status} {key}: {value}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    rt_passed = all(v for k, v in rt_results.items() if k != 'sanity_check_details' and isinstance(v, bool))
    alloc_passed = all(v for k, v in alloc_results.items() if isinstance(v, bool))
    
    print(f"Risk Targeting: {'✓ PASS' if rt_passed else '✗ FAIL'}")
    print(f"Allocator: {'✓ PASS' if alloc_passed else '✗ FAIL'}")
    
    if rt_passed and alloc_passed:
        print("\n✓ All artifact validations passed!")
        return 0
    else:
        print("\n✗ Some validations failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

