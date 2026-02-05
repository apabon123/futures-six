"""
Phase 3B Baseline Verification Script

Verifies all 6 checkpoints for Phase 3B baselines:
1. Artifact existence
2. Hard-pass contract
3. Allocator meta coherence
4. Identity check (artifacts_only)
5. Sidecar absence
6. Waterfall sanity (basic check)
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def verify_checkpoints(run_id: str, base_dir: str = "reports/runs") -> Dict:
    """Verify all 6 checkpoints for a Phase 3B baseline."""
    
    run_dir = Path(base_dir) / run_id
    
    if not run_dir.exists():
        return {
            'run_id': run_id,
            'status': 'error',
            'error': f"Run directory not found: {run_dir}"
        }
    
    results = {
        'run_id': run_id,
        'status': 'verifying',
        'checkpoints': {}
    }
    
    # Checkpoint 1: Artifact existence
    required_artifacts = [
        'weights_post_construction.csv',
        'weights_post_risk_targeting.csv',
        'weights_post_allocator.csv',
        'weights_used_for_portfolio_returns.csv',
        'portfolio_returns.csv'
    ]
    
    artifacts_exist = {}
    all_exist = True
    for artifact in required_artifacts:
        exists = (run_dir / artifact).exists()
        artifacts_exist[artifact] = exists
        if not exists:
            all_exist = False
    
    results['checkpoints']['1_artifacts_exist'] = {
        'passed': all_exist,
        'details': artifacts_exist
    }
    
    if not all_exist:
        results['status'] = 'failed'
        results['error'] = "Missing required artifacts"
        return results
    
    # Checkpoint 2: Hard-pass contract (portfolio_returns_base == sum(weights_used × instrument_returns))
    try:
        # Phase 3B: Use portfolio_returns_base.csv (the exact vector used before any transforms)
        portfolio_returns_base_path = run_dir / 'portfolio_returns_base.csv'
        if portfolio_returns_base_path.exists():
            portfolio_returns_base = pd.read_csv(portfolio_returns_base_path, index_col=0, parse_dates=True)
        else:
            # Fallback to portfolio_returns.csv if base doesn't exist (legacy)
            portfolio_returns_base = pd.read_csv(run_dir / 'portfolio_returns.csv', index_col=0, parse_dates=True)
        
        weights_used = pd.read_csv(run_dir / 'weights_used_for_portfolio_returns.csv', index_col=0, parse_dates=True)
        asset_returns = pd.read_csv(run_dir / 'asset_returns.csv', index_col=0, parse_dates=True)
        
        # Align and compute portfolio returns from weights
        weights_daily = weights_used.reindex(asset_returns.index).ffill().fillna(0.0)
        common_symbols = weights_daily.columns.intersection(asset_returns.columns)
        
        if len(common_symbols) > 0:
            weights_aligned = weights_daily[common_symbols]
            returns_aligned = asset_returns[common_symbols]
            
            # Phase 3B: asset_returns.csv contains SIMPLE returns (already converted from log in exec_sim.py line 2087-2091)
            # Runtime computes: portfolio_log = sum(weights * log_returns), then converts to simple: exp(portfolio_log) - 1
            # To reconstruct, we need to convert simple returns back to log, then compute portfolio_log, then convert to simple
            # log_return = log(1 + simple_return)
            # portfolio_log = sum(weights * log_returns)
            # portfolio_simple = exp(portfolio_log) - 1
            returns_log = np.log(1 + returns_aligned)  # Convert simple returns back to log returns
            portfolio_returns_log = (weights_aligned * returns_log).sum(axis=1)
            portfolio_returns_computed = np.exp(portfolio_returns_log) - 1.0  # Convert portfolio log return to simple
            
            # Align dates with portfolio_returns_base
            portfolio_returns_base_aligned = portfolio_returns_base['ret'].reindex(portfolio_returns_computed.index)
            common_dates = portfolio_returns_computed.index.intersection(portfolio_returns_base_aligned.index)
            
            if len(common_dates) > 0:
                computed_aligned = portfolio_returns_computed.loc[common_dates]
                actual_aligned = portfolio_returns_base_aligned.loc[common_dates]
                
                diff = computed_aligned - actual_aligned
                abs_diff = diff.abs()
                max_abs_diff = abs_diff.max()
                max_rel_diff = (abs_diff / (actual_aligned.abs() + 1e-8)).max()
                
                # Phase 3B: Find first mismatch location for diagnostics
                first_mismatch_info = {}
                if max_abs_diff > 1e-6 or max_rel_diff > 0.01:
                    first_mismatch_idx = abs_diff.idxmax()
                    first_mismatch_date = first_mismatch_idx
                    first_mismatch_info = {
                        'first_mismatch_date': str(first_mismatch_date),
                        'first_mismatch_computed': float(computed_aligned.loc[first_mismatch_date]),
                        'first_mismatch_actual': float(actual_aligned.loc[first_mismatch_date]),
                        'first_mismatch_abs_diff': float(abs_diff.loc[first_mismatch_date])
                    }
                
                contract_passed = max_abs_diff < 1e-6 and max_rel_diff < 0.01
                
                checkpoint_details = {
                    'max_abs_diff': float(max_abs_diff),
                    'max_rel_diff': float(max_rel_diff),
                    'details': {
                        'computed_count': len(computed_aligned),
                        'actual_count': len(actual_aligned),
                        'common_dates': len(common_dates)
                    }
                }
                
                if first_mismatch_info:
                    checkpoint_details['first_mismatch'] = first_mismatch_info
                
                results['checkpoints']['2_contract_hard_pass'] = {
                    'passed': contract_passed,
                    **checkpoint_details
                }
                
                if not contract_passed:
                    results['status'] = 'failed'
                    error_msg = f"Contract violation: max_abs_diff={max_abs_diff:.8f}, max_rel_diff={max_rel_diff:.4%}"
                    if first_mismatch_info:
                        error_msg += f"\nFirst mismatch on {first_mismatch_info['first_mismatch_date']}: computed={first_mismatch_info['first_mismatch_computed']:.8f}, actual={first_mismatch_info['first_mismatch_actual']:.8f}, diff={first_mismatch_info['first_mismatch_abs_diff']:.8f}"
                    results['error'] = error_msg
            else:
                results['checkpoints']['2_contract_hard_pass'] = {
                    'passed': False,
                    'error': "No common dates between computed and actual returns"
                }
                results['status'] = 'failed'
        else:
            results['checkpoints']['2_contract_hard_pass'] = {
                'passed': False,
                'error': "No common symbols between weights and asset returns"
            }
            results['status'] = 'failed'
    except Exception as e:
        results['checkpoints']['2_contract_hard_pass'] = {
            'passed': False,
            'error': str(e)
        }
        results['status'] = 'failed'
    
    # Checkpoint 3: Allocator meta coherence
    try:
        with open(run_dir / 'meta.json', 'r') as f:
            meta = json.load(f)
        
        allocator_meta = meta.get('allocator_v1', {})
        canonical_mode = allocator_meta.get('canonical_mode')
        applied_to_weights = allocator_meta.get('applied_to_weights', False)
        apply_enabled = allocator_meta.get('apply_enabled', True)
        had_effect = allocator_meta.get('had_effect', False)
        had_clip = allocator_meta.get('had_clip', False)
        
        # Verify coherence
        mode_coherent = True
        if apply_enabled is False:
            mode_coherent = (canonical_mode == 'artifacts_only') and (applied_to_weights == False)
        elif canonical_mode == 'applied':
            mode_coherent = applied_to_weights == True
        
        results['checkpoints']['3_allocator_meta_coherent'] = {
            'passed': mode_coherent,
            'canonical_mode': canonical_mode,
            'apply_enabled': apply_enabled,
            'applied_to_weights': applied_to_weights,
            'had_effect': had_effect,
            'had_clip': had_clip
        }
    except Exception as e:
        results['checkpoints']['3_allocator_meta_coherent'] = {
            'passed': False,
            'error': str(e)
        }
    
    # Checkpoint 4: Identity check (if artifacts_only)
    if canonical_mode == 'artifacts_only' or apply_enabled is False:
        try:
            weights_post_rt = pd.read_csv(run_dir / 'weights_post_risk_targeting.csv', index_col=0, parse_dates=True)
            weights_post_alloc = pd.read_csv(run_dir / 'weights_post_allocator.csv', index_col=0, parse_dates=True)
            
            # Align indices
            common_dates = weights_post_rt.index.intersection(weights_post_alloc.index)
            if len(common_dates) > 0:
                weights_rt_aligned = weights_post_rt.loc[common_dates]
                weights_alloc_aligned = weights_post_alloc.loc[common_dates]
                
                # Check if they match (allowing for small numerical differences)
                max_abs_diff = (weights_rt_aligned - weights_alloc_aligned).abs().max().max()
                max_rel_diff = ((weights_rt_aligned - weights_alloc_aligned).abs() / (weights_rt_aligned.abs() + 1e-8)).max().max()
                
                identity_passed = max_abs_diff < 1e-6 and max_rel_diff < 0.01
                
                results['checkpoints']['4_identity_check'] = {
                    'passed': identity_passed,
                    'max_abs_diff': float(max_abs_diff),
                    'max_rel_diff': float(max_rel_diff),
                    'common_dates': len(common_dates)
                }
            else:
                results['checkpoints']['4_identity_check'] = {
                    'passed': False,
                    'error': "No common dates between post-RT and post-allocator weights"
                }
        except Exception as e:
            results['checkpoints']['4_identity_check'] = {
                'passed': False,
                'error': str(e)
            }
    else:
        results['checkpoints']['4_identity_check'] = {
            'passed': True,
            'skipped': True,
            'reason': f"Not artifacts_only mode (canonical_mode={canonical_mode})"
        }
    
    # Checkpoint 5: Sidecar absence (portfolio_returns == portfolio_returns_base)
    # Phase 3B: Verify portfolio_returns.csv equals portfolio_returns_base.csv exactly
    try:
        portfolio_returns = pd.read_csv(run_dir / 'portfolio_returns.csv', index_col=0, parse_dates=True)
        portfolio_returns_base_path = run_dir / 'portfolio_returns_base.csv'
        
        if portfolio_returns_base_path.exists():
            portfolio_returns_base = pd.read_csv(portfolio_returns_base_path, index_col=0, parse_dates=True)
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(portfolio_returns_base.index)
            if len(common_dates) > 0:
                returns_aligned = portfolio_returns['ret'].loc[common_dates]
                base_aligned = portfolio_returns_base['ret'].loc[common_dates]
                
                diff = returns_aligned - base_aligned
                abs_diff = diff.abs()
                max_abs_diff = abs_diff.max()
                max_rel_diff = (abs_diff / (base_aligned.abs() + 1e-8)).max()
                
                sidecar_absent = max_abs_diff < 1e-6 and max_rel_diff < 0.01
                
                results['checkpoints']['5_sidecar_absent'] = {
                    'passed': sidecar_absent,
                    'max_abs_diff': float(max_abs_diff),
                    'max_rel_diff': float(max_rel_diff),
                    'note': 'Verifies portfolio_returns == portfolio_returns_base (no sidecar additions)'
                }
                
                if not sidecar_absent:
                    results['status'] = 'failed'
            else:
                results['checkpoints']['5_sidecar_absent'] = {
                    'passed': False,
                    'error': 'No common dates between portfolio_returns and portfolio_returns_base'
                }
                results['status'] = 'failed'
        else:
            # Fallback: use checkpoint 2 result if base file doesn't exist
            if '2_contract_hard_pass' in results['checkpoints']:
                contract_result = results['checkpoints']['2_contract_hard_pass']
                results['checkpoints']['5_sidecar_absent'] = {
                    'passed': contract_result.get('passed', False),
                    'max_abs_diff': contract_result.get('max_abs_diff'),
                    'max_rel_diff': contract_result.get('max_rel_diff'),
                    'note': 'portfolio_returns_base.csv not found, using checkpoint 2 result'
                }
            else:
                results['checkpoints']['5_sidecar_absent'] = {
                    'passed': False,
                    'error': 'portfolio_returns_base.csv not found and contract check failed'
                }
    except Exception as e:
        results['checkpoints']['5_sidecar_absent'] = {
            'passed': False,
            'error': str(e)
        }
        results['status'] = 'failed'
    
    # Checkpoint 6: Waterfall sanity (basic - just check if files can be loaded)
    try:
        # This is a basic sanity check - full waterfall would need to run the diagnostic
        # We just verify that the stage artifacts can be loaded and have reasonable values
        weights_post_rt = pd.read_csv(run_dir / 'weights_post_risk_targeting.csv', index_col=0, parse_dates=True)
        weights_post_alloc = pd.read_csv(run_dir / 'weights_post_allocator.csv', index_col=0, parse_dates=True)
        
        rt_nonzero = (weights_post_rt.abs() > 1e-6).any().any()
        alloc_nonzero = (weights_post_alloc.abs() > 1e-6).any().any()
        
        if canonical_mode == 'artifacts_only':
            # Should be similar
            common_dates = weights_post_rt.index.intersection(weights_post_alloc.index)
            if len(common_dates) > 0:
                diff = (weights_post_rt.loc[common_dates] - weights_post_alloc.loc[common_dates]).abs().max().max()
                waterfall_sane = diff < 1e-4  # Allow small differences
            else:
                waterfall_sane = False
        else:
            # Applied mode - weights should differ if allocator had effect
            common_dates = weights_post_rt.index.intersection(weights_post_alloc.index)
            if len(common_dates) > 0 and had_clip:
                diff = (weights_post_rt.loc[common_dates] - weights_post_alloc.loc[common_dates]).abs().max().max()
                waterfall_sane = diff > 1e-6  # Should differ if clipping occurred
            else:
                waterfall_sane = True  # If no clipping, similarity is fine
        
        results['checkpoints']['6_waterfall_sanity'] = {
            'passed': waterfall_sane,
            'canonical_mode': canonical_mode,
            'had_clip': had_clip,
            'rt_has_weights': rt_nonzero,
            'alloc_has_weights': alloc_nonzero
        }
    except Exception as e:
        results['checkpoints']['6_waterfall_sanity'] = {
            'passed': False,
            'error': str(e)
        }
    
    # Overall status
    all_passed = all(
        cp.get('passed', False) for cp in results['checkpoints'].values()
    )
    
    if results['status'] != 'failed':
        results['status'] = 'passed' if all_passed else 'warning'
    
    return results


def print_results(results: Dict):
    """Print verification results in a readable format."""
    print("=" * 80)
    print(f"Phase 3B Baseline Verification: {results['run_id']}")
    print("=" * 80)
    print(f"Status: {results['status'].upper()}")
    print()
    
    for checkpoint_name, checkpoint_result in sorted(results['checkpoints'].items()):
        passed = checkpoint_result.get('passed', False)
        status_icon = "[OK]" if passed else "[FAIL]"
        print(f"{status_icon} {checkpoint_name}: {'PASSED' if passed else 'FAILED'}")
        
        if not passed and 'error' in checkpoint_result:
            print(f"   Error: {checkpoint_result['error']}")
        
        # Print relevant details
        if 'max_abs_diff' in checkpoint_result:
            print(f"   Max abs diff: {checkpoint_result['max_abs_diff']:.8f}")
        if 'max_rel_diff' in checkpoint_result:
            print(f"   Max rel diff: {checkpoint_result['max_rel_diff']:.4%}")
        if 'first_mismatch' in checkpoint_result:
            mismatch = checkpoint_result['first_mismatch']
            print(f"   First mismatch date: {mismatch['first_mismatch_date']}")
            print(f"   First mismatch computed: {mismatch['first_mismatch_computed']:.8f}")
            print(f"   First mismatch actual: {mismatch['first_mismatch_actual']:.8f}")
            print(f"   First mismatch abs diff: {mismatch['first_mismatch_abs_diff']:.8f}")
        if 'canonical_mode' in checkpoint_result:
            print(f"   Canonical mode: {checkpoint_result['canonical_mode']}")
        if 'applied_to_weights' in checkpoint_result:
            print(f"   Applied to weights: {checkpoint_result['applied_to_weights']}")
        if 'had_clip' in checkpoint_result:
            print(f"   Had clip: {checkpoint_result['had_clip']}")
        print()
    
    if results['status'] == 'passed':
        print("[OK] All checkpoints passed!")
    elif results['status'] == 'failed':
        print("[FAIL] Some checkpoints failed - see details above")
    print("=" * 80)


def main():
    """Verify both Phase 3B baselines."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Phase 3B baseline checkpoints")
    parser.add_argument('--run_id', type=str, help='Single run ID to verify')
    parser.add_argument('--both', action='store_true', help='Verify both artifacts_only and traded baselines')
    
    args = parser.parse_args()
    
    if args.run_id:
        results = verify_checkpoints(args.run_id)
        print_results(results)
    elif args.both:
        # Find latest baselines
        base_dir = Path("reports/runs")
        artifacts_only_runs = sorted([d for d in base_dir.glob("phase3b_baseline_artifacts_only_*") if d.is_dir()])
        traded_runs = sorted([d for d in base_dir.glob("phase3b_baseline_traded_*") if d.is_dir()])
        
        if artifacts_only_runs:
            latest_artifacts = artifacts_only_runs[-1].name
            print("\n" + "="*80)
            print("VERIFYING ARTIFACTS-ONLY BASELINE")
            print("="*80 + "\n")
            results1 = verify_checkpoints(latest_artifacts)
            print_results(results1)
        
        if traded_runs:
            latest_traded = traded_runs[-1].name
            print("\n" + "="*80)
            print("VERIFYING TRADED BASELINE")
            print("="*80 + "\n")
            results2 = verify_checkpoints(latest_traded)
            print_results(results2)
    else:
        # Default: verify both latest
        base_dir = Path("reports/runs")
        artifacts_only_runs = sorted([d for d in base_dir.glob("phase3b_baseline_artifacts_only_*") if d.is_dir()])
        traded_runs = sorted([d for d in base_dir.glob("phase3b_baseline_traded_*") if d.is_dir()])
        
        if artifacts_only_runs:
            latest_artifacts = artifacts_only_runs[-1].name
            print("\n" + "="*80)
            print("VERIFYING ARTIFACTS-ONLY BASELINE")
            print("="*80 + "\n")
            results1 = verify_checkpoints(latest_artifacts)
            print_results(results1)
        
        if traded_runs:
            latest_traded = traded_runs[-1].name
            print("\n" + "="*80)
            print("VERIFYING TRADED BASELINE")
            print("="*80 + "\n")
            results2 = verify_checkpoints(latest_traded)
            print_results(results2)


if __name__ == "__main__":
    main()
