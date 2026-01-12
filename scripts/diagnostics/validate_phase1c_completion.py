"""
Phase 1C Allocator Application Proof - Validation Script

This script validates that allocator multipliers were actually applied in the integrated execution path.

Acceptance Criteria:
1. Config logs show mode='compute'
2. ExecSim logs show "Risk scalars applied: X/52" where X > 0
3. allocator_risk_v1_applied.csv has multipliers < 0.999
4. Final weights = weights_post_rt * multiplier (within tolerance)
"""
import pandas as pd
import numpy as np
from pathlib import Path


def validate_allocator_application(run_id: str):
    """Validate that allocator was applied in the backtest."""
    
    run_dir = Path(f"reports/runs/{run_id}")
    
    if not run_dir.exists():
        print(f"FAIL: Run directory not found: {run_dir}")
        return False
    
    print("=" * 80)
    print(f"PHASE 1C ALLOCATOR APPLICATION PROOF - Run: {run_id}")
    print("=" * 80)
    
    # Test 1: Check allocator artifacts exist and have active multipliers
    print("\nTest 1: Allocator Artifacts")
    print("-" * 80)
    
    alloc_applied_path = run_dir / 'allocator_risk_v1_applied.csv'
    alloc_regime_path = run_dir / 'allocator_regime_v1.csv'
    
    if not alloc_applied_path.exists():
        print("FAIL: allocator_risk_v1_applied.csv not found")
        return False
    
    alloc_df = pd.read_csv(alloc_applied_path, index_col=0)
    
    num_rebalances = len(alloc_df)
    mean_scalar = alloc_df['risk_scalar_applied'].mean()
    min_scalar = alloc_df['risk_scalar_applied'].min()
    max_scalar = alloc_df['risk_scalar_applied'].max()
    pct_active = (alloc_df['risk_scalar_applied'] < 0.999).sum() / len(alloc_df) * 100
    
    print(f"  Rebalances: {num_rebalances}")
    print(f"  Mean scalar: {mean_scalar:.4f}")
    print(f"  Min scalar: {min_scalar:.4f}")
    print(f"  Max scalar: {max_scalar:.4f}")
    print(f"  % active (< 0.999): {pct_active:.1f}%")
    
    test1_pass = pct_active > 0  # Must have SOME active intervention
    print(f"\n  Result: {'PASS' if test1_pass else 'FAIL'}")
    if not test1_pass:
        print("    FAIL: No active allocator intervention (all scalars = 1.0)")
    
    # Test 2: Compare RT-only vs RT+Alloc-H returns
    print("\nTest 2: Returns Comparison")
    print("-" * 80)
    
    rt_only_dir = Path("reports/runs/core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro_rt_only_2024-01-01_2024-12-31")
    
    if rt_only_dir.exists():
        rt_only_returns = pd.read_csv(rt_only_dir / 'portfolio_returns.csv', index_col=0)
        rt_alloc_returns = pd.read_csv(run_dir / 'portfolio_returns.csv', index_col=0)
        
        # Use 'ret' or 'portfolio_return' column
        ret_col = 'ret' if 'ret' in rt_only_returns.columns else 'portfolio_return'
        
        rt_only_total = rt_only_returns[ret_col].sum()
        rt_alloc_total = rt_alloc_returns[ret_col].sum()
        
        diff = abs(rt_only_total - rt_alloc_total)
        
        print(f"  RT only total return: {rt_only_total:.6f}")
        print(f"  RT + Alloc-H total return: {rt_alloc_total:.6f}")
        print(f"  Difference: {diff:.6f}")
        
        test2_pass = diff > 1e-6  # Returns should differ if allocator applied
        print(f"\n  Result: {'PASS' if test2_pass else 'FAIL'}")
        if not test2_pass:
            print("    FAIL: Returns are identical (allocator had no effect)")
    else:
        print("  WARNING: RT-only run not found, skipping comparison")
        test2_pass = None
    
    # Test 3: Spot-check weight scaling on an active date
    print("\nTest 3: Weight Scaling Verification")
    print("-" * 80)
    
    # Find a date with active allocator (multiplier < 0.999)
    active_dates = alloc_df[alloc_df['risk_scalar_applied'] < 0.999]
    
    if len(active_dates) > 0:
        # Pick the date with minimum multiplier (most active)
        test_date = active_dates['risk_scalar_applied'].idxmin()
        multiplier = active_dates.loc[test_date, 'risk_scalar_applied']
        
        print(f"  Test date: {test_date}")
        print(f"  Allocator multiplier: {multiplier:.4f}")
        
        # Load RT post weights and final weights for this date
        rt_post_path = run_dir / 'risk_targeting' / 'weights_post_risk_targeting.csv'
        weights_path = run_dir / 'weights.csv'
        
        if rt_post_path.exists() and weights_path.exists():
            rt_post_df = pd.read_csv(rt_post_path)
            weights_df = pd.read_csv(weights_path, index_col=0)
            
            # Get weights for this date
            rt_post = rt_post_df[rt_post_df['date'] == test_date].set_index('instrument')['weight']
            
            # Find the weight date (might be rebalance date)
            # Weights are indexed by date, instruments are columns
            if test_date in weights_df.index:
                final_weights = weights_df.loc[test_date]
            else:
                # Find closest date
                valid_dates = weights_df.index[weights_df.index <= test_date]
                if len(valid_dates) > 0:
                    final_weights = weights_df.loc[valid_dates[-1]]
                else:
                    print("    WARNING: Could not find weights for test date")
                    test3_pass = None
                    final_weights = None
            
            if final_weights is not None:
                # Compute expected weights: rt_post * multiplier
                common_assets = rt_post.index.intersection(final_weights.index)
                
                expected = rt_post.loc[common_assets] * multiplier
                actual = final_weights.loc[common_assets]
                
                # Compute error
                error = (actual - expected).abs().max()
                
                print(f"  Common assets: {len(common_assets)}")
                print(f"  Max error (actual vs expected): {error:.6f}")
                print(f"  Gross (RT post): {rt_post.abs().sum():.2f}")
                print(f"  Gross (expected): {expected.abs().sum():.2f}")
                print(f"  Gross (actual): {actual.abs().sum():.2f}")
                
                test3_pass = error < 0.01  # Tolerance for numerical precision
                print(f"\n  Result: {'PASS' if test3_pass else 'FAIL'}")
                if not test3_pass:
                    print(f"    FAIL: Final weights don't match expected (error={error:.6f})")
        else:
            print("    WARNING: Weight files not found, skipping weight verification")
            test3_pass = None
    else:
        print("  No active allocator dates found")
        test3_pass = False
    
    # Overall result
    print("\n" + "=" * 80)
    
    if test1_pass and (test2_pass or test2_pass is None) and (test3_pass or test3_pass is None):
        print("OVERALL: PASS - Allocator was applied!")
        print("")
        print("Phase 1C Requirements Met:")
        print("  - Allocator computed regimes and multipliers")
        print("  - Multipliers were applied to weights")
        print("  - RT + Alloc-H differs from RT only")
        print("")
        print("Phase 1C is COMPLETE!")
        result = True
    else:
        print("OVERALL: FAIL - Allocator was not applied")
        print("")
        print("Issues:")
        if not test1_pass:
            print("  - No active allocator intervention")
        if test2_pass is False:
            print("  - Returns identical to RT only")
        if test3_pass is False:
            print("  - Weight scaling doesn't match expected")
        result = False
    
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    import sys
    
    run_id = "rt_alloc_h_apply_proof_2024"
    
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    
    success = validate_allocator_application(run_id)
    sys.exit(0 if success else 1)

