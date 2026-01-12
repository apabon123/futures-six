"""
Quick acceptance test for RT artifact fix.

Validates that weights_pre/post artifacts now contain full instrument set.
"""
import pandas as pd
from pathlib import Path

def test_rt_artifacts(run_id: str):
    """Test RT artifacts for correctness."""
    run_dir = Path(f"reports/runs/{run_id}")
    
    if not run_dir.exists():
        print(f"FAIL: Run directory not found: {run_dir}")
        return False
    
    # Load artifacts
    leverage_path = run_dir / 'risk_targeting' / 'leverage_series.csv'
    weights_pre_path = run_dir / 'risk_targeting' / 'weights_pre_risk_targeting.csv'
    weights_post_path = run_dir / 'risk_targeting' / 'weights_post_risk_targeting.csv'
    
    if not all([p.exists() for p in [leverage_path, weights_pre_path, weights_post_path]]):
        print("FAIL: RT artifacts not found")
        return False
    
    leverage_df = pd.read_csv(leverage_path)
    weights_pre_df = pd.read_csv(weights_pre_path)
    weights_post_df = pd.read_csv(weights_post_path)
    
    print("="*80)
    print(f"RT Artifact Acceptance Test - Run: {run_id}")
    print("="*80)
    
    # Test 1: Check first rebalance date
    first_date = leverage_df.iloc[0]['date']
    first_leverage = leverage_df.iloc[0]['leverage']
    
    pre = weights_pre_df[weights_pre_df['date'] == first_date]
    post = weights_post_df[weights_post_df['date'] == first_date]
    
    num_instruments_pre = len(pre)
    num_instruments_post = len(post)
    gross_pre = pre['weight'].abs().sum()
    gross_post = post['weight'].abs().sum()
    
    print(f"\nTest 1: First Rebalance Date ({first_date})")
    print(f"  Leverage (artifact):     {first_leverage:.2f}x")
    print(f"  Instruments (pre):       {num_instruments_pre}")
    print(f"  Instruments (post):      {num_instruments_post}")
    print(f"  Gross (pre):             {gross_pre:.2f}")
    print(f"  Gross (post):            {gross_post:.2f}")
    print(f"  Expected gross (post):   ~{first_leverage:.2f}x (since we normalize to unit gross)")
    
    # Pass criteria
    test1_pass = (
        num_instruments_pre > 5 and  # Should have many instruments, not just 1!
        num_instruments_post > 5 and
        abs(gross_post - first_leverage) < 0.5  # Gross should be close to leverage
    )
    
    print(f"\n  Result: {'PASS' if test1_pass else 'FAIL'}")
    if not test1_pass:
        if num_instruments_pre <= 5:
            print(f"    - FAIL: weights_pre has only {num_instruments_pre} instruments (bug not fixed!)")
        if abs(gross_post - first_leverage) >= 0.5:
            print(f"    - FAIL: gross_post={gross_post:.2f} != leverage={first_leverage:.2f}")
    
    # Test 2: Check all dates have multiple instruments
    print(f"\nTest 2: Panel Data Structure")
    dates = weights_pre_df['date'].unique()
    instruments_per_date = weights_pre_df.groupby('date').size()
    
    min_instruments = instruments_per_date.min()
    max_instruments = instruments_per_date.max()
    mean_instruments = instruments_per_date.mean()
    
    print(f"  Num dates:               {len(dates)}")
    print(f"  Instruments per date:    min={min_instruments}, max={max_instruments}, mean={mean_instruments:.1f}")
    
    test2_pass = min_instruments > 5  # Every date should have many instruments
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")
    if not test2_pass:
        print(f"    - FAIL: Some dates have only {min_instruments} instruments")
    
    # Test 3: Gross consistency across all dates
    print(f"\nTest 3: Gross Consistency")
    errors = []
    for date in dates[:3]:  # Check first 3 dates
        lev = leverage_df[leverage_df['date'] == date]['leverage'].iloc[0]
        pre = weights_pre_df[weights_pre_df['date'] == date]
        post = weights_post_df[weights_post_df['date'] == date]
        
        gross_pre_val = pre['weight'].abs().sum()
        gross_post_val = post['weight'].abs().sum()
        
        error = abs(gross_post_val - lev)
        errors.append(error)
        
        print(f"  {date}: lev={lev:.2f}, gross_pre={gross_pre_val:.2f}, gross_post={gross_post_val:.2f}, error={error:.3f}")
    
    test3_pass = max(errors) < 0.5
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")
    
    # Overall result
    print("\n" + "="*80)
    all_pass = test1_pass and test2_pass and test3_pass
    if all_pass:
        print("OVERALL: PASS - RT artifacts are correct!")
    else:
        print("OVERALL: FAIL - RT artifacts still have issues")
    print("="*80)
    
    return all_pass


if __name__ == "__main__":
    import sys
    
    # Test the artifact fix run
    run_id = "test_rt_artifact_fix_2024-01-01_2024-01-15"
    
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    
    success = test_rt_artifacts(run_id)
    sys.exit(0 if success else 1)

