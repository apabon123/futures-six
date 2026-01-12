"""Verify Risk Targeting weights alignment - spot check one date."""
import pandas as pd
from pathlib import Path

# Use the existing RT only run from full year 2024
run_dir = Path("reports/runs/core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro_rt_only_2024-01-01_2024-12-31")

# Pick a representative date (a rebalance date - Friday closest to top drawdown)
test_date = '2024-11-15'  # Friday rebalance

# Load artifacts
weights_pre_df = pd.read_csv(run_dir / 'risk_targeting' / 'weights_pre_risk_targeting.csv')
weights_post_df = pd.read_csv(run_dir / 'risk_targeting' / 'weights_post_risk_targeting.csv')
leverage_series = pd.read_csv(run_dir / 'risk_targeting' / 'leverage_series.csv')

# Convert to wide format for easier analysis
weights_pre_df['date'] = pd.to_datetime(weights_pre_df['date'])
weights_post_df['date'] = pd.to_datetime(weights_post_df['date'])
leverage_series['date'] = pd.to_datetime(leverage_series['date'])

# Extract values for test date
test_date_dt = pd.to_datetime(test_date)
try:
    leverage = leverage_series[leverage_series['date'] == test_date_dt]['leverage'].iloc[0]
    pre = weights_pre_df[weights_pre_df['date'] == test_date_dt].set_index('instrument')['weight']
    post = weights_post_df[weights_post_df['date'] == test_date_dt].set_index('instrument')['weight']
except (KeyError, IndexError) as e:
    print(f"Error: Date {test_date} not found in artifacts")
    print(f"Available dates: {weights_pre_df['date'].unique()[:5]} ... {weights_pre_df['date'].unique()[-5:]}")
    raise

# Compute statistics
gross_pre = pre.abs().sum()
gross_post = post.abs().sum()
expected_gross_post = gross_pre * leverage

# Compute weight ratios
weight_ratios = (post / pre).replace([float('inf'), -float('inf')], None).dropna()

print("=" * 80)
print(f"Risk Targeting Weights Alignment Check - Date: {test_date}")
print("=" * 80)
print(f"\n1. Leverage:")
print(f"   Leverage scalar: {leverage:.4f}×")

print(f"\n2. Gross Exposure:")
print(f"   Gross (pre-RT):  {gross_pre:.4f}")
print(f"   Gross (post-RT): {gross_post:.4f}")
print(f"   Expected:        {expected_gross_post:.4f}")
print(f"   Match: {'✅ YES' if abs(gross_post - expected_gross_post) < 0.01 else '❌ NO'}")
print(f"   Error: {abs(gross_post - expected_gross_post):.6f}")

print(f"\n3. Weight Ratios (post/pre):")
print(f"   Unique ratios: {weight_ratios.unique()[:10]}")  # Show first 10
print(f"   All ratios equal to leverage? {'✅ YES' if weight_ratios.nunique() == 1 and abs(weight_ratios.iloc[0] - leverage) < 0.001 else '❌ NO'}")
print(f"   Mean ratio: {weight_ratios.mean():.4f}")
print(f"   Std ratio: {weight_ratios.std():.6f}")

print(f"\n4. Sample Assets:")
assets_to_show = ['ES_FRONT_CALENDAR_2D', 'ZN_FRONT_VOLUME', '6E_FRONT_CALENDAR']
for asset in assets_to_show:
    if asset in pre.index and asset in post.index:
        print(f"   {asset}:")
        print(f"      Pre:   {pre[asset]:8.4f}")
        print(f"      Post:  {post[asset]:8.4f}")
        print(f"      Ratio: {post[asset] / pre[asset] if pre[asset] != 0 else 0:8.4f}")

print("\n" + "=" * 80)
print("CONCLUSION:")
if abs(gross_post - expected_gross_post) < 0.01 and weight_ratios.nunique() == 1:
    print("✅ RT is working correctly:")
    print("   - Gross exposure scales by leverage")
    print("   - All asset weights scale uniformly")
else:
    print("❌ RT has issues:")
    if abs(gross_post - expected_gross_post) >= 0.01:
        print(f"   - Gross exposure mismatch ({gross_post:.2f} vs {expected_gross_post:.2f})")
    if weight_ratios.nunique() > 1:
        print(f"   - Weights not scaling uniformly (ratios: {weight_ratios.unique()[:5]})")
print("=" * 80)

