"""Check alignment between state features and regime series."""
import pandas as pd
import sys
from pathlib import Path

run_id = sys.argv[1] if len(sys.argv) > 1 else "allocator_calib_allocator_calib_v2_regime_aware_rt_p75_20260115_181323"
run_dir = Path("reports/runs") / run_id

# Load state features (daily)
state_daily = pd.read_csv(run_dir / "allocator_state_v1.csv", parse_dates=True, index_col=0)

# Load regime series (rebalance dates only)
regime_df = pd.read_csv(run_dir / "allocator/regime_series.csv", parse_dates=['date'], index_col='date')

print("=== DATE ALIGNMENT CHECK ===")
print(f"State file (daily): {len(state_daily)} dates, {state_daily.index.min()} to {state_daily.index.max()}")
print(f"Regime series (rebalance): {len(regime_df)} dates, {regime_df.index.min()} to {regime_df.index.max()}")

# Align to rebalance dates only
state_rebalance = state_daily.reindex(regime_df.index)

print(f"\nState features on rebalance dates: {len(state_rebalance)} dates")
print(f"Missing state features on rebalance dates: {state_rebalance.isnull().any(axis=1).sum()} dates")

# Compute risk scores on rebalance dates
if not state_rebalance.empty and all(col in state_rebalance.columns for col in ['vol_accel', 'corr_shock', 'dd_level', 'dd_slope_10d']):
    risk_scores_rebal = (
        (state_rebalance['vol_accel'] >= 1.70).astype(int) +
        (state_rebalance['corr_shock'] >= 0.20).astype(int) +
        (state_rebalance['dd_level'] <= -0.14).astype(int) +
        (state_rebalance['dd_slope_10d'] <= -0.10).astype(int)
    )
    
    print(f"\n=== RISK SCORE ON REBALANCE DATES ===")
    print(f"Risk score distribution:")
    print(risk_scores_rebal.value_counts().sort_index())
    print(f"\nRisk score >= 3: {(risk_scores_rebal >= 3).sum()} days ({(risk_scores_rebal >= 3).mean()*100:.1f}%)")
    
    print(f"\n=== CRISIS CONDITIONS ON REBALANCE DATES ===")
    crisis_dd = (state_rebalance['dd_level'] <= -0.20).sum()
    crisis_risk = (risk_scores_rebal >= 3).sum()
    crisis_triple = (
        (state_rebalance['vol_accel'] >= 1.70) &
        (state_rebalance['corr_shock'] >= 0.20) &
        (state_rebalance['dd_slope_10d'] <= -0.10)
    ).sum()
    
    print(f"  dd_level <= -0.20: {crisis_dd} days ({crisis_dd/len(regime_df)*100:.1f}%)")
    print(f"  risk_score >= 3: {crisis_risk} days ({crisis_risk/len(regime_df)*100:.1f}%)")
    print(f"  triple condition: {crisis_triple} days ({crisis_triple/len(regime_df)*100:.1f}%)")
    
    print(f"\n=== ACTUAL REGIME DISTRIBUTION ===")
    print(regime_df['regime'].value_counts())
    print(f"\nCRISIS days: {(regime_df['regime'] == 'CRISIS').sum()} ({(regime_df['regime'] == 'CRISIS').mean()*100:.1f}%)")
    
    # Check for mismatch
    if (regime_df['regime'] == 'CRISIS').sum() > 0:
        crisis_dates = regime_df[regime_df['regime'] == 'CRISIS'].index
        print(f"\n=== INSPECTING CRISIS DATES ===")
        print(f"First 10 CRISIS dates:")
        for date in crisis_dates[:10]:
            if date in state_rebalance.index:
                row = state_rebalance.loc[date]
                rs = risk_scores_rebal.loc[date]
                print(f"  {date}: dd={row['dd_level']:.3f}, risk_score={rs}, "
                      f"vol_accel={row['vol_accel']:.3f}, corr_shock={row['corr_shock']:.3f}")
