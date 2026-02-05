"""
Analyze allocator regime attribution to understand why CRISIS/ELEVATED/STRESS are assigned.

This script provides:
1. Regime attribution summary (reason code breakdown)
2. Predicate trigger rates
3. Feature distribution sanity checks
4. Stress window analysis
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

run_id = sys.argv[1] if len(sys.argv) > 1 else None
if not run_id:
    print("Usage: python analyze_allocator_attribution.py <run_id>")
    sys.exit(1)

run_dir = Path("reports/runs") / run_id

print("=" * 80)
print("ALLOCATOR REGIME ATTRIBUTION ANALYSIS")
print("=" * 80)
print(f"Run ID: {run_id}\n")

# Load attribution DataFrame
attribution_file = run_dir / "allocator_regime_v1_attribution.csv"
if not attribution_file.exists():
    print(f"ERROR: Attribution file not found: {attribution_file}")
    print("This run may not have attribution data. Run with updated code.")
    sys.exit(1)

attribution_df = pd.read_csv(attribution_file, parse_dates=True, index_col=0)
print(f"Loaded attribution data: {len(attribution_df)} dates\n")

# Load state file for feature distributions
state_file = run_dir / "allocator_state_v1.csv"
if state_file.exists():
    state_df = pd.read_csv(state_file, parse_dates=True, index_col=0)
    print(f"Loaded state data: {len(state_df)} dates\n")
else:
    state_df = None
    print("WARNING: State file not found, skipping feature distribution analysis\n")

# 1) REGIME ATTRIBUTION SUMMARY
print("=" * 80)
print("1. REGIME ATTRIBUTION SUMMARY")
print("=" * 80)

regime_counts = attribution_df['regime'].value_counts()
regime_pct = (regime_counts / len(attribution_df) * 100).apply(lambda x: round(x, 1))
print(f"\nRegime Distribution:")
for regime, count in regime_counts.items():
    print(f"  {regime}: {count} ({regime_pct[regime]}%)")

# CRISIS attribution
crisis_mask = attribution_df['regime'] == 'CRISIS'
if crisis_mask.any():
    crisis_df = attribution_df[crisis_mask]
    crisis_total = len(crisis_df)
    
    print(f"\nCRISIS Attribution ({crisis_total} days):")
    crisis_reason_counts = crisis_df['crisis_reason'].value_counts()
    for reason, count in crisis_reason_counts.items():
        pct = round(count / crisis_total * 100, 1)
        print(f"  {reason}: {count} ({pct}%)")
    
    # Check persistence
    persistence_count = (crisis_df['crisis_reason'] == 'PERSISTENCE').sum()
    if persistence_count > 0:
        print(f"\n  [WARNING] PERSISTENCE: {persistence_count} days ({persistence_count/crisis_total*100:.1f}%)")
        print(f"     These are days trapped in CRISIS without meeting CRISIS conditions")

# STRESS attribution
stress_mask = attribution_df['regime'] == 'STRESS'
if stress_mask.any():
    stress_df = attribution_df[stress_mask]
    stress_total = len(stress_df)
    
    print(f"\nSTRESS Attribution ({stress_total} days):")
    stress_reason_counts = stress_df['stress_reason'].value_counts()
    for reason, count in stress_reason_counts.items():
        pct = round(count / stress_total * 100, 1)
        print(f"  {reason}: {count} ({pct}%)")

# 2) PREDICATE TRIGGER RATES
print("\n" + "=" * 80)
print("2. PREDICATE TRIGGER RATES")
print("=" * 80)

total_days = len(attribution_df)
print(f"\nOverall trigger rates (% of all {total_days} days):")
predicates = ['p_crisis_dd', 'p_crisis_score', 'p_crisis_triple', 
              'p_stress_dd', 'p_stress_volaccel', 'p_stress_corr', 
              'p_stress_score', 'p_elevated_score']

for pred in predicates:
    if pred in attribution_df.columns:
        trigger_count = attribution_df[pred].sum()
        trigger_pct = round(trigger_count / total_days * 100, 1)
        print(f"  {pred}: {trigger_count} ({trigger_pct}%)")

if crisis_mask.any():
    print(f"\nTrigger rates on CRISIS days ({crisis_mask.sum()} days):")
    for pred in predicates:
        if pred in attribution_df.columns:
            crisis_trigger = (attribution_df[crisis_mask][pred]).sum()
            crisis_total = crisis_mask.sum()
            crisis_pct = round(crisis_trigger / crisis_total * 100, 1) if crisis_total > 0 else 0
            print(f"  {pred}: {crisis_trigger} ({crisis_pct}%)")

# 3) HWM UPDATE FREQUENCY (Critical diagnostic for dd_level correctness)
if state_df is not None and 'equity_used_for_dd' in state_df.columns and 'hwm_used_for_dd' in state_df.columns:
    print("\n" + "=" * 80)
    print("3. HWM UPDATE FREQUENCY (Drawdown Baseline Diagnostic)")
    print("=" * 80)
    
    equity_series = state_df['equity_used_for_dd']
    hwm_series = state_df['hwm_used_for_dd']
    
    # Check if HWM updates (equity == HWM within epsilon)
    epsilon = 1e-6
    hwm_updates = (abs(equity_series - hwm_series) < epsilon)
    pct_hwm_updates = (hwm_updates.sum() / len(hwm_updates) * 100)
    
    # Compute consecutive days without HWM update
    no_update_streaks = []
    current_streak = 0
    for updated in hwm_updates:
        if updated:
            if current_streak > 0:
                no_update_streaks.append(current_streak)
            current_streak = 0
        else:
            current_streak += 1
    if current_streak > 0:
        no_update_streaks.append(current_streak)
    
    max_consecutive_no_update = max(no_update_streaks) if no_update_streaks else 0
    mean_consecutive_no_update = np.mean(no_update_streaks) if no_update_streaks else 0
    
    print(f"\nHWM Update Statistics:")
    print(f"  % days HWM updates (equity == HWM): {pct_hwm_updates:.1f}%")
    print(f"  Max consecutive days without HWM update: {max_consecutive_no_update}")
    print(f"  Mean consecutive days without HWM update: {mean_consecutive_no_update:.1f}")
    print(f"  Total HWM update events: {hwm_updates.sum()} out of {len(hwm_updates)} days")
    
    # Verification: dd_level should match recomputed value
    if 'dd_level_recomputed' in state_df.columns:
        dd_match = (abs(state_df['dd_level'] - state_df['dd_level_recomputed']) < 1e-6)
        mismatch_count = (~dd_match).sum()
        if mismatch_count > 0:
            print(f"\n  [ERROR] dd_level mismatch: {mismatch_count} dates where dd_level != dd_level_recomputed")
        else:
            print(f"\n  [OK] dd_level matches recomputed value for all dates")
    
    # Equity/HWM statistics
    print(f"\nEquity Series Statistics:")
    print(f"  min: {equity_series.min():.6f}")
    print(f"  p50: {equity_series.median():.6f}")
    print(f"  max: {equity_series.max():.6f}")
    print(f"\nHWM Series Statistics:")
    print(f"  min: {hwm_series.min():.6f}")
    print(f"  p50: {hwm_series.median():.6f}")
    print(f"  max: {hwm_series.max():.6f}")
    print(f"  Final HWM: {hwm_series.iloc[-1]:.6f}")
    print(f"  Final Equity: {equity_series.iloc[-1]:.6f}")
    
    # Red flag checks
    if pct_hwm_updates < 5:
        print(f"\n  [RED FLAG] HWM updates <5% of days - likely stuck at early peak")
    if max_consecutive_no_update > 500:
        print(f"\n  [RED FLAG] Max consecutive days without HWM update >500 - HWM likely stuck")
    if hwm_series.iloc[-1] / hwm_series.iloc[0] < 0.5:
        print(f"\n  [WARNING] HWM declined significantly from start - check if using correct equity curve")

# 4) FEATURE DISTRIBUTION SANITY
if state_df is not None:
    print("\n" + "=" * 80)
    print("4. FEATURE DISTRIBUTION SANITY")
    print("=" * 80)
    
    features = ['dd_level', 'dd_slope_10d', 'vol_accel', 'corr_shock']
    
    # Compute risk_score from state features
    from src.allocator.regime_rules_v1 import get_default_thresholds
    thresholds = get_default_thresholds()
    
    state_df['risk_score'] = (
        (state_df['vol_accel'] >= thresholds['vol_accel_enter']).astype(int) +
        (state_df['corr_shock'] >= thresholds['corr_shock_enter']).astype(int) +
        (state_df['dd_level'] <= thresholds['dd_enter']).astype(int) +
        (state_df['dd_slope_10d'] <= thresholds['dd_slope_enter']).astype(int)
    )
    features.append('risk_score')
    
    print(f"\nFeature Statistics:")
    for feat in features:
        if feat in state_df.columns:
            feat_data = state_df[feat].dropna()
            if len(feat_data) > 0:
                print(f"\n  {feat}:")
                print(f"    min: {feat_data.min():.6f}")
                print(f"    p5: {feat_data.quantile(0.05):.6f}")
                print(f"    p50: {feat_data.quantile(0.50):.6f}")
                print(f"    p95: {feat_data.quantile(0.95):.6f}")
                print(f"    max: {feat_data.max():.6f}")
                print(f"    NaN count: {state_df[feat].isna().sum()}")
                
                # Risk score specific analysis
                if feat == 'risk_score':
                    print(f"    Distribution:")
                    for score in range(5):
                        count = (feat_data == score).sum()
                        pct = (count / len(feat_data) * 100).round(1)
                        print(f"      {score}: {count} ({pct}%)")
                    print(f"    % >= 1: {(feat_data >= 1).sum() / len(feat_data) * 100:.1f}%")
                    print(f"    % >= 2: {(feat_data >= 2).sum() / len(feat_data) * 100:.1f}%")
                    print(f"    % >= 3: {(feat_data >= 3).sum() / len(feat_data) * 100:.1f}%")

# 5) STRESS WINDOW ANALYSIS
print("\n" + "=" * 80)
print("5. STRESS WINDOW ANALYSIS")
print("=" * 80)

# Load multiplier series for active % calculation
multiplier_file = run_dir / "allocator" / "multiplier_series.csv"
if multiplier_file.exists():
    multiplier_df = pd.read_csv(multiplier_file, parse_dates=['date'], index_col='date')
    
    windows = {
        '2020 Q1': ('2020-01-01', '2020-03-31'),
        '2022 H1': ('2022-01-01', '2022-06-30'),
        '2022 Q1': ('2022-01-01', '2022-03-31'),
        '2022 Q2': ('2022-04-01', '2022-06-30')
    }
    
    print(f"\nStress Window Analysis:")
    for window_name, (start, end) in windows.items():
        window_mask = (attribution_df.index >= start) & (attribution_df.index <= end)
        if window_mask.any():
            window_attribution = attribution_df[window_mask]
            window_multiplier = multiplier_df[multiplier_df.index.isin(window_attribution.index)]
            
            regime_counts = window_attribution['regime'].value_counts()
            active_pct = round((window_multiplier['multiplier'] < 1.0).sum() / len(window_multiplier) * 100, 1) if len(window_multiplier) > 0 else 0
            median_scalar = window_multiplier['multiplier'].median() if len(window_multiplier) > 0 else 1.0
            
            print(f"\n  {window_name}:")
            print(f"    Dates: {len(window_attribution)}")
            print(f"    Regime distribution:")
            for regime, count in regime_counts.items():
                pct = round(count / len(window_attribution) * 100, 1)
                print(f"      {regime}: {count} ({pct}%)")
            print(f"    Active %: {active_pct}%")
            print(f"    Median scalar: {median_scalar:.3f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
