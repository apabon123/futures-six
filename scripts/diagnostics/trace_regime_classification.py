"""Trace regime classification to find first CRISIS assignment."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.allocator.regime_rules_v1 import get_default_thresholds

def check_initial_classification(run_id: str):
    """Step 1: Check first date's initial regime."""
    print("=" * 80)
    print("STEP 1: CHECK INITIAL CLASSIFICATION")
    print("=" * 80)
    
    run_dir = project_root / "reports" / "runs" / run_id
    
    # Load state and regime
    state_df = pd.read_csv(run_dir / "allocator_state_v1.csv", parse_dates=True, index_col=0)
    regime_df = pd.read_csv(run_dir / "allocator/regime_series.csv", parse_dates=['date'], index_col='date')
    
    # Align to rebalance dates
    state_rebal = state_df.reindex(regime_df.index)
    
    # Get first date
    first_date = regime_df.index[0]
    first_regime = regime_df.loc[first_date, 'regime']
    
    print(f"First date: {first_date}")
    print(f"First regime (from artifact): {first_regime}")
    
    # Compute regime statelessly on first date
    if first_date in state_rebal.index:
        row = state_rebal.loc[first_date]
        thresholds = get_default_thresholds()
        
        # Compute conditions
        s_vol_fast = row['vol_accel'] >= thresholds['vol_accel_enter']
        s_corr_spike = row['corr_shock'] >= thresholds['corr_shock_enter']
        s_dd_deep = row['dd_level'] <= thresholds['dd_enter']
        s_dd_worsening = row['dd_slope_10d'] <= thresholds['dd_slope_enter']
        
        risk_score = sum([s_vol_fast, s_corr_spike, s_dd_deep, s_dd_worsening])
        
        print(f"\nState features on first date:")
        print(f"  vol_accel: {row['vol_accel']:.3f}")
        print(f"  corr_shock: {row['corr_shock']:.3f}")
        print(f"  dd_level: {row['dd_level']:.3f}")
        print(f"  dd_slope_10d: {row['dd_slope_10d']:.3f}")
        
        print(f"\nConditions (ENTER thresholds):")
        print(f"  vol_accel >= {thresholds['vol_accel_enter']}: {s_vol_fast}")
        print(f"  corr_shock >= {thresholds['corr_shock_enter']}: {s_corr_spike}")
        print(f"  dd_level <= {thresholds['dd_enter']}: {s_dd_deep}")
        print(f"  dd_slope <= {thresholds['dd_slope_enter']}: {s_dd_worsening}")
        print(f"  risk_score: {risk_score}")
        
        # Check CRISIS conditions
        crisis_dd = row['dd_level'] <= thresholds['dd_crisis_enter']
        crisis_risk = risk_score >= 3
        crisis_triple = s_vol_fast and s_corr_spike and s_dd_worsening
        
        print(f"\nCRISIS conditions:")
        print(f"  dd_level <= {thresholds['dd_crisis_enter']}: {crisis_dd}")
        print(f"  risk_score >= 3: {crisis_risk}")
        print(f"  triple (vol AND corr AND dd_slope): {crisis_triple}")
        
        should_be_crisis = crisis_dd or crisis_risk or crisis_triple
        
        if should_be_crisis:
            print(f"\n[OK] First date SHOULD be CRISIS based on conditions")
        elif first_regime == 'CRISIS':
            print(f"\n[BUG] First date is CRISIS but SHOULD NOT be based on conditions!")
            print(f"      This is an initialization bug.")
        else:
            # Check what it should be
            if risk_score >= 2 or (s_vol_fast and s_corr_spike) or row['dd_level'] <= thresholds['dd_stress_enter']:
                should_be = 'STRESS'
            elif risk_score >= 1:
                should_be = 'ELEVATED'
            else:
                should_be = 'NORMAL'
            
            if first_regime == should_be:
                print(f"\n[OK] First date correctly classified as {first_regime}")
            else:
                print(f"\n[BUG] First date misclassified: {first_regime} (should be {should_be})")


def find_first_crisis_assignment(run_id: str):
    """Step 2: Find first CRISIS assignment and reason."""
    print("\n" + "=" * 80)
    print("STEP 2: FIND FIRST CRISIS ASSIGNMENT")
    print("=" * 80)
    
    run_dir = project_root / "reports" / "runs" / run_id
    
    # Load state and regime
    state_df = pd.read_csv(run_dir / "allocator_state_v1.csv", parse_dates=True, index_col=0)
    regime_df = pd.read_csv(run_dir / "allocator/regime_series.csv", parse_dates=['date'], index_col='date')
    
    # Align to rebalance dates
    state_rebal = state_df.reindex(regime_df.index)
    
    # Find first transition to CRISIS
    prev_regime = None
    thresholds = get_default_thresholds()
    
    for i, date in enumerate(regime_df.index):
        current_regime = regime_df.loc[date, 'regime']
        
        if prev_regime != 'CRISIS' and current_regime == 'CRISIS':
            print(f"\nFirst CRISIS assignment on: {date}")
            print(f"Previous regime: {prev_regime}")
            
            if date in state_rebal.index:
                row = state_rebal.loc[date]
                
                # Compute conditions
                s_vol_fast = row['vol_accel'] >= thresholds['vol_accel_enter']
                s_corr_spike = row['corr_shock'] >= thresholds['corr_shock_enter']
                s_dd_deep = row['dd_level'] <= thresholds['dd_enter']
                s_dd_worsening = row['dd_slope_10d'] <= thresholds['dd_slope_enter']
                
                risk_score = sum([s_vol_fast, s_corr_spike, s_dd_deep, s_dd_worsening])
                
                print(f"\nState features:")
                print(f"  vol_accel: {row['vol_accel']:.3f} (NaN: {pd.isna(row['vol_accel'])})")
                print(f"  corr_shock: {row['corr_shock']:.3f} (NaN: {pd.isna(row['corr_shock'])})")
                print(f"  dd_level: {row['dd_level']:.3f} (NaN: {pd.isna(row['dd_level'])})")
                print(f"  dd_slope_10d: {row['dd_slope_10d']:.3f} (NaN: {pd.isna(row['dd_slope_10d'])})")
                
                print(f"\nConditions:")
                print(f"  vol_accel >= {thresholds['vol_accel_enter']}: {s_vol_fast} (value: {row['vol_accel']:.3f})")
                print(f"  corr_shock >= {thresholds['corr_shock_enter']}: {s_corr_spike} (value: {row['corr_shock']:.3f})")
                print(f"  dd_level <= {thresholds['dd_enter']}: {s_dd_deep} (value: {row['dd_level']:.3f})")
                print(f"  dd_slope <= {thresholds['dd_slope_enter']}: {s_dd_worsening} (value: {row['dd_slope_10d']:.3f})")
                print(f"  risk_score: {risk_score}")
                
                # Check CRISIS conditions
                crisis_dd = row['dd_level'] <= thresholds['dd_crisis_enter']
                crisis_risk = risk_score >= 3
                crisis_triple = s_vol_fast and s_corr_spike and s_dd_worsening
                
                print(f"\nCRISIS trigger conditions:")
                print(f"  [1] dd_level <= {thresholds['dd_crisis_enter']}: {crisis_dd}")
                print(f"  [2] risk_score >= 3: {crisis_risk}")
                print(f"  [3] triple (vol AND corr AND dd_slope): {crisis_triple}")
                
                if crisis_dd:
                    print(f"\n[REASON] CRISIS triggered by: dd_level <= {thresholds['dd_crisis_enter']}")
                elif crisis_risk:
                    print(f"\n[REASON] CRISIS triggered by: risk_score >= 3")
                elif crisis_triple:
                    print(f"\n[REASON] CRISIS triggered by: triple condition")
                else:
                    print(f"\n[BUG] CRISIS assigned but NO conditions are true!")
                    print(f"      This is a classification logic bug.")
                
                break
        
        prev_regime = current_regime


def inspect_downgrade_logic(run_id: str):
    """Step 3: Inspect _can_downgrade for trapped states."""
    print("\n" + "=" * 80)
    print("STEP 3: INSPECT DOWNGRADE LOGIC")
    print("=" * 80)
    
    run_dir = project_root / "reports" / "runs" / run_id
    
    # Load state and regime
    state_df = pd.read_csv(run_dir / "allocator_state_v1.csv", parse_dates=True, index_col=0)
    regime_df = pd.read_csv(run_dir / "allocator/regime_series.csv", parse_dates=['date'], index_col='date')
    
    # Align to rebalance dates
    state_rebal = state_df.reindex(regime_df.index)
    
    thresholds = get_default_thresholds()
    
    # Find all CRISIS dates that don't meet CRISIS conditions
    crisis_dates = regime_df[regime_df['regime'] == 'CRISIS'].index
    
    print(f"Total CRISIS dates: {len(crisis_dates)}")
    
    # Check first 10 CRISIS dates that don't meet conditions
    trapped_count = 0
    consecutive_crisis = 0
    max_consecutive = 0
    
    prev_regime = None
    days_in_crisis = 0
    
    for i, date in enumerate(regime_df.index):
        current_regime = regime_df.loc[date, 'regime']
        
        if current_regime == 'CRISIS':
            if prev_regime == 'CRISIS':
                consecutive_crisis += 1
            else:
                consecutive_crisis = 1
                days_in_crisis = 1
            
            max_consecutive = max(max_consecutive, consecutive_crisis)
            
            # Check if conditions are met
            if date in state_rebal.index:
                row = state_rebal.loc[date]
                
                # EXIT conditions
                s_vol_exit = row['vol_accel'] >= thresholds['vol_accel_exit']
                s_corr_exit = row['corr_shock'] >= thresholds['corr_shock_exit']
                s_dd_exit = row['dd_level'] <= thresholds['dd_exit']
                s_dd_slope_exit = row['dd_slope_10d'] <= thresholds['dd_slope_exit']
                
                risk_score_exit = sum([s_vol_exit, s_corr_exit, s_dd_exit, s_dd_slope_exit])
                
                # ENTER conditions
                s_vol_enter = row['vol_accel'] >= thresholds['vol_accel_enter']
                s_corr_enter = row['corr_shock'] >= thresholds['corr_shock_enter']
                s_dd_enter = row['dd_level'] <= thresholds['dd_enter']
                s_dd_slope_enter = row['dd_slope_10d'] <= thresholds['dd_slope_enter']
                
                risk_score_enter = sum([s_vol_enter, s_corr_enter, s_dd_enter, s_dd_slope_enter])
                
                # CRISIS conditions
                crisis_dd = row['dd_level'] <= thresholds['dd_crisis_enter']
                crisis_risk = risk_score_enter >= 3
                crisis_triple = s_vol_enter and s_corr_enter and s_dd_slope_enter
                
                meets_crisis = crisis_dd or crisis_risk or crisis_triple
                
                if not meets_crisis and trapped_count < 5:
                    trapped_count += 1
                    print(f"\n[TRAPPED] Date: {date}, days_in_crisis: {consecutive_crisis}")
                    print(f"  ENTER risk_score: {risk_score_enter}, EXIT risk_score: {risk_score_exit}")
                    print(f"  Meets CRISIS conditions: {meets_crisis}")
                    print(f"  EXIT risk_score < 3: {risk_score_exit < 3}")
                    print(f"  days_in_crisis >= {thresholds['min_days_in_regime']}: {consecutive_crisis >= thresholds['min_days_in_regime']}")
                    print(f"  Should downgrade: {risk_score_exit < 3 and consecutive_crisis >= thresholds['min_days_in_regime']}")
        else:
            consecutive_crisis = 0
            days_in_crisis = 0
        
        prev_regime = current_regime
    
    print(f"\nMax consecutive days in CRISIS: {max_consecutive}")
    print(f"MIN_DAYS_IN_REGIME: {thresholds['min_days_in_regime']}")


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "allocator_calib_allocator_calib_v2_regime_aware_rt_p75_20260115_181323"
    
    check_initial_classification(run_id)
    find_first_crisis_assignment(run_id)
    inspect_downgrade_logic(run_id)
