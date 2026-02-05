"""Extract CRISIS-related logs from a run's console output or check artifacts."""
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

run_id = sys.argv[1] if len(sys.argv) > 1 else "allocator_calib_allocator_calib_v4_full_logging_20260115_210632"
run_dir = Path("reports/runs") / run_id

print("=" * 80)
print("EXTRACTING CRISIS DIAGNOSTIC INFORMATION")
print("=" * 80)

# Check if state_hash is in artifacts
state_file = run_dir / "allocator_state_v1.csv"
if state_file.exists():
    state_df = pd.read_csv(state_file, parse_dates=True, index_col=0)
    print(f"\nState file: {len(state_df)} rows")
    print(f"Columns: {list(state_df.columns)}")
    print(f"Has state_hash: {'state_hash' in state_df.columns}")
    
    if 'state_hash' in state_df.columns:
        print(f"Sample state_hash values:")
        print(state_df[['dd_level', 'state_hash']].head(10))
else:
    print(f"State file not found: {state_file}")

# Check regime series
regime_file = run_dir / "allocator" / "regime_series.csv"
if regime_file.exists():
    regime_df = pd.read_csv(regime_file, parse_dates=['date'], index_col='date')
    print(f"\nRegime series: {len(regime_df)} rows")
    print(f"Regime distribution:")
    print(regime_df['regime'].value_counts())
    
    # Find first CRISIS
    crisis_dates = regime_df[regime_df['regime'] == 'CRISIS'].index
    if len(crisis_dates) > 0:
        first_crisis = crisis_dates[0]
        print(f"\nFirst CRISIS date: {first_crisis}")
        
        # Check state on that date
        if state_file.exists() and first_crisis in state_df.index:
            row = state_df.loc[first_crisis]
            print(f"\nState features on first CRISIS date ({first_crisis}):")
            print(f"  dd_level: {row.get('dd_level', 'N/A')}")
            print(f"  vol_accel: {row.get('vol_accel', 'N/A')}")
            print(f"  corr_shock: {row.get('corr_shock', 'N/A')}")
            print(f"  dd_slope_10d: {row.get('dd_slope_10d', 'N/A')}")
            if 'state_hash' in row:
                print(f"  state_hash: {row['state_hash']}")
            
            # Check what regime should be
            from src.allocator.regime_rules_v1 import get_default_thresholds
            thresholds = get_default_thresholds()
            
            vol_accel = row.get('vol_accel', 0)
            corr_shock = row.get('corr_shock', 0)
            dd_level = row.get('dd_level', 0)
            dd_slope = row.get('dd_slope_10d', 0)
            
            s_vol = vol_accel >= thresholds['vol_accel_enter']
            s_corr = corr_shock >= thresholds['corr_shock_enter']
            s_dd = dd_level <= thresholds['dd_enter']
            s_slope = dd_slope <= thresholds['dd_slope_enter']
            
            risk_score = sum([s_vol, s_corr, s_dd, s_slope])
            
            print(f"\nConditions (ENTER thresholds):")
            print(f"  vol_accel >= {thresholds['vol_accel_enter']}: {s_vol} (value: {vol_accel})")
            print(f"  corr_shock >= {thresholds['corr_shock_enter']}: {s_corr} (value: {corr_shock})")
            print(f"  dd_level <= {thresholds['dd_enter']}: {s_dd} (value: {dd_level})")
            print(f"  dd_slope <= {thresholds['dd_slope_enter']}: {s_slope} (value: {dd_slope})")
            print(f"  risk_score: {risk_score}")
            
            # CRISIS conditions
            crisis_dd = dd_level <= thresholds['dd_crisis_enter']
            crisis_risk = risk_score >= 3
            crisis_triple = s_vol and s_corr and s_slope
            
            print(f"\nCRISIS predicates:")
            print(f"  dd_level <= {thresholds['dd_crisis_enter']}: {crisis_dd}")
            print(f"  risk_score >= 3: {crisis_risk}")
            print(f"  triple (vol AND corr AND dd_slope): {crisis_triple}")
            print(f"  Should be CRISIS: {crisis_dd or crisis_risk or crisis_triple}")
            
            # STRESS conditions
            stress_dd = dd_level <= thresholds['dd_stress_enter']
            print(f"\nSTRESS predicate:")
            print(f"  dd_level <= {thresholds['dd_stress_enter']}: {stress_dd}")
            print(f"  Should be STRESS: {stress_dd or risk_score >= 2 or (s_vol and s_corr)}")
