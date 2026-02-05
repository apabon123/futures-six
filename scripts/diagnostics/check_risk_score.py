"""Quick check of risk score components."""
import pandas as pd
import sys
from pathlib import Path

run_id = sys.argv[1] if len(sys.argv) > 1 else "allocator_calib_allocator_calib_v2_regime_aware_rt_p75_20260115_181323"
run_dir = Path("reports/runs") / run_id

df = pd.read_csv(run_dir / "allocator_state_v1.csv", parse_dates=True, index_col=0)

print("=== CHECKING RISK SCORE COMPONENTS ===")
print(f"Total dates: {len(df)}")
print(f"\nIndividual conditions (ENTER thresholds):")
print(f"  vol_accel >= 1.70: {(df['vol_accel'] >= 1.70).sum()} days ({(df['vol_accel'] >= 1.70).mean()*100:.1f}%)")
print(f"  corr_shock >= 0.20: {(df['corr_shock'] >= 0.20).sum()} days ({(df['corr_shock'] >= 0.20).mean()*100:.1f}%)")
print(f"  dd_level <= -0.14: {(df['dd_level'] <= -0.14).sum()} days ({(df['dd_level'] <= -0.14).mean()*100:.1f}%)")
print(f"  dd_slope <= -0.10: {(df['dd_slope_10d'] <= -0.10).sum()} days ({(df['dd_slope_10d'] <= -0.10).mean()*100:.1f}%)")

risk_scores = (
    (df['vol_accel'] >= 1.70).astype(int) +
    (df['corr_shock'] >= 0.20).astype(int) +
    (df['dd_level'] <= -0.14).astype(int) +
    (df['dd_slope_10d'] <= -0.10).astype(int)
)

print(f"\nRisk score distribution:")
print(risk_scores.value_counts().sort_index())
print(f"\nRisk score >= 3: {(risk_scores >= 3).sum()} days ({(risk_scores >= 3).mean()*100:.1f}%)")

triple = (
    (df['vol_accel'] >= 1.70) &
    (df['corr_shock'] >= 0.20) &
    (df['dd_slope_10d'] <= -0.10)
)
print(f"\nTriple condition (vol AND corr AND dd_slope):")
print(f"  Triple condition: {triple.sum()} days ({triple.mean()*100:.1f}%)")

# Check what's causing CRISIS
crisis_conditions = {
    'dd_level <= -0.20': (df['dd_level'] <= -0.20).sum(),
    'risk_score >= 3': (risk_scores >= 3).sum(),
    'triple condition': triple.sum()
}
print(f"\nCRISIS trigger conditions:")
for cond, count in crisis_conditions.items():
    print(f"  {cond}: {count} days ({count/len(df)*100:.1f}%)")
