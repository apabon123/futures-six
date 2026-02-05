"""Check regime transitions."""
import pandas as pd
from pathlib import Path

run_id = "allocator_calib_allocator_calib_v2_regime_aware_rt_p75_20260115_181323"
run_dir = Path("reports/runs") / run_id

df = pd.read_csv(run_dir / "allocator/regime_series.csv", parse_dates=['date'], index_col='date')

print("Regime on dates around 2022-12-02:")
print(df.loc['2022-11-18':'2022-12-16'])

print("\nRegime transitions:")
prev = None
for date in df.index:
    curr = df.loc[date, 'regime']
    if prev != curr:
        print(f"{date}: {prev} -> {curr}")
    prev = curr
