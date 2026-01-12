"""Check allocator application in RT + Alloc-H run."""
import pandas as pd
from pathlib import Path

run_dir = Path('reports/runs/core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro_rt_alloc_h_2024-01-01_2024-12-31')

# Check if allocator files exist
alloc_applied = run_dir / 'allocator_risk_v1_applied.csv'
alloc_regime = run_dir / 'allocator_regime_v1.csv'

print('Allocator files exist:')
print(f'  applied: {alloc_applied.exists()}')
print(f'  regime: {alloc_regime.exists()}')

if alloc_applied.exists():
    df = pd.read_csv(alloc_applied, index_col=0)
    print(f'\nAllocator scalars applied:')
    print(f'  Count: {len(df)}')
    print(f'  Mean: {df["risk_scalar_applied"].mean():.4f}')
    print(f'  Min: {df["risk_scalar_applied"].min():.4f}')
    print(f'  Max: {df["risk_scalar_applied"].max():.4f}')
    print(f'  % < 0.999: {(df["risk_scalar_applied"] < 0.999).sum() / len(df) * 100:.1f}%')
    print(f'\nFirst 5 scalars:')
    print(df.head())

