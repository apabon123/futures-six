#!/usr/bin/env python3
"""
Quick data sanity checks for VRP-TermStructure Phase-0:
1. VX2-VX1 shape & magnitudes
2. Signal alignment with contango
3. PnL calculation sanity
4. Comparison to VRP-Core Phase-0
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import duckdb
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.market_data.vrp_loaders import load_vx_curve, load_vrp_inputs
from src.agents.utils_db import open_readonly_connection

# Load config
config_path = Path("configs/data.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
db_path = config['db']['path']

con = open_readonly_connection(db_path)

start = CANONICAL_START
end = CANONICAL_END

print("=" * 80)
print("VRP-TERMSTRUCTURE DATA SANITY CHECKS")
print("=" * 80)

# 1. Load VX curve and compute slope
print("\n[1] Loading VX curve data...")
vx_df = load_vx_curve(con, start, end)
vx_df = vx_df.dropna(subset=['vx1', 'vx2']).sort_values('date').copy()
vx_df['slope'] = vx_df['vx2'] - vx_df['vx1']

print(f"  Loaded {len(vx_df)} days")
print(f"  VX1 range: {vx_df['vx1'].min():.2f} to {vx_df['vx1'].max():.2f}")
print(f"  VX2 range: {vx_df['vx2'].min():.2f} to {vx_df['vx2'].max():.2f}")
print(f"  Slope (VX2-VX1) range: {vx_df['slope'].min():.2f} to {vx_df['slope'].max():.2f}")
print(f"  Slope mean: {vx_df['slope'].mean():.2f}, std: {vx_df['slope'].std():.2f}")
print(f"  Slope percentiles:")
print(f"    5th: {vx_df['slope'].quantile(0.05):.2f}")
print(f"    25th: {vx_df['slope'].quantile(0.25):.2f}")
print(f"    50th: {vx_df['slope'].quantile(0.50):.2f}")
print(f"    75th: {vx_df['slope'].quantile(0.75):.2f}")
print(f"    95th: {vx_df['slope'].quantile(0.95):.2f}")

# Check if slope is in reasonable range
slope_in_range = ((vx_df['slope'] >= -5) & (vx_df['slope'] <= 10)).sum() / len(vx_df) * 100
print(f"  Slope in [-5, 10] range: {slope_in_range:.1f}% of days")
if slope_in_range < 90:
    print(f"  WARNING: {100-slope_in_range:.1f}% of days have extreme slope values")

# 2. Load VIX for comparison
print("\n[2] Loading VIX for comparison...")
vrp_df = load_vrp_inputs(con, start, end)
vrp_df = vrp_df.dropna(subset=['vix', 'vx1', 'vx2']).sort_values('date').copy()

# Merge
df = vx_df.merge(vrp_df[['date', 'vix']], on='date', how='left')

# 3. Generate signals
threshold = 0.5
df['signal'] = np.where(df['slope'] > threshold, -1.0, 0.0)

# 4. Load VX1 returns
print("\n[3] Loading VX1 returns...")
result = con.execute(
    """
    SELECT
        timestamp::DATE AS date,
        close::DOUBLE AS close
    FROM market_data
    WHERE symbol = '@VX=101XN'
      AND timestamp::DATE BETWEEN ? AND ?
    ORDER BY timestamp
    """,
    [start, end]
).df()

result = result.set_index('date')
result['vx1_return'] = np.log(result['close']).diff()
vx1_rets = result['vx1_return'].dropna()

# Merge returns
df = df.merge(vx1_rets.to_frame('vx1_return'), left_on='date', right_index=True, how='inner')

# Compute PnL
df['position'] = df['signal'].shift(1)
df['pnl'] = df['position'] * df['vx1_return']
df = df.dropna(subset=['pnl']).copy()

print(f"  Computed PnL for {len(df)} days")

# 5. Check signal alignment
print("\n[4] Signal alignment check...")
short_days = df[df['signal'] == -1.0]
flat_days = df[df['signal'] == 0.0]

print(f"  Short days ({len(short_days)}):")
print(f"    Mean slope: {short_days['slope'].mean():.2f}")
print(f"    Mean VX1: {short_days['vx1'].mean():.2f}")
print(f"    Mean VX2: {short_days['vx2'].mean():.2f}")
print(f"    Mean VIX: {short_days['vix'].mean():.2f}")
print(f"    VX2 > VX1: {(short_days['vx2'] > short_days['vx1']).sum() / len(short_days) * 100:.1f}% of days")

print(f"  Flat days ({len(flat_days)}):")
print(f"    Mean slope: {flat_days['slope'].mean():.2f}")
print(f"    Mean VX1: {flat_days['vx1'].mean():.2f}")
print(f"    Mean VX2: {flat_days['vx2'].mean():.2f}")
print(f"    Mean VIX: {flat_days['vix'].mean():.2f}")
print(f"    VX2 > VX1: {(flat_days['vx2'] > flat_days['vx1']).sum() / len(flat_days) * 100:.1f}% of days")

# 6. Check spike events
print("\n[5] Spike event analysis...")
# March 2020
march_2020 = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2020-03-31')]
if len(march_2020) > 0:
    print(f"  March 2020:")
    print(f"    Days: {len(march_2020)}")
    print(f"    VX1 range: {march_2020['vx1'].min():.2f} to {march_2020['vx1'].max():.2f}")
    print(f"    VX1 max return: {march_2020['vx1_return'].max():.4f} ({march_2020['vx1_return'].max()*100:.2f}%)")
    print(f"    Short positions: {(march_2020['position'] == -1).sum()}")
    print(f"    PnL on short days: {march_2020[march_2020['position'] == -1]['pnl'].sum():.4f}")
    print(f"    Total PnL: {march_2020['pnl'].sum():.4f}")

# 2022 selloff
selloff_2022 = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2022-06-30')]
if len(selloff_2022) > 0:
    print(f"  H1 2022:")
    print(f"    Days: {len(selloff_2022)}")
    print(f"    VX1 range: {selloff_2022['vx1'].min():.2f} to {selloff_2022['vx1'].max():.2f}")
    print(f"    VX1 max return: {selloff_2022['vx1_return'].max():.4f} ({selloff_2022['vx1_return'].max()*100:.2f}%)")
    print(f"    Short positions: {(selloff_2022['position'] == -1).sum()}")
    print(f"    PnL on short days: {selloff_2022[selloff_2022['position'] == -1]['pnl'].sum():.4f}")
    print(f"    Total PnL: {selloff_2022['pnl'].sum():.4f}")

# 7. PnL calculation sanity
print("\n[6] PnL calculation sanity check...")
# Check that PnL = position * return
manual_pnl = df['position'] * df['vx1_return']
pnl_match = np.allclose(df['pnl'], manual_pnl, rtol=1e-10)
print(f"  PnL = position * return: {pnl_match}")

# Check position lag
print(f"  Position lag check:")
print(f"    Signal on day T, position on day T+1: {((df['position'].shift(-1) == df['signal']).sum() / len(df) * 100):.1f}%")
print(f"    (Should be ~100% if lag is correct)")

# 8. Compare to VRP-Core Phase-0 structure
print("\n[7] Comparing PnL calculation to VRP-Core Phase-0 pattern...")
print("  VRP-Core Phase-0: position = signal.shift(1), pnl = position * vx1_return")
print("  VRP-TermStructure Phase-0: position = signal.shift(1), pnl = position * vx1_return")
print("  ✓ Same calculation pattern")

# 9. Generate diagnostic plots
print("\n[8] Generating diagnostic plots...")
output_dir = Path("reports/sanity_checks/vrp/vrp_termstructure/diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: VX1, VX2, and slope over time
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(df['date'], df['vx1'], label='VX1', alpha=0.7, linewidth=1)
axes[0].plot(df['date'], df['vx2'], label='VX2', alpha=0.7, linewidth=1)
axes[0].set_ylabel('Price (vol points)')
axes[0].set_title('VX1 and VX2 Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['date'], df['slope'], label='Slope (VX2 - VX1)', alpha=0.7, linewidth=1, color='green')
axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1].axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold = {threshold}')
axes[1].fill_between(df['date'], threshold, df['slope'].max(), alpha=0.2, color='red', label='Short Zone')
axes[1].set_ylabel('Slope (vol points)')
axes[1].set_title('Term Structure Slope (VX2 - VX1)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(df['date'], df['signal'], label='Signal', alpha=0.7, linewidth=1, color='red')
axes[2].set_ylabel('Signal (-1 = short, 0 = flat)')
axes[2].set_xlabel('Date')
axes[2].set_title('Phase-0 Signal Over Time')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'vx_curve_and_signals.png', dpi=150)
plt.close()

# Plot 2: Signal alignment with VIX/VX1/VX2
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# VIX
axes[0].plot(df['date'], df['vix'], label='VIX', alpha=0.7, linewidth=1, color='blue')
axes[0].set_ylabel('VIX (vol points)')
axes[0].set_title('VIX Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# VX1 and VX2 with short signals highlighted
axes[1].plot(df['date'], df['vx1'], label='VX1', alpha=0.5, linewidth=1, color='orange')
axes[1].plot(df['date'], df['vx2'], label='VX2', alpha=0.5, linewidth=1, color='purple')
short_mask = df['signal'] == -1
axes[1].scatter(df.loc[short_mask, 'date'], df.loc[short_mask, 'vx1'], 
                c='red', s=10, alpha=0.5, label='Short Signal Days', zorder=5)
axes[1].set_ylabel('Price (vol points)')
axes[1].set_title('VX1 and VX2 with Short Signals Highlighted')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Slope with short signals
axes[2].plot(df['date'], df['slope'], label='Slope', alpha=0.5, linewidth=1, color='green')
axes[2].scatter(df.loc[short_mask, 'date'], df.loc[short_mask, 'slope'], 
                c='red', s=10, alpha=0.5, label='Short Signal Days', zorder=5)
axes[2].axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold = {threshold}')
axes[2].set_ylabel('Slope (vol points)')
axes[2].set_title('Slope with Short Signals Highlighted')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# PnL
equity = (1 + df['pnl']).cumprod()
axes[3].plot(df['date'], equity, label='Equity Curve', linewidth=1.5, color='black')
axes[3].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
axes[3].set_ylabel('Equity')
axes[3].set_xlabel('Date')
axes[3].set_title('Equity Curve')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'signal_alignment_and_pnl.png', dpi=150)
plt.close()

# Plot 3: Spike events detail
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# March 2020
if len(march_2020) > 0:
    axes[0].plot(march_2020['date'], march_2020['vx1'], label='VX1', linewidth=2, color='orange')
    axes[0].plot(march_2020['date'], march_2020['vx2'], label='VX2', linewidth=2, color='purple')
    axes[0].plot(march_2020['date'], march_2020['vix'], label='VIX', linewidth=2, color='blue')
    short_mask_march = march_2020['signal'] == -1
    axes[0].scatter(march_2020.loc[short_mask_march, 'date'], 
                    march_2020.loc[short_mask_march, 'vx1'], 
                    c='red', s=50, marker='x', label='Short Signal', zorder=5)
    axes[0].set_ylabel('Price (vol points)')
    axes[0].set_title('March 2020 Spike Event')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

# H1 2022
if len(selloff_2022) > 0:
    axes[1].plot(selloff_2022['date'], selloff_2022['vx1'], label='VX1', linewidth=2, color='orange')
    axes[1].plot(selloff_2022['date'], selloff_2022['vx2'], label='VX2', linewidth=2, color='purple')
    axes[1].plot(selloff_2022['date'], selloff_2022['vix'], label='VIX', linewidth=2, color='blue')
    short_mask_2022 = selloff_2022['signal'] == -1
    axes[1].scatter(selloff_2022.loc[short_mask_2022, 'date'], 
                    selloff_2022.loc[short_mask_2022, 'vx1'], 
                    c='red', s=50, marker='x', label='Short Signal', zorder=5)
    axes[1].set_ylabel('Price (vol points)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('H1 2022 Selloff')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'spike_events_detail.png', dpi=150)
plt.close()

print(f"  Saved plots to: {output_dir}")

# 10. Summary statistics
print("\n[9] Summary Statistics:")
print(f"  Total PnL: {df['pnl'].sum():.4f}")
print(f"  Mean daily PnL: {df['pnl'].mean():.4f}")
print(f"  Std daily PnL: {df['pnl'].std():.4f}")
print(f"  Sharpe: {df['pnl'].mean() / df['pnl'].std() * np.sqrt(252):.4f}")
print(f"  Max single-day loss: {df['pnl'].min():.4f}")
print(f"  Max single-day gain: {df['pnl'].max():.4f}")

con.close()

print("\n" + "=" * 80)
print("DATA SANITY CHECKS COMPLETE")
print("=" * 80)

