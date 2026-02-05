"""
Plot equity curves and HWM to diagnose drawdown computation issue.

Shows:
1. Post-RT, pre-allocator equity (what allocator sees)
2. Post-allocator equity (what actually gets traded)
3. High-water mark used for dd computation
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

run_id = sys.argv[1] if len(sys.argv) > 1 else None
if not run_id:
    print("Usage: python plot_equity_hwm_diagnostic.py <run_id>")
    sys.exit(1)

run_dir = Path("reports/runs") / run_id

print("=" * 80)
print("EQUITY & HWM DIAGNOSTIC PLOT")
print("=" * 80)
print(f"Run ID: {run_id}\n")

# Load state file (contains equity_used_for_dd and hwm_used_for_dd)
state_file = run_dir / "allocator_state_v1.csv"
if not state_file.exists():
    print(f"ERROR: State file not found: {state_file}")
    sys.exit(1)

state_df = pd.read_csv(state_file, parse_dates=True, index_col=0)
print(f"Loaded state data: {len(state_df)} dates\n")

# Load post-allocator equity curve
equity_file = run_dir / "equity_curve.csv"
if not equity_file.exists():
    print(f"ERROR: Equity curve file not found: {equity_file}")
    sys.exit(1)

equity_df = pd.read_csv(equity_file, parse_dates=True, index_col=0)
print(f"Loaded equity curve: {len(equity_df)} dates\n")

# Extract series
if 'equity_used_for_dd' not in state_df.columns or 'hwm_used_for_dd' not in state_df.columns:
    print("ERROR: equity_used_for_dd or hwm_used_for_dd not found in state file")
    sys.exit(1)

pre_alloc_equity = state_df['equity_used_for_dd']
hwm = state_df['hwm_used_for_dd']

# Get post-allocator equity (align to same dates)
if 'equity' in equity_df.columns:
    post_alloc_equity = equity_df['equity']
else:
    # Try first column
    post_alloc_equity = equity_df.iloc[:, 0]

# Align post-allocator equity to state dates
post_alloc_aligned = post_alloc_equity.reindex(pre_alloc_equity.index)

# Create plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot the three series
ax.plot(pre_alloc_equity.index, pre_alloc_equity.values, 
        label='Post-RT, Pre-Allocator Equity (allocator sees)', 
        linewidth=2, color='blue', alpha=0.8)
ax.plot(post_alloc_aligned.index, post_alloc_aligned.values, 
        label='Post-Allocator Equity (actually traded)', 
        linewidth=2, color='green', alpha=0.8)
ax.plot(hwm.index, hwm.values, 
        label='High-Water Mark (used for dd)', 
        linewidth=2, color='red', linestyle='--', alpha=0.8)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Equity Value', fontsize=12)
ax.set_title(f'Equity & HWM Diagnostic - {run_id}', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Add annotations for key statistics
pre_alloc_start = pre_alloc_equity.iloc[0]
pre_alloc_end = pre_alloc_equity.iloc[-1]
pre_alloc_max = pre_alloc_equity.max()
hwm_final = hwm.iloc[-1]
hwm_max = hwm.max()

stats_text = (
    f"Pre-Alloc Equity: Start={pre_alloc_start:.3f}, End={pre_alloc_end:.3f}, Max={pre_alloc_max:.3f}\n"
    f"HWM: Final={hwm_final:.3f}, Max={hwm_max:.3f}\n"
    f"Gap (End Equity - Final HWM): {pre_alloc_end - hwm_final:.3f}"
)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=9, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save plot
output_file = run_dir / 'equity_hwm_diagnostic.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved plot to: {output_file}")

# Also print key diagnostics
print("\n" + "=" * 80)
print("KEY DIAGNOSTICS")
print("=" * 80)
print(f"\nPre-Allocator Equity:")
print(f"  Start: {pre_alloc_start:.6f}")
print(f"  End: {pre_alloc_end:.6f}")
print(f"  Max: {pre_alloc_max:.6f}")
print(f"  % change: {(pre_alloc_end/pre_alloc_start - 1)*100:.2f}%")

print(f"\nHigh-Water Mark:")
print(f"  Start: {hwm.iloc[0]:.6f}")
print(f"  End: {hwm_final:.6f}")
print(f"  Max: {hwm_max:.6f}")
print(f"  Updates: {(hwm.diff() != 0).sum()} times")

print(f"\nGap Analysis:")
print(f"  Final Equity - Final HWM: {pre_alloc_end - hwm_final:.6f}")
print(f"  Max Equity - Max HWM: {pre_alloc_max - hwm_max:.6f}")

if pre_alloc_equity.max() > hwm.max():
    print(f"\n  [RED FLAG] Equity max ({pre_alloc_max:.6f}) > HWM max ({hwm_max:.6f})")
    print(f"  This indicates HWM is stuck and not updating with new equity highs")
elif abs(pre_alloc_equity.max() - hwm.max()) < 1e-6:
    print(f"\n  [OK] Equity max matches HWM max (both at {pre_alloc_max:.6f})")
else:
    print(f"\n  [WARNING] HWM max ({hwm_max:.6f}) > Equity max ({pre_alloc_max:.6f})")
    print(f"  This suggests HWM was computed from a different series")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print(f"\nView the plot: {output_file}")
