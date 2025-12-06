"""
Minimal Charts: Basic diagnostic visualizations.

Provides two essential charts:
1. Rolling Sharpe (252 days) - catches regime explosions
2. Weight Heatmap (last 500 days) - catches weird sign flips and huge leverage
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def plot_rolling_sharpe(returns: pd.Series, out_path: str):
    """
    Plot rolling Sharpe ratio at the actual data frequency.
    
    For daily returns: rolling 252-day Sharpe (annualized)
    For weekly/rebalance returns: rolling 52-week Sharpe (annualized)
    
    Args:
        returns: Returns Series at actual frequency (daily or rebalance frequency)
        out_path: Output file path for the plot
    """
    if returns.empty:
        print(f"[WARN] Returns Series is empty")
        return
    
    # Determine data frequency
    date_diff = returns.index.to_series().diff().dt.days
    median_freq = date_diff.median()
    
    # Choose window size based on frequency
    # Daily: 252 days (1 year)
    # Weekly (~7 days): 52 weeks (1 year)
    # For other frequencies, approximate based on median
    if median_freq <= 2:
        # Daily or near-daily data
        window = 252
        period_label = "252-day"
        annualization = np.sqrt(252)
    else:
        # Weekly or rebalance frequency
        # Use 52 periods for weekly (52 weeks = 1 year)
        window = 52
        period_label = "52-week"
        # Annualize assuming ~52 periods per year
        annualization = np.sqrt(52)
    
    if len(returns) < window:
        print(f"[WARN] Insufficient data for rolling Sharpe (need >= {window} periods, got {len(returns)})")
        return
    
    # Calculate rolling Sharpe at actual frequency
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    # Avoid division by zero
    rolling_sharpe = rolling_mean / rolling_std * annualization
    rolling_sharpe = rolling_sharpe.dropna()
    
    if rolling_sharpe.empty:
        print("[WARN] No valid rolling Sharpe values")
        return
    
    # Create plot
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Sharpe')
    plt.title(f"Rolling {period_label} Sharpe (Annualized)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Ensure output directory exists
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"[OK] Saved rolling {period_label} Sharpe plot to {out_path}")


def plot_weight_heatmap(weights: pd.DataFrame, out_path: str, tail: int = 500):
    """
    Plot weights heatmap for the last N rebalance periods.
    
    Args:
        weights: Weights DataFrame [date x symbol] at rebalance frequency
        out_path: Output file path for the plot
        tail: Number of rebalance periods to plot (default: 500)
    """
    if weights.empty:
        print("[WARN] Weights DataFrame is empty")
        return
    
    # Determine if we should show all data or just tail
    # If tail is very large or exceeds data length, show all
    if tail >= len(weights):
        w = weights
        period_label = f"{len(w)} rebalance periods"
    else:
        # Take last N rebalance periods
        w = weights.tail(tail)
        period_label = f"last {len(w)} rebalance periods"
    
    if w.empty:
        print("[WARN] No data after tail selection")
        return
    
    # Determine actual frequency for labeling
    date_diff = w.index.to_series().diff().dt.days
    median_freq = date_diff.median()
    
    if median_freq <= 2:
        freq_label = "daily"
    elif median_freq <= 8:
        freq_label = "weekly"
    else:
        freq_label = "rebalance"
    
    # Transpose for heatmap (symbols on y-axis, dates on x-axis)
    w_t = w.T
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(
        w_t.values,
        aspect='auto',
        interpolation='none',
        cmap='RdBu_r',
        vmin=-w_t.abs().max().max(),
        vmax=w_t.abs().max().max()
    )
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Weight')
    
    # Set y-axis labels (symbols)
    if len(w_t.index) <= 50:  # Only show labels if not too many symbols
        ax.set_yticks(range(len(w_t.index)))
        ax.set_yticklabels(w_t.index)
    else:
        ax.set_yticks([])
    
    # Set x-axis labels (dates) - show every Nth date
    n_dates = len(w_t.columns)
    if n_dates > 20:
        step = max(1, n_dates // 10)
        date_indices = list(range(0, n_dates, step))
        if date_indices[-1] != n_dates - 1:
            date_indices.append(n_dates - 1)
        ax.set_xticks(date_indices)
        ax.set_xticklabels([w_t.columns[i].strftime('%Y-%m-%d') if hasattr(w_t.columns[i], 'strftime') else str(w_t.columns[i]) for i in date_indices], rotation=45, ha='right')
    else:
        ax.set_xticks(range(n_dates))
        ax.set_xticklabels([d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in w_t.columns], rotation=45, ha='right')
    
    ax.set_title(f"Weights Heatmap ({period_label}, {freq_label} rebalancing)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Symbol")
    
    plt.tight_layout()
    
    # Ensure output directory exists
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"[OK] Saved weight heatmap to {out_path}")


def run_minimal_charts(
    weights: pd.DataFrame,
    returns: pd.Series,
    out_dir: str = "reports/minimal"
):
    """
    Generate all minimal diagnostic charts.
    
    Args:
        weights: Weights DataFrame [date x symbol]
        returns: Daily returns Series
        out_dir: Output directory for charts
    """
    print("=" * 70)
    print("Generating Minimal Charts")
    print("=" * 70)
    
    # Ensure output directory exists
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Plot rolling Sharpe
    print("\n[1/2] Plotting rolling Sharpe...")
    plot_rolling_sharpe(returns, str(out_dir_path / "rolling_sharpe.png"))
    
    # Plot weight heatmap
    print("\n[2/2] Plotting weight heatmap...")
    plot_weight_heatmap(weights, str(out_dir_path / "weights_heatmap.png"))
    
    print("\n" + "=" * 70)
    print(f"[OK] Charts saved to {out_dir}")

