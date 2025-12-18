#!/usr/bin/env python3
"""
VRP Phase-0 Signal Test + Data Diagnostics Script

This script performs TWO distinct tasks:

1. VRP DATA DIAGNOSTICS (NOT Phase-0):
   - Coverage checks for VIX, VIX3M, VX1/2/3
   - Basic spreads: VIX - VX1, VIX3M - VIX, VX2 - VX1
   - Summary stats and plots
   - This is purely data readiness, not a signal test

2. VRP-CORE PHASE-0 SIGNAL TEST:
   - Economic spec: VIX (30d IV) vs 21-day realized ES volatility
   - Toy rule: short VX1 when VIX > RV_21, otherwise flat
   - No z-scores, no clipping, no vol targeting
   - Raw economic signal sanity check (analogous to Trend/CSMOM Phase-0)

Phase-0 Definition:
- Simple, non-engineered rule to test if economic idea has edge
- Pass criteria: Sharpe ≥ 0.1-0.2, reasonable drawdown profile

Usage:
    python scripts/diagnostics/run_vrp_phase0.py
    python scripts/diagnostics/run_vrp_phase0.py --start 2020-01-01 --end 2025-10-31
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import logging
import duckdb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.market_data.vrp_loaders import load_vrp_inputs
from src.agents.utils_db import open_readonly_connection
from src.agents.data_broker import MarketData
from src.diagnostics.tsmom_sanity import compute_summary_stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# VRP data starts when VIX3M begins (first obs)
VRP_START = "2009-09-18"

# Phase-0 threshold: fixed constant for documentation only
# This is NOT a tunable parameter
# This is ONLY for Phase-0 documentation
# This is NOT used in Phase-1 or Phase-2
PHASE0_THRESHOLD = 1.5  # vol points


def compute_vrp_spread_diagnostics(df: pd.DataFrame) -> dict:
    """
    Compute VRP spread diagnostics (DATA DIAGNOSTICS, NOT Phase-0).
    
    Args:
        df: DataFrame with columns [date, vix, vix3m, vx1, vx2, vx3]
    
    Returns:
        Dict with computed spreads and summary statistics
    """
    # Compute spreads
    df['vrp_vix_vx1'] = df['vix'] - df['vx1']
    df['term_vix3m_vix'] = df['vix3m'] - df['vix']
    df['slope_vx2_vx1'] = df['vx2'] - df['vx1']
    
    # Coverage metrics
    n_total = len(df)
    coverage = {
        'n_total': n_total,
        'n_vix': df['vix'].notna().sum(),
        'n_vix3m': df['vix3m'].notna().sum(),
        'n_vx1': df['vx1'].notna().sum(),
        'n_vx2': df['vx2'].notna().sum(),
        'n_vx3': df['vx3'].notna().sum(),
        'n_vrp_vix_vx1': df['vrp_vix_vx1'].notna().sum(),
        'n_term_vix3m_vix': df['term_vix3m_vix'].notna().sum(),
        'n_slope_vx2_vx1': df['slope_vx2_vx1'].notna().sum(),
    }
    
    # Percent coverage
    pct_coverage = {
        k.replace('n_', 'pct_'): (v / n_total * 100) if n_total > 0 else 0
        for k, v in coverage.items()
    }
    
    # Summary stats for each series
    summary_stats = {}
    
    for col in ['vix', 'vix3m', 'vx1', 'vx2', 'vx3', 
                'vrp_vix_vx1', 'term_vix3m_vix', 'slope_vx2_vx1']:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75)),
                }
    
    # VRP-specific metrics
    vrp_metrics = {}
    
    if 'vrp_vix_vx1' in df.columns:
        vrp_series = df['vrp_vix_vx1'].dropna()
        if len(vrp_series) > 0:
            vrp_metrics['pct_positive'] = float((vrp_series > 0).sum() / len(vrp_series) * 100)
            vrp_metrics['pct_negative'] = float((vrp_series < 0).sum() / len(vrp_series) * 100)
            vrp_metrics['pct_near_zero'] = float((vrp_series.abs() < 0.1).sum() / len(vrp_series) * 100)
    
    if 'term_vix3m_vix' in df.columns:
        term_series = df['term_vix3m_vix'].dropna()
        if len(term_series) > 0:
            vrp_metrics['term_pct_positive'] = float((term_series > 0).sum() / len(term_series) * 100)
    
    if 'slope_vx2_vx1' in df.columns:
        slope_series = df['slope_vx2_vx1'].dropna()
        if len(slope_series) > 0:
            vrp_metrics['slope_pct_positive'] = float((slope_series > 0).sum() / len(slope_series) * 100)
    
    return {
        'coverage': coverage,
        'pct_coverage': pct_coverage,
        'summary_stats': summary_stats,
        'vrp_metrics': vrp_metrics,
        'df': df
    }


def generate_data_diagnostic_plots(df: pd.DataFrame, output_dir: Path):
    """
    Generate VRP data diagnostic plots (NOT Phase-0 signal test).
    
    Plots:
    1. VIX vs VX1 time series
    2. VIX3M - VIX spread time series
    3. VX2 - VX1 slope time series
    4. VRP (VIX - VX1) histogram
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    logger.info("  Generating data diagnostic plots...")
    
    # Plot 1: VIX vs VX1
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['date'], df['vix'], label='VIX (1M)', alpha=0.7)
    ax.plot(df['date'], df['vx1'], label='VX1 (Front)', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.set_title('VIX vs VX1 Front Month (Data Diagnostic)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'vix_vs_vx1.png', dpi=150)
    plt.close()
    
    # Plot 2: VIX3M - VIX (term structure)
    if 'term_vix3m_vix' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['date'], df['term_vix3m_vix'], label='VIX3M - VIX', color='blue', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('VIX3M - VIX (pts)')
        ax.set_title('VIX Term Structure (Data Diagnostic)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'vix3m_minus_vix.png', dpi=150)
        plt.close()
    
    # Plot 3: VX2 - VX1 (curve slope)
    if 'slope_vx2_vx1' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['date'], df['slope_vx2_vx1'], label='VX2 - VX1', color='green', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('VX2 - VX1 (pts)')
        ax.set_title('VX Curve Slope (Data Diagnostic)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'vx2_minus_vx1.png', dpi=150)
        plt.close()
    
    # Plot 4: VRP (VIX - VX1) histogram
    if 'vrp_vix_vx1' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        vrp_data = df['vrp_vix_vx1'].dropna()
        ax.hist(vrp_data, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(vrp_data.mean(), color='red', linestyle='--', label=f'Mean: {vrp_data.mean():.2f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('VIX - VX1 (pts)')
        ax.set_ylabel('Frequency')
        ax.set_title('VRP Distribution (Data Diagnostic)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'vrp_vix_vx1_histogram.png', dpi=150)
        plt.close()
    
    logger.info(f"  Saved 4 data diagnostic plots to {output_dir}")


def run_vrp_data_diagnostics(con: duckdb.DuckDBPyConnection, start: str, end: str, output_dir: Path) -> None:
    """
    Run VRP data diagnostics: coverage, spreads, summary stats.
    
    NOTE: This is NOT Phase-0. This is purely data + spread sanity check.
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        output_dir: Output directory for data diagnostics
    """
    logger.info("\n" + "=" * 80)
    logger.info("VRP DATA DIAGNOSTICS (NOT Phase-0)")
    logger.info("=" * 80)
    
    # Load VRP inputs
    logger.info("Loading VRP data...")
    df = load_vrp_inputs(con, start, end)
    logger.info(f"  Loaded {len(df)} rows")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Compute diagnostics
    logger.info("Computing VRP spread diagnostics...")
    diagnostics = compute_vrp_spread_diagnostics(df)
    
    # Create data diagnostics subdirectory
    data_dir = output_dir / "data_diagnostics"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    logger.info("Saving data diagnostic results...")
    
    # Save data
    vrp_df = diagnostics['df']
    vrp_df.to_parquet(data_dir / 'vrp_inputs.parquet', index=False)
    vrp_df.to_csv(data_dir / 'vrp_inputs.csv', index=False)
    
    # Convert numpy types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save summary stats
    summary = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start,
        'end_date': end,
        'note': 'Data diagnostics only, not a Phase-0 signal test',
        'coverage': convert_to_native(diagnostics['coverage']),
        'pct_coverage': convert_to_native(diagnostics['pct_coverage']),
        'summary_stats': convert_to_native(diagnostics['summary_stats']),
        'vrp_metrics': convert_to_native(diagnostics['vrp_metrics']),
    }
    
    with open(data_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots
    generate_data_diagnostic_plots(vrp_df, data_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VRP DATA DIAGNOSTICS SUMMARY (NOT Phase-0)")
    print("=" * 80)
    print(f"\nData Coverage (n={diagnostics['coverage']['n_total']} days):")
    print(f"  VIX:       {diagnostics['pct_coverage']['pct_vix']:.1f}%")
    print(f"  VIX3M:     {diagnostics['pct_coverage']['pct_vix3m']:.1f}%")
    print(f"  VX1:       {diagnostics['pct_coverage']['pct_vx1']:.1f}%")
    print(f"  VX2:       {diagnostics['pct_coverage']['pct_vx2']:.1f}%")
    print(f"  VX3:       {diagnostics['pct_coverage']['pct_vx3']:.1f}%")
    
    print("\nVRP Spreads:")
    if 'vrp_vix_vx1' in diagnostics['summary_stats']:
        stats = diagnostics['summary_stats']['vrp_vix_vx1']
        print(f"  VIX - VX1:      mean={stats['mean']:6.2f}, std={stats['std']:5.2f}")
        if 'pct_positive' in diagnostics['vrp_metrics']:
            print(f"                  {diagnostics['vrp_metrics']['pct_positive']:.1f}% positive")
    
    if 'term_vix3m_vix' in diagnostics['summary_stats']:
        stats = diagnostics['summary_stats']['term_vix3m_vix']
        print(f"  VIX3M - VIX:    mean={stats['mean']:6.2f}, std={stats['std']:5.2f}")
        if 'term_pct_positive' in diagnostics['vrp_metrics']:
            print(f"                  {diagnostics['vrp_metrics']['term_pct_positive']:.1f}% positive")
    
    if 'slope_vx2_vx1' in diagnostics['summary_stats']:
        stats = diagnostics['summary_stats']['slope_vx2_vx1']
        print(f"  VX2 - VX1:      mean={stats['mean']:6.2f}, std={stats['std']:5.2f}")
        if 'slope_pct_positive' in diagnostics['vrp_metrics']:
            print(f"                  {diagnostics['vrp_metrics']['slope_pct_positive']:.1f}% positive")
    
    print(f"\nData diagnostics saved to: {data_dir}")


def load_vx1_returns(con: duckdb.DuckDBPyConnection, start: str, end: str) -> pd.Series:
    """
    Load VX1 returns from canonical DB.
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        
    Returns:
        Series of daily log returns indexed by date
    """
    result = con.execute(
        """
        SELECT
            timestamp::DATE AS date,
            close::DOUBLE AS close
        FROM market_data
        WHERE symbol = '@VX=101XN'
          AND timestamp::DATE BETWEEN ? AND ?
          AND close IS NOT NULL
        ORDER BY timestamp
        """,
        [start, end]
    ).df()
    
    if result.empty:
        return pd.Series(dtype=float, name='vx1_ret')
    
    # Compute log returns
    result = result.set_index('date')
    result['vx1_ret'] = np.log(result['close']).diff()
    
    return result['vx1_ret'].dropna()


def generate_phase0_plots(df: pd.DataFrame, output_dir: Path):
    """Generate Phase-0 signal test plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    # Plot 1: Cumulative PnL (equity curve)
    fig, ax = plt.subplots(figsize=(14, 6))
    equity = (1 + df['pnl']).cumprod()
    ax.plot(df['date'], equity, label='VRP-Core Phase-0', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title(f'VRP-Core Phase-0: Equity Curve (Signal = -1 when VRP spread > {PHASE0_THRESHOLD})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase0_equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: VRP spread and signals
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # VRP spread over time
    axes[0].plot(df['date'], df['vrp_spread_21'], label='VRP Spread (VIX - RV_21)', alpha=0.7)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].axhline(PHASE0_THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'Threshold = {PHASE0_THRESHOLD}')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('VRP Spread (pts)')
    axes[0].set_title('VRP Spread Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Signals over time
    axes[1].fill_between(df['date'], 0, df['signal_phase0'], alpha=0.5, label='Signal')
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Signal (-1 = short VX1, 0 = flat)')
    axes[1].set_title('Phase-0 Signals')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase0_spreads_signals.png', dpi=150)
    plt.close()
    
    # Plot 3: PnL histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['pnl'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(df['pnl'].mean(), color='red', linestyle='--', label=f'Mean: {df["pnl"].mean():.4f}')
    ax.set_xlabel('Daily PnL')
    ax.set_ylabel('Frequency')
    ax.set_title('VRP-Core Phase-0: PnL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase0_pnl_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved 3 Phase-0 plots to {output_dir}")


def run_vrp_core_phase0_signal_test(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    output_dir: Path
) -> dict:
    """
    Canonical VRP-Core Phase-0 signal test.
    
    Economic spec: VIX (30d IV) vs 21-day realized ES volatility
    Toy rule: short VX1 when (VIX - RV_21) > 1.5 vol points, otherwise flat
    No z-scores, no clipping, no vol targeting
    
    Threshold of 1.5 is fixed for Phase-0 documentation only and NOT used in Phase-1/Phase-2.
    
    This is a raw economic signal sanity check, analogous to Trend/CSMOM Phase-0.
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        output_dir: Output directory for Phase-0 signal test
        
    Returns:
        Dict with Phase-0 metrics (Sharpe, CAGR, MaxDD, hit rate, etc.)
    """
    logger.info("\n" + "=" * 80)
    logger.info("VRP-CORE PHASE-0 SIGNAL TEST")
    logger.info("=" * 80)
    logger.info("Economic spec: VIX - 21d realized ES vol")
    logger.info(f"Toy rule: short VX1 when VRP spread > {PHASE0_THRESHOLD} vol pts, else flat")
    
    # 1) Load VRP inputs
    logger.info("\n[1/6] Loading VRP data...")
    df_vrp = load_vrp_inputs(con, start, end)
    logger.info(f"  Loaded {len(df_vrp)} VRP rows")
    
    # 2) Load ES returns for realized vol
    logger.info("\n[2/6] Loading ES returns for realized vol calculation...")
    market = MarketData()
    try:
        es_symbol = "ES_FRONT_CALENDAR_2D"
        
        if es_symbol not in market.returns_cont.columns:
            raise ValueError(f"{es_symbol} not found in market data")
        
        # Get ES returns
        es_returns = market.returns_cont[[es_symbol]].copy()
        
        # Filter by date
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        es_returns = es_returns[(es_returns.index >= start_dt) & (es_returns.index <= end_dt)]
        
        # Compute 21-day realized vol (annualized)
        # Note: This gives decimal form (e.g., 0.18 for 18%)
        rv_es_21 = es_returns[es_symbol].rolling(window=21, min_periods=21).std() * np.sqrt(252)
        
        df_es = pd.DataFrame({
            'date': rv_es_21.index,
            'rv_es_21': rv_es_21.values  # In decimal form (0.18 = 18%)
        })
        
        logger.info(f"  Computed RV for {len(df_es)} days")
    finally:
        market.close()
    
    # 3) Merge VRP and RV data
    logger.info("\n[3/6] Merging VRP and RV data...")
    df = df_vrp.merge(df_es, on='date', how='inner')
    df = df.dropna(subset=['vix', 'rv_es_21', 'vx1']).sort_values('date').copy()
    logger.info(f"  Merged data: {len(df)} days")
    
    # 4) Compute Phase-0 VRP spread and signal
    logger.info("\n[4/6] Computing Phase-0 signal (toy rule)...")
    # VIX is in vol points (20 = 20%), realized vol is in decimals (0.20 = 20%)
    # Convert realized vol to vol points by multiplying by 100
    df['vrp_spread_21'] = df['vix'] - (df['rv_es_21'] * 100.0)
    
    # Toy signal: -1 (short VX1) when VRP spread > threshold, else 0 (flat)
    # Threshold = 1.5 vol points (fixed for Phase-0 documentation only)
    df['signal_phase0'] = np.where(df['vrp_spread_21'] > PHASE0_THRESHOLD, -1.0, 0.0)
    
    logger.info(f"  VRP spread: mean={df['vrp_spread_21'].mean():.2f}, std={df['vrp_spread_21'].std():.2f}")
    pct_short = (df['signal_phase0'] < 0).sum() / len(df) * 100
    logger.info(f"  Signal: {pct_short:.1f}% short, {100-pct_short:.1f}% flat (threshold={PHASE0_THRESHOLD})")
    
    # 5) Load VX1 returns and compute PnL
    logger.info("\n[5/6] Loading VX1 returns and computing PnL...")
    vx1_rets = load_vx1_returns(con, start, end)
    
    # Merge VX1 returns
    df = df.merge(vx1_rets.to_frame('vx1_return'), left_on='date', right_index=True, how='inner')
    
    # Compute PnL with 1-day lag (avoid lookahead)
    df['position'] = df['signal_phase0'].shift(1)
    df['pnl'] = df['position'] * df['vx1_return']
    df = df.dropna(subset=['pnl']).copy()
    
    logger.info(f"  Computed PnL for {len(df)} days")
    
    # 6) Compute Phase-0 metrics
    logger.info("\n[6/6] Computing Phase-0 metrics...")
    
    # Use existing summary stats helper (wraps portfolio returns as DataFrame for compatibility)
    portfolio_rets = df['pnl']
    equity = (1 + portfolio_rets).cumprod()
    asset_strategy_returns = pd.DataFrame({'VX1': portfolio_rets})
    
    stats = compute_summary_stats(
        portfolio_returns=portfolio_rets,
        equity_curve=equity,
        asset_strategy_returns=asset_strategy_returns
    )
    
    metrics = stats['portfolio']
    
    # Create Phase-0 signal test subdirectory
    phase0_dir = output_dir / "phase0_signal_test"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Phase-0 results
    logger.info("Saving Phase-0 signal test results...")
    
    # Save timeseries
    df[['date', 'vix', 'rv_es_21', 'vrp_spread_21', 'signal_phase0', 'vx1_return', 'pnl']].to_parquet(
        phase0_dir / 'vrp_core_phase0_timeseries.parquet', index=False
    )
    df[['date', 'vix', 'rv_es_21', 'vrp_spread_21', 'signal_phase0', 'vx1_return', 'pnl']].to_csv(
        phase0_dir / 'vrp_core_phase0_timeseries.csv', index=False
    )
    
    # Save metrics
    metrics_to_save = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start,
        'end_date': end,
        'phase0_threshold': PHASE0_THRESHOLD,
        'threshold_description': 'Fixed 1.5 vol-point VRP spread threshold for Phase-0 sanity test. Not used in Phase-1 or production.',
        'description': f'VRP-Core Phase-0 signal test: short VX1 when (VIX - RV_21) > {PHASE0_THRESHOLD}, else flat',
        'metrics': metrics
    }
    
    with open(phase0_dir / 'vrp_core_phase0_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Generate Phase-0 plots
    generate_phase0_plots(df, phase0_dir)
    
    # Register in phase index
    phase_index_dir = Path("reports/phase_index/vrp")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / "phase0.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"# Phase-0: VRP-Core Signal Test (Toy Rule: Short VX1 when VRP spread > {PHASE0_THRESHOLD})\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"threshold: {PHASE0_THRESHOLD}\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"sharpe: {metrics.get('Sharpe', float('nan')):.4f}\n")
        f.write(f"cagr: {metrics.get('CAGR', float('nan')):.4f}\n")
        f.write(f"max_dd: {metrics.get('MaxDD', float('nan')):.4f}\n")
        f.write(f"path: {phase0_dir}\n")
    
    logger.info(f"  Registered in: {phase0_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VRP-CORE PHASE-0 SIGNAL TEST SUMMARY")
    print("=" * 80)
    print(f"Rule: Short VX1 when VRP spread > {PHASE0_THRESHOLD} vol pts, else flat")
    print(f"\nMetrics:")
    print(f"  CAGR:      {metrics.get('CAGR', float('nan')):8.4f} ({metrics.get('CAGR', 0)*100:6.2f}%)")
    print(f"  Vol:       {metrics.get('Vol', float('nan')):8.4f} ({metrics.get('Vol', 0)*100:6.2f}%)")
    print(f"  Sharpe:    {metrics.get('Sharpe', float('nan')):8.4f}")
    print(f"  MaxDD:     {metrics.get('MaxDD', float('nan')):8.4f} ({metrics.get('MaxDD', 0)*100:6.2f}%)")
    print(f"  HitRate:   {metrics.get('HitRate', float('nan')):8.4f} ({metrics.get('HitRate', 0)*100:6.2f}%)")
    print(f"  n_days:    {metrics.get('n_days', 0):8d}")
    print(f"  years:     {metrics.get('years', float('nan')):8.2f}")
    
    # Pass criteria
    sharpe = metrics.get('Sharpe', float('nan'))
    print(f"\nPhase-0 Pass Criteria:")
    if not pd.isna(sharpe):
        if sharpe >= 0.1:
            print(f"  ✓ Sharpe ≥ 0.1: {sharpe:.4f} (PASS)")
        else:
            print(f"  ✗ Sharpe < 0.1: {sharpe:.4f} (FAIL)")
    else:
        print(f"  ✗ Sharpe could not be computed (FAIL)")
    
    print(f"\nPhase-0 signal test saved to: {phase0_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="VRP Phase-0 Signal Test + Data Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs TWO tasks:

1. VRP Data Diagnostics (NOT Phase-0):
   - Coverage checks, spreads, summary stats
   - Outputs: data_diagnostics/ subdirectory

2. VRP-Core Phase-0 Signal Test:
   - Toy rule: short VX1 when VIX > RV_21, else flat
   - Outputs: phase0_signal_test/ subdirectory

Examples:
  # Run both diagnostics and Phase-0 signal test
  python scripts/diagnostics/run_vrp_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Use full VRP history
  python scripts/diagnostics/run_vrp_phase0.py --start 2009-09-18 --end 2025-10-31
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=VRP_START,
        help=f"Start date (YYYY-MM-DD), default: {VRP_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date (YYYY-MM-DD), default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Path to canonical DuckDB (default: from configs/data.yaml)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/diagnostics/vrp_phase0)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("VRP DIAGNOSTICS + PHASE-0 SIGNAL TEST")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        
        # Determine DB path
        if args.db_path:
            db_path = args.db_path
        else:
            # Load from config
            import yaml
            config_path = Path("configs/data.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                db_path = config['db']['path']
            else:
                logger.error("configs/data.yaml not found and --db_path not specified")
                sys.exit(1)
        
        logger.info(f"Database: {db_path}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path("data/diagnostics/vrp_phase0")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Connect to database
        con = open_readonly_connection(db_path)
        
        try:
            # 1) Run VRP data diagnostics (NOT Phase-0)
            run_vrp_data_diagnostics(con, args.start, args.end, output_dir)
            
            # 2) Run VRP-Core Phase-0 signal test
            metrics = run_vrp_core_phase0_signal_test(con, args.start, args.end, output_dir)
            
        finally:
            con.close()
        
        print("\n" + "=" * 80)
        print("ALL DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"  - data_diagnostics/     (coverage, spreads)")
        print(f"  - phase0_signal_test/   (toy signal test)")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
