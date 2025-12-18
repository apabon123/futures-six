#!/usr/bin/env python3
"""
VRP-Convergence Phase-0 Signal Test Script

This script performs a Phase-0 signal sanity test for VRP-Convergence:
- Economic spec: VIX (spot) vs VX1 (front-month futures) convergence
- Toy rule: Long VX1 when (VIX - VX1) > +T, Short VX1 when (VIX - VX1) < -T, else flat
- No z-scores, no clipping, no vol targeting
- Raw economic signal sanity check

Phase-0 Definition:
- Simple, non-engineered rule to test if economic idea has edge
- Pass criteria: Sharpe ≥ 0.1, reasonable drawdown profile, non-degenerate signal distribution

Usage:
    python scripts/diagnostics/run_vrp_convergence_phase0.py
    python scripts/diagnostics/run_vrp_convergence_phase0.py --start 2020-01-01 --end 2025-10-31
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
from src.diagnostics.tsmom_sanity import compute_summary_stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# VRP data starts when VIX3M begins (first obs)
VRP_START = "2009-09-18"

# Phase-0 thresholds: test multiple values for short-only convergence rule
# These are fixed constants for Phase-0 documentation only
# NOT used in Phase-1 or Phase-2
# Short-only rule: Short VX1 when (VIX - VX1) < -T (VX1 too rich vs VIX), else flat
# Positive spreads (VIX > VX1) do NOT imply mean reversion - they often indicate momentum/expansion
PHASE0_THRESHOLDS = [1.0, 1.5, 2.0]  # vol points to test


def load_vx1_returns(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    symbol: str = "@VX=101XN"
) -> pd.Series:
    """
    Load VX1 returns from canonical DB.
    
    Args:
        con: DuckDB connection
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        symbol: VX1 symbol (default: @VX=101XN)
        
    Returns:
        Series of daily log returns indexed by date
    """
    result = con.execute(
        """
        SELECT
            timestamp::DATE AS date,
            close::DOUBLE AS close
        FROM market_data
        WHERE symbol = ?
          AND timestamp::DATE BETWEEN ? AND ?
        ORDER BY timestamp
        """,
        [symbol, start, end]
    ).df()
    
    if result.empty:
        return pd.Series(dtype=float, name='vx1_ret')
    
    # Compute log returns
    result = result.set_index('date')
    result['vx1_ret'] = np.log(result['close']).diff()
    
    return result['vx1_ret'].dropna()


def generate_phase0_plots(df: pd.DataFrame, output_dir: Path, threshold: float):
    """Generate Phase-0 signal test plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    # Compute equity curve
    equity = (1 + df['pnl']).cumprod()
    
    # Plot 1: Cumulative PnL (equity curve)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['date'], equity, label='VRP-Convergence Phase-0 (Short-Only)', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title(f'VRP-Convergence Phase-0: Equity Curve (Short Threshold = -{threshold})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase0_equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Spread and signals
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Convergence spread over time
    axes[0].plot(df['date'], df['spread_conv'], label='Convergence Spread (VIX - VX1)', alpha=0.7, color='blue')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].axhline(-threshold, color='red', linestyle='--', alpha=0.7, label=f'Short Threshold = -{threshold}')
    axes[0].fill_between(df['date'], -threshold, df['spread_conv'].min(), alpha=0.2, color='red', label='Short Zone')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Spread (vol points)')
    axes[0].set_title('VIX - VX1 Convergence Spread Over Time (Short-Only Rule)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Signals over time
    axes[1].fill_between(df['date'], 0, df['signal'], alpha=0.5, label='Signal', color='red')
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Signal (-1 = short VX1, 0 = flat)')
    axes[1].set_title('Phase-0 Convergence Signals (Short-Only)')
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
    ax.set_title('VRP-Convergence Phase-0: PnL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase0_pnl_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved 3 Phase-0 plots to {output_dir}")


def run_vrp_convergence_phase0_signal_test(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    output_dir: Path,
    threshold: float = 1.0
) -> dict:
    """
    Canonical VRP-Convergence Phase-0 signal test.
    
    Economic spec: VIX (spot) vs VX1 (front-month futures) convergence
    Toy rule: SHORT-ONLY convergence rule
        - Short VX1 when (VIX - VX1) < -T (VX1 too rich vs VIX, expect convergence down)
        - Flat otherwise
    No z-scores, no clipping, no vol targeting
    
    IMPORTANT: Positive spreads (VIX > VX1) do NOT imply mean reversion.
    They often indicate momentum/expansion and are not mean-reverting.
    This is why we use short-only rule, similar to VRP-Core Phase-0.
    
    Threshold T is fixed for Phase-0 documentation only and NOT used in Phase-1/Phase-2.
    
    This is a raw economic signal sanity check, analogous to VRP-Core Phase-0.
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        output_dir: Output directory for Phase-0 signal test
        threshold: Threshold for short-only rule (default: 1.0)
        
    Returns:
        Dict with Phase-0 metrics (Sharpe, CAGR, MaxDD, hit rate, etc.)
    """
    logger.info("\n" + "=" * 80)
    logger.info("VRP-CONVERGENCE PHASE-0 SIGNAL TEST (SHORT-ONLY)")
    logger.info("=" * 80)
    logger.info("Economic spec: VIX (spot) vs VX1 (front-month futures) convergence")
    logger.info(f"Toy rule: Short VX1 when spread < -{threshold} (VX1 too rich), else flat")
    logger.info("Note: Positive spreads do NOT imply mean reversion (momentum/expansion regime)")
    
    # 1) Load VRP inputs
    logger.info("\n[1/5] Loading VRP data...")
    df_vrp = load_vrp_inputs(con, start, end)
    logger.info(f"  Loaded {len(df_vrp)} VRP rows")
    
    # 2) Compute convergence spread
    logger.info("\n[2/5] Computing convergence spread...")
    df = df_vrp.dropna(subset=['vix', 'vx1']).sort_values('date').copy()
    
    # VIX and VX1 are both in vol points (same units)
    # VIX is spot implied vol, VX1 is front-month futures price (already in vol points from loader)
    df['spread_conv'] = df['vix'] - df['vx1']
    
    logger.info(f"  Convergence spread: mean={df['spread_conv'].mean():.2f}, std={df['spread_conv'].std():.2f}")
    logger.info(f"  Data: {len(df)} days")
    
    # 3) Generate Phase-0 signal (short-only convergence rule)
    logger.info("\n[3/5] Computing Phase-0 signal (short-only toy rule)...")
    
    # Signal logic:
    # spread_conv = VIX - VX1
    # if spread_conv < -T: VX1 "too rich" vs VIX → SHORT VX1 (expect convergence down)
    # else: FLAT
    # NOTE: Positive spreads (VIX > VX1) do NOT imply mean reversion
    # They often indicate momentum/expansion and are not mean-reverting
    df['signal'] = np.where(df['spread_conv'] < -threshold, -1.0, 0.0)  # Short VX1 or flat
    
    pct_short = (df['signal'] < 0).sum() / len(df) * 100
    pct_flat = (df['signal'] == 0).sum() / len(df) * 100
    logger.info(f"  Signal distribution: {pct_short:.1f}% short, {pct_flat:.1f}% flat")
    
    # 4) Load VX1 returns and compute PnL
    logger.info("\n[4/5] Loading VX1 returns and computing PnL...")
    vx1_rets = load_vx1_returns(con, start, end)
    
    # Merge VX1 returns
    df = df.merge(vx1_rets.to_frame('vx1_return'), left_on='date', right_index=True, how='inner')
    
    # Compute PnL with 1-day lag (avoid lookahead)
    df['position'] = df['signal'].shift(1)
    df['pnl'] = df['position'] * df['vx1_return']
    df = df.dropna(subset=['pnl']).copy()
    
    # Compute equity curve
    df['equity'] = (1 + df['pnl']).cumprod()
    if len(df) > 0:
        df.loc[df.index[0], 'equity'] = 1.0
    
    logger.info(f"  Computed PnL for {len(df)} days")
    
    # 5) Compute Phase-0 metrics
    logger.info("\n[5/5] Computing Phase-0 metrics...")
    
    portfolio_rets = df['pnl']
    equity = df['equity']
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
    df[['date', 'vix', 'vx1', 'spread_conv', 'signal', 'position', 'pnl', 'equity']].to_parquet(
        phase0_dir / 'vrp_convergence_phase0_timeseries.parquet', index=False
    )
    df[['date', 'vix', 'vx1', 'spread_conv', 'signal', 'position', 'pnl', 'equity']].to_csv(
        phase0_dir / 'vrp_convergence_phase0_timeseries.csv', index=False
    )
    
    # Save metrics
    metrics_to_save = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start,
        'end_date': end,
        'phase0_threshold': threshold,
        'threshold_description': f'Fixed {threshold} vol-point convergence spread threshold for Phase-0 sanity test. Not used in Phase-1 or production.',
        'description': f'VRP-Convergence Phase-0 signal test (SHORT-ONLY): Short VX1 when (VIX - VX1) < -{threshold}, else flat. Positive spreads do NOT imply mean reversion.',
        'rule_type': 'short_only',
        'rationale': 'Negative spread (VX1 > VIX) indicates VX1 too rich vs spot, expect convergence down. Positive spreads indicate momentum/expansion, not mean-reverting.',
        'metrics': metrics,
        'signal_distribution': {
            'pct_short': pct_short,
            'pct_flat': pct_flat
        }
    }
    
    with open(phase0_dir / 'vrp_convergence_phase0_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Generate Phase-0 plots
    generate_phase0_plots(df, phase0_dir, threshold)
    
    # Register in phase index
    phase_index_dir = Path("reports/phase_index/vrp")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / f"vrp_convergence_phase0_threshold_{threshold:.1f}.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"# Phase-0: VRP-Convergence Signal Test (SHORT-ONLY: Short VX1 when (VIX - VX1) < -{threshold})\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"threshold: {threshold}\n")
        f.write(f"rule_type: short_only\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"sharpe: {metrics.get('Sharpe', float('nan')):.4f}\n")
        f.write(f"cagr: {metrics.get('CAGR', float('nan')):.4f}\n")
        f.write(f"max_dd: {metrics.get('MaxDD', float('nan')):.4f}\n")
        f.write(f"path: {phase0_dir}\n")
    
    logger.info(f"  Registered in: {phase0_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VRP-CONVERGENCE PHASE-0 SIGNAL TEST SUMMARY (SHORT-ONLY)")
    print("=" * 80)
    print(f"Rule: Short VX1 when (VIX - VX1) < -{threshold}, else flat")
    print(f"Note: Positive spreads do NOT imply mean reversion (momentum/expansion regime)")
    print(f"\nMetrics:")
    print(f"  CAGR:      {metrics.get('CAGR', float('nan')):8.4f} ({metrics.get('CAGR', 0)*100:6.2f}%)")
    print(f"  Vol:       {metrics.get('Vol', float('nan')):8.4f} ({metrics.get('Vol', 0)*100:6.2f}%)")
    print(f"  Sharpe:    {metrics.get('Sharpe', float('nan')):8.4f}")
    print(f"  MaxDD:     {metrics.get('MaxDD', float('nan')):8.4f} ({metrics.get('MaxDD', 0)*100:6.2f}%)")
    print(f"  HitRate:   {metrics.get('HitRate', float('nan')):8.4f} ({metrics.get('HitRate', 0)*100:6.2f}%)")
    print(f"  n_days:    {metrics.get('n_days', 0):8d}")
    print(f"  years:     {metrics.get('years', float('nan')):8.2f}")
    print(f"\nSignal Distribution:")
    print(f"  Short:     {pct_short:6.1f}%")
    print(f"  Flat:      {pct_flat:6.1f}%")
    
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
    
    # Check signal distribution (non-degenerate - should have meaningful short signals)
    if pct_short > 5:
        print(f"  ✓ Signal distribution non-degenerate (short signals used) (PASS)")
    else:
        print(f"  ✗ Signal distribution degenerate (stuck at flat) (FAIL)")
    
    print(f"\nPhase-0 signal test saved to: {phase0_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="VRP-Convergence Phase-0 Signal Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/diagnostics/run_vrp_convergence_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Use full VRP history
  python scripts/diagnostics/run_vrp_convergence_phase0.py --start 2009-09-18 --end 2025-10-31
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date (YYYY-MM-DD), default: {CANONICAL_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date (YYYY-MM-DD), default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (empty=default: data/diagnostics/vrp_convergence_phase0)"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="",
        help="Path to canonical DuckDB (empty=from config)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("VRP-CONVERGENCE PHASE-0 SIGNAL TEST")
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
            output_dir = Path("data/diagnostics/vrp_convergence_phase0")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Connect to database
        con = open_readonly_connection(db_path)
        
        try:
            # Run VRP-Convergence Phase-0 signal test for multiple thresholds
            all_results = {}
            best_sharpe = float('-inf')
            best_threshold = None
            
            for threshold in PHASE0_THRESHOLDS:
                logger.info("\n" + "=" * 80)
                logger.info(f"Testing threshold: {threshold}")
                logger.info("=" * 80)
                
                # Create threshold-specific output directory
                threshold_output_dir = output_dir / f"threshold_{threshold:.1f}"
                threshold_output_dir.mkdir(parents=True, exist_ok=True)
                
                metrics = run_vrp_convergence_phase0_signal_test(
                    con, args.start, args.end, threshold_output_dir, threshold=threshold
                )
                
                all_results[threshold] = metrics
                
                sharpe = metrics.get('Sharpe', float('-inf'))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_threshold = threshold
            
            # Print summary of all thresholds
            print("\n" + "=" * 80)
            print("VRP-CONVERGENCE PHASE-0 THRESHOLD COMPARISON")
            print("=" * 80)
            print(f"{'Threshold':<12} {'Sharpe':<10} {'CAGR':<12} {'MaxDD':<12} {'HitRate':<10}")
            print("-" * 80)
            for threshold in sorted(all_results.keys()):
                m = all_results[threshold]
                print(f"{threshold:<12.1f} {m.get('Sharpe', float('nan')):<10.4f} "
                      f"{m.get('CAGR', float('nan'))*100:<12.2f}% "
                      f"{m.get('MaxDD', float('nan'))*100:<12.2f}% "
                      f"{m.get('HitRate', float('nan'))*100:<10.2f}%")
            
            if best_threshold is not None:
                print(f"\nBest threshold: {best_threshold} (Sharpe: {best_sharpe:.4f})")
            
        finally:
            con.close()
        
        print("\n" + "=" * 80)
        print("ALL DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"  - threshold_1.0/   (threshold = 1.0)")
        print(f"  - threshold_1.5/   (threshold = 1.5)")
        print(f"  - threshold_2.0/   (threshold = 2.0)")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

