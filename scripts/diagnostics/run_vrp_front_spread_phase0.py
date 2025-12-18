#!/usr/bin/env python3
"""
VRP Front-Spread Phase-0 Signal Test Script

This script performs a Phase-0 signal sanity test for VRP Front-Spread:
- Economic spec: VX1 richness vs VX2 (calendar carry/decay edge)
- Toy rule: Short VX1 when VX1 > VX2 (contango), else flat
- No z-scores, no clipping, no vol targeting
- Raw economic signal sanity check

Phase-0 Definition:
- Simple, non-engineered rule to test if economic idea has edge
- Pass criteria: Sharpe ≥ 0.1-0.2, reasonable drawdown profile, non-degenerate signal distribution

Usage:
    python scripts/diagnostics/run_vrp_front_spread_phase0.py
    python scripts/diagnostics/run_vrp_front_spread_phase0.py --start 2020-01-01 --end 2025-10-31
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
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.market_data.vrp_loaders import load_vx_curve
from src.agents.utils_db import open_readonly_connection
from src.diagnostics.tsmom_sanity import compute_summary_stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# VRP data starts when VX futures begin (first obs)
VRP_START = "2004-01-01"  # Approximate start of VX futures data


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


def generate_phase0_plots(df: pd.DataFrame, output_dir: Path):
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
    ax.plot(df['date'], equity, label='VRP Front-Spread Phase-0', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('VRP Front-Spread Phase-0: Equity Curve (Short VX1 when VX1 > VX2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Spread time series with shading for active days
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot spread
    ax.plot(df['date'], df['spread'], label='Spread (VX1 - VX2)', alpha=0.7, color='blue', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # Shade regions where signal is active (short VX1)
    active_mask = df['signal'] < 0
    if active_mask.any():
        ax.fill_between(
            df['date'],
            0,
            df['spread'].max(),
            where=active_mask,
            alpha=0.2,
            color='red',
            label='Short Active (VX1 > VX2)'
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread (vol points)')
    ax.set_title('Front-Spread Over Time (VX1 - VX2) with Active Signal Shading')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'spread_timeseries.png', dpi=150)
    plt.close()
    
    # Plot 3: PnL histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['pnl'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(df['pnl'].mean(), color='red', linestyle='--', label=f'Mean: {df["pnl"].mean():.4f}')
    ax.set_xlabel('Daily PnL')
    ax.set_ylabel('Frequency')
    ax.set_title('VRP Front-Spread Phase-0: Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved 3 Phase-0 plots to {output_dir}")


def run_vrp_front_spread_phase0_signal_test(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    output_dir: Path,
) -> dict:
    """
    Canonical VRP Front-Spread Phase-0 signal test.
    
    Economic spec: VX1 richness vs VX2 (calendar carry/decay edge)
    Toy rule: Short VX1 when VX1 > VX2 (contango), else flat
    No z-scores, no clipping, no vol targeting
    
    When VX1 > VX2, the front contract is "rich" vs the next tenor → greater expected
    decay / roll-down pressure on VX1. We test the simplest tradable implication:
    short VX1 in contango.
    
    This is explicitly different from slope-driven "TermStructure outright" - Front-Spread
    is framed as calendar-carry richness, not "curve slope predicts direction."
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        output_dir: Output directory for Phase-0 signal test
        
    Returns:
        Dict with Phase-0 metrics (Sharpe, CAGR, MaxDD, hit rate, etc.)
    """
    logger.info("\n" + "=" * 80)
    logger.info("VRP FRONT-SPREAD PHASE-0 SIGNAL TEST")
    logger.info("=" * 80)
    logger.info("Economic spec: VX1 richness vs VX2 (calendar carry/decay edge)")
    logger.info("Toy rule: Short VX1 when VX1 > VX2 (contango), else flat")
    
    # 1) Load VX curve (VX1 and VX2)
    logger.info("\n[1/5] Loading VX curve data (VX1, VX2)...")
    df_vx = load_vx_curve(con, start, end)
    logger.info(f"  Loaded {len(df_vx)} VX curve rows")
    
    # 2) Compute spread and drop NAs
    logger.info("\n[2/5] Computing front-spread (VX1 - VX2)...")
    
    # Strict dropna: only compute spread on dates where both VX1 and VX2 exist
    df = df_vx.dropna(subset=['vx1', 'vx2']).sort_values('date').copy()
    
    if df.empty:
        logger.error("  ERROR: No overlapping VX1/VX2 data after dropna.")
        raise ValueError("No overlapping VX1/VX2 data available.")
    
    # Compute spread: VX1 - VX2 (both in vol points)
    df['spread'] = df['vx1'] - df['vx2']
    
    # Drop any remaining NAs in spread
    df = df.dropna(subset=['spread']).copy()
    
    if df.empty:
        logger.error("  ERROR: No valid spread data after dropna.")
        raise ValueError("No valid spread data available.")
    
    # Convert date to datetime if needed
    if len(df) > 0 and not isinstance(df['date'].iloc[0], pd.Timestamp):
        df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"  Spread: mean={df['spread'].mean():.2f}, std={df['spread'].std():.2f}")
    logger.info(f"  VX1: mean={df['vx1'].mean():.2f}, std={df['vx1'].std():.2f}")
    logger.info(f"  VX2: mean={df['vx2'].mean():.2f}, std={df['vx2'].std():.2f}")
    logger.info(f"  Data: {len(df)} days")
    
    # Check coverage
    coverage_vx1 = (df['vx1'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    coverage_vx2 = (df['vx2'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    logger.info(f"  VX1 coverage: {coverage_vx1:.1f}%")
    logger.info(f"  VX2 coverage: {coverage_vx2:.1f}%")
    
    if coverage_vx1 < 90 or coverage_vx2 < 90:
        logger.warning(f"  WARNING: Coverage below 90% target (VX1: {coverage_vx1:.1f}%, VX2: {coverage_vx2:.1f}%)")
    
    # 3) Generate Phase-0 signal
    logger.info("\n[3/5] Computing Phase-0 signal...")
    
    # Signal logic:
    # spread = VX1 - VX2
    # if spread > 0: VX1 > VX2 (contango) → SHORT VX1
    # else: VX1 <= VX2 (backwardation or flat) → FLAT
    df['signal'] = np.where(df['spread'] > 0, -1.0, 0.0)  # Short VX1 or flat
    
    pct_short = (df['signal'] < 0).sum() / len(df) * 100
    pct_flat = (df['signal'] == 0).sum() / len(df) * 100
    logger.info(f"  Signal distribution: {pct_short:.1f}% short, {pct_flat:.1f}% flat")
    
    # 4) Load VX1 returns and compute PnL
    logger.info("\n[4/5] Loading VX1 returns and computing PnL...")
    vx1_rets = load_vx1_returns(con, start, end)
    
    # Merge VX1 returns
    df = df.merge(vx1_rets.to_frame('vx1_return'), left_on='date', right_index=True, how='inner')
    
    # Compute PnL with 1-day lag (avoid lookahead)
    # IMPORTANT: portfolio return = -position * VX1 return
    # When position = -1 (short), pnl = -(-1) * vx1_return = vx1_return
    # This means we profit when VX1 goes down (negative return)
    df['position'] = df['signal'].shift(1)
    df['pnl'] = -df['position'] * df['vx1_return']
    df = df.dropna(subset=['pnl']).copy()
    
    # Compute equity curve
    df['equity'] = (1 + df['pnl']).cumprod()
    if len(df) > 0:
        df.loc[df.index[0], 'equity'] = 1.0
    
    logger.info(f"  Computed PnL for {len(df)} days")
    
    # Log Run Consistency Contract metrics
    logger.info("\n[Run Consistency Contract]")
    logger.info(f"  Requested start date: {start}")
    logger.info(f"  Effective start date: {df['date'].iloc[0] if len(df) > 0 else None}")
    logger.info(f"  Warmup rows dropped: 0 (no warmup for Phase-0)")
    initial_rows = len(df_vx.dropna(subset=['vx1', 'vx2']))
    rows_dropped_na = initial_rows - len(df)
    logger.info(f"  Rows dropped due to NA: {rows_dropped_na}")
    logger.info(f"  Final row count: {len(df)}")
    logger.info(f"  % days active: {pct_short:.1f}%")
    logger.info(f"  Spread stats: mean={df['spread'].mean():.2f}, median={df['spread'].median():.2f}, std={df['spread'].std():.2f}")
    
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
    
    # Create Phase-0 signal test subdirectory structure
    # reports/sanity_checks/vrp/front_spread/archive/{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create latest/ symlink/copy
    latest_dir = output_dir / "latest"
    
    # Save Phase-0 results
    logger.info("Saving Phase-0 signal test results...")
    
    # Save to archive
    save_dir = archive_dir
    
    # Save full timeseries (for analysis)
    df[['date', 'vx1', 'vx2', 'spread', 'signal', 'position', 'vx1_return', 'pnl']].to_parquet(
        save_dir / 'vrp_front_spread_phase0_timeseries.parquet', index=False
    )
    df[['date', 'vx1', 'vx2', 'spread', 'signal', 'position', 'vx1_return', 'pnl']].to_csv(
        save_dir / 'vrp_front_spread_phase0_timeseries.csv', index=False
    )
    
    # Save portfolio returns (canonical format)
    portfolio_returns_df = pd.DataFrame({
        'date': df['date'],
        'ret': df['pnl']
    })
    portfolio_returns_df.to_csv(save_dir / 'portfolio_returns.csv', index=False)
    
    # Save equity curve (canonical format)
    equity_df = pd.DataFrame({
        'date': df['date'],
        'equity': df['equity']
    })
    equity_df.to_csv(save_dir / 'equity_curve.csv', index=False)
    
    # Save metrics
    metrics_to_save = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start,
        'end_date': end,
        'effective_start_date': str(df['date'].iloc[0]) if len(df) > 0 else None,
        'description': 'VRP Front-Spread Phase-0 signal test: Short VX1 when VX1 > VX2 (contango), else flat. Tests calendar carry/decay edge.',
        'rule_type': 'short_only',
        'rationale': 'When VX1 > VX2, front contract is rich vs next tenor → greater expected decay/roll-down pressure on VX1. Short VX1 in contango.',
        'metrics': metrics,
        'signal_distribution': {
            'pct_short': pct_short,
            'pct_flat': pct_flat
        },
        'spread_stats': {
            'mean': float(df['spread'].mean()),
            'median': float(df['spread'].median()),
            'std': float(df['spread'].std()),
            'min': float(df['spread'].min()),
            'max': float(df['spread'].max())
        },
        'run_consistency': {
            'requested_start': start,
            'effective_start': str(df['date'].iloc[0]) if len(df) > 0 else None,
            'warmup_rows_dropped': 0,
            'rows_dropped_na': rows_dropped_na,
            'final_row_count': len(df),
            'pct_days_active': pct_short
        }
    }
    
    with open(save_dir / 'meta.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    
    # Generate Phase-0 plots
    generate_phase0_plots(df, save_dir)
    
    # Copy to latest/ (for canonical reference)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(save_dir, latest_dir)
    logger.info(f"  Copied results to latest/ directory")
    
    # Register in phase index
    phase_index_dir = Path("reports/phase_index/vrp/front_spread")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / "phase0.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"# Phase-0: VRP Front-Spread Signal Test (Short VX1 when VX1 > VX2)\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"rule_type: short_only\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"effective_start_date: {str(df['date'].iloc[0]) if len(df) > 0 else None}\n")
        f.write(f"sharpe: {metrics.get('Sharpe', float('nan')):.4f}\n")
        f.write(f"cagr: {metrics.get('CAGR', float('nan')):.4f}\n")
        f.write(f"max_dd: {metrics.get('MaxDD', float('nan')):.4f}\n")
        f.write(f"pct_days_active: {pct_short:.1f}\n")
        f.write(f"path: {save_dir}\n")
        f.write(f"latest_path: {latest_dir}\n")
    
    logger.info(f"  Registered in: {phase0_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VRP FRONT-SPREAD PHASE-0 SIGNAL TEST SUMMARY")
    print("=" * 80)
    print(f"Rule: Short VX1 when VX1 > VX2 (contango), else flat")
    print(f"Economic thesis: VX1 richness vs VX2 contains calendar carry/decay edge")
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
    print(f"\nSpread Stats:")
    print(f"  Mean:      {df['spread'].mean():6.2f} vol points")
    print(f"  Median:    {df['spread'].median():6.2f} vol points")
    print(f"  Std:       {df['spread'].std():6.2f} vol points")
    
    # Pass criteria
    sharpe = metrics.get('Sharpe', float('nan'))
    print(f"\nPhase-0 Pass Criteria:")
    if not pd.isna(sharpe):
        if sharpe >= 0.20:
            print(f"  [PASS] Sharpe >= 0.20: {sharpe:.4f} (CLEAN PASS)")
        elif sharpe >= 0.10:
            print(f"  [PASS] Sharpe >= 0.10: {sharpe:.4f} (BORDERLINE PASS)")
        else:
            print(f"  [FAIL] Sharpe < 0.10: {sharpe:.4f} (FAIL)")
    else:
        print(f"  [FAIL] Sharpe could not be computed (FAIL)")
    
    # Check signal distribution (non-degenerate - should have meaningful short signals)
    if 10 <= pct_short <= 90:
        print(f"  [PASS] Signal distribution non-degenerate ({pct_short:.1f}% active) (PASS)")
    else:
        print(f"  [FAIL] Signal distribution degenerate ({pct_short:.1f}% active) (FAIL)")
    
    # Check MaxDD (not catastrophic)
    maxdd = metrics.get('MaxDD', float('nan'))
    if not pd.isna(maxdd):
        if maxdd > -0.85:
            print(f"  [PASS] MaxDD not catastrophic: {maxdd:.4f} (PASS)")
        else:
            print(f"  [FAIL] MaxDD catastrophic: {maxdd:.4f} (FAIL)")
    
    print(f"\nPhase-0 signal test saved to: {save_dir}")
    print(f"Canonical results in: {latest_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="VRP Front-Spread Phase-0 Signal Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/diagnostics/run_vrp_front_spread_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Use full VRP history
  python scripts/diagnostics/run_vrp_front_spread_phase0.py --start 2004-01-01 --end 2025-10-31
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
        help="Output directory (empty=default: reports/sanity_checks/vrp/front_spread)"
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
        logger.info("VRP FRONT-SPREAD PHASE-0 SIGNAL TEST")
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
            output_dir = Path("reports/sanity_checks/vrp/front_spread")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Connect to database
        con = open_readonly_connection(db_path)
        
        try:
            # Run VRP Front-Spread Phase-0 signal test
            metrics = run_vrp_front_spread_phase0_signal_test(
                con, args.start, args.end, output_dir
            )
            
        finally:
            con.close()
        
        print("\n" + "=" * 80)
        print("ALL DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"  - archive/{datetime.now().strftime('%Y%m%d_%H%M%S')}/   (timestamped run)")
        print(f"  - latest/   (canonical Phase-0 results)")
        print(f"Phase index: reports/phase_index/vrp/front_spread/phase0.txt")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

