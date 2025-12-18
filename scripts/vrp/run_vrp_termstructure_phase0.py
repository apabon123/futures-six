#!/usr/bin/env python3
"""
VRP-TermStructure Phase-0 Signal Test Script

This script performs a Phase-0 signal sanity test for VRP-TermStructure:
- Economic spec: VX2 - VX1 slope (term structure slope)
- Toy rule: Short VX1 when slope > 0.5 vol points (contango), else flat
- No z-scores, no clipping, no vol targeting
- Raw economic signal sanity check

Phase-0 Definition:
- Simple, non-engineered rule to test if economic idea has edge
- Pass criteria: Sharpe ≥ 0.1, reasonable drawdown profile, non-degenerate signal distribution

Usage:
    python scripts/vrp/run_vrp_termstructure_phase0.py
    python scripts/vrp/run_vrp_termstructure_phase0.py --start 2020-01-01 --end 2025-10-31
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

# Phase-0 threshold: fixed constant for documentation only
# This is NOT a tunable parameter
# This is ONLY for Phase-0 documentation
# This is NOT used in Phase-1 or Phase-2
PHASE0_THRESHOLD = 0.5  # vol points


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
    ax.plot(df['date'], equity, label='VRP-TermStructure Phase-0 (Short-Only)', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title(f'VRP-TermStructure Phase-0: Equity Curve (Slope Threshold = {threshold})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Slope timeseries
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['date'], df['slope'], label='Term Structure Slope (VX2 - VX1)', alpha=0.7, color='blue')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Short Threshold = {threshold}')
    ax.fill_between(df['date'], threshold, df['slope'].max(), alpha=0.2, color='red', label='Short Zone (Contango)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Slope (vol points)')
    ax.set_title('VX Term Structure Slope Over Time (VX2 - VX1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'slope_timeseries.png', dpi=150)
    plt.close()
    
    # Plot 3: PnL histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['pnl'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(df['pnl'].mean(), color='red', linestyle='--', label=f'Mean: {df["pnl"].mean():.4f}')
    ax.set_xlabel('Daily PnL')
    ax.set_ylabel('Frequency')
    ax.set_title('VRP-TermStructure Phase-0: Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved 3 Phase-0 plots to {output_dir}")


def run_vrp_termstructure_phase0_signal_test(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    output_dir: Path,
    threshold: float = 0.5
) -> dict:
    """
    Canonical VRP-TermStructure Phase-0 signal test.
    
    Economic spec: VX2 - VX1 slope (term structure slope)
    Toy rule: SHORT-ONLY term structure rule
        - Short VX1 when slope > T (contango, strong downward roll pressure)
        - Flat otherwise (backwardation = crisis regime, no entry)
    No z-scores, no clipping, no vol targeting
    
    IMPORTANT: This is an asymmetric VRP sleeve - we only trade contango.
    Backwardation (VX1 > VX2) indicates crisis regime and should be treated as "no entry".
    This is consistent with earlier VRP exceptions in Phase-0 (e.g., VRP-Convergence short-only)
    and permissible under the asymmetry rules in PROCEDURES.md.
    
    Threshold T is fixed for Phase-0 documentation only and NOT used in Phase-1/Phase-2.
    
    This is a raw economic signal sanity check, analogous to VRP-Core Phase-0.
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        output_dir: Output directory for Phase-0 signal test
        threshold: Threshold for short-only rule (default: 0.5)
        
    Returns:
        Dict with Phase-0 metrics (Sharpe, CAGR, MaxDD, hit rate, etc.)
    """
    logger.info("\n" + "=" * 80)
    logger.info("VRP-TERMSTRUCTURE PHASE-0 SIGNAL TEST (SHORT-ONLY)")
    logger.info("=" * 80)
    logger.info("Economic spec: VX2 - VX1 slope (term structure slope)")
    logger.info(f"Toy rule: Short VX1 when slope > {threshold} (contango), else flat")
    logger.info("Note: Backwardation (VX1 > VX2) indicates crisis regime, no entry")
    
    # 1) Load VX curve
    logger.info("\n[1/5] Loading VX curve data...")
    df_vx = load_vx_curve(con, start, end)
    logger.info(f"  Loaded {len(df_vx)} VX curve rows")
    
    # 2) Compute slope
    logger.info("\n[2/5] Computing term structure slope...")
    # Drop rows with NA after computing slope (per Run Consistency Contract §2.2)
    df = df_vx.dropna(subset=['vx1', 'vx2']).sort_values('date').copy()
    
    # Compute slope: VX2 - VX1
    df['slope'] = df['vx2'] - df['vx1']
    
    logger.info(f"  Term structure slope: mean={df['slope'].mean():.2f}, std={df['slope'].std():.2f}")
    logger.info(f"  Data: {len(df)} days")
    
    # Check coverage
    coverage_vx1 = (df['vx1'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    coverage_vx2 = (df['vx2'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    logger.info(f"  VX1 coverage: {coverage_vx1:.1f}%")
    logger.info(f"  VX2 coverage: {coverage_vx2:.1f}%")
    
    if coverage_vx1 < 90 or coverage_vx2 < 90:
        logger.warning(f"  WARNING: Coverage below 90% target (VX1: {coverage_vx1:.1f}%, VX2: {coverage_vx2:.1f}%)")
    
    # 3) Generate Phase-0 signal (short-only term structure rule)
    logger.info("\n[3/5] Computing Phase-0 signal (short-only toy rule)...")
    
    # Signal logic:
    # slope = VX2 - VX1
    # if slope > T: Contango → SHORT VX1 (strong downward roll pressure)
    # else: Backwardation or neutral → FLAT (crisis regime, no entry)
    df['signal'] = np.where(df['slope'] > threshold, -1.0, 0.0)  # Short VX1 or flat
    
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
    
    # Log Run Consistency Contract metrics
    logger.info("\n[Run Consistency Contract]")
    logger.info(f"  Requested start date: {start}")
    logger.info(f"  Effective start date: {df['date'].iloc[0]}")
    logger.info(f"  Warmup rows dropped: 0 (no warmup for Phase-0)")
    rows_dropped_na = len(df_vx) - len(df)
    logger.info(f"  Rows dropped due to NA: {rows_dropped_na}")
    logger.info(f"  Final row count: {len(df)}")
    
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
    # reports/sanity_checks/vrp/vrp_termstructure/archive/{timestamp}/
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
    df[['date', 'vx1', 'vx2', 'slope', 'signal', 'position', 'pnl', 'equity']].to_parquet(
        save_dir / 'vrp_termstructure_phase0_timeseries.parquet', index=False
    )
    df[['date', 'vx1', 'vx2', 'slope', 'signal', 'position', 'pnl', 'equity']].to_csv(
        save_dir / 'vrp_termstructure_phase0_timeseries.csv', index=False
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
        'phase0_threshold': threshold,
        'threshold_description': f'Fixed {threshold} vol-point term structure slope threshold for Phase-0 sanity test. Not used in Phase-1 or production.',
        'description': f'VRP-TermStructure Phase-0 signal test (SHORT-ONLY): Short VX1 when (VX2 - VX1) > {threshold}, else flat. Backwardation indicates crisis regime, no entry.',
        'rule_type': 'short_only',
        'rationale': 'Contango (VX2 > VX1) indicates strong downward roll pressure → short-vol bias. Backwardation (VX1 > VX2) indicates crisis regime → flatten/neutralize exposure.',
        'metrics': metrics,
        'signal_distribution': {
            'pct_short': pct_short,
            'pct_flat': pct_flat
        },
        'run_consistency': {
            'requested_start': start,
            'effective_start': str(df['date'].iloc[0]) if len(df) > 0 else None,
            'warmup_rows_dropped': 0,
            'rows_dropped_na': rows_dropped_na,
            'final_row_count': len(df)
        }
    }
    
    with open(save_dir / 'meta.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    
    # Generate Phase-0 plots
    generate_phase0_plots(df, save_dir, threshold)
    
    # Copy to latest/ (for canonical reference)
    import shutil
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(save_dir, latest_dir)
    logger.info(f"  Copied results to latest/ directory")
    
    # Register in phase index
    phase_index_dir = Path("reports/phase_index/vrp/vrp_termstructure")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / "phase0.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"# Phase-0: VRP-TermStructure Signal Test (SHORT-ONLY: Short VX1 when (VX2 - VX1) > {threshold})\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"threshold: {threshold}\n")
        f.write(f"rule_type: short_only\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"effective_start_date: {str(df['date'].iloc[0]) if len(df) > 0 else None}\n")
        f.write(f"sharpe: {metrics.get('Sharpe', float('nan')):.4f}\n")
        f.write(f"cagr: {metrics.get('CAGR', float('nan')):.4f}\n")
        f.write(f"max_dd: {metrics.get('MaxDD', float('nan')):.4f}\n")
        f.write(f"path: {save_dir}\n")
        f.write(f"latest_path: {latest_dir}\n")
    
    logger.info(f"  Registered in: {phase0_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VRP-TERMSTRUCTURE PHASE-0 SIGNAL TEST SUMMARY (SHORT-ONLY)")
    print("=" * 80)
    print(f"Rule: Short VX1 when (VX2 - VX1) > {threshold}, else flat")
    print(f"Note: Backwardation (VX1 > VX2) indicates crisis regime, no entry")
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
    
    # Check MaxDD (not catastrophic)
    maxdd = metrics.get('MaxDD', float('nan'))
    if not pd.isna(maxdd):
        if maxdd > -0.85:
            print(f"  ✓ MaxDD not catastrophic: {maxdd:.4f} (PASS)")
        else:
            print(f"  ✗ MaxDD catastrophic: {maxdd:.4f} (FAIL)")
    
    print(f"\nPhase-0 signal test saved to: {save_dir}")
    print(f"Canonical results in: {latest_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="VRP-TermStructure Phase-0 Signal Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/vrp/run_vrp_termstructure_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Use full VRP history
  python scripts/vrp/run_vrp_termstructure_phase0.py --start 2004-01-01 --end 2025-10-31
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
        help="Output directory (empty=default: reports/sanity_checks/vrp/vrp_termstructure)"
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
        logger.info("VRP-TERMSTRUCTURE PHASE-0 SIGNAL TEST")
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
            output_dir = Path("reports/sanity_checks/vrp/vrp_termstructure")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Connect to database
        con = open_readonly_connection(db_path)
        
        try:
            # Run VRP-TermStructure Phase-0 signal test
            metrics = run_vrp_termstructure_phase0_signal_test(
                con, args.start, args.end, output_dir, threshold=PHASE0_THRESHOLD
            )
            
        finally:
            con.close()
        
        print("\n" + "=" * 80)
        print("ALL DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"  - archive/{datetime.now().strftime('%Y%m%d_%H%M%S')}/   (timestamped run)")
        print(f"  - latest/   (canonical Phase-0 results)")
        print(f"Phase index: reports/phase_index/vrp/vrp_termstructure/phase0.txt")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

