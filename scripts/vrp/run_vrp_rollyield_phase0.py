#!/usr/bin/env python3
"""
VRP-RollYield Phase-0 Signal Test Script

This script performs a Phase-0 signal sanity test for VRP-RollYield:
- Economic spec: VX1 roll-down toward VIX as expiration approaches
- Toy rule: Short VX1 when roll_yield > 0, else flat
- Roll yield = (VX1 - VIX) / days_to_expiry
- No z-scores, no clipping, no vol targeting
- Raw economic signal sanity check

Phase-0 Definition:
- Simple, non-engineered rule to test if economic idea has edge
- Pass criteria: Sharpe ≥ 0.1, reasonable drawdown profile, non-degenerate signal distribution

Usage:
    python scripts/vrp/run_vrp_rollyield_phase0.py
    python scripts/vrp/run_vrp_rollyield_phase0.py --start 2020-01-01 --end 2025-10-31
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import logging
import duckdb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.market_data.vrp_loaders import load_vix, load_vx_curve
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


def get_vx_expiry_date(year: int, month: int) -> datetime:
    """
    Calculate VX futures expiry date (third Wednesday of the month).
    
    VX futures expire on the third Wednesday of each month.
    
    Args:
        year: Year
        month: Month (1-12)
        
    Returns:
        datetime object for the third Wednesday of the month
    """
    # First day of the month
    first_day = datetime(year, month, 1)
    
    # Find first Wednesday
    # weekday() returns 0=Monday, 1=Tuesday, 2=Wednesday, etc.
    # So Wednesday = 2
    first_wednesday = 2  # Wednesday
    
    # Calculate days to add to get to first Wednesday
    # If first day is Wednesday, days_to_first_wed = 0
    # Otherwise, calculate days needed to reach next Wednesday
    days_to_first_wed = (first_wednesday - first_day.weekday()) % 7
    
    first_wed = first_day + timedelta(days=days_to_first_wed)
    
    # Third Wednesday is 14 days after first Wednesday
    third_wed = first_wed + timedelta(days=14)
    
    return third_wed


def calculate_days_to_expiry(date: pd.Timestamp) -> int:
    """
    Calculate days to expiry for VX1 front month contract.
    
    VX futures expire on the third Wednesday of each month.
    On any given date, we need to find the next expiry date.
    
    Args:
        date: Current trading date
        
    Returns:
        Number of days until the next VX expiry
    """
    current_date = date.to_pydatetime() if isinstance(date, pd.Timestamp) else date
    
    # Get current month and year
    year = current_date.year
    month = current_date.month
    
    # Calculate expiry for current month
    expiry_this_month = get_vx_expiry_date(year, month)
    
    # If we're past this month's expiry, use next month's expiry
    if current_date >= expiry_this_month:
        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        expiry_this_month = get_vx_expiry_date(year, month)
    
    # Calculate days to expiry
    days_to_exp = (expiry_this_month - current_date).days
    
    # Ensure non-negative (shouldn't happen, but safety check)
    return max(0, days_to_exp)


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
    ax.plot(df['date'], equity, label='VRP-RollYield Phase-0 (Short-Only)', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('VRP-RollYield Phase-0: Equity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Roll yield timeseries
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['date'], df['roll_yield'], label='Roll Yield ((VX1 - VIX) / days_to_expiry)', alpha=0.7, color='blue')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.fill_between(df['date'], 0, df['roll_yield'].max(), alpha=0.2, color='red', label='Short Zone (Positive Roll Yield)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Roll Yield (vol points / day)')
    ax.set_title('VX1 Roll Yield Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roll_yield_timeseries.png', dpi=150)
    plt.close()
    
    # Plot 3: PnL histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['pnl'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(df['pnl'].mean(), color='red', linestyle='--', label=f'Mean: {df["pnl"].mean():.4f}')
    ax.set_xlabel('Daily PnL')
    ax.set_ylabel('Frequency')
    ax.set_title('VRP-RollYield Phase-0: Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved 3 Phase-0 plots to {output_dir}")


def run_vrp_rollyield_phase0_signal_test(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    output_dir: Path,
) -> dict:
    """
    Canonical VRP-RollYield Phase-0 signal test.
    
    Economic spec: VX1 roll-down toward VIX as expiration approaches
    Toy rule: SHORT-ONLY roll yield rule
        - Short VX1 when roll_yield > 0 (positive roll yield = expected downward drift)
        - Flat otherwise (negative roll yield = volatility spike, no entry)
    No z-scores, no clipping, no vol targeting
    
    Roll yield = (VX1 - VIX) / days_to_expiry
    
    IMPORTANT: This is an asymmetric VRP sleeve - we only trade positive roll yield.
    Negative roll yield (VX1 < VIX) indicates volatility spike/crisis regime.
    
    This is a raw economic signal sanity check, analogous to VRP-Core Phase-0.
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        output_dir: Output directory for Phase-0 signal test
        
    Returns:
        Dict with Phase-0 metrics (Sharpe, CAGR, MaxDD, hit rate, etc.)
    """
    logger.info("\n" + "=" * 80)
    logger.info("VRP-ROLLYIELD PHASE-0 SIGNAL TEST (SHORT-ONLY)")
    logger.info("=" * 80)
    logger.info("Economic spec: VX1 roll-down toward VIX as expiration approaches")
    logger.info("Toy rule: Short VX1 when roll_yield > 0, else flat")
    logger.info("Note: Negative roll yield indicates volatility spike/crisis regime, no entry")
    
    # 1) Load VIX and VX curve
    logger.info("\n[1/5] Loading VIX and VX curve data...")
    df_vix = load_vix(con, start, end)
    df_vx = load_vx_curve(con, start, end)
    logger.info(f"  Loaded {len(df_vix)} VIX rows")
    logger.info(f"  Loaded {len(df_vx)} VX curve rows")
    
    # 2) Merge data and compute roll yield
    logger.info("\n[2/5] Computing roll yield...")
    df = df_vix.merge(df_vx, on="date", how="inner")
    df = df.dropna(subset=['vix', 'vx1']).sort_values('date').copy()
    
    # Convert date to datetime if needed
    if not isinstance(df['date'].iloc[0], pd.Timestamp):
        df['date'] = pd.to_datetime(df['date'])
    
    # Calculate days to expiry for each date
    df['days_to_expiry'] = df['date'].apply(calculate_days_to_expiry)
    
    # Compute roll = VX1 - VIX
    df['roll'] = df['vx1'] - df['vix']
    
    # Compute roll yield = roll / days_to_expiry
    # Avoid division by zero (shouldn't happen, but safety check)
    df['roll_yield'] = np.where(
        df['days_to_expiry'] > 0,
        df['roll'] / df['days_to_expiry'],
        np.nan
    )
    
    # Drop rows with NaN roll_yield
    df = df.dropna(subset=['roll_yield']).copy()
    
    logger.info(f"  Roll yield: mean={df['roll_yield'].mean():.4f}, std={df['roll_yield'].std():.4f}")
    logger.info(f"  Days to expiry: mean={df['days_to_expiry'].mean():.1f}, min={df['days_to_expiry'].min():.0f}, max={df['days_to_expiry'].max():.0f}")
    logger.info(f"  Data: {len(df)} days")
    
    # Check coverage
    coverage_vix = (df['vix'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    coverage_vx1 = (df['vx1'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    logger.info(f"  VIX coverage: {coverage_vix:.1f}%")
    logger.info(f"  VX1 coverage: {coverage_vx1:.1f}%")
    
    if coverage_vix < 90 or coverage_vx1 < 90:
        logger.warning(f"  WARNING: Coverage below 90% target (VIX: {coverage_vix:.1f}%, VX1: {coverage_vx1:.1f}%)")
    
    # 3) Generate Phase-0 signal (short-only roll yield rule)
    logger.info("\n[3/5] Computing Phase-0 signal (short-only toy rule)...")
    
    # Signal logic:
    # roll_yield = (VX1 - VIX) / days_to_expiry
    # if roll_yield > 0: Positive roll yield → SHORT VX1 (expected downward drift)
    # else: Negative roll yield → FLAT (volatility spike/crisis regime, no entry)
    df['signal'] = np.where(df['roll_yield'] > 0, -1.0, 0.0)  # Short VX1 or flat
    
    pct_short = (df['signal'] < 0).sum() / len(df) * 100
    pct_flat = (df['signal'] == 0).sum() / len(df) * 100
    logger.info(f"  Signal distribution: {pct_short:.1f}% short, {pct_flat:.1f}% flat")
    
    # 4) Load VX1 returns and compute PnL
    logger.info("\n[4/5] Loading VX1 returns and computing PnL...")
    vx1_rets = load_vx1_returns(con, start, end)
    
    # Merge VX1 returns
    df = df.merge(vx1_rets.to_frame('vx1_return'), left_on='date', right_index=True, how='inner')
    
    # Compute PnL with 1-day lag (avoid lookahead)
    # IMPORTANT: portfolio return = -position * VX1 return (per spec)
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
    logger.info(f"  Effective start date: {df['date'].iloc[0]}")
    logger.info(f"  Warmup rows dropped: 0 (no warmup for Phase-0)")
    initial_rows = len(df_vix.merge(df_vx, on="date", how="inner"))
    rows_dropped_na = initial_rows - len(df)
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
    # reports/sanity_checks/vrp/vrp_rollyield/archive/{timestamp}/
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
    df[['date', 'vix', 'vx1', 'vx1_return', 'days_to_expiry', 'roll', 'roll_yield', 'signal', 'position', 'pnl']].to_parquet(
        save_dir / 'vrp_rollyield_phase0_timeseries.parquet', index=False
    )
    df[['date', 'vix', 'vx1', 'vx1_return', 'days_to_expiry', 'roll', 'roll_yield', 'signal', 'position', 'pnl']].to_csv(
        save_dir / 'vrp_rollyield_phase0_timeseries.csv', index=False
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
        'description': 'VRP-RollYield Phase-0 signal test (SHORT-ONLY): Short VX1 when roll_yield > 0, else flat. Roll yield = (VX1 - VIX) / days_to_expiry.',
        'rule_type': 'short_only',
        'rationale': 'Positive roll yield (VX1 > VIX) indicates expected downward drift as VX1 converges to VIX → short-vol bias. Negative roll yield indicates volatility spike/crisis regime → flatten/neutralize exposure.',
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
    generate_phase0_plots(df, save_dir)
    
    # Copy to latest/ (for canonical reference)
    import shutil
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(save_dir, latest_dir)
    logger.info(f"  Copied results to latest/ directory")
    
    # Register in phase index
    phase_index_dir = Path("reports/phase_index/vrp/vrp_rollyield")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / "phase0.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"# Phase-0: VRP-RollYield Signal Test (SHORT-ONLY: Short VX1 when roll_yield > 0)\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    print("VRP-ROLLYIELD PHASE-0 SIGNAL TEST SUMMARY (SHORT-ONLY)")
    print("=" * 80)
    print(f"Rule: Short VX1 when roll_yield > 0, else flat")
    print(f"Roll yield = (VX1 - VIX) / days_to_expiry")
    print(f"Note: Negative roll yield indicates volatility spike/crisis regime, no entry")
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
    if pct_short > 20:
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
        description="VRP-RollYield Phase-0 Signal Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/vrp/run_vrp_rollyield_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Use full VRP history
  python scripts/vrp/run_vrp_rollyield_phase0.py --start 2004-01-01 --end 2025-10-31
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
        help="Output directory (empty=default: reports/sanity_checks/vrp/vrp_rollyield)"
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
        logger.info("VRP-ROLLYIELD PHASE-0 SIGNAL TEST")
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
            output_dir = Path("reports/sanity_checks/vrp/vrp_rollyield")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Connect to database
        con = open_readonly_connection(db_path)
        
        try:
            # Run VRP-RollYield Phase-0 signal test
            metrics = run_vrp_rollyield_phase0_signal_test(
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
        print(f"Phase index: reports/phase_index/vrp/vrp_rollyield/phase0.txt")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

