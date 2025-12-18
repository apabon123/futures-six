#!/usr/bin/env python3
"""
VRP-Alt (VRP-Richness) Phase-0 Signal Test Script

This script performs a Phase-0 signal sanity test for VRP-Alt (Alt-VRP using RV5):
- Economic spec: VIX vs short-term realized volatility (RV5)
- Toy rule: Short VX1 when VIX - RV5 > 0, else flat
- No z-scores, no clipping, no vol targeting
- Raw economic signal sanity check

Phase-0 Definition:
- Simple, non-engineered rule to test if economic idea has edge
- Pass criteria: Sharpe ≥ 0.1, reasonable drawdown profile, non-degenerate signal distribution

Usage:
    python scripts/vrp/run_vrp_alt_phase0.py
    python scripts/vrp/run_vrp_alt_phase0.py --start 2020-01-01 --end 2025-10-31
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
from src.market_data.vrp_loaders import load_vix, load_vx_curve
from src.agents.utils_db import open_readonly_connection
from src.agents.data_broker import MarketData
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
    ax.plot(df['date'], equity, label='VRP-Alt Phase-0 (Short-Only)', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('VRP-Alt Phase-0: Equity Curve (VIX - RV5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Alt-VRP timeseries (VIX, RV5, and spread)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: VIX and RV5
    axes[0].plot(df['date'], df['vix'], label='VIX', alpha=0.7, color='blue', linewidth=1.5)
    axes[0].plot(df['date'], df['rv5'], label='RV5', alpha=0.7, color='red', linewidth=1.5)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Vol (vol points)')
    axes[0].set_title('VIX vs RV5 (5-Day Realized Volatility)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: Alt-VRP spread (VIX - RV5)
    axes[1].plot(df['date'], df['alt_vrp'], label='Alt-VRP (VIX - RV5)', alpha=0.7, color='green', linewidth=1.5)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1].fill_between(df['date'], 0, df['alt_vrp'].max(), alpha=0.2, color='red', label='Short Zone (VIX > RV5)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Spread (vol points)')
    axes[1].set_title('Alt-VRP Spread Over Time (VIX - RV5)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alt_vrp_timeseries.png', dpi=150)
    plt.close()
    
    # Plot 3: PnL histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['pnl'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(df['pnl'].mean(), color='red', linestyle='--', label=f'Mean: {df["pnl"].mean():.4f}')
    ax.set_xlabel('Daily PnL')
    ax.set_ylabel('Frequency')
    ax.set_title('VRP-Alt Phase-0: Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved 3 Phase-0 plots to {output_dir}")


def run_vrp_alt_phase0_signal_test(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    output_dir: Path,
) -> dict:
    """
    Canonical VRP-Alt Phase-0 signal test.
    
    Economic spec: VIX vs short-term realized volatility (RV5)
    Toy rule: SHORT-ONLY alt-VRP rule
        - Short VX1 when VIX - RV5 > 0 (VIX rich vs short-term realized)
        - Flat otherwise (VIX cheap vs short-term realized, no entry)
    No z-scores, no clipping, no vol targeting
    
    Alt-VRP = VIX - RV5
    
    IMPORTANT: This is an asymmetric VRP sleeve - we only trade when VIX > RV5.
    When VIX < RV5, short-term realized vol is elevated (volatility spike), no entry.
    
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
    logger.info("VRP-ALT PHASE-0 SIGNAL TEST (SHORT-ONLY)")
    logger.info("=" * 80)
    logger.info("Economic spec: VIX vs short-term realized volatility (RV5)")
    logger.info("Toy rule: Short VX1 when VIX - RV5 > 0, else flat")
    logger.info("Note: When VIX < RV5, short-term realized vol is elevated (volatility spike), no entry")
    
    # 1) Load VIX and VX1
    logger.info("\n[1/5] Loading VIX and VX1 data...")
    df_vix = load_vix(con, start, end)
    df_vx = load_vx_curve(con, start, end)
    logger.info(f"  Loaded {len(df_vix)} VIX rows")
    logger.info(f"  Loaded {len(df_vx)} VX curve rows")
    
    # Load ES returns using MarketData (consistent with VRP-Core)
    logger.info("  Loading ES returns for RV5 calculation...")
    import yaml
    config_path = Path("configs/data.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        db_path = config['db']['path']
    else:
        logger.error("configs/data.yaml not found")
        raise ValueError("Cannot load config for MarketData")
    
    market = MarketData(db_path=db_path)
    es_symbol = "ES_FRONT_CALENDAR_2D"
    
    try:
        es_returns = market.get_returns(
            symbols=(es_symbol,),
            method="log",
            start=start,
            end=end
        )
        
        if es_returns.empty or es_symbol not in es_returns.columns:
            logger.error(f"  ERROR: No ES returns data for {es_symbol}")
            raise ValueError(f"No ES returns available for {es_symbol}")
        
        logger.info(f"  Loaded {len(es_returns)} ES return rows")
        
        # Compute RV5 from ES returns
        # RV5 = rolling std of daily log returns * sqrt(252) * 100 (vol points)
        rv5 = es_returns[es_symbol].rolling(
            window=5,
            min_periods=5
        ).std() * np.sqrt(252) * 100.0
        
        # Convert to DataFrame
        df_rv = pd.DataFrame({
            'date': rv5.index,
            'rv5': rv5.values
        })
        df_rv = df_rv.dropna().reset_index(drop=True)
        logger.info(f"  Computed {len(df_rv)} RV5 rows")
        
    finally:
        market.close()
    
    # 2) Merge data and compute alt-VRP
    logger.info("\n[2/5] Computing alt-VRP (VIX - RV5)...")
    
    if df_rv.empty:
        logger.error("  ERROR: No RV5 data computed.")
        raise ValueError("No RV5 data available. Cannot compute alt-VRP.")
    
    df = df_vix.merge(df_rv, on="date", how="inner")
    df = df.merge(df_vx[['date', 'vx1']], on="date", how="inner")
    df = df.dropna(subset=['vix', 'rv5', 'vx1']).sort_values('date').copy()
    
    if df.empty:
        logger.error("  ERROR: No overlapping data after merging VIX, RV5, and VX1.")
        raise ValueError("No overlapping data available. Cannot compute alt-VRP.")
    
    # Convert date to datetime if needed
    if len(df) > 0 and not isinstance(df['date'].iloc[0], pd.Timestamp):
        df['date'] = pd.to_datetime(df['date'])
    
    # Compute alt-VRP = VIX - RV5
    df['alt_vrp'] = df['vix'] - df['rv5']
    
    # Drop rows with NaN alt_vrp
    df = df.dropna(subset=['alt_vrp']).copy()
    
    logger.info(f"  Alt-VRP: mean={df['alt_vrp'].mean():.2f}, std={df['alt_vrp'].std():.2f}")
    logger.info(f"  VIX: mean={df['vix'].mean():.2f}, std={df['vix'].std():.2f}")
    logger.info(f"  RV5: mean={df['rv5'].mean():.2f}, std={df['rv5'].std():.2f}")
    logger.info(f"  Data: {len(df)} days")
    
    # Check coverage
    coverage_vix = (df['vix'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    coverage_rv5 = (df['rv5'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    coverage_vx1 = (df['vx1'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    logger.info(f"  VIX coverage: {coverage_vix:.1f}%")
    logger.info(f"  RV5 coverage: {coverage_rv5:.1f}%")
    logger.info(f"  VX1 coverage: {coverage_vx1:.1f}%")
    
    if coverage_vix < 90 or coverage_rv5 < 90 or coverage_vx1 < 90:
        logger.warning(f"  WARNING: Coverage below 90% target (VIX: {coverage_vix:.1f}%, RV5: {coverage_rv5:.1f}%, VX1: {coverage_vx1:.1f}%)")
    
    # 3) Generate Phase-0 signal (short-only alt-VRP rule)
    logger.info("\n[3/5] Computing Phase-0 signal (short-only toy rule)...")
    
    # Signal logic:
    # alt_vrp = VIX - RV5
    # if alt_vrp > 0: VIX rich vs short-term realized → SHORT VX1
    # else: VIX cheap vs short-term realized (volatility spike) → FLAT
    df['signal'] = np.where(df['alt_vrp'] > 0, -1.0, 0.0)  # Short VX1 or flat
    
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
    initial_rows = len(df_vix.merge(df_rv, on="date", how="inner").merge(df_vx[['date', 'vx1']], on="date", how="inner"))
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
    # reports/sanity_checks/vrp/vrp_alt/archive/{timestamp}/
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
    df[['date', 'vix', 'rv5', 'vx1', 'vx1_return', 'alt_vrp', 'signal', 'position', 'pnl']].to_parquet(
        save_dir / 'vrp_alt_phase0_timeseries.parquet', index=False
    )
    df[['date', 'vix', 'rv5', 'vx1', 'vx1_return', 'alt_vrp', 'signal', 'position', 'pnl']].to_csv(
        save_dir / 'vrp_alt_phase0_timeseries.csv', index=False
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
        'description': 'VRP-Alt Phase-0 signal test (SHORT-ONLY): Short VX1 when VIX - RV5 > 0, else flat. Alt-VRP = VIX - RV5 (short-term realized vol).',
        'rule_type': 'short_only',
        'rationale': 'Positive alt-VRP (VIX > RV5) indicates VIX rich vs short-term realized → short-vol bias. Negative alt-VRP indicates volatility spike/crisis regime → flatten/neutralize exposure.',
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
    phase_index_dir = Path("reports/phase_index/vrp/vrp_alt")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / "phase0.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"reports/sanity_checks/vrp/vrp_alt/latest/\n")
    
    logger.info(f"  Registered in: {phase0_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VRP-ALT PHASE-0 SIGNAL TEST SUMMARY (SHORT-ONLY)")
    print("=" * 80)
    print(f"Rule: Short VX1 when VIX - RV5 > 0, else flat")
    print(f"Alt-VRP = VIX - RV5 (short-term realized volatility)")
    print(f"Note: When VIX < RV5, short-term realized vol is elevated (volatility spike), no entry")
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
        description="VRP-Alt Phase-0 Signal Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/vrp/run_vrp_alt_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Use full VRP history
  python scripts/vrp/run_vrp_alt_phase0.py --start 2004-01-01 --end 2025-10-31
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
        help="Output directory (empty=default: reports/sanity_checks/vrp/vrp_alt)"
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
        logger.info("VRP-ALT PHASE-0 SIGNAL TEST")
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
            output_dir = Path("reports/sanity_checks/vrp/vrp_alt")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Connect to database
        con = open_readonly_connection(db_path)
        
        try:
            # Run VRP-Alt Phase-0 signal test
            metrics = run_vrp_alt_phase0_signal_test(
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
        print(f"Phase index: reports/phase_index/vrp/vrp_alt/phase0.txt")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

