#!/usr/bin/env python3
"""
VRP-Structural (RV252) Phase-0 Signal Test Script

This script performs Phase-0 signal sanity tests for VRP-Structural using RV252:
- Economic spec: Long-horizon implied vs realized volatility (VIX - RV252)
- Toy rule: Short VX when VIX > RV252, else flat
- Tests three variants: VX1, VX2, VX3
- No z-scores, no clipping, no vol targeting
- Raw economic signal sanity check

Phase-0 Definition:
- Simple, non-engineered rule to test if economic idea has edge
- Pass criteria: Sharpe ≥ 0.1-0.2, reasonable drawdown profile, non-degenerate signal distribution

Usage:
    python scripts/diagnostics/run_vrp_structural_rv252_phase0.py
    python scripts/diagnostics/run_vrp_structural_rv252_phase0.py --start 2020-01-01 --end 2025-10-31
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
import duckdb
import shutil

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




def generate_phase0_plots(df: pd.DataFrame, output_dir: Path, tenor: str):
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
    ax.plot(df['date'], equity, label=f'VRP-Structural (RV252) Phase-0 - {tenor}', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title(f'VRP-Structural (RV252) Phase-0: Equity Curve ({tenor})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Spread time series (VIX - RV252)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot spread
    ax.plot(df['date'], df['spread'], label='Spread (VIX - RV252)', alpha=0.7, color='blue', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # Shade regions where signal is active (short VX)
    active_mask = df['signal'] < 0
    if active_mask.any():
        ax.fill_between(
            df['date'],
            0,
            df['spread'].max(),
            where=active_mask,
            alpha=0.2,
            color='red',
            label='Short Active (VIX > RV252)'
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread (vol points)')
    ax.set_title(f'VRP-Structural Spread Over Time (VIX - RV252) with Active Signal Shading ({tenor})')
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
    ax.set_title(f'VRP-Structural (RV252) Phase-0: Return Distribution ({tenor})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved 3 Phase-0 plots to {output_dir}")


def run_vrp_structural_rv252_variant(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    output_dir: Path,
    tenor: int,  # 1, 2, or 3
    df_base: pd.DataFrame,  # Pre-loaded VIX, RV252, VX curve data
) -> Dict[str, Any]:
    """
    Run Phase-0 signal test for a single VX tenor variant.
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        output_dir: Base output directory
        tenor: VX tenor (1, 2, or 3)
        df_base: DataFrame with VIX, RV252, and VX curve data
        
    Returns:
        Dict with Phase-0 metrics and metadata
    """
    tenor_name = f"VX{tenor}"
    vx_col = f"vx{tenor}"
    
    logger.info("\n" + "=" * 80)
    logger.info(f"VRP-STRUCTURAL (RV252) PHASE-0 SIGNAL TEST - {tenor_name}")
    logger.info("=" * 80)
    logger.info("Economic spec: Long-horizon implied vs realized volatility (VIX - RV252)")
    logger.info(f"Toy rule: Short {tenor_name} when VIX > RV252, else flat")
    
    # Extract data for this variant
    df = df_base.copy()
    
    # Check if VX column exists
    if vx_col not in df.columns:
        logger.error(f"  ERROR: {vx_col} not found in VX curve data")
        raise ValueError(f"{vx_col} not available")
    
    # Drop NAs for this variant
    df = df.dropna(subset=['vix', 'rv252', vx_col]).sort_values('date').copy()
    
    if df.empty:
        logger.error(f"  ERROR: No overlapping data for {tenor_name} after dropna")
        raise ValueError(f"No overlapping VIX/RV252/{vx_col} data available")
    
    logger.info(f"  Loaded {len(df)} days of data")
    logger.info(f"  VIX: mean={df['vix'].mean():.2f}, std={df['vix'].std():.2f}")
    logger.info(f"  RV252: mean={df['rv252'].mean():.2f}, std={df['rv252'].std():.2f}")
    logger.info(f"  {tenor_name}: mean={df[vx_col].mean():.2f}, std={df[vx_col].std():.2f}")
    
    # Compute spread and signal
    logger.info(f"\n[2/4] Computing Phase-0 signal...")
    df['spread'] = df['vix'] - df['rv252']
    
    # Signal: -1 if spread > 0, else 0
    df['signal'] = np.where(df['spread'] > 0, -1.0, 0.0)
    
    pct_short = (df['signal'] < 0).sum() / len(df) * 100
    pct_flat = (df['signal'] == 0).sum() / len(df) * 100
    logger.info(f"  Signal distribution: {pct_short:.1f}% short, {pct_flat:.1f}% flat")
    logger.info(f"  Spread: mean={df['spread'].mean():.2f}, median={df['spread'].median():.2f}, std={df['spread'].std():.2f}")
    
    # Compute VX returns from prices and compute PnL
    logger.info(f"\n[3/4] Computing {tenor_name} returns and PnL...")
    df['vx_return'] = np.log(df[vx_col]).diff()
    
    # Compute PnL with 1-day lag (avoid lookahead)
    df['position'] = df['signal'].shift(1)
    df['pnl'] = -df['position'] * df['vx_return']
    df = df.dropna(subset=['pnl']).copy()
    
    # Compute equity curve
    df['equity'] = (1 + df['pnl']).cumprod()
    if len(df) > 0:
        df.loc[df.index[0], 'equity'] = 1.0
    
    logger.info(f"  Computed PnL for {len(df)} days")
    
    # Log Run Consistency Contract metrics
    logger.info(f"\n[Run Consistency Contract]")
    logger.info(f"  Variant: {tenor_name}")
    logger.info(f"  Requested start date: {start}")
    logger.info(f"  Effective start date: {df['date'].iloc[0] if len(df) > 0 else None}")
    logger.info(f"  Warmup rows dropped: 0 (no warmup for Phase-0)")
    initial_rows = len(df_base.dropna(subset=['vix', 'rv252', vx_col]))
    rows_dropped_na = initial_rows - len(df)
    logger.info(f"  Rows dropped due to NA: {rows_dropped_na}")
    logger.info(f"  Final row count: {len(df)}")
    logger.info(f"  % days active: {pct_short:.1f}%")
    logger.info(f"  Spread stats: mean={df['spread'].mean():.2f}, median={df['spread'].median():.2f}, std={df['spread'].std():.2f}")
    
    # Compute Phase-0 metrics
    logger.info(f"\n[4/4] Computing Phase-0 metrics...")
    
    portfolio_rets = df['pnl']
    equity = df['equity']
    asset_strategy_returns = pd.DataFrame({tenor_name: portfolio_rets})
    
    stats = compute_summary_stats(
        portfolio_returns=portfolio_rets,
        equity_curve=equity,
        asset_strategy_returns=asset_strategy_returns
    )
    
    metrics = stats['portfolio']
    
    # Create Phase-0 signal test subdirectory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    latest_dir = output_dir / "latest"
    
    # Save Phase-0 results
    logger.info("Saving Phase-0 signal test results...")
    
    save_dir = archive_dir
    
    # Save full timeseries
    df[['date', 'vix', 'rv252', vx_col, 'spread', 'signal', 'position', 'vx_return', 'pnl']].to_parquet(
        save_dir / f'vrp_structural_rv252_{tenor_name.lower()}_phase0_timeseries.parquet', index=False
    )
    df[['date', 'vix', 'rv252', vx_col, 'spread', 'signal', 'position', 'vx_return', 'pnl']].to_csv(
        save_dir / f'vrp_structural_rv252_{tenor_name.lower()}_phase0_timeseries.csv', index=False
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
        'description': f'VRP-Structural (RV252) Phase-0 signal test: Short {tenor_name} when VIX > RV252, else flat. Tests long-horizon implied vs realized volatility premium.',
        'rule_type': 'short_only',
        'rationale': 'Long-horizon implied volatility (VIX) vs long-horizon realized volatility (RV252) should contain a tradable volatility risk premium.',
        'traded_instrument': tenor_name,
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
            'requested_end': end,
            'effective_start': str(df['date'].iloc[0]) if len(df) > 0 else None,
            'warmup_rows_dropped': 0,
            'rows_dropped_na': rows_dropped_na,
            'valid_rows': len(df),
            'final_row_count': len(df),
            'pct_days_active': pct_short
        }
    }
    
    with open(save_dir / 'meta.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    
    # Generate Phase-0 plots
    generate_phase0_plots(df, save_dir, tenor_name)
    
    # Copy to latest/ (for canonical reference)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(save_dir, latest_dir)
    logger.info(f"  Copied results to latest/ directory")
    
    # Register in phase index
    phase_index_dir = Path(f"reports/phase_index/vrp/structural_rv252_{tenor_name.lower()}")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / "phase0.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"reports/sanity_checks/vrp/structural_rv252_{tenor_name.lower()}/latest/\n")
    
    logger.info(f"  Registered in: {phase0_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"VRP-STRUCTURAL (RV252) PHASE-0 SIGNAL TEST SUMMARY - {tenor_name}")
    print("=" * 80)
    print(f"Rule: Short {tenor_name} when VIX > RV252, else flat")
    print(f"Economic thesis: Long-horizon implied vs realized volatility premium")
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
    
    # Check signal distribution
    if 10 <= pct_short <= 90:
        print(f"  [PASS] Signal distribution non-degenerate ({pct_short:.1f}% active) (PASS)")
    else:
        print(f"  [FAIL] Signal distribution degenerate ({pct_short:.1f}% active) (FAIL)")
    
    # Check MaxDD
    maxdd = metrics.get('MaxDD', float('nan'))
    if not pd.isna(maxdd):
        if maxdd > -0.85:
            print(f"  [PASS] MaxDD not catastrophic: {maxdd:.4f} (PASS)")
        else:
            print(f"  [FAIL] MaxDD catastrophic: {maxdd:.4f} (FAIL)")
    
    print(f"\nPhase-0 signal test saved to: {save_dir}")
    print(f"Canonical results in: {latest_dir}")
    
    return {
        'tenor': tenor_name,
        'metrics': metrics,
        'pct_active': pct_short,
        'effective_start': str(df['date'].iloc[0]) if len(df) > 0 else None,
        'path': str(save_dir)
    }


def run_vrp_structural_rv252_phase0(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    base_output_dir: Path,
) -> Dict[str, Any]:
    """
    Run VRP-Structural (RV252) Phase-0 signal test for all three variants (VX1, VX2, VX3).
    
    Args:
        con: DuckDB connection
        start: Start date
        end: End date
        base_output_dir: Base output directory (will create subdirectories per variant)
        
    Returns:
        Dict with comparison summary
    """
    logger.info("\n" + "=" * 80)
    logger.info("VRP-STRUCTURAL (RV252) PHASE-0 SIGNAL TEST")
    logger.info("=" * 80)
    logger.info("Economic spec: Long-horizon implied vs realized volatility (VIX - RV252)")
    logger.info("Toy rule: Short VX when VIX > RV252, else flat")
    logger.info("Testing three variants: VX1, VX2, VX3")
    
    # 1) Load VIX
    logger.info("\n[1/5] Loading VIX data...")
    df_vix = load_vix(con, start, end)
    logger.info(f"  Loaded {len(df_vix)} VIX rows")
    
    # 2) Load RV252 from ES returns (using MarketData, consistent with VRP-Core/Alt)
    logger.info("\n[2/5] Loading RV252 (252-day realized volatility) from ES returns...")
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
        
        # Compute RV252 from ES returns
        # RV252 = rolling std of daily log returns * sqrt(252) * 100 (vol points)
        rv252 = es_returns[es_symbol].rolling(
            window=252,
            min_periods=252
        ).std() * np.sqrt(252) * 100.0
        
        # Convert to DataFrame
        df_rv = pd.DataFrame({
            'date': rv252.index,
            'rv252': rv252.values
        })
        df_rv = df_rv.dropna().reset_index(drop=True)
        
        # Convert date to datetime if needed
        if len(df_rv) > 0 and not isinstance(df_rv['date'].iloc[0], pd.Timestamp):
            df_rv['date'] = pd.to_datetime(df_rv['date'])
        
        logger.info(f"  Computed {len(df_rv)} RV252 rows")
        logger.info(f"  RV252: mean={df_rv['rv252'].mean():.2f}, std={df_rv['rv252'].std():.2f}")
        
    finally:
        market.close()
    
    # 3) Load VX curve
    logger.info("\n[3/5] Loading VX curve data (VX1, VX2, VX3)...")
    df_vx = load_vx_curve(con, start, end)
    logger.info(f"  Loaded {len(df_vx)} VX curve rows")
    
    # 4) Merge base data
    logger.info("\n[4/5] Merging base data...")
    df_base = df_vix.merge(df_rv, on="date", how="inner")
    df_base = df_base.merge(df_vx, on="date", how="inner")
    
    # Convert date to datetime if needed
    if len(df_base) > 0 and not isinstance(df_base['date'].iloc[0], pd.Timestamp):
        df_base['date'] = pd.to_datetime(df_base['date'])
    
    logger.info(f"  Merged base data: {len(df_base)} days")
    
    # 5) Run three variants
    logger.info("\n[5/5] Running Phase-0 tests for all three variants...")
    
    results = {}
    tenors = [1, 2, 3]
    
    for tenor in tenors:
        tenor_name = f"VX{tenor}"
        output_dir = base_output_dir / f"structural_rv252_{tenor_name.lower()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = run_vrp_structural_rv252_variant(
                con, start, end, output_dir, tenor, df_base
            )
            results[tenor_name] = result
        except Exception as e:
            logger.error(f"  ERROR running {tenor_name}: {e}")
            results[tenor_name] = {'error': str(e)}
    
    # Create comparison summary
    logger.info("\n" + "=" * 80)
    logger.info("Creating comparison summary...")
    
    compare_dir = base_output_dir / "structural_rv252_compare" / "latest"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start,
        'end_date': end,
        'variants': {}
    }
    
    for tenor_name, result in results.items():
        if 'error' in result:
            summary['variants'][tenor_name] = {'error': result['error']}
        else:
            metrics = result['metrics']
            summary['variants'][tenor_name] = {
                'sharpe': float(metrics.get('Sharpe', float('nan'))),
                'cagr': float(metrics.get('CAGR', float('nan'))),
                'maxdd': float(metrics.get('MaxDD', float('nan'))),
                'active_pct': float(result['pct_active']),
                'effective_start': result['effective_start'],
                'path': result['path']
            }
    
    # Determine recommended winner
    valid_results = {
        k: v for k, v in summary['variants'].items()
        if 'error' not in v and not pd.isna(v.get('sharpe', float('nan'))) and v.get('maxdd', -999) > -0.85
    }
    
    if valid_results:
        # Sort by Sharpe (primary), then by MaxDD (secondary)
        winner = max(valid_results.items(), key=lambda x: (x[1]['sharpe'], x[1]['maxdd']))
        summary['recommended_winner'] = winner[0]
        summary['recommended_winner_sharpe'] = winner[1]['sharpe']
        summary['recommended_winner_maxdd'] = winner[1]['maxdd']
        logger.info(f"  Recommended winner: {winner[0]} (Sharpe: {winner[1]['sharpe']:.4f}, MaxDD: {winner[1]['maxdd']:.4f})")
    else:
        summary['recommended_winner'] = None
        logger.warning("  No valid winner (all variants failed or have catastrophic MaxDD)")
    
    # Save comparison summary
    with open(compare_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"  Comparison summary saved to: {compare_dir / 'summary.json'}")
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("VRP-STRUCTURAL (RV252) PHASE-0 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Window: {start} to {end}")
    print(f"\nVariant Comparison:")
    for tenor_name in ['VX1', 'VX2', 'VX3']:
        if tenor_name in summary['variants']:
            variant = summary['variants'][tenor_name]
            if 'error' in variant:
                print(f"  {tenor_name}: ERROR - {variant['error']}")
            else:
                print(f"  {tenor_name}:")
                print(f"    Sharpe:  {variant['sharpe']:8.4f}")
                print(f"    CAGR:    {variant['cagr']:8.4f} ({variant['cagr']*100:6.2f}%)")
                print(f"    MaxDD:   {variant['maxdd']:8.4f} ({variant['maxdd']*100:6.2f}%)")
                print(f"    Active:  {variant['active_pct']:6.1f}%")
                print(f"    Start:   {variant['effective_start']}")
    
    if summary.get('recommended_winner'):
        print(f"\nRecommended Winner: {summary['recommended_winner']}")
        print(f"  Sharpe: {summary['recommended_winner_sharpe']:.4f}")
        print(f"  MaxDD:  {summary['recommended_winner_maxdd']:.4f}")
    else:
        print(f"\nNo recommended winner (all variants failed or have catastrophic MaxDD)")
    
    print(f"\nComparison summary: {compare_dir / 'summary.json'}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="VRP-Structural (RV252) Phase-0 Signal Test (VX1/VX2/VX3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/diagnostics/run_vrp_structural_rv252_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Use full VRP history
  python scripts/diagnostics/run_vrp_structural_rv252_phase0.py --start 2004-01-01 --end 2025-10-31
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
        help="Output directory (empty=default: reports/sanity_checks/vrp)"
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
        logger.info("VRP-STRUCTURAL (RV252) PHASE-0 SIGNAL TEST")
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
            base_output_dir = Path(args.output_dir)
        else:
            base_output_dir = Path("reports/sanity_checks/vrp")
        
        base_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {base_output_dir}")
        
        # Connect to database
        con = open_readonly_connection(db_path)
        
        try:
            # Run VRP-Structural Phase-0 signal test (all variants)
            summary = run_vrp_structural_rv252_phase0(
                con, args.start, args.end, base_output_dir
            )
            
        finally:
            con.close()
        
        print("\n" + "=" * 80)
        print("ALL DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {base_output_dir}")
        print(f"  - structural_rv252_vx1/latest/   (VX1 variant)")
        print(f"  - structural_rv252_vx2/latest/   (VX2 variant)")
        print(f"  - structural_rv252_vx3/latest/   (VX3 variant)")
        print(f"  - structural_rv252_compare/latest/summary.json   (comparison summary)")
        print(f"\nPhase index entries:")
        print(f"  - reports/phase_index/vrp/structural_rv252_vx1/phase0.txt")
        print(f"  - reports/phase_index/vrp/structural_rv252_vx2/phase0.txt")
        print(f"  - reports/phase_index/vrp/structural_rv252_vx3/phase0.txt")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

