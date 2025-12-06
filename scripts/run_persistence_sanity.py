"""
CLI script for running Persistence Sign-Only Sanity Check.

This script implements a Phase-0 diagnostic for the persistence (momentum-of-momentum) idea:
- Return acceleration: ret_84[t] - ret_84[t-21]
- Slope acceleration: (EMA20 - EMA84)[t] - (EMA20 - EMA84)[t-21]
- Sign-only signals, equal-weighted across assets
- No overlays, no vol targeting, no z-scoring

Usage:
    python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31 --variant return_accel --lookback 84 --acceleration_window 21
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import MarketData
from src.diagnostics.persistence_sanity import (
    run_persistence,
    save_results,
    generate_plots
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_universe(universe_str: Optional[str]) -> Optional[list]:
    """
    Parse comma-separated universe string into list.
    
    Args:
        universe_str: Comma-separated string like "ES,NQ,RTY" or None
        
    Returns:
        List of symbols or None
    """
    if universe_str is None:
        return None
    return [s.strip() for s in universe_str.split(',') if s.strip()]


def main():
    """Main entry point for persistence sanity check."""
    parser = argparse.ArgumentParser(
        description='Run Persistence Sign-Only Sanity Check (Phase-0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Return acceleration variant (default)
  python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31
  
  # Slope acceleration variant
  python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31 --variant slope_accel
  
  # Breakout acceleration variant
  python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31 --variant breakout_accel
  
  # Custom parameters
  python scripts/run_persistence_sanity.py \\
    --start 2021-01-01 \\
    --end 2025-10-31 \\
    --variant return_accel \\
    --lookback 84 \\
    --acceleration_window 21
  
  # Custom output directory
  python scripts/run_persistence_sanity.py \\
    --start 2021-01-01 \\
    --end 2025-10-31 \\
    --output_dir reports/sanity_checks/trend/persistence/my_run
        """
    )
    
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--variant',
        type=str,
        choices=['return_accel', 'slope_accel', 'breakout_accel'],
        default='return_accel',
        help='Persistence variant: return_accel (ret_84[t] - ret_84[t-21]), slope_accel (slope acceleration), or breakout_accel (breakout_126[t] - breakout_126[t-21])'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=84,
        help='Base lookback window in trading days (default: 84 for ret_84)'
    )
    
    parser.add_argument(
        '--acceleration_window',
        type=int,
        default=21,
        help='Window for acceleration calculation in trading days (default: 21)'
    )
    
    parser.add_argument(
        '--universe',
        type=str,
        default=None,
        help='Comma-separated list of symbols (e.g., "ES,NQ,RTY"). If not provided, uses all available assets.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory. If not provided, uses reports/sanity_checks/trend/persistence/<timestamp>/'
    )
    
    args = parser.parse_args()
    
    # Parse universe
    universe = parse_universe(args.universe)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'reports/sanity_checks/trend/persistence/{timestamp}')
    
    logger.info("=" * 80)
    logger.info("PERSISTENCE SIGN-ONLY SANITY CHECK (Phase-0)")
    logger.info("=" * 80)
    logger.info(f"Variant: {args.variant}")
    logger.info(f"Lookback: {args.lookback} days")
    logger.info(f"Acceleration Window: {args.acceleration_window} days")
    logger.info(f"Start Date: {args.start}")
    logger.info(f"End Date: {args.end}")
    logger.info(f"Universe: {universe if universe else 'All available assets'}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)
    
    # Load market data
    logger.info("Loading market data...")
    market = MarketData()
    logger.info(f"  MarketData universe: {len(market.universe)} symbols")
    
    # Load continuous prices
    logger.info("Loading continuous prices...")
    prices_df = market.prices_cont
    logger.info(f"  Prices shape: {prices_df.shape}")
    logger.info(f"  Date range: {prices_df.index.min()} to {prices_df.index.max()}")
    
    # If universe was specified, filter to those symbols
    if universe:
        # Check which symbols are available
        available = [s for s in universe if s in prices_df.columns]
        if not available:
            logger.error(f"None of the specified universe symbols are available in prices_cont")
            logger.error(f"Requested: {universe}")
            logger.error(f"Available columns: {list(prices_df.columns)}")
            sys.exit(1)
        prices_df = prices_df[available]
        logger.info(f"  Filtered to {len(available)} assets: {available}")
    
    # Run persistence strategy
    logger.info(f"\nRunning persistence strategy ({args.variant})...")
    results = run_persistence(
        prices=prices_df,
        variant=args.variant,
        lookback=args.lookback,
        acceleration_window=args.acceleration_window,
        start_date=args.start,
        end_date=args.end,
        universe=universe
    )
    
    # Print summary
    metrics = results['metrics']
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"CAGR:        {metrics['cagr']:.4f} ({metrics['cagr']*100:.2f}%)")
    logger.info(f"Volatility:  {metrics['vol']:.4f} ({metrics['vol']*100:.2f}%)")
    logger.info(f"Sharpe:      {metrics['sharpe']:.4f}")
    logger.info(f"Max Drawdown: {metrics['maxdd']:.4f} ({metrics['maxdd']*100:.2f}%)")
    logger.info(f"Hit Rate:    {metrics['hit_rate']:.4f} ({metrics['hit_rate']*100:.2f}%)")
    logger.info(f"Trading Days: {metrics['n_days']}")
    logger.info(f"Years:       {metrics['years']:.2f}")
    logger.info("=" * 80)
    
    # Phase-0 pass criteria
    if metrics['sharpe'] >= 0.2:
        logger.info("✅ Phase-0 PASS: Sharpe >= 0.2. Eligible for Phase-1.")
    else:
        logger.info("❌ Phase-0 FAIL: Sharpe < 0.2. Redesign required before Phase-1.")
    
    # Print top/bottom assets
    if not results['per_asset_stats'].empty:
        logger.info("\nTop 5 Assets by CAGR:")
        top_assets = results['per_asset_stats'].head(5)
        for _, row in top_assets.iterrows():
            logger.info(f"  {row['symbol']:30s} CAGR: {row['cagr']*100:6.2f}%, Sharpe: {row['sharpe']:6.3f}")
        
        logger.info("\nBottom 5 Assets by CAGR:")
        bottom_assets = results['per_asset_stats'].tail(5)
        for _, row in bottom_assets.iterrows():
            logger.info(f"  {row['symbol']:30s} CAGR: {row['cagr']*100:6.2f}%, Sharpe: {row['sharpe']:6.3f}")
    
    # Save results
    logger.info(f"\nSaving results to {output_dir}...")
    save_results(
        results=results,
        output_dir=output_dir,
        variant=args.variant,
        lookback=args.lookback,
        acceleration_window=args.acceleration_window
    )
    
    # Generate plots
    logger.info("Generating plots...")
    generate_plots(
        results=results,
        output_dir=output_dir,
        variant=args.variant
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("PERSISTENCE SANITY CHECK COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

