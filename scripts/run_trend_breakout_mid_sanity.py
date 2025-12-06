"""
CLI script for running Breakout Mid (50-100d) Sign-Only Sanity Check.

This script implements a Phase-0 diagnostic for the breakout mid idea:
- 50-day and 100-day breakout strength combined
- Donchian-style range breakouts
- Sign-only signals, equal-weighted across assets
- No overlays, no vol targeting, no z-scoring

Usage:
    python scripts/run_trend_breakout_mid_sanity.py --start 2020-01-01 --end 2025-11-19 --lookback_50 50 --lookback_100 100
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
from src.diagnostics.breakout_mid_sanity import (
    run_breakout_mid,
    save_results,
    generate_plots
)
from src.utils.phase_index import (
    get_sleeve_dirs,
    copy_to_latest,
    update_phase_index
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
    
    # Split by comma and strip whitespace
    symbols = [s.strip() for s in universe_str.split(',') if s.strip()]
    
    # Map short names to database symbols
    fx_symbols = {'6E', '6B', '6J'}
    equity_symbols = {'ES', 'NQ', 'RTY'}
    rates_volume = {'ZT', 'ZF', 'ZN', 'UB'}
    rates_calendar = {'SR3'}
    commodities = {'CL', 'GC'}
    
    db_symbols = []
    for sym in symbols:
        if sym in equity_symbols:
            db_symbols.append(f"{sym}_FRONT_CALENDAR_2D")
        elif sym in rates_volume:
            db_symbols.append(f"{sym}_FRONT_VOLUME")
        elif sym in rates_calendar:
            db_symbols.append(f"{sym}_FRONT_CALENDAR")
        elif sym in fx_symbols:
            db_symbols.append(f"{sym}_FRONT_CALENDAR")
        elif sym in commodities:
            db_symbols.append(f"{sym}_FRONT_VOLUME")
        else:
            # Assume it's already a database symbol
            db_symbols.append(sym)
    
    return db_symbols


def main():
    parser = argparse.ArgumentParser(
        description="Run Breakout Mid (50-100d) Sign-Only Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python scripts/run_trend_breakout_mid_sanity.py --start 2020-01-01 --end 2025-11-19
  
  # Custom lookback periods and thresholds
  python scripts/run_trend_breakout_mid_sanity.py --start 2020-01-01 --end 2025-11-19 --lookback_50 50 --lookback_100 100 --upper 0.55 --lower 0.45
  
  # Specific universe
  python scripts/run_trend_breakout_mid_sanity.py --start 2020-01-01 --end 2025-11-19 --universe ES,NQ,RTY,CL,GC
  
  # Custom output directory
  python scripts/run_trend_breakout_mid_sanity.py --start 2020-01-01 --end 2025-11-19 --output_dir reports/sanity_checks/trend/breakout_mid_50_100/my_run
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD). If not specified, uses latest available data."
    )
    parser.add_argument(
        "--lookback_50",
        type=int,
        default=50,
        help="50-day lookback period in days (default: 50)"
    )
    parser.add_argument(
        "--lookback_100",
        type=int,
        default=100,
        help="100-day lookback period in days (default: 100)"
    )
    parser.add_argument(
        "--upper",
        type=float,
        default=0.55,
        help="Upper breakout threshold for +1 signal (default: 0.55)"
    )
    parser.add_argument(
        "--lower",
        type=float,
        default=0.45,
        help="Lower breakout threshold for -1 signal (default: 0.45)"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Comma-separated list of symbols (e.g., ES,NQ,RTY,ZT,ZN,CL,GC,6E,6B,6J). "
             "If not specified, uses all assets from MarketData."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: reports/sanity_checks/trend/breakout_mid_50_100/<timestamp>)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("BREAKOUT MID (50-100d) SIGN-ONLY SANITY CHECK")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end or 'latest available'}")
        logger.info(f"Lookback 50: {args.lookback_50} days")
        logger.info(f"Lookback 100: {args.lookback_100} days")
        logger.info(f"Upper threshold: {args.upper}")
        logger.info(f"Lower threshold: {args.lower}")
        
        # Parse universe
        universe = parse_universe(args.universe)
        if universe:
            logger.info(f"Universe: {universe}")
        else:
            logger.info("Universe: All assets from MarketData")
        
        # Initialize MarketData
        logger.info("\n[1/4] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  MarketData universe: {len(market.universe)} symbols")
        
        # Load continuous prices
        logger.info("\n[2/4] Loading continuous prices...")
        prices_cont = market.prices_cont
        logger.info(f"  Prices shape: {prices_cont.shape}")
        logger.info(f"  Date range: {prices_cont.index.min()} to {prices_cont.index.max()}")
        
        # Determine end date if not specified
        if args.end is None:
            end_date = prices_cont.index.max().strftime("%Y-%m-%d")
            logger.info(f"  Using latest available date: {end_date}")
        else:
            end_date = args.end
        
        # If universe was specified, filter to those symbols
        if universe:
            # Check which symbols are available
            available = [s for s in universe if s in prices_cont.columns]
            if not available:
                logger.error(f"None of the specified universe symbols are available in prices_cont")
                logger.error(f"Requested: {universe}")
                logger.error(f"Available columns: {list(prices_cont.columns)}")
                sys.exit(1)
            prices_cont = prices_cont[available]
            logger.info(f"  Filtered to {len(available)} assets: {available}")
        
        # Determine output directory structure
        if args.output_dir:
            # Custom output directory (for testing/debugging)
            output_dir = Path(args.output_dir)
            archive_dir = output_dir
            latest_dir = None
        else:
            # Standard structure: archive/{timestamp}/ and latest/
            sleeve_dirs = get_sleeve_dirs("trend", "breakout_mid_50_100")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = sleeve_dirs["archive"] / timestamp
            latest_dir = sleeve_dirs["latest"]
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        output_dir = archive_dir  # Use archive_dir for all writes
        
        # Run strategy
        logger.info("\n[3/4] Computing breakout mid strategy...")
        result = run_breakout_mid(
            prices=prices_cont,
            lookback_50=args.lookback_50,
            lookback_100=args.lookback_100,
            upper_threshold=args.upper,
            lower_threshold=args.lower,
            start_date=args.start,
            end_date=end_date,
            universe=None  # Already filtered above
        )
        
        # Prepare stats dict for save_results
        stats = {
            'portfolio': result['metrics'],
            'per_asset': result['per_asset_stats']
        }
        
        # Save results
        logger.info("\n[4/4] Saving results...")
        save_results(
            results=result,
            stats=stats,
            output_dir=output_dir,
            start_date=args.start,
            end_date=end_date,
            lookback_50=args.lookback_50,
            lookback_100=args.lookback_100,
            upper_threshold=args.upper,
            lower_threshold=args.lower,
            universe=universe if universe else list(prices_cont.columns)
        )
        
        # Generate plots
        logger.info("  Generating plots...")
        generate_plots(
            results=result,
            stats=stats,
            output_dir=output_dir,
            prices=prices_cont
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("BREAKOUT MID (50-100d) SIGN-ONLY SANITY CHECK RESULTS")
        print("=" * 80)
        metrics = result['metrics']
        print(f"\nPortfolio Metrics:")
        print(f"  CAGR           : {metrics.get('CAGR', float('nan')):8.4f} ({metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  Vol            : {metrics.get('Vol', float('nan')):8.4f} ({metrics.get('Vol', 0)*100:6.2f}%)")
        print(f"  Sharpe         : {metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  MaxDD          : {metrics.get('MaxDD', float('nan')):8.4f} ({metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  HitRate        : {metrics.get('HitRate', float('nan')):8.4f} ({metrics.get('HitRate', 0)*100:6.2f}%)")
        print(f"  Trading Days   : {metrics.get('n_days', 0)}")
        print(f"  Years          : {metrics.get('years', float('nan')):.2f}")
        
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        sharpe = metrics.get('Sharpe', float('nan'))
        if not pd.isna(sharpe):
            if sharpe >= 0.2:
                verdict = "✓ PASS (Sharpe >= 0.2)"
                comment = "Breakout mid idea shows positive alpha. Eligible for Phase-1."
            elif sharpe >= 0.0:
                verdict = "⚠ WEAK (0.0 <= Sharpe < 0.2)"
                comment = "Breakout mid idea shows weak or zero alpha. May need refinement."
            else:
                verdict = "✗ FAIL (Sharpe < 0.0)"
                comment = "Breakout mid idea shows negative alpha. Redesign required."
            print(f"\n  {verdict}")
            print(f"  {comment}")
        else:
            print("\n  ⚠ Unable to compute Sharpe ratio (insufficient data)")
        
        # Copy key files to latest/ if using standard structure
        if latest_dir is not None:
            logger.info("\n[5/5] Updating canonical latest/ directory...")
            copy_to_latest(archive_dir, latest_dir)
            update_phase_index("trend", "breakout_mid_50_100", "phase0")
            logger.info(f"  Canonical Phase-0 results: {latest_dir}")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {archive_dir}")
        if latest_dir is not None:
            print(f"Canonical Phase-0: {latest_dir}")
        print(f"  - portfolio_returns.csv")
        print(f"  - equity_curve.csv")
        print(f"  - asset_strategy_returns.csv")
        print(f"  - per_asset_stats.csv")
        print(f"  - meta.json")
        print(f"  - equity_curve.png")
        print(f"  - return_histogram.png")
        
        # Close market connection
        market.close()
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

