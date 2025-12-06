"""
CLI script for running Residual Trend Sign-Only Sanity Check.

This script implements a Phase-0 diagnostic for the residual trend idea:
- Long-horizon trend (e.g., 252 days) minus short-term movement (e.g., 21 days)
- Sign-only signals, equal-weighted across assets
- No overlays, no vol targeting, no z-scoring

Usage:
    python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --long_lookback 252 --short_lookback 21
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
from src.diagnostics.residual_trend_sanity import (
    run_residual_trend,
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
    # Based on MarketData mapping logic:
    # - Equities (ES, NQ, RTY): *_FRONT_CALENDAR_2D
    # - Rates volume (ZT, ZF, ZN, UB): *_FRONT_VOLUME
    # - Rates calendar (SR3): SR3_FRONT_CALENDAR
    # - FX (6E, 6B, 6J): *_FRONT_CALENDAR
    # - Commodities (CL, GC): *_FRONT_VOLUME
    
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
        description="Run Residual Trend Sign-Only Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31
  
  # Custom lookback periods
  python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --long_lookback 252 --short_lookback 21
  
  # Specific universe
  python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --universe ES,NQ,RTY,CL,GC
  
  # Custom output directory
  python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/trend/residual_trend/my_run
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
        "--long_lookback",
        type=int,
        default=252,
        help="Long-horizon lookback period in days (default: 252)"
    )
    parser.add_argument(
        "--short_lookback",
        type=int,
        default=21,
        help="Short-horizon lookback period in days (default: 21)"
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
        help="Output directory (default: reports/sanity_checks/trend/residual_trend/<timestamp>)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("RESIDUAL TREND SIGN-ONLY SANITY CHECK")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end or 'latest available'}")
        logger.info(f"Long lookback: {args.long_lookback} days")
        logger.info(f"Short lookback: {args.short_lookback} days")
        
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
            sleeve_dirs = get_sleeve_dirs("trend", "residual_trend")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = sleeve_dirs["archive"] / timestamp
            latest_dir = sleeve_dirs["latest"]
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        output_dir = archive_dir  # Use archive_dir for all writes
        
        # Run strategy
        logger.info("\n[3/4] Computing residual trend strategy...")
        result = run_residual_trend(
            prices=prices_cont,
            long_lookback=args.long_lookback,
            short_lookback=args.short_lookback,
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
            long_lookback=args.long_lookback,
            short_lookback=args.short_lookback,
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
        print("RESIDUAL TREND SIGN-ONLY SANITY CHECK RESULTS")
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
                comment = "Residual trend idea shows positive alpha. Eligible for Phase-1."
            elif sharpe >= 0.0:
                verdict = "⚠ WEAK (0.0 <= Sharpe < 0.2)"
                comment = "Residual trend idea shows weak or zero alpha. May need refinement."
            else:
                verdict = "✗ FAIL (Sharpe < 0.0)"
                comment = "Residual trend idea shows negative alpha. Redesign required."
            print(f"\n  {verdict}")
            print(f"  {comment}")
        else:
            print("\n  ⚠ Unable to compute Sharpe ratio (insufficient data)")
        
        # Copy key files to latest/ if using standard structure
        if latest_dir is not None:
            logger.info("\n[5/5] Updating canonical latest/ directory...")
            copy_to_latest(archive_dir, latest_dir)
            update_phase_index("trend", "residual_trend", "phase0")
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

