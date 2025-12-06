"""
CLI script for running CSMOM Phase-1 Diagnostics.

This script runs Phase-1 cross-sectional momentum with multi-horizon z-scored momentum
and volatility-aware cross-sectional ranking.

Usage:
    python scripts/run_csmom_phase1.py --start 2020-01-01 --end 2025-10-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.csmom_phase1 import run_csmom_phase1

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_universe(universe_str: str) -> list | None:
    """Parse comma-separated universe string into list."""
    if not universe_str or universe_str.strip() == "":
        return None
    
    symbols = [s.strip() for s in universe_str.split(',') if s.strip()]
    
    if not symbols:
        return None
    
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
            db_symbols.append(sym)
    
    return db_symbols


def main():
    parser = argparse.ArgumentParser(
        description="Run CSMOM Phase-1 Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python scripts/run_csmom_phase1.py --start 2020-01-01 --end 2025-10-31
  
  # Custom universe
  python scripts/run_csmom_phase1.py --start 2020-01-01 --end 2025-10-31 --universe ES,NQ,RTY,CL,GC
  
  # Custom output directory
  python scripts/run_csmom_phase1.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/csmom/phase1/my_run
        """
    )
    
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--universe", type=str, default="", help="Comma-separated symbols (empty=default)")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory (empty=auto-generate)")
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("CSMOM PHASE-1 DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        
        universe = parse_universe(args.universe) if args.universe else None
        if universe:
            logger.info(f"Universe: {universe}")
        else:
            logger.info("Universe: All assets from MarketData")
        
        outdir = args.output_dir if args.output_dir else None
        
        # Run Phase-1 diagnostics
        logger.info("\n[1/3] Computing Phase-1 CSMOM signals...")
        results = run_csmom_phase1(
            start=args.start,
            end=args.end,
            universe=universe,
            outdir=outdir
        )
        
        # Print summary
        logger.info("\n[2/3] Summary metrics:")
        print("\n" + "=" * 80)
        print("CSMOM PHASE-1 SUMMARY")
        print("=" * 80)
        
        summary = results['summary']
        print(f"  CAGR           : {summary.get('CAGR', float('nan')):10.4f} ({summary.get('CAGR', 0)*100:6.2f}%)")
        print(f"  Vol            : {summary.get('Vol', float('nan')):10.4f} ({summary.get('Vol', 0)*100:6.2f}%)")
        print(f"  Sharpe         : {summary.get('Sharpe', float('nan')):10.4f}")
        print(f"  MaxDD          : {summary.get('MaxDD', float('nan')):10.4f} ({summary.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  HitRate        : {summary.get('HitRate', float('nan')):10.4f} ({summary.get('HitRate', 0)*100:6.2f}%)")
        print(f"  n_days         : {summary.get('n_days', 0):10d}")
        print(f"  years          : {summary.get('years', float('nan')):10.2f}")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {results['outdir']}")
        print(f"  - portfolio_returns.csv")
        print(f"  - equity_curve.csv")
        print(f"  - asset_returns.csv")
        print(f"  - asset_strategy_returns.csv")
        print(f"  - weights.csv")
        print(f"  - signals.csv")
        print(f"  - per_asset_stats.csv")
        print(f"  - summary_metrics.csv")
        print(f"  - meta.json")
        print(f"  - equity_curve.png")
        print(f"  - return_histogram.png")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

