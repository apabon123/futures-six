"""
CLI script for running CSMOM Sign-Only Sanity Check.

This script implements a deliberately simple, academic-style cross-sectional momentum strategy
to verify that the data and P&L machinery are working correctly.

Usage:
    python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --lookback 126
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.csmom_sanity import run_sign_only_csmom

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
    if universe_str is None or universe_str.strip() == "":
        return None
    
    # Split by comma and strip whitespace
    symbols = [s.strip() for s in universe_str.split(',') if s.strip()]
    
    if not symbols:
        return None
    
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
        description="Run CSMOM Sign-Only Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31
  
  # Custom lookback period
  python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --lookback 126
  
  # Custom fractions
  python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --top_frac 0.30 --bottom_frac 0.30
  
  # Specific universe
  python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --universe ES,NQ,RTY,CL,GC
  
  # Custom output directory
  python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/csmom/phase0/my_run
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
        required=True,
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=126,
        help="Lookback period in days (default: 126)"
    )
    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.33,
        help="Top fraction to long (0-1, default: 0.33)"
    )
    parser.add_argument(
        "--bottom_frac",
        type=float,
        default=0.33,
        help="Bottom fraction to short (0-1, default: 0.33)"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        help="Comma-separated list of symbols (e.g., ES,NQ,RTY,ZT,ZN,CL,GC,6E,6B,6J). "
             "If not specified, uses all assets from MarketData."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: reports/sanity_checks/csmom/phase0/<timestamp>)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("CSMOM SIGN-ONLY SANITY CHECK")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Lookback: {args.lookback} days")
        logger.info(f"Top fraction: {args.top_frac}")
        logger.info(f"Bottom fraction: {args.bottom_frac}")
        
        # Parse universe
        universe = parse_universe(args.universe) if args.universe else None
        if universe:
            logger.info(f"Universe: {universe}")
        else:
            logger.info("Universe: All assets from MarketData")
        
        # Determine output directory
        if args.output_dir:
            outdir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = f"reports/sanity_checks/csmom/phase0/{timestamp}"
        
        logger.info(f"Output directory: {outdir}")
        
        # Run strategy
        logger.info("\n[1/3] Computing sign-only CSMOM...")
        results = run_sign_only_csmom(
            lookback=args.lookback,
            top_frac=args.top_frac,
            bottom_frac=args.bottom_frac,
            start=args.start,
            end=args.end,
            universe=universe,
            outdir=outdir,
        )
        
        # Print summary
        logger.info("\n[2/3] Summary metrics:")
        print("\n" + "=" * 80)
        print("SIGN-ONLY CSMOM SUMMARY")
        print("=" * 80)
        
        summary = results.summary
        print(f"  CAGR           : {summary.get('CAGR', float('nan')):10.4f} ({summary.get('CAGR', 0)*100:6.2f}%)")
        print(f"  Vol            : {summary.get('Vol', float('nan')):10.4f} ({summary.get('Vol', 0)*100:6.2f}%)")
        print(f"  Sharpe         : {summary.get('Sharpe', float('nan')):10.4f}")
        print(f"  MaxDD          : {summary.get('MaxDD', float('nan')):10.4f} ({summary.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  HitRate        : {summary.get('HitRate', float('nan')):10.4f} ({summary.get('HitRate', 0)*100:6.2f}%)")
        print(f"  n_days         : {summary.get('n_days', 0):10d}")
        print(f"  years          : {summary.get('years', float('nan')):10.2f}")
        
        # Interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        sharpe = summary.get('Sharpe', float('nan'))
        if not (pd.isna(sharpe) if hasattr(pd, 'isna') else sharpe != sharpe):
            if sharpe >= 0.2:
                comment = "✓ PASS (Sharpe >= 0.2)"
                print(f"  Sharpe = {sharpe:.4f} - {comment}")
                print("  Cross-sectional momentum edge validated. Proceed to Phase-1 (clean implementation).")
            elif sharpe >= 0.0:
                comment = "⚠ WEAK (0.0 <= Sharpe < 0.2)"
                print(f"  Sharpe = {sharpe:.4f} - {comment}")
                print("  Weak edge. May need different construction or down-weighting.")
            else:
                comment = "✗ FAIL (Sharpe < 0.0)"
                print(f"  Sharpe = {sharpe:.4f} - {comment}")
                print("  Negative Sharpe. Sleeve remains disabled until redesigned.")
        else:
            print("  Sharpe = NaN - Unable to compute")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {outdir}")
        print(f"  - portfolio_returns.csv")
        print(f"  - equity_curve.csv")
        print(f"  - asset_returns.csv")
        print(f"  - asset_strategy_returns.csv")
        print(f"  - weights.csv")
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

