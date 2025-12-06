"""
CLI script for running TSMOM Sign-Only Sanity Check.

This script implements a deliberately simple, academic-style trend-following strategy
to verify that the data and P&L machinery are working correctly.

Usage:
    python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --lookback 252
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import MarketData
from src.diagnostics.tsmom_sanity import (
    run_sign_only_momentum,
    HORIZONS,
    save_results,
    generate_plots
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_horizons(horizons_str: Optional[str]) -> Dict[str, int]:
    """
    Parse comma-separated horizons string into dict.
    
    Args:
        horizons_str: Comma-separated string like "long_252,med_84,short_21" or None
        
    Returns:
        Dict mapping horizon names to lookback days
    """
    if horizons_str is None:
        return HORIZONS.copy()
    
    # Split by comma and strip whitespace
    horizon_names = [h.strip() for h in horizons_str.split(',') if h.strip()]
    
    # Build dict from requested horizons
    horizons_dict = {}
    for name in horizon_names:
        if name in HORIZONS:
            horizons_dict[name] = HORIZONS[name]
        else:
            logger.warning(f"Unknown horizon name: {name}, skipping")
    
    if not horizons_dict:
        logger.warning("No valid horizons specified, using defaults")
        return HORIZONS.copy()
    
    return horizons_dict


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
        description="Run TSMOM Sign-Only Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31
  
  # Custom lookback period
  python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --lookback 126
  
  # Specific universe
  python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --universe ES,NQ,RTY,CL,GC
  
  # Custom output directory
  python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/trend/my_run
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
        default=None,
        help="Lookback period in days (if specified, runs single horizon; otherwise runs all horizons)"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated list of horizons to test (e.g., long_252,med_84,short_21). "
             "If not specified, uses default horizons."
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
        help="Output directory (default: reports/sanity_checks/trend/<atomic_sleeve>/<timestamp>)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("TSMOM SIGN-ONLY SANITY CHECK (Multi-Horizon)")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        
        # Determine horizons to test
        if args.lookback is not None:
            # Single horizon mode (backward compatibility)
            horizons = {"single": args.lookback}
            logger.info(f"Single lookback: {args.lookback} days")
        else:
            # Multi-horizon mode
            horizons = parse_horizons(args.horizons)
            logger.info(f"Horizons to test: {list(horizons.keys())}")
            for name, lb in horizons.items():
                logger.info(f"  {name}: {lb} days")
        
        # Parse universe
        universe = parse_universe(args.universe)
        if universe:
            logger.info(f"Universe: {universe}")
        else:
            logger.info("Universe: All assets from MarketData")
        
        # Initialize MarketData
        logger.info("\n[1/5] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  MarketData universe: {len(market.universe)} symbols")
        
        # Load continuous prices
        logger.info("\n[2/5] Loading continuous prices...")
        prices_cont = market.prices_cont
        logger.info(f"  Prices shape: {prices_cont.shape}")
        logger.info(f"  Date range: {prices_cont.index.min()} to {prices_cont.index.max()}")
        
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
        
        # Determine base output directory (Meta-Sleeve level)
        meta_sleeve_dir = Path("reports/sanity_checks/trend")
        meta_sleeve_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp once for this run (shared across all atomic sleeves)
        if args.output_dir:
            base_output_path = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_path = None
        
        # Loop over horizons (Atomic Sleeves)
        logger.info("\n[3/5] Computing sign-only TSMOM for each atomic sleeve...")
        all_metrics = []
        
        # Map horizon names to atomic sleeve names
        atomic_sleeve_map = {
            'long_252': 'long_term',
            'med_84': 'medium_term',
            'short_21': 'short_term'
        }
        
        for horizon_name, lookback in horizons.items():
            logger.info(f"\n  Processing {horizon_name} (lookback={lookback} days)...")
            
            # Run strategy
            result = run_sign_only_momentum(
                prices=prices_cont,
                lookback=lookback,
                start_date=args.start,
                end_date=args.end,
                universe=None  # Already filtered above
            )
            
            # Create atomic sleeve directory structure: Meta-Sleeve → Atomic Sleeve → Timestamp
            atomic_sleeve_name = atomic_sleeve_map.get(horizon_name, horizon_name)
            atomic_sleeve_dir = meta_sleeve_dir / atomic_sleeve_name
            atomic_sleeve_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamp subdirectory for this run
            if base_output_path:
                timestamp_dir = base_output_path / atomic_sleeve_name
            else:
                timestamp_dir = atomic_sleeve_dir / timestamp
            
            timestamp_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare stats dict for save_results
            stats = {
                'portfolio': result['metrics'],
                'per_asset': result['per_asset_stats']
            }
            
            # Save results
            save_results(
                results=result,
                stats=stats,
                output_dir=timestamp_dir,
                start_date=args.start,
                end_date=args.end,
                lookback=lookback,
                universe=universe if universe else list(prices_cont.columns),
                horizon_name=horizon_name
            )
            
            # Generate plots for this atomic sleeve
            logger.info(f"    Generating plots for {atomic_sleeve_name}...")
            generate_plots(
                results=result,
                stats=stats,
                output_dir=timestamp_dir,
                prices=prices_cont
            )
            
            # Collect metrics for summary
            metrics = result['metrics']
            all_metrics.append({
                'horizon': horizon_name,
                'lookback': lookback,
                'cagr': metrics.get('CAGR', float('nan')),
                'vol': metrics.get('Vol', float('nan')),
                'sharpe': metrics.get('Sharpe', float('nan')),
                'maxdd': metrics.get('MaxDD', float('nan')),
                'hit_rate': metrics.get('HitRate', float('nan')),
                'n_days': metrics.get('n_days', 0),
                'years': metrics.get('years', float('nan'))
            })
        
        # Create and save summary table at meta-sleeve level
        logger.info("\n[4/5] Creating summary table...")
        summary_df = pd.DataFrame(all_metrics)
        if base_output_path is None:
            summary_path = meta_sleeve_dir / f"summary_{timestamp}.csv"
        else:
            summary_path = base_output_path / "summary_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary table
        print("\n" + "=" * 80)
        print("SIGN-ONLY TSMOM HORIZON SUMMARY")
        print("=" * 80)
        display_cols = ['horizon', 'lookback', 'cagr', 'vol', 'sharpe', 'maxdd', 'hit_rate']
        summary_display = summary_df[display_cols].copy()
        
        # Format for display
        for col in ['cagr', 'vol', 'maxdd']:
            if col in summary_display.columns:
                summary_display[col] = summary_display[col].apply(
                    lambda x: f"{x:.4f} ({x*100:.2f}%)" if not pd.isna(x) else "NaN"
                )
        for col in ['sharpe', 'hit_rate']:
            if col in summary_display.columns:
                summary_display[col] = summary_display[col].apply(
                    lambda x: f"{x:.4f}" if not pd.isna(x) else "NaN"
                )
        
        print(summary_display.to_string(index=False))
        
        # Add interpretation comments
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        
        for _, row in summary_df.iterrows():
            horizon = row['horizon']
            sharpe = row['sharpe']
            if not pd.isna(sharpe):
                if sharpe >= 0.3:
                    comment = "✓ Strong"
                elif sharpe >= 0.0:
                    comment = "⚠ Weak"
                else:
                    comment = "✗ Negative"
                print(f"  {horizon:15} (lookback={int(row['lookback']):3d}): Sharpe={sharpe:6.4f} - {comment}")
        
        print("\n" + "=" * 80)
        print("HORIZON ANALYSIS")
        print("=" * 80)
        
        strong_horizons = summary_df[summary_df['sharpe'] >= 0.3]
        weak_horizons = summary_df[(summary_df['sharpe'] >= 0.0) & (summary_df['sharpe'] < 0.3)]
        negative_horizons = summary_df[summary_df['sharpe'] < 0.0]
        
        if len(strong_horizons) > 0:
            print(f"\n✓ Strong horizons (Sharpe >= 0.3): {len(strong_horizons)}")
            for _, row in strong_horizons.iterrows():
                print(f"  - {row['horizon']} (lookback={int(row['lookback'])}): Sharpe={row['sharpe']:.4f}")
            print("  These horizons show reasonable positive Sharpe and are valid 'economic ideas'.")
            print("  They strongly support using these horizons as sleeves or sub-horizons.")
        
        if len(weak_horizons) > 0:
            print(f"\n⚠ Weak horizons (0.0 <= Sharpe < 0.3): {len(weak_horizons)}")
            for _, row in weak_horizons.iterrows():
                print(f"  - {row['horizon']} (lookback={int(row['lookback'])}): Sharpe={row['sharpe']:.4f}")
            print("  These horizons show weak or zero Sharpe.")
            print("  May need different construction (e.g., trend + reversal mix, filters) or down-weighting.")
        
        if len(negative_horizons) > 0:
            print(f"\n✗ Negative horizons (Sharpe < 0.0): {len(negative_horizons)}")
            for _, row in negative_horizons.iterrows():
                print(f"  - {row['horizon']} (lookback={int(row['lookback'])}): Sharpe={row['sharpe']:.4f}")
            print("  These horizons show negative Sharpe.")
            print("  Consider disabling or heavily down-weighting these horizons.")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {meta_sleeve_dir}")
        print(f"  - Atomic sleeves: {', '.join([atomic_sleeve_map.get(h, h) for h in horizons.keys()])}")
        print(f"Summary table: {summary_path}")
        
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

