"""
CLI script for running FX/Commodity Carry Sign-Only Sanity Check.

This script implements a deliberately simple, academic-style carry trading strategy
to verify that the roll yield idea and P&L machinery are working correctly.

Usage:
    python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import MarketData
from src.diagnostics.carry_sanity import (
    run_sign_only_carry,
    compute_subperiod_stats,
    save_results,
    generate_plots,
    CARRY_UNIVERSE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run FX/Commodity Carry Sign-Only Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31
  
  # Custom universe
  python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31 --universe CL,GC,6E
  
  # Custom output directory
  python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/carry/fx_commodity/my_run
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
        "--universe",
        type=str,
        default=None,
        help="Comma-separated list of root symbols (e.g., CL,GC,6E,6B,6J). "
             "If not specified, uses all assets in carry universe."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: reports/sanity_checks/carry/fx_commodity/<timestamp>)"
    )
    parser.add_argument(
        "--break_date",
        type=str,
        default="2022-01-01",
        help="Date to split subperiods for analysis (default: 2022-01-01)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("FX/COMMODITY CARRY SIGN-ONLY SANITY CHECK")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Subperiod break: {args.break_date}")
        
        # Parse universe
        if args.universe:
            universe = [s.strip() for s in args.universe.split(',') if s.strip()]
            logger.info(f"Universe: {universe}")
        else:
            universe = None
            logger.info(f"Universe: All assets in carry universe ({list(CARRY_UNIVERSE.keys())})")
        
        # Initialize MarketData
        logger.info("\n[1/6] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  MarketData universe: {len(market.universe)} symbols")
        
        # Run sign-only carry strategy
        logger.info("\n[2/6] Running sign-only carry strategy...")
        result = run_sign_only_carry(
            market=market,
            start_date=args.start,
            end_date=args.end,
            universe=universe
        )
        
        # Compute subperiod stats
        logger.info("\n[3/6] Computing subperiod statistics...")
        subperiod_stats = compute_subperiod_stats(
            portfolio_returns=result['portfolio_returns'],
            equity_curve=result['equity_curve'],
            break_date=args.break_date
        )
        
        # Prepare stats dict for save_results
        stats = {
            'portfolio': result['metrics'],
            'per_asset': result['per_asset_stats']
        }
        
        # Determine output directory structure: Meta-Sleeve → Atomic Sleeve → Timestamp
        meta_sleeve_dir = Path("reports/sanity_checks/carry")
        meta_sleeve_dir.mkdir(parents=True, exist_ok=True)
        
        # For now, FX/Commodity Carry is the atomic sleeve (SR3 can be added later)
        atomic_sleeve_dir = meta_sleeve_dir / "fx_commodity"
        atomic_sleeve_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this run
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = atomic_sleeve_dir / timestamp
        
        # Save results
        logger.info(f"\n[4/6] Saving results to {output_dir}...")
        save_results(
            results=result,
            stats=stats,
            subperiod_stats=subperiod_stats,
            output_dir=output_dir,
            start_date=args.start,
            end_date=args.end
        )
        
        # Generate plots
        logger.info("[5/6] Generating plots...")
        generate_plots(
            results=result,
            stats=stats,
            subperiod_stats=subperiod_stats,
            output_dir=output_dir
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("FX/COMMODITY CARRY SIGN-ONLY SANITY CHECK RESULTS")
        print("=" * 80)
        
        # Portfolio metrics
        portfolio_metrics = result['metrics']
        print("\nPortfolio (Equal-Weight):")
        print(f"  CAGR:      {portfolio_metrics.get('CAGR', 0)*100:8.2f}%")
        print(f"  Vol:       {portfolio_metrics.get('Vol', 0)*100:8.2f}%")
        print(f"  Sharpe:    {portfolio_metrics.get('Sharpe', 0):8.4f}")
        print(f"  MaxDD:     {portfolio_metrics.get('MaxDD', 0)*100:8.2f}%")
        print(f"  HitRate:   {portfolio_metrics.get('HitRate', 0)*100:8.2f}%")
        print(f"  n_days:    {portfolio_metrics.get('n_days', 0):8d}")
        print(f"  years:     {portfolio_metrics.get('years', 0):8.2f}")
        
        # Per-asset metrics
        per_asset_stats = result['per_asset_stats']
        if not per_asset_stats.empty:
            print("\nPer-Asset Metrics:")
            for sym, row in per_asset_stats.iterrows():
                print(f"\n  {sym}:")
                print(f"    AnnRet:   {row.get('AnnRet', 0)*100:8.2f}%")
                print(f"    AnnVol:   {row.get('AnnVol', 0)*100:8.2f}%")
                print(f"    Sharpe:   {row.get('Sharpe', 0):8.4f}")
        
        # Subperiod stats
        print("\nSubperiod Analysis:")
        for period_name, period_stats in subperiod_stats.items():
            if not period_stats:
                continue
            period_label = "Pre-2022" if period_name == "pre" else "Post-2022"
            print(f"\n  {period_label}:")
            print(f"    CAGR:     {period_stats.get('CAGR', 0)*100:8.2f}%")
            print(f"    Vol:      {period_stats.get('Vol', 0)*100:8.2f}%")
            print(f"    Sharpe:   {period_stats.get('Sharpe', 0):8.4f}")
            print(f"    MaxDD:    {period_stats.get('MaxDD', 0)*100:8.2f}%")
            print(f"    HitRate:  {period_stats.get('HitRate', 0)*100:8.2f}%")
            print(f"    n_days:   {period_stats.get('n_days', 0):8d}")
        
        # Interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        
        portfolio_sharpe = portfolio_metrics.get('Sharpe', 0)
        if not pd.isna(portfolio_sharpe):
            if portfolio_sharpe >= 0.3:
                comment = "✓ Strong - Core carry idea has legs"
                print(f"\nPortfolio Sharpe: {portfolio_sharpe:.4f} - {comment}")
                print("  → The core roll yield idea is valid.")
                print("  → Can proceed with adding complexity (cross-sectional, momentum filters).")
            elif portfolio_sharpe >= 0.2:
                comment = "✓ Reasonable - Carry idea has moderate edge"
                print(f"\nPortfolio Sharpe: {portfolio_sharpe:.4f} - {comment}")
                print("  → The roll yield idea shows moderate edge.")
                print("  → May benefit from additional features (cross-sectional, momentum).")
            elif portfolio_sharpe >= 0.0:
                comment = "⚠ Weak - Carry idea has limited edge"
                print(f"\nPortfolio Sharpe: {portfolio_sharpe:.4f} - {comment}")
                print("  → The roll yield idea shows weak or zero edge.")
                print("  → May need different construction or down-weighting.")
            else:
                comment = "✗ Negative - Question the carry hypothesis"
                print(f"\nPortfolio Sharpe: {portfolio_sharpe:.4f} - {comment}")
                print("  → Should question:")
                print("     - The roll yield calculation")
                print("     - The data quality (rank 0 vs rank 1 prices)")
                print("     - The hypothesis that carry is still a tradable edge")
                print("     - Regime dependency (USD strength, commodity cycles)")
        
        # Asset attribution
        if not per_asset_stats.empty:
            print("\nAsset Attribution:")
            sorted_assets = per_asset_stats.sort_values('Sharpe', ascending=False)
            for sym, row in sorted_assets.iterrows():
                sharpe = row.get('Sharpe', 0)
                if not pd.isna(sharpe):
                    if sharpe >= 0.3:
                        status = "✓ Strong"
                    elif sharpe >= 0.0:
                        status = "⚠ Weak"
                    else:
                        status = "✗ Negative"
                    print(f"  {sym:3s}: Sharpe={sharpe:7.4f} - {status}")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"  - portfolio_returns.csv")
        print(f"  - equity_curve.csv")
        print(f"  - asset_strategy_returns.csv")
        print(f"  - roll_yields.csv")
        print(f"  - meta.json")
        print(f"  - equity_curves.png")
        print(f"  - return_histograms.png")
        print(f"  - subperiod_comparison.png")
        
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

