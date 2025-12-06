"""
CLI script for running Rates Curve Sign-Only Sanity Check.

This script implements a deliberately simple, academic-style curve trading strategy
to verify that the rates curve features and P&L machinery are working correctly.

Usage:
    python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31
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
from src.agents.feature_service import FeatureService
from src.diagnostics.rates_curve_sanity import (
    run_sign_only_curve,
    compute_subperiod_stats,
    save_results,
    generate_plots,
    RATES_SYMBOLS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Rates Curve Sign-Only Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults (DV01-neutral)
  python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31
  
  # Equal notional weighting
  python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31 --equal_notional
  
  # Custom output directory
  python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/rates_curve_rv/my_run
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
        "--equal_notional",
        action="store_true",
        help="Use equal notional weighting instead of DV01-neutral (default: DV01-neutral)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: reports/sanity_checks/rates_curve_rv/<atomic_sleeve>/<timestamp>)"
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
        logger.info("RATES CURVE SIGN-ONLY SANITY CHECK")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Weighting: {'Equal Notional' if args.equal_notional else 'DV01-Neutral'}")
        logger.info(f"Subperiod break: {args.break_date}")
        
        # Initialize MarketData
        logger.info("\n[1/6] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  MarketData universe: {len(market.universe)} symbols")
        
        # Load continuous prices for rates symbols
        logger.info("\n[2/6] Loading continuous prices for rates symbols...")
        prices_cont = market.prices_cont
        logger.info(f"  Prices shape: {prices_cont.shape}")
        logger.info(f"  Date range: {prices_cont.index.min()} to {prices_cont.index.max()}")
        
        # Check required symbols
        required_symbols = list(RATES_SYMBOLS.values())
        missing = [s for s in required_symbols if s not in prices_cont.columns]
        if missing:
            logger.error(f"Missing required symbols in prices_cont: {missing}")
            logger.error(f"Available columns: {list(prices_cont.columns)}")
            sys.exit(1)
        
        rates_prices = prices_cont[required_symbols].copy()
        logger.info(f"  Loaded {len(required_symbols)} rates symbols: {required_symbols}")
        
        # Initialize FeatureService
        logger.info("\n[3/6] Initializing FeatureService...")
        feature_service = FeatureService(market, config={})
        logger.info("  FeatureService initialized")
        
        # Compute rates curve features
        logger.info("\n[4/6] Computing rates curve features...")
        features_dict = feature_service.get_features(
            end_date=args.end,
            feature_types=["RATES_CURVE"]
        )
        
        if "RATES_CURVE" not in features_dict:
            logger.error("Failed to compute rates curve features")
            sys.exit(1)
        
        curve_features = features_dict["RATES_CURVE"]
        logger.info(f"  Curve features shape: {curve_features.shape}")
        logger.info(f"  Feature columns: {list(curve_features.columns)}")
        logger.info(f"  Date range: {curve_features.index.min()} to {curve_features.index.max()}")
        
        # Check required features
        required_features = ['curve_2s10s_z', 'curve_5s30s_z']
        missing_features = [f for f in required_features if f not in curve_features.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            logger.error(f"Available columns: {list(curve_features.columns)}")
            sys.exit(1)
        
        # Determine output directory structure: Meta-Sleeve → Atomic Sleeve → Timestamp
        meta_sleeve_dir = Path("reports/sanity_checks/rates_curve_rv")
        meta_sleeve_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this run
        if args.output_dir:
            base_output_path = Path(args.output_dir)
            timestamp = None
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_path = None
        
        # Create atomic sleeve directories
        atomic_sleeve_2s10s_dir = meta_sleeve_dir / "2s10s"
        atomic_sleeve_5s30s_dir = meta_sleeve_dir / "5s30s"
        portfolio_dir = meta_sleeve_dir / "portfolio"
        
        atomic_sleeve_2s10s_dir.mkdir(parents=True, exist_ok=True)
        atomic_sleeve_5s30s_dir.mkdir(parents=True, exist_ok=True)
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        
        # Main output directory (portfolio level - combined results)
        if base_output_path:
            output_dir = base_output_path / "portfolio"
        else:
            output_dir = portfolio_dir / timestamp
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run sign-only curve strategy
        logger.info("\n[5/6] Running sign-only curve strategy...")
        use_dv01_neutral = not args.equal_notional
        
        result = run_sign_only_curve(
            prices=rates_prices,
            curve_features=curve_features,
            start_date=args.start,
            end_date=args.end,
            use_dv01_neutral=use_dv01_neutral
        )
        
        # Compute subperiod stats
        logger.info("\n[6/6] Computing subperiod statistics...")
        subperiod_stats = compute_subperiod_stats(
            portfolio_returns=result['portfolio_returns'],
            equity_curve=result['equity_curve'],
            break_date=args.break_date
        )
        
        # Prepare stats dict for save_results
        stats = {
            'portfolio': result['metrics'],
            'legs': result['leg_metrics']
        }
        
        # Save portfolio results (combined 2s10s + 5s30s)
        logger.info(f"\nSaving portfolio results to {output_dir}...")
        save_results(
            results=result,
            stats=stats,
            subperiod_stats=subperiod_stats,
            output_dir=output_dir,
            start_date=args.start,
            end_date=args.end,
            use_dv01_neutral=use_dv01_neutral
        )
        
        # Save atomic sleeve results (individual legs)
        if timestamp:
            atomic_2s10s_dir = atomic_sleeve_2s10s_dir / timestamp
            atomic_5s30s_dir = atomic_sleeve_5s30s_dir / timestamp
        else:
            atomic_2s10s_dir = base_output_path / "2s10s"
            atomic_5s30s_dir = base_output_path / "5s30s"
        
        atomic_2s10s_dir.mkdir(parents=True, exist_ok=True)
        atomic_5s30s_dir.mkdir(parents=True, exist_ok=True)
        
        # Save 2s10s atomic sleeve results
        leg_2s10s_result = {
            'portfolio_returns': result['leg_2s10s_returns'],
            'equity_curve': result['equity_2s10s'],
            'metrics': result['leg_metrics']['2s10s']
        }
        leg_2s10s_stats = {'portfolio': result['leg_metrics']['2s10s']}
        save_results(
            results=leg_2s10s_result,
            stats=leg_2s10s_stats,
            subperiod_stats=subperiod_stats,
            output_dir=atomic_2s10s_dir,
            start_date=args.start,
            end_date=args.end,
            use_dv01_neutral=use_dv01_neutral
        )
        
        # Save 5s30s atomic sleeve results
        leg_5s30s_result = {
            'portfolio_returns': result['leg_5s30s_returns'],
            'equity_curve': result['equity_5s30s'],
            'metrics': result['leg_metrics']['5s30s']
        }
        leg_5s30s_stats = {'portfolio': result['leg_metrics']['5s30s']}
        save_results(
            results=leg_5s30s_result,
            stats=leg_5s30s_stats,
            subperiod_stats=subperiod_stats,
            output_dir=atomic_5s30s_dir,
            start_date=args.start,
            end_date=args.end,
            use_dv01_neutral=use_dv01_neutral
        )
        
        # Generate plots
        logger.info("Generating plots...")
        generate_plots(
            results=result,
            stats=stats,
            subperiod_stats=subperiod_stats,
            output_dir=output_dir
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("RATES CURVE SIGN-ONLY SANITY CHECK RESULTS")
        print("=" * 80)
        
        # Portfolio metrics
        portfolio_metrics = result['metrics']
        print("\nPortfolio (Curve Sleeve - 50/50 blend):")
        print(f"  CAGR:      {portfolio_metrics.get('CAGR', 0)*100:8.2f}%")
        print(f"  Vol:       {portfolio_metrics.get('Vol', 0)*100:8.2f}%")
        print(f"  Sharpe:    {portfolio_metrics.get('Sharpe', 0):8.4f}")
        print(f"  MaxDD:     {portfolio_metrics.get('MaxDD', 0)*100:8.2f}%")
        print(f"  HitRate:   {portfolio_metrics.get('HitRate', 0)*100:8.2f}%")
        print(f"  n_days:    {portfolio_metrics.get('n_days', 0):8d}")
        print(f"  years:     {portfolio_metrics.get('years', 0):8.2f}")
        
        # Per-leg metrics
        leg_metrics = result['leg_metrics']
        print("\nPer-Leg Metrics:")
        for leg_name, leg_stats in leg_metrics.items():
            print(f"\n  {leg_name.upper()}:")
            print(f"    AnnRet:   {leg_stats.get('AnnRet', 0)*100:8.2f}%")
            print(f"    AnnVol:   {leg_stats.get('AnnVol', 0)*100:8.2f}%")
            print(f"    Sharpe:   {leg_stats.get('Sharpe', 0):8.4f}")
        
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
                comment = "✓ Strong - Core curve idea has legs"
                print(f"\nPortfolio Sharpe: {portfolio_sharpe:.4f} - {comment}")
                print("  → The core curve trading idea is valid.")
                print("  → Negative PnL in full strategy likely came from:")
                print("     - Over-complicating with curvature and extra weights")
                print("     - Interaction with other sleeves (allocator, vol overlay)")
                print("     - Implementation details (feature alignment, z-score windows)")
            elif portfolio_sharpe >= 0.0:
                comment = "⚠ Weak - Curve idea has limited edge"
                print(f"\nPortfolio Sharpe: {portfolio_sharpe:.4f} - {comment}")
                print("  → The curve trading idea shows weak or zero edge.")
                print("  → May need different construction or down-weighting.")
            else:
                comment = "✗ Negative - Question the curve hypothesis"
                print(f"\nPortfolio Sharpe: {portfolio_sharpe:.4f} - {comment}")
                print("  → Should question:")
                print("     - The yield reconstruction (FRED + futures)")
                print("     - DV01 inputs")
                print("     - The hypothesis that 'steep vs flat vs hump' is still tradable")
        
        # Leg analysis
        print("\nLeg Analysis:")
        for leg_name, leg_stats in leg_metrics.items():
            leg_sharpe = leg_stats.get('Sharpe', 0)
            if not pd.isna(leg_sharpe):
                if leg_sharpe >= 0.2:
                    comment = "✓ Reasonable"
                elif leg_sharpe >= 0.0:
                    comment = "⚠ Weak"
                else:
                    comment = "✗ Negative"
                print(f"  {leg_name.upper()}: Sharpe={leg_sharpe:.4f} - {comment}")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"  - portfolio_returns.csv")
        print(f"  - equity_curve.csv")
        print(f"  - leg_returns.csv")
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

