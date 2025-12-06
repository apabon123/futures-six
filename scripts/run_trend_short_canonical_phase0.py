"""
CLI script for running Canonical Short-Term (21d) TSMOM Sign-Only Sanity Check (Phase-0).

This script implements a deliberately simple, academic-style trend-following strategy
to verify that the canonical short-term horizon (21 days) shows positive Sharpe on
our data before proceeding with the full multi-feature implementation.

Strategy (Phase-0):
- For each asset, compute 21-day return momentum (skip 5 days)
- Take the sign of that lookback return (+1 if > 0, -1 if < 0, 0 if ≈ 0)
- Use that sign as the position for the next day
- Daily strategy return = sign * daily_return
- Equal-weight portfolio across assets

Pass Criteria:
- Sharpe >= 0.2 over full window
- Reasonable behavior across years and assets

Usage:
    python scripts/run_trend_short_canonical_phase0.py --start 2020-01-01 --end 2025-10-31
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

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.agents import MarketData
from src.diagnostics.tsmom_sanity import (
    save_results,
    generate_plots,
    compute_summary_stats
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Canonical Short-Term (21d) TSMOM Sign-Only Sanity Check (Phase-0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults (21d lookback, 5d skip)
  python scripts/run_trend_short_canonical_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Custom universe
  python scripts/run_trend_short_canonical_phase0.py --start 2020-01-01 --end 2025-10-31 --universe ES,NQ,RTY,CL,GC
  
  # Custom output directory
  python scripts/run_trend_short_canonical_phase0.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/trend/short_canonical/custom
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date for backtest (YYYY-MM-DD), default: {CANONICAL_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date for backtest (YYYY-MM-DD), default: {CANONICAL_END}"
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
        help="Output directory (default: reports/sanity_checks/trend/short_canonical/archive/<timestamp>)"
    )
    
    args = parser.parse_args()
    
    # Canonical short-term parameters
    LOOKBACK = 21  # Canonical short-term horizon
    SKIP_RECENT = 5  # Skip recent days to avoid noise
    HORIZON_NAME = "short_canonical_21"
    
    try:
        logger.info("=" * 80)
        logger.info("CANONICAL SHORT-TERM (21d) TSMOM SIGN-ONLY SANITY CHECK (Phase-0)")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Lookback: {LOOKBACK} days (canonical short-term)")
        logger.info(f"Skip recent: {SKIP_RECENT} days")
        
        # Parse universe
        universe = None
        if args.universe:
            # Split by comma and strip whitespace
            symbols = [s.strip() for s in args.universe.split(',') if s.strip()]
            
            # Map short names to database symbols
            fx_symbols = {'6E', '6B', '6J'}
            equity_symbols = {'ES', 'NQ', 'RTY'}
            rates_volume = {'ZT', 'ZF', 'ZN', 'UB'}
            rates_calendar = {'SR3'}
            commodities = {'CL', 'GC'}
            
            universe = []
            for sym in symbols:
                if sym in equity_symbols:
                    universe.append(f"{sym}_FRONT_CALENDAR_2D")
                elif sym in rates_volume:
                    universe.append(f"{sym}_FRONT_VOLUME")
                elif sym in rates_calendar:
                    universe.append(f"{sym}_FRONT_CALENDAR")
                elif sym in fx_symbols:
                    universe.append(f"{sym}_FRONT_CALENDAR")
                elif sym in commodities:
                    universe.append(f"{sym}_FRONT_VOLUME")
                else:
                    # Assume it's already a database symbol
                    universe.append(sym)
            
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
        
        # Determine output directory structure
        meta_sleeve_dir = Path("reports/sanity_checks/trend")
        meta_sleeve_dir.mkdir(parents=True, exist_ok=True)
        
        atomic_sleeve_dir = meta_sleeve_dir / "short_canonical"
        atomic_sleeve_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp subdirectory for this run
        if args.output_dir:
            timestamp_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = atomic_sleeve_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp_dir = archive_dir / timestamp
        
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n[3/5] Computing sign-only TSMOM for canonical short-term (21d, skip={SKIP_RECENT}d)...")
        
        # Compute lookback returns with skip
        logger.info(f"  Computing 21d returns with {SKIP_RECENT}d skip...")
        
        # We need: return from (t-5-21) to (t-5)
        # Method: Shift prices by skip_recent, then compute 21d return
        # This gives us: price[t-5] / price[t-5-21] - 1
        prices_shifted = prices_cont.shift(SKIP_RECENT)
        lookback_returns_with_skip = prices_shifted.pct_change(periods=LOOKBACK, fill_method=None)
        
        # Compute daily returns (no shift needed)
        daily_returns = prices_cont.pct_change(fill_method=None)
        
        # Align returns (need common dates where both have data)
        common_idx = lookback_returns_with_skip.index.intersection(daily_returns.index)
        lookback_returns_with_skip = lookback_returns_with_skip.loc[common_idx]
        daily_returns = daily_returns.loc[common_idx]
        
        # Now drop rows where ALL assets are NaN (but keep rows where at least one asset has data)
        lookback_returns_with_skip = lookback_returns_with_skip.dropna(how='all')
        daily_returns = daily_returns.dropna(how='all')
        
        # Re-align after dropna
        common_idx = lookback_returns_with_skip.index.intersection(daily_returns.index)
        lookback_returns_with_skip = lookback_returns_with_skip.loc[common_idx]
        daily_returns = daily_returns.loc[common_idx]
        
        logger.info(f"  Aligned data before date filtering: {len(common_idx)} days")
        
        # Filter by date range (after computing signals, but before generating positions)
        if args.start:
            start_dt = pd.to_datetime(args.start)
            mask = common_idx >= start_dt
            common_idx = common_idx[mask]
            lookback_returns_with_skip = lookback_returns_with_skip.loc[common_idx]
            daily_returns = daily_returns.loc[common_idx]
        
        if args.end:
            end_dt = pd.to_datetime(args.end)
            mask = common_idx <= end_dt
            common_idx = common_idx[mask]
            lookback_returns_with_skip = lookback_returns_with_skip.loc[common_idx]
            daily_returns = daily_returns.loc[common_idx]
        
        logger.info(f"  Aligned data after date filtering: {len(common_idx)} days")
        
        # Generate sign-only positions
        # position_t = sign(lookback_return_with_skip_t)
        positions = lookback_returns_with_skip.copy()
        zero_threshold = 1e-8
        positions[positions.abs() < zero_threshold] = 0.0
        positions[positions > zero_threshold] = 1.0
        positions[positions < -zero_threshold] = -1.0
        
        # Compute per-asset strategy returns
        # strategy_ret_asset_t = position_t * daily_return_t
        asset_strategy_returns = positions * daily_returns
        
        # Aggregate to portfolio (equal-weight across assets each day)
        # portfolio_ret_t = mean(strategy_ret_asset_t across assets)
        portfolio_returns = asset_strategy_returns.mean(axis=1)
        
        # Compute cumulative equity
        equity_curve = (1 + portfolio_returns).cumprod()
        
        result = {
            'portfolio_returns': portfolio_returns,
            'equity_curve': equity_curve,
            'asset_returns': daily_returns,
            'asset_strategy_returns': asset_strategy_returns,
            'positions': positions,
            'lookback_returns': lookback_returns_with_skip
        }
        
        stats = compute_summary_stats(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_strategy_returns=asset_strategy_returns
        )
        
        result['metrics'] = stats['portfolio']
        result['per_asset_stats'] = stats['per_asset']
        
        logger.info("\n[4/5] Saving results...")
        
        # Save results
        save_results(
            results=result,
            stats=stats,
            output_dir=timestamp_dir,
            start_date=args.start,
            end_date=args.end,
            lookback=LOOKBACK,
            universe=universe if universe else list(prices_cont.columns),
            horizon_name=HORIZON_NAME
        )
        
        # Generate plots
        logger.info(f"  Generating plots...")
        generate_plots(
            results=result,
            stats=stats,
            output_dir=timestamp_dir,
            prices=prices_cont
        )
        
        # Update latest symlink (if on Unix-like system)
        try:
            latest_link = atomic_sleeve_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(timestamp_dir.relative_to(atomic_sleeve_dir), target_is_directory=True)
            logger.info(f"  Updated 'latest' symlink to point to this run")
        except Exception as e:
            logger.warning(f"  Could not create 'latest' symlink: {e}")
        
        # Register Phase-0 run in phase_index
        logger.info("\n[5/5] Registering Phase-0 run in phase_index...")
        phase_index_dir = Path("reports/phase_index/trend/short_canonical")
        phase_index_dir.mkdir(parents=True, exist_ok=True)
        
        phase0_file = phase_index_dir / "phase0.txt"
        with open(phase0_file, 'w') as f:
            f.write(f"# Phase-0: Canonical Short-Term (21d) Sign-Only Sanity Check\n")
            f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"run_id: {timestamp_dir.name}\n")
            f.write(f"lookback: {LOOKBACK}\n")
            f.write(f"skip_recent: {SKIP_RECENT}\n")
            f.write(f"sharpe: {result['metrics'].get('Sharpe', float('nan')):.4f}\n")
            f.write(f"cagr: {result['metrics'].get('CAGR', float('nan')):.4f}\n")
            f.write(f"path: {timestamp_dir}\n")
        
        logger.info(f"  Registered in: {phase0_file}")
        
        # Print summary
        metrics = result['metrics']
        print("\n" + "=" * 80)
        print("CANONICAL SHORT-TERM (21d) SIGN-ONLY TSMOM SUMMARY (Phase-0)")
        print("=" * 80)
        print(f"Horizon: {HORIZON_NAME} (lookback={LOOKBACK}, skip={SKIP_RECENT})")
        print(f"CAGR:    {metrics.get('CAGR', float('nan')):.4f} ({metrics.get('CAGR', 0)*100:.2f}%)")
        print(f"Vol:     {metrics.get('Vol', float('nan')):.4f} ({metrics.get('Vol', 0)*100:.2f}%)")
        print(f"Sharpe:  {metrics.get('Sharpe', float('nan')):.4f}")
        print(f"MaxDD:   {metrics.get('MaxDD', float('nan')):.4f} ({metrics.get('MaxDD', 0)*100:.2f}%)")
        print(f"HitRate: {metrics.get('HitRate', float('nan')):.4f}")
        print(f"N Days:  {metrics.get('n_days', 0)}")
        print(f"Years:   {metrics.get('years', float('nan')):.2f}")
        
        # Phase-0 pass criteria
        print("\n" + "=" * 80)
        print("PHASE-0 PASS CRITERIA EVALUATION")
        print("=" * 80)
        
        sharpe = metrics.get('Sharpe', float('nan'))
        if not pd.isna(sharpe):
            if sharpe >= 0.2:
                print(f"✓ Sharpe >= 0.2: {sharpe:.4f} (PASS)")
            else:
                print(f"✗ Sharpe < 0.2: {sharpe:.4f} (FAIL)")
        else:
            print("✗ Sharpe could not be computed (FAIL)")
        
        print("\n" + "=" * 80)
        print("Phase-0 diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {timestamp_dir}")
        print(f"Phase-0 registered in: {phase0_file}")
        
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

