#!/usr/bin/env python3
"""
CLI script for running VRP Convergence Phase-1 Diagnostics.

This script runs Phase-1 VRP Convergence strategy with z-scored convergence spread (VIX - VX1).

Usage:
    python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.diagnostics.vrp_convergence_phase1 import run_vrp_convergence_phase1
from src.config.backtest_window import CANONICAL_START, CANONICAL_END

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# VRP data starts when VIX3M begins (first obs)
VRP_START = "2009-09-18"


def main():
    parser = argparse.ArgumentParser(
        description="Run VRP Convergence Phase-1 Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults (252d z-score window)
  python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31
  
  # Custom parameters
  python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --zscore_window 252 --clip 3.0
  
  # Custom output directory
  python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --output_dir data/diagnostics/vrp_convergence_phase1/my_run
  
  # Use tanh signal transformation
  python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --signal_mode tanh
  
  # Custom vol targeting
  python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --target_vol 0.12 --vol_lookback 63
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=VRP_START,
        help=f"Start date (YYYY-MM-DD), default: {VRP_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date (YYYY-MM-DD), default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--zscore_window",
        type=int,
        default=252,
        help="Z-score rolling window in days (default: 252)"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=3.0,
        help="Z-score clipping bounds (default: 3.0)"
    )
    parser.add_argument(
        "--signal_mode",
        type=str,
        default="zscore",
        choices=["zscore", "tanh"],
        help="Signal transformation mode (default: zscore)"
    )
    parser.add_argument(
        "--target_vol",
        type=float,
        default=0.10,
        help="Target annualized volatility (default: 0.10 = 10%)"
    )
    parser.add_argument(
        "--vol_lookback",
        type=int,
        default=63,
        help="Volatility lookback for vol targeting in days (default: 63)"
    )
    parser.add_argument(
        "--vol_floor",
        type=float,
        default=0.05,
        help="Minimum volatility floor (default: 0.05 = 5%)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (empty=auto-generate)"
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
        logger.info("VRP CONVERGENCE PHASE-1 DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Z-score window: {args.zscore_window} days")
        logger.info(f"Clip: ±{args.clip}")
        logger.info(f"Signal mode: {args.signal_mode}")
        logger.info(f"Target vol: {args.target_vol*100:.1f}%")
        logger.info(f"Vol lookback: {args.vol_lookback} days")
        logger.info(f"Vol floor: {args.vol_floor*100:.1f}%")
        
        outdir = args.output_dir if args.output_dir else None
        db_path = args.db_path if args.db_path else None
        
        # Run Phase-1 diagnostics
        logger.info("\n[1/3] Computing Phase-1 VRP Convergence signals...")
        results = run_vrp_convergence_phase1(
            start=args.start,
            end=args.end,
            outdir=outdir,
            zscore_window=args.zscore_window,
            clip=args.clip,
            signal_mode=args.signal_mode,
            target_vol=args.target_vol,
            vol_lookback=args.vol_lookback,
            vol_floor=args.vol_floor,
            db_path=db_path
        )
        
        # Print summary
        logger.info("\n[2/3] Summary metrics:")
        print("\n" + "=" * 80)
        print("VRP CONVERGENCE PHASE-1 SUMMARY")
        print("=" * 80)
        
        summary = results['summary']
        print(f"  CAGR           : {summary.get('CAGR', float('nan')):10.4f} ({summary.get('CAGR', 0)*100:6.2f}%)")
        print(f"  Vol            : {summary.get('Vol', float('nan')):10.4f} ({summary.get('Vol', 0)*100:6.2f}%)")
        print(f"  Sharpe         : {summary.get('Sharpe', float('nan')):10.4f}")
        print(f"  MaxDD          : {summary.get('MaxDD', float('nan')):10.4f} ({summary.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  HitRate        : {summary.get('HitRate', float('nan')):10.4f} ({summary.get('HitRate', 0)*100:6.2f}%)")
        print(f"  n_days         : {summary.get('n_days', 0):10d}")
        print(f"  years          : {summary.get('years', float('nan')):10.2f}")
        
        # Signal stats
        signals = results['signals']
        print(f"\nSignal Statistics:")
        print(f"  Mean           : {signals.mean():10.4f}")
        print(f"  Std            : {signals.std():10.4f}")
        print(f"  Min            : {signals.min():10.4f}")
        print(f"  Max            : {signals.max():10.4f}")
        print(f"  % Long (>0.1)  : {(signals > 0.1).sum() / len(signals) * 100:10.1f}%")
        print(f"  % Short (<-0.1): {(signals < -0.1).sum() / len(signals) * 100:10.1f}%")
        print(f"  % Neutral      : {((signals >= -0.1) & (signals <= 0.1)).sum() / len(signals) * 100:10.1f}%")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        print(f"Results saved to: {results['outdir']}")
        print(f"  - portfolio_returns.csv")
        print(f"  - equity_curve.csv")
        print(f"  - vx1_returns.csv")
        print(f"  - signals.csv")
        print(f"  - positions.csv")
        print(f"  - spread_conv_timeseries.csv")
        print(f"  - summary_metrics.csv")
        print(f"  - vrp_convergence_phase1_metrics.json")
        print(f"  - equity_curve.png")
        print(f"  - spread_z_and_signals.png")
        print(f"  - pnl_histogram.png")
        
        logger.info("\n[3/3] Phase-1 registered in reports/phase_index/vrp/vrp_convergence_phase1.txt")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

