"""
Phase-1 Diagnostics Script for Persistence Atomic Sleeve.

Runs three strategy profiles:
1. Baseline Trend (core_v3_no_macro)
2. Standalone Persistence (persistence_phase1)
3. External blend (core_v3_trend_plus_persistence_external)

Saves results to reports/runs/<run_id>/ for comparison.
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import run_strategy
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_strategy import main as run_strategy_main


def run_strategy_profile(profile_name: str, run_id: str, start_date: str, end_date: str):
    """
    Run a strategy profile and save results.
    
    Args:
        profile_name: Strategy profile name from configs/strategies.yaml
        run_id: Run identifier for saving artifacts
        start_date: Start date for backtest
        end_date: End date for backtest
    """
    logger.info("=" * 80)
    logger.info(f"Running strategy profile: {profile_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info("=" * 80)
    
    # Build command-line arguments for run_strategy.py
    sys.argv = [
        "run_strategy.py",
        "--strategy_profile", profile_name,
        "--run_id", run_id,
        "--start", start_date,
        "--end", end_date
    ]
    
    try:
        run_strategy_main()
        logger.info(f"✓ Completed run: {run_id}")
    except Exception as e:
        logger.error(f"✗ Failed run: {run_id}")
        logger.error(f"Error: {e}")
        raise


def main():
    """
    Run Phase-1 diagnostics for Persistence atomic sleeve.
    
    This script runs three strategy profiles:
    1. Baseline Trend (core_v3_no_macro)
    2. Standalone Persistence (persistence_phase1 - custom profile)
    3. External blend (core_v3_trend_plus_persistence_external)
    
    Results are saved to reports/runs/<run_id>/ for comparison.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase-1 diagnostics for Persistence atomic sleeve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default run_ids
  python scripts/run_persistence_phase1.py
  
  # Custom date range
  python scripts/run_persistence_phase1.py \\
    --start 2021-01-01 \\
    --end 2025-10-31
  
  # Custom run_id prefix
  python scripts/run_persistence_phase1.py \\
    --run_id_prefix persistence_phase1_v1
        """
    )
    
    parser.add_argument(
        "--run_id_prefix",
        type=str,
        default="persistence_phase1",
        help="Prefix for run identifiers (default: persistence_phase1)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2021-01-01",
        help="Start date for backtest (default: 2021-01-01)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2025-10-31",
        help="End date for backtest (default: 2025-10-31)"
    )
    
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline run (if already completed)"
    )
    
    parser.add_argument(
        "--skip_standalone",
        action="store_true",
        help="Skip standalone persistence run"
    )
    
    parser.add_argument(
        "--skip_external",
        action="store_true",
        help="Skip external blend run"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PERSISTENCE PHASE-1 DIAGNOSTICS")
    logger.info("=" * 80)
    logger.info(f"Run ID Prefix: {args.run_id_prefix}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Baseline Trend (core_v3_no_macro)
    if not args.skip_baseline:
        baseline_run_id = f"{args.run_id_prefix}_baseline_{timestamp}"
        logger.info("\n[1/3] Running Baseline Trend (core_v3_no_macro)...")
        run_strategy_profile(
            profile_name="core_v3_no_macro",
            run_id=baseline_run_id,
            start_date=args.start,
            end_date=args.end
        )
    else:
        logger.info("\n[1/3] Skipping Baseline Trend (already completed)")
        baseline_run_id = None
    
    # 2. Standalone Persistence
    # Create a custom profile on-the-fly by modifying config or use a pre-defined one
    # For now, we'll create a simple persistence-only profile
    if not args.skip_standalone:
        standalone_run_id = f"{args.run_id_prefix}_standalone_{timestamp}"
        logger.info("\n[2/3] Running Standalone Persistence...")
        # Note: We need to create a persistence-only profile in strategies.yaml
        # For now, we'll use a workaround by creating a minimal config
        run_strategy_profile(
            profile_name="persistence_phase1",
            run_id=standalone_run_id,
            start_date=args.start,
            end_date=args.end
        )
    else:
        logger.info("\n[2/3] Skipping Standalone Persistence")
        standalone_run_id = None
    
    # 3. External blend (Trend + Persistence)
    if not args.skip_external:
        external_run_id = f"{args.run_id_prefix}_external_{timestamp}"
        logger.info("\n[3/3] Running External Blend (Trend + Persistence)...")
        run_strategy_profile(
            profile_name="core_v3_trend_plus_persistence_external",
            run_id=external_run_id,
            start_date=args.start,
            end_date=args.end
        )
    else:
        logger.info("\n[3/3] Skipping External Blend")
        external_run_id = None
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE-1 DIAGNOSTICS COMPLETE")
    logger.info("=" * 80)
    logger.info("\nRun IDs:")
    if baseline_run_id:
        logger.info(f"  Baseline Trend: {baseline_run_id}")
    if standalone_run_id:
        logger.info(f"  Standalone Persistence: {standalone_run_id}")
    if external_run_id:
        logger.info(f"  External Blend: {external_run_id}")
    
    logger.info("\nNext steps:")
    logger.info("  1. Run performance diagnostics:")
    if external_run_id and baseline_run_id:
        logger.info(f"     python scripts/run_perf_diagnostics.py --run_id {external_run_id} --baseline_id {baseline_run_id}")
    logger.info("  2. Compare metrics in reports/runs/<run_id>/")
    logger.info("  3. Review per-asset statistics")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

