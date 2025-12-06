"""
Phase-1 Diagnostics Script for Canonical Short-Term (21d) Atomic Sleeve.

Runs two strategy profiles:
1. Standalone Canonical Short-Term with equal-weight composite (1/3, 1/3, 1/3)
2. Legacy Short-Term (for comparison, using legacy weights 0.5, 0.3, 0.2)

Saves results to reports/runs/<run_id>/ for comparison.

This validates the canonical short-term sleeve as a standalone atomic sleeve
before integrating it into the Trend Meta-Sleeve in Phase-2.
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import run_strategy
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
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
    Run Phase-1 diagnostics for Canonical Short-Term (21d) atomic sleeve.
    
    This script runs two strategy profiles:
    1. Standalone Canonical Short-Term (equal-weight 1/3, 1/3, 1/3)
    2. Legacy Short-Term (legacy weights 0.5, 0.3, 0.2 for comparison)
    
    Results are saved to reports/runs/<run_id>/ for comparison.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase-1 diagnostics for Canonical Short-Term (21d) atomic sleeve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default run_ids
  python scripts/run_trend_short_canonical_phase1.py
  
  # Custom date range
  python scripts/run_trend_short_canonical_phase1.py \\
    --start 2020-01-01 \\
    --end 2025-10-31
  
  # Custom run_id prefix
  python scripts/run_trend_short_canonical_phase1.py \\
    --run_id_prefix short_canonical_phase1_v1
        """
    )
    
    parser.add_argument(
        "--run_id_prefix",
        type=str,
        default="short_canonical_phase1",
        help="Prefix for run identifiers (default: short_canonical_phase1)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date for backtest (default: {CANONICAL_START})"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date for backtest (default: {CANONICAL_END})"
    )
    
    parser.add_argument(
        "--skip_standalone",
        action="store_true",
        help="Skip standalone canonical short-term run"
    )
    
    parser.add_argument(
        "--skip_legacy",
        action="store_true",
        help="Skip legacy short-term run (for comparison)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("CANONICAL SHORT-TERM (21d) PHASE-1 DIAGNOSTICS")
    logger.info("=" * 80)
    logger.info(f"Run ID Prefix: {args.run_id_prefix}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # We need to create temporary strategy profiles for Phase-1 testing
    # These profiles will use tsmom_short with different variant settings
    # For now, we'll use a workaround where we use the existing tsmom_short strategy
    # but override it to test canonical vs legacy
    
    # 1. Standalone Canonical Short-Term
    if not args.skip_standalone:
        standalone_run_id = f"{args.run_id_prefix}_standalone_{timestamp}"
        logger.info("\n[1/2] Running Standalone Canonical Short-Term (21d, equal-weight 1/3, 1/3, 1/3)...")
        
        # NOTE: We need to create a profile for this in strategies.yaml
        # For now, we'll use momentum_stack_v1 but with only short-term enabled
        # This is a workaround until we have explicit short_canonical_phase1 profile
        logger.warning("  Using tsmom_short with canonical variant (weights will be 1/3, 1/3, 1/3)")
        
        # Create a temporary profile or use existing one
        # We'll assume a profile "short_canonical_phase1" should exist in strategies.yaml
        try:
            run_strategy_profile(
                profile_name="short_canonical_phase1",
                run_id=standalone_run_id,
                start_date=args.start,
                end_date=args.end
            )
        except Exception as e:
            logger.error(f"Failed to run canonical short-term: {e}")
            logger.error("Make sure 'short_canonical_phase1' profile exists in configs/strategies.yaml")
            raise
    else:
        logger.info("\n[1/2] Skipping Standalone Canonical Short-Term")
        standalone_run_id = None
    
    # 2. Legacy Short-Term (for comparison)
    if not args.skip_legacy:
        legacy_run_id = f"{args.run_id_prefix}_legacy_{timestamp}"
        logger.info("\n[2/2] Running Legacy Short-Term (for comparison, weights 0.5, 0.3, 0.2)...")
        
        # Use existing momentum_stack_v1 profile with only short-term enabled
        # Or create a dedicated short_legacy_phase1 profile in strategies.yaml
        try:
            run_strategy_profile(
                profile_name="short_legacy_phase1",
                run_id=legacy_run_id,
                start_date=args.start,
                end_date=args.end
            )
        except Exception as e:
            logger.error(f"Failed to run legacy short-term: {e}")
            logger.error("Make sure 'short_legacy_phase1' profile exists in configs/strategies.yaml")
            raise
    else:
        logger.info("\n[2/2] Skipping Legacy Short-Term")
        legacy_run_id = None
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE-1 DIAGNOSTICS COMPLETE")
    logger.info("=" * 80)
    logger.info("\nRun IDs:")
    if standalone_run_id:
        logger.info(f"  Standalone Canonical Short-Term: {standalone_run_id}")
    if legacy_run_id:
        logger.info(f"  Legacy Short-Term: {legacy_run_id}")
    
    logger.info("\nNext Steps:")
    if standalone_run_id and legacy_run_id:
        logger.info("  1. Compare metrics using run_perf_diagnostics.py")
        logger.info(f"     python scripts/run_perf_diagnostics.py --run_id {standalone_run_id} --baseline_id {legacy_run_id}")
    logger.info("  2. Review reports in reports/runs/")
    logger.info("  3. If Phase-1 passes, proceed to Phase-2 integration")
    
    # Register Phase-1 run in phase_index
    logger.info("\nRegistering Phase-1 run in phase_index...")
    phase_index_dir = Path("reports/phase_index/trend/short_canonical")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase1_file = phase_index_dir / "phase1.txt"
    with open(phase1_file, 'w') as f:
        f.write(f"# Phase-1: Canonical Short-Term (21d) Standalone Atomic Sleeve\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if standalone_run_id:
            f.write(f"standalone_run_id: {standalone_run_id}\n")
        if legacy_run_id:
            f.write(f"legacy_run_id: {legacy_run_id}\n")
        f.write(f"start_date: {args.start}\n")
        f.write(f"end_date: {args.end}\n")
    
    logger.info(f"  Registered in: {phase1_file}")
    logger.info("\nPhase-1 diagnostics script complete!")


if __name__ == "__main__":
    main()

