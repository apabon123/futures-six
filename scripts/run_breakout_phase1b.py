"""
Phase-1B Diagnostics Script for Breakout Mid Atomic Sleeve.

Runs multiple strategy profiles with different configurations:
1. Baseline Trend (core_v3_no_macro)
2. Phase-1B Test 1: 70/30 feature weights
3. Phase-1B Test 2: 30/70 feature weights
4. Phase-1B Test 3: Pure 50d (100/0)
5. Phase-1B Test 4: Pure 100d (0/100)

All Phase-1B tests use 3% horizon weight for breakout.

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
    Run Phase-1B diagnostics for Breakout Mid atomic sleeve.
    
    This script runs multiple strategy profiles:
    1. Baseline Trend (core_v3_no_macro)
    2. Phase-1B Test 1: 70/30 feature weights (core_v3_trend_breakout_1b_7030)
    3. Phase-1B Test 2: 30/70 feature weights (core_v3_trend_breakout_1b_3070)
    4. Phase-1B Test 3: Pure 50d (core_v3_trend_breakout_1b_1000)
    5. Phase-1B Test 4: Pure 100d (core_v3_trend_breakout_1b_0100)
    
    Results are saved to reports/runs/<run_id>/ for comparison.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase-1B diagnostics for Breakout Mid atomic sleeve"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2021-01-01",
        help="Start date for backtest (YYYY-MM-DD). Default: 2021-01-01"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-11-19",
        help="End date for backtest (YYYY-MM-DD). Default: 2025-11-19"
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Only run baseline (skip Phase-1B tests)"
    )
    parser.add_argument(
        "--test_only",
        type=str,
        choices=["7030", "3070", "1000", "0100"],
        help="Run only a specific Phase-1B test"
    )
    
    args = parser.parse_args()
    
    start_date = args.start
    end_date = args.end
    
    logger.info("=" * 80)
    logger.info("PHASE-1B: BREAKOUT MID REFINEMENT TESTS")
    logger.info("=" * 80)
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info("")
    
    # Test configurations
    tests = [
        {
            "profile": "core_v3_no_macro",
            "run_id": "core_v3_no_macro_phase1b_baseline",
            "description": "Baseline Trend (no breakout)"
        },
        {
            "profile": "core_v3_trend_breakout_1b_7030",
            "run_id": "breakout_1b_7030",
            "description": "Phase-1B Test 1: 70/30 feature weights (50d/100d)"
        },
        {
            "profile": "core_v3_trend_breakout_1b_3070",
            "run_id": "breakout_1b_3070",
            "description": "Phase-1B Test 2: 30/70 feature weights (50d/100d)"
        },
        {
            "profile": "core_v3_trend_breakout_1b_1000",
            "run_id": "breakout_1b_1000",
            "description": "Phase-1B Test 3: Pure 50d breakout (100/0)"
        },
        {
            "profile": "core_v3_trend_breakout_1b_0100",
            "run_id": "breakout_1b_0100",
            "description": "Phase-1B Test 4: Pure 100d breakout (0/100)"
        }
    ]
    
    # Filter tests based on arguments
    if args.baseline_only:
        tests = [tests[0]]  # Only baseline
    elif args.test_only:
        # Run baseline + specific test
        test_map = {
            "7030": tests[1],
            "3070": tests[2],
            "1000": tests[3],
            "0100": tests[4]
        }
        tests = [tests[0], test_map[args.test_only]]
    
    # Run tests
    for i, test in enumerate(tests, 1):
        logger.info("")
        logger.info(f"[{i}/{len(tests)}] {test['description']}")
        logger.info(f"Profile: {test['profile']}")
        logger.info(f"Run ID: {test['run_id']}")
        logger.info("")
        
        try:
            run_strategy_profile(
                profile_name=test['profile'],
                run_id=test['run_id'],
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"Failed to run {test['profile']}: {e}")
            logger.error("Continuing with next test...")
            continue
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE-1B TESTS COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run diagnostics to compare results:")
    logger.info("   python scripts/run_perf_diagnostics.py --run_id breakout_1b_7030 --baseline_id core_v3_no_macro_phase1b_baseline")
    logger.info("   python scripts/run_perf_diagnostics.py --run_id breakout_1b_3070 --baseline_id core_v3_no_macro_phase1b_baseline")
    logger.info("   python scripts/run_perf_diagnostics.py --run_id breakout_1b_1000 --baseline_id core_v3_no_macro_phase1b_baseline")
    logger.info("   python scripts/run_perf_diagnostics.py --run_id breakout_1b_0100 --baseline_id core_v3_no_macro_phase1b_baseline")
    logger.info("")
    logger.info("2. Evaluate results:")
    logger.info("   - Sharpe >= baseline")
    logger.info("   - MaxDD <= baseline")
    logger.info("   - Acceptable correlations")
    logger.info("")
    logger.info("3. If any test passes criteria, promote to Phase-2")


if __name__ == "__main__":
    main()

