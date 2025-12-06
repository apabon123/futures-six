"""
Phase-1 Diagnostics Script for Residual Trend Atomic Sleeve.

Runs the experimental strategy profile (core_v3_trend_plus_residual_experiment)
and saves results to reports/runs/<run_id>/ for comparison with baseline.
"""

import sys
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import run_strategy
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_strategy import main as run_strategy_main


def main():
    """
    Run Phase-1 diagnostics for Residual Trend atomic sleeve.
    
    This script is a wrapper around run_strategy.py that:
    1. Uses the experimental strategy profile: core_v3_trend_plus_residual_experiment
    2. Runs over default window (2021-01-01 to 2025-10-31) unless specified
    3. Saves results to reports/runs/<run_id>/
    """
    parser = argparse.ArgumentParser(
        description="Run Phase-1 diagnostics for Residual Trend atomic sleeve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default run_id
  python scripts/run_residual_trend_phase1.py --run_id residual_trend_phase1_v1
  
  # Custom date range
  python scripts/run_residual_trend_phase1.py \\
    --run_id residual_trend_phase1_v1 \\
    --start 2021-01-01 \\
    --end 2025-10-31
  
  # Custom strategy profile (if needed)
  python scripts/run_residual_trend_phase1.py \\
    --run_id residual_trend_phase1_v1 \\
    --strategy_profile core_v3_trend_plus_residual_experiment
        """
    )
    
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run identifier for saving artifacts (e.g., 'residual_trend_phase1_v1')"
    )
    
    parser.add_argument(
        "--strategy_profile",
        type=str,
        default="core_v3_trend_plus_residual_experiment",
        help="Strategy profile to use (default: core_v3_trend_plus_residual_experiment)"
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
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("RESIDUAL TREND PHASE-1 DIAGNOSTICS")
    logger.info("=" * 80)
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Strategy Profile: {args.strategy_profile}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info("=" * 80)
    
    # Create a mock sys.argv for run_strategy.main()
    # run_strategy.main() uses argparse internally, so we need to pass args correctly
    # We'll call the internal logic directly instead
    
    # Import the main function components
    import sys
    from pathlib import Path
    import pandas as pd
    import yaml
    
    from src.agents import MarketData
    from src.agents.strat_combined import CombinedStrategy
    from src.agents.feature_service import FeatureService
    from src.agents.overlay_volmanaged import VolManagedOverlay
    from src.agents.overlay_macro_regime import MacroRegimeFilter
    from src.agents.risk_vol import RiskVol
    from src.agents.allocator import Allocator
    from src.agents.exec_sim import ExecSim
    
    # Load config
    config_path = Path("configs/strategies.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load strategy profile
    strategy_profiles = config.get("strategy_profiles", {})
    if args.strategy_profile not in strategy_profiles:
        logger.error(f"Strategy profile '{args.strategy_profile}' not found in config")
        logger.info(f"Available profiles: {list(strategy_profiles.keys())}")
        return None
    
    profile_config = strategy_profiles[args.strategy_profile]
    
    # Override strategies config with profile
    strategies_cfg = profile_config.get("strategies", config.get("strategies", {}))
    
    # Override macro_regime config with profile
    profile_macro = profile_config.get("macro_regime", {})
    if profile_macro:
        macro_cfg = {**config.get("macro_regime", {}), **profile_macro}
    else:
        macro_cfg = config.get("macro_regime", {})
    
    features_cfg = config.get("features", {}) if config else {}
    
    # Initialize components (reuse run_strategy logic)
    # We'll use a simplified approach: just call run_strategy.main() with modified sys.argv
    # But since run_strategy.main() doesn't take args, we need to modify sys.argv temporarily
    
    # Save original argv
    original_argv = sys.argv.copy()
    
    # Set up sys.argv for run_strategy.main()
    sys.argv = [
        "run_strategy.py",
        "--strategy_profile", args.strategy_profile,
        "--start", args.start,
        "--end", args.end,
        "--run_id", args.run_id
    ]
    
    try:
        # Call run_strategy.main()
        results = run_strategy_main()
        
        if results is None:
            logger.error("Strategy run failed")
            return None
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE-1 DIAGNOSTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: reports/runs/{args.run_id}/")
        logger.info("\nNext steps:")
        logger.info(f"  1. Run diagnostics comparison:")
        logger.info(f"     python scripts/run_perf_diagnostics.py \\")
        logger.info(f"       --run_id {args.run_id} \\")
        logger.info(f"       --baseline_id <baseline_run_id>")
        logger.info("=" * 80)
        
        return results
        
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results is not None else 1)

