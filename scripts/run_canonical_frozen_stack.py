"""
Run Canonical Frozen Stack - Step 1
===================================

This script executes Step 1 of the three-step process:
1. Freeze the stack and run the full system end-to-end (historical)
2. Produce a System Characterization Report
3. System Cleanup

Step 1 establishes the first ever canonical performance record of the entire
layered system, exactly as it will run in paper.

Configuration:
- Core v9 engines
- Engine Policy v1 (Trend + VRP)
- Risk Targeting ON
- Allocator-H ON
- Discretion OFF
- Precomputed decision flow (compute → apply)
- Canonical evaluation window: 2020-01-06 to 2025-10-31

This is NOT optimization. This is "What does the finished system actually look like?"
"""

import sys
import argparse
import yaml
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.canonical_window import load_canonical_window
from run_strategy import main as run_strategy_main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deep_merge_dict(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_and_merge_config(base_config_path: Path, override_config_path: Path) -> dict:
    """Load base config and merge with override config."""
    # Load base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f) or {}
    
    # Load override config
    with open(override_config_path, 'r', encoding='utf-8') as f:
        override_config = yaml.safe_load(f) or {}
    
    # Deep merge configs (override takes precedence)
    merged_config = deep_merge_dict(base_config, override_config)
    
    return merged_config


def write_merged_config(config: dict, output_path: Path):
    """Write merged config to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote merged config to {output_path}")


def update_precomputed_config(precomputed_config_path: Path, compute_run_id: str):
    """Update precomputed config with compute run ID."""
    with open(precomputed_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Update engine policy precomputed run ID
    if 'engine_policy_v1' in config:
        config['engine_policy_v1']['precomputed_run_id'] = compute_run_id
    
    # Update allocator precomputed run ID
    if 'allocator_v1' in config:
        config['allocator_v1']['precomputed_run_id'] = compute_run_id
    
    with open(precomputed_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Updated precomputed config with run_id: {compute_run_id}")


def run_compute_mode(
    base_config_path: Path,
    override_config_path: Path,
    strategy_profile: str,
    start_date: str,
    end_date: str,
    run_id: str = None
) -> str:
    """Run in compute mode to generate artifacts."""
    logger.info("=" * 80)
    logger.info("STEP 1: COMPUTE MODE - Generating Artifacts")
    logger.info("=" * 80)
    
    # Load and merge configs
    merged_config = load_and_merge_config(base_config_path, override_config_path)
    
    # Write temporary merged config
    temp_config_path = project_root / "configs" / "temp_canonical_compute.yaml"
    write_merged_config(merged_config, temp_config_path)
    
    # Generate run_id if not provided
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"canonical_frozen_stack_compute_{timestamp}"
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Strategy Profile: {strategy_profile}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Mode: COMPUTE (generating artifacts)")
    
    # Temporarily modify sys.argv to pass to run_strategy.main
    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            'run_strategy.py',
            '--strategy_profile', strategy_profile,
            '--start', start_date,
            '--end', end_date,
            '--run_id', run_id,
            '--config_path', str(temp_config_path)
        ]
        
        # Run strategy
        results = run_strategy_main()
        
        if results is None:
            raise RuntimeError("Strategy run failed")
        
        logger.info("=" * 80)
        logger.info("COMPUTE MODE COMPLETE")
        logger.info(f"Run ID: {run_id}")
        logger.info("=" * 80)
        
        return run_id
        
    finally:
        sys.argv = original_argv


def run_precomputed_mode(
    base_config_path: Path,
    override_config_path: Path,
    strategy_profile: str,
    start_date: str,
    end_date: str,
    compute_run_id: str,
    run_id: str = None
) -> str:
    """Run in precomputed mode to apply artifacts."""
    logger.info("=" * 80)
    logger.info("STEP 1: PRECOMPUTED MODE - Applying Artifacts")
    logger.info("=" * 80)
    
    # Update precomputed config with compute run ID
    update_precomputed_config(override_config_path, compute_run_id)
    
    # Load and merge configs
    merged_config = load_and_merge_config(base_config_path, override_config_path)
    
    # Write temporary merged config
    temp_config_path = project_root / "configs" / "temp_canonical_precomputed.yaml"
    write_merged_config(merged_config, temp_config_path)
    
    # Generate run_id if not provided
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"canonical_frozen_stack_precomputed_{timestamp}"
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Strategy Profile: {strategy_profile}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Mode: PRECOMPUTED (applying artifacts from {compute_run_id})")
    
    # Temporarily modify sys.argv to pass to run_strategy.main
    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            'run_strategy.py',
            '--strategy_profile', strategy_profile,
            '--start', start_date,
            '--end', end_date,
            '--run_id', run_id,
            '--config_path', str(temp_config_path)
        ]
        
        # Run strategy
        results = run_strategy_main()
        
        if results is None:
            raise RuntimeError("Strategy run failed")
        
        logger.info("=" * 80)
        logger.info("PRECOMPUTED MODE COMPLETE")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Baseline Run ID: {compute_run_id}")
        logger.info("=" * 80)
        
        return run_id
        
    finally:
        sys.argv = original_argv


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Canonical Frozen Stack - Step 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the canonical frozen stack in two passes:
1. COMPUTE MODE: Generates allocator and engine policy artifacts
2. PRECOMPUTED MODE: Applies artifacts to generate final baseline

The canonical evaluation window is 2020-01-06 to 2025-10-31.

Examples:
  # Run with default settings
  python scripts/run_canonical_frozen_stack.py
  
  # Run with custom run IDs
  python scripts/run_canonical_frozen_stack.py \\
    --compute_run_id my_compute_run \\
    --precomputed_run_id my_precomputed_run
        """
    )
    
    parser.add_argument(
        '--compute_run_id',
        type=str,
        default=None,
        help='Run ID for compute mode (default: auto-generated)'
    )
    
    parser.add_argument(
        '--precomputed_run_id',
        type=str,
        default=None,
        help='Run ID for precomputed mode (default: auto-generated)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD). Default: canonical window start (2020-01-06)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD). Default: canonical window end (2025-10-31)'
    )
    
    parser.add_argument(
        '--strategy_profile',
        type=str,
        default='core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro',
        help='Strategy profile name (default: core_v9)'
    )
    
    parser.add_argument(
        '--skip_compute',
        action='store_true',
        help='Skip compute mode (use existing compute run)'
    )
    
    parser.add_argument(
        '--existing_compute_run_id',
        type=str,
        default=None,
        help='Use existing compute run ID (required if --skip_compute)'
    )
    
    args = parser.parse_args()
    
    # Load canonical window
    canonical_start, canonical_end = load_canonical_window()
    start_date = args.start or canonical_start
    end_date = args.end or canonical_end
    
    # Paths
    base_config_path = project_root / "configs" / "strategies.yaml"
    compute_config_path = project_root / "configs" / "canonical_frozen_stack_compute.yaml"
    precomputed_config_path = project_root / "configs" / "canonical_frozen_stack_precomputed.yaml"
    
    # Validate config files exist
    if not base_config_path.exists():
        logger.error(f"Base config not found: {base_config_path}")
        return 1
    
    if not compute_config_path.exists():
        logger.error(f"Compute config not found: {compute_config_path}")
        return 1
    
    if not precomputed_config_path.exists():
        logger.error(f"Precomputed config not found: {precomputed_config_path}")
        return 1
    
    try:
        # Step 1a: Run compute mode
        if args.skip_compute:
            if not args.existing_compute_run_id:
                logger.error("--existing_compute_run_id required when --skip_compute")
                return 1
            compute_run_id = args.existing_compute_run_id
            logger.info(f"Using existing compute run ID: {compute_run_id}")
        else:
            compute_run_id = run_compute_mode(
                base_config_path=base_config_path,
                override_config_path=compute_config_path,
                strategy_profile=args.strategy_profile,
                start_date=start_date,
                end_date=end_date,
                run_id=args.compute_run_id
            )
        
        # Step 1b: Run precomputed mode
        precomputed_run_id = run_precomputed_mode(
            base_config_path=base_config_path,
            override_config_path=precomputed_config_path,
            strategy_profile=args.strategy_profile,
            start_date=start_date,
            end_date=end_date,
            compute_run_id=compute_run_id,
            run_id=args.precomputed_run_id
        )
        
        logger.info("=" * 80)
        logger.info("STEP 1 COMPLETE: CANONICAL FROZEN STACK BASELINE ESTABLISHED")
        logger.info("=" * 80)
        logger.info(f"Compute Run ID: {compute_run_id}")
        logger.info(f"Precomputed Run ID: {precomputed_run_id}")
        logger.info(f"Results saved to: reports/runs/{precomputed_run_id}/")
        logger.info("=" * 80)
        logger.info("Next Steps:")
        logger.info("1. Review artifacts in reports/runs/{}/".format(precomputed_run_id))
        logger.info("2. Run Step 2: System Characterization Report")
        logger.info("3. Run Step 3: System Cleanup")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

