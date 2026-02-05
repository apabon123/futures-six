"""
Phase 3B Baseline Rerun Script

Runs two canonical baselines back-to-back:
1. artifacts_only: allocator computed but not applied (isolates engine+RT)
2. traded: allocator applied (actual traded curve)

Both use same window, same pinned governance, same frozen stack.
"""

import sys
import yaml
from pathlib import Path
import logging
from datetime import datetime
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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


def run_baseline(apply_enabled: bool, run_id: str) -> str:
    """Run a single baseline with specified allocator_apply_enabled setting."""
    logger.info("=" * 80)
    logger.info(f"Running Phase 3B baseline: {run_id}")
    logger.info(f"Allocator apply_enabled: {apply_enabled}")
    logger.info("=" * 80)
    
    # Load base config
    base_config_path = project_root / "configs" / "strategies.yaml"
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f) or {}
    
    # Load override config
    override_config_path = project_root / "configs" / "canonical_frozen_stack_precomputed.yaml"
    with open(override_config_path, 'r', encoding='utf-8') as f:
        override_config = yaml.safe_load(f) or {}
    
    # Set allocator_apply_enabled
    if 'allocator_v1' not in override_config:
        override_config['allocator_v1'] = {}
    override_config['allocator_v1']['apply_enabled'] = apply_enabled
    
    # Merge configs
    merged_config = deep_merge_dict(base_config, override_config)
    
    # Write temp merged config
    temp_config_path = project_root / "configs" / f"temp_phase3b_{run_id}.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created merged config: {temp_config_path}")
    
    # Run backtest
    cmd = [
        sys.executable,
        str(project_root / "run_strategy.py"),
        "--strategy_profile", "core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro",
        "--start", "2020-01-06",
        "--end", "2025-10-31",
        "--run_id", run_id,
        "--config_path", str(temp_config_path)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Backtest failed for {run_id}")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Backtest failed for {run_id}")
    
    logger.info(f"✓ Completed baseline: {run_id}")
    return run_id


def main():
    """Run both Phase 3B baselines."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run artifacts-only baseline
    artifacts_only_run_id = f"phase3b_baseline_artifacts_only_{timestamp}"
    run_baseline(apply_enabled=False, run_id=artifacts_only_run_id)
    
    # Run traded baseline
    traded_run_id = f"phase3b_baseline_traded_{timestamp}"
    run_baseline(apply_enabled=True, run_id=traded_run_id)
    
    logger.info("=" * 80)
    logger.info("Phase 3B Baseline Rerun Complete")
    logger.info(f"Artifacts-only: {artifacts_only_run_id}")
    logger.info(f"Traded: {traded_run_id}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
