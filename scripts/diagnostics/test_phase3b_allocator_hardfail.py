"""
Pre-flight Test: Verify Allocator Hard-Fail for Canonical Baselines

Tests that canonical baselines hard-fail when allocator is requested but scalar loading fails.
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


def test_preflight_hardfail():
    """Test that canonical baseline hard-fails when scalar loading fails."""
    logger.info("=" * 80)
    logger.info("PRE-FLIGHT TEST: Allocator Hard-Fail for Canonical Baselines")
    logger.info("=" * 80)
    
    # Load base config
    base_config_path = project_root / "configs" / "strategies.yaml"
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f) or {}
    
    # Load override config
    override_config_path = project_root / "configs" / "canonical_frozen_stack_precomputed.yaml"
    with open(override_config_path, 'r', encoding='utf-8') as f:
        override_config = yaml.safe_load(f) or {}
    
    # Intentionally break scalar loading with non-existent run_id
    override_config['allocator_v1']['precomputed_run_id'] = 'NONEXISTENT_RUN_ID_FOR_TESTING'
    
    # Merge configs
    merged_config = deep_merge_dict(base_config, override_config)
    
    # Write temp merged config
    temp_config_path = project_root / "configs" / "temp_preflight_test_hardfail.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created test config with broken precomputed_run_id: {temp_config_path}")
    
    # Run backtest - should hard-fail
    run_id = f"preflight_test_hardfail_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    logger.info("Expected: Hard-fail with detailed error message")
    
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    
    # Verify it failed
    if result.returncode == 0:
        logger.error("❌ PRE-FLIGHT TEST FAILED: Run should have hard-failed but succeeded!")
        logger.error("This indicates the hard-fail mechanism is not working.")
        return False
    
    # Verify error message contains expected details
    error_output = result.stdout + result.stderr
    required_keywords = [
        'CANONICAL BASELINE ALLOCATOR LOAD FAILURE',
        'Expected scalar path',
        'Source run ID',
        'NONEXISTENT_RUN_ID_FOR_TESTING'
    ]
    
    missing_keywords = []
    for keyword in required_keywords:
        if keyword not in error_output:
            missing_keywords.append(keyword)
    
    if missing_keywords:
        logger.error(f"❌ PRE-FLIGHT TEST FAILED: Error message missing keywords: {missing_keywords}")
        logger.error(f"Error output:\n{error_output}")
        return False
    
    logger.info("✅ PRE-FLIGHT TEST PASSED: Hard-fail triggered with proper error message")
    logger.info("Error message contains all required details:")
    for keyword in required_keywords:
        logger.info(f"  ✓ {keyword}")
    
    # Cleanup
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    return True


if __name__ == "__main__":
    success = test_preflight_hardfail()
    sys.exit(0 if success else 1)
