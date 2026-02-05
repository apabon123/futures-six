"""
Decoupling Test Script

Tests that the allocator (Layer 6) does not feed back into portfolio construction (Layer 3).

Expected behavior after fix:
- weights_post_construction.csv should be IDENTICAL between artifacts-only and traded runs
- weights_post_allocator.csv should DIFFER when allocator has teeth (scalars < 1.0)

This ensures layer separation is preserved and attribution is not corrupted.
"""

import subprocess
import sys
import hashlib
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent


def deep_merge_dict(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def hash_csv(path: Path) -> str:
    """Compute SHA256 hash of CSV contents."""
    df = pd.read_csv(path)
    content = df.to_csv(index=False, float_format='%.15g')
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def run_baseline(run_id: str, config_path: Path) -> bool:
    """Run a single baseline."""
    cmd = [
        sys.executable,
        str(project_root / 'run_strategy.py'),
        '--strategy_profile', 'core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro',
        '--start', '2020-01-06',
        '--end', '2025-10-31',
        '--run_id', run_id,
        '--config_path', str(config_path)
    ]
    
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f'FAILED: {run_id}')
        print(result.stderr[-2000:] if result.stderr else 'No stderr')
        return False
    return True


def main():
    print("=" * 70)
    print("DECOUPLING TEST")
    print("Verifying allocator (Layer 6) does not affect construction (Layer 3)")
    print("=" * 70)
    print()
    
    # Load and merge configs
    with open(project_root / 'configs/strategies.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f) or {}
    with open(project_root / 'configs/canonical_frozen_stack_precomputed.yaml', 'r', encoding='utf-8') as f:
        override_config = yaml.safe_load(f) or {}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create artifacts-only config
    merged_ao = deep_merge_dict(base_config, override_config)
    merged_ao['allocator_v1']['apply_enabled'] = False
    temp_ao = project_root / f'configs/temp_decoupling_test_ao_{timestamp}.yaml'
    with open(temp_ao, 'w', encoding='utf-8') as f:
        yaml.dump(merged_ao, f, default_flow_style=False)
    
    # Create traded config  
    merged_tr = deep_merge_dict(base_config, override_config)
    merged_tr['allocator_v1']['apply_enabled'] = True
    temp_tr = project_root / f'configs/temp_decoupling_test_tr_{timestamp}.yaml'
    with open(temp_tr, 'w', encoding='utf-8') as f:
        yaml.dump(merged_tr, f, default_flow_style=False)
    
    run_id_ao = f'decoupling_test_ao_{timestamp}'
    run_id_tr = f'decoupling_test_tr_{timestamp}'
    
    # Run artifacts-only
    print(f'Running artifacts-only: {run_id_ao}...')
    ok_ao = run_baseline(run_id_ao, temp_ao)
    status_ao = 'OK' if ok_ao else 'FAILED'
    print(f'  Result: {status_ao}')
    
    # Run traded
    print(f'Running traded: {run_id_tr}...')
    ok_tr = run_baseline(run_id_tr, temp_tr)
    status_tr = 'OK' if ok_tr else 'FAILED'
    print(f'  Result: {status_tr}')
    
    if not (ok_ao and ok_tr):
        print('\nOne or both runs failed. Cannot verify decoupling.')
        return 1
    
    print()
    print("=" * 70)
    print("COMPARING ARTIFACTS")
    print("=" * 70)
    
    run_ao_dir = project_root / 'reports/runs' / run_id_ao
    run_tr_dir = project_root / 'reports/runs' / run_id_tr
    
    # Test 1: weights_post_construction should be IDENTICAL
    h_ao = hash_csv(run_ao_dir / 'weights_post_construction.csv')
    h_tr = hash_csv(run_tr_dir / 'weights_post_construction.csv')
    
    construction_match = h_ao == h_tr
    
    print()
    print('Test 1: weights_post_construction (should be IDENTICAL)')
    print(f'  artifacts-only: {h_ao}')
    print(f'  traded:         {h_tr}')
    print(f'  MATCH: {construction_match}')
    
    if not construction_match:
        # Show diff magnitude
        df_ao = pd.read_csv(run_ao_dir / 'weights_post_construction.csv', index_col=0)
        df_tr = pd.read_csv(run_tr_dir / 'weights_post_construction.csv', index_col=0)
        diff = (df_ao - df_tr).abs()
        print(f'  Max abs diff: {diff.max().max():.2e}')
        print('  FAIL: Allocator is still feeding back into construction!')
    
    # Test 2: weights_post_allocator should DIFFER (when allocator has teeth)
    h_alloc_ao = hash_csv(run_ao_dir / 'weights_post_allocator.csv')
    h_alloc_tr = hash_csv(run_tr_dir / 'weights_post_allocator.csv')
    
    allocator_differ = h_alloc_ao != h_alloc_tr
    
    print()
    print('Test 2: weights_post_allocator (should DIFFER when allocator has teeth)')
    print(f'  artifacts-only: {h_alloc_ao}')
    print(f'  traded:         {h_alloc_tr}')
    print(f'  DIFFER: {allocator_differ}')
    
    if not allocator_differ:
        print('  WARNING: Allocator effect not visible. Check if scalars < 1.0 exist.')
    
    # Summary
    print()
    print("=" * 70)
    if construction_match and allocator_differ:
        print("RESULT: PASS - Layer separation preserved")
        print("  - Construction (Layer 3) is decoupled from Allocator (Layer 6)")
        print("  - Allocator effect is isolated to post-allocator stage")
        print("=" * 70)
        return 0
    elif construction_match:
        print("RESULT: PARTIAL PASS - Construction decoupled, but allocator effect not visible")
        print("=" * 70)
        return 0
    else:
        print("RESULT: FAIL - Layer separation violated")
        print("  - Allocator is feeding back into construction stage")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
