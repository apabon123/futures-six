"""
Determinism Test Script

Runs the same canonical baseline twice in separate processes and checks
if weights_post_construction.csv is identical (bitwise determinism).

This tests whether the pipeline has any non-deterministic operations like:
- Parallel BLAS reductions with different thread orderings
- Unordered dict/set iteration
- Hash randomization affecting iteration order
- Groupby/merge operations without explicit sort
"""

import subprocess
import sys
import hashlib
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import shutil

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
    print("DETERMINISM TEST")
    print("Testing if two identical runs produce identical weights")
    print("=" * 70)
    print()
    
    # Create merged config
    with open(project_root / 'configs/strategies.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f) or {}
    with open(project_root / 'configs/canonical_frozen_stack_precomputed.yaml', 'r', encoding='utf-8') as f:
        override_config = yaml.safe_load(f) or {}
    
    merged = deep_merge_dict(base_config, override_config)
    
    temp_path = project_root / 'configs/temp_determinism_test.yaml'
    with open(temp_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged, f, default_flow_style=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_a = f"determinism_test_A_{timestamp}"
    run_id_b = f"determinism_test_B_{timestamp}"
    
    # Run A
    print(f"Running test A: {run_id_a}...")
    ok_a = run_baseline(run_id_a, temp_path)
    
    # Run B
    print(f"Running test B: {run_id_b}...")
    ok_b = run_baseline(run_id_b, temp_path)
    
    if not (ok_a and ok_b):
        print("\nOne or both runs failed. Cannot test determinism.")
        return 1
    
    # Compare artifacts
    print()
    print("=" * 70)
    print("COMPARING ARTIFACTS")
    print("=" * 70)
    
    artifacts_to_check = [
        'weights_raw.csv',
        'weights_post_construction.csv',
        'weights_post_risk_targeting.csv',
        'sleeve_returns.csv',
        'asset_returns.csv',
    ]
    
    run_a_dir = project_root / 'reports/runs' / run_id_a
    run_b_dir = project_root / 'reports/runs' / run_id_b
    
    all_match = True
    
    for artifact in artifacts_to_check:
        path_a = run_a_dir / artifact
        path_b = run_b_dir / artifact
        
        if not path_a.exists() or not path_b.exists():
            print(f"  {artifact}: MISSING")
            all_match = False
            continue
        
        h_a = hash_csv(path_a)
        h_b = hash_csv(path_b)
        
        if h_a == h_b:
            print(f"  {artifact}: MATCH ({h_a})")
        else:
            print(f"  {artifact}: MISMATCH")
            print(f"    A: {h_a}")
            print(f"    B: {h_b}")
            all_match = False
            
            # Show first difference
            df_a = pd.read_csv(path_a, index_col=0, parse_dates=True)
            df_b = pd.read_csv(path_b, index_col=0, parse_dates=True)
            diff = (df_a - df_b).abs()
            max_diff = diff.max().max()
            print(f"    Max abs diff: {max_diff:.2e}")
            
            if max_diff > 0:
                first_diff_idx = diff[diff > 1e-15].any(axis=1)
                if first_diff_idx.any():
                    first_diff = first_diff_idx.idxmax()
                    print(f"    First diff date: {first_diff}")
    
    print()
    print("=" * 70)
    if all_match:
        print("RESULT: DETERMINISTIC - All artifacts match")
        print("=" * 70)
        return 0
    else:
        print("RESULT: NON-DETERMINISTIC - Artifacts differ between runs")
        print("=" * 70)
        print()
        print("This indicates the pipeline has non-deterministic operations.")
        print("Common causes:")
        print("  1. BLAS parallel reductions (multi-threaded NumPy)")
        print("  2. Unordered dict/set iteration")
        print("  3. Python hash randomization (PYTHONHASHSEED)")
        print("  4. Pandas groupby/merge without explicit sort")
        return 1


if __name__ == "__main__":
    sys.exit(main())
