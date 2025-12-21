#!/usr/bin/env python3
"""
Two-Pass Allocator Audit Orchestration

Runs:
1. Pass 1: Baseline run (allocator disabled)
2. Pass 2: Scaled run (precomputed scalars from Pass 1)
3. Generate comparison report

Usage:
    python scripts/diagnostics/run_allocator_two_pass.py \
        --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
        --start 2024-01-01 \
        --end 2024-12-15
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import yaml


def run_cmd(cmd: list[str]) -> None:
    """Run a shell command, printing and checking for errors."""
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_yaml(path: Path) -> Dict:
    """Load YAML configuration file."""
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def dump_yaml(path: Path, obj: Dict) -> None:
    """Save YAML configuration file."""
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')


def make_temp_config_with_allocator_precomputed(
    base_config_path: Path,
    baseline_run_id: str,
    scalar_filename: str,
    apply_missing_as: float,
) -> Path:
    """Create temporary config with allocator in precomputed mode."""
    cfg = load_yaml(base_config_path)

    # Adjust allocator_v1 settings for Pass 2
    alloc = cfg.get("allocator_v1", {})
    alloc["enabled"] = True
    alloc["mode"] = "precomputed"
    alloc["precomputed_run_id"] = baseline_run_id
    alloc["precomputed_scalar_filename"] = scalar_filename
    alloc["apply_missing_scalar_as"] = apply_missing_as
    cfg["allocator_v1"] = alloc

    # Create temporary config file
    tmp_dir = Path(tempfile.mkdtemp(prefix="alloc_two_pass_"))
    tmp_cfg = tmp_dir / base_config_path.name
    dump_yaml(tmp_cfg, cfg)
    
    print(f"Created temporary config: {tmp_cfg}")
    return tmp_cfg


def main():
    ap = argparse.ArgumentParser(
        description="Two-pass allocator audit: baseline → scaled → compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full two-pass audit
  python scripts/diagnostics/run_allocator_two_pass.py \\
      --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \\
      --start 2024-01-01 \\
      --end 2024-12-15

  # Skip Pass 1 if baseline already exists
  python scripts/diagnostics/run_allocator_two_pass.py \\
      --strategy_profile core_v9 \\
      --start 2024-01-01 \\
      --end 2024-12-15 \\
      --baseline_run_id existing_baseline \\
      --skip_pass1
        """
    )
    ap.add_argument("--strategy_profile", required=True, help="Strategy profile name")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--config_path", default="configs/strategies.yaml", help="Config file path")
    ap.add_argument("--baseline_run_id", default=None, help="Baseline run ID (auto-generated if not provided)")
    ap.add_argument("--scaled_run_id", default=None, help="Scaled run ID (auto-generated if not provided)")
    ap.add_argument("--scalar_filename", default="allocator_risk_v1_applied.csv", help="Scalar filename to load from baseline")
    ap.add_argument("--apply_missing_scalar_as", type=float, default=1.0, help="Default scalar for missing dates")
    ap.add_argument("--runs_root", default="reports/runs", help="Root directory for run artifacts")
    ap.add_argument("--skip_pass1", action="store_true", help="Skip Pass 1 (baseline already exists)")
    ap.add_argument("--skip_compare", action="store_true", help="Skip comparison report generation")
    args = ap.parse_args()

    # Generate run IDs if not provided
    base_run_id = args.baseline_run_id or f"baseline_{args.start}_{args.end}".replace("-", "")
    scaled_run_id = args.scaled_run_id or f"allocpass2_{args.start}_{args.end}".replace("-", "")

    config_path = Path(args.config_path)

    print("=" * 80)
    print("Two-Pass Allocator Audit")
    print("=" * 80)
    print(f"Strategy: {args.strategy_profile}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Baseline run ID: {base_run_id}")
    print(f"Scaled run ID: {scaled_run_id}")
    print("=" * 80)

    # Pass 1 (baseline)
    if not args.skip_pass1:
        print("\n" + "=" * 80)
        print("PASS 1: Baseline (allocator disabled)")
        print("=" * 80)
        
        # Ensure allocator is off for baseline
        # The default config should have enabled: false, mode: "off"
        run_cmd([
            "python", "run_strategy.py",
            "--strategy_profile", args.strategy_profile,
            "--start", args.start,
            "--end", args.end,
            "--run_id", base_run_id,
        ])
        
        print(f"\n[PASS 1 COMPLETE] Run ID: {base_run_id}")
    else:
        print(f"\n[SKIPPING PASS 1] Using existing baseline: {base_run_id}")

    # Pass 2 (scaled with precomputed scalars from pass 1)
    print("\n" + "=" * 80)
    print("PASS 2: Scaled (precomputed scalars from Pass 1)")
    print("=" * 80)
    
    tmp_cfg = make_temp_config_with_allocator_precomputed(
        base_config_path=config_path,
        baseline_run_id=base_run_id,
        scalar_filename=args.scalar_filename,
        apply_missing_as=args.apply_missing_scalar_as,
    )

    try:
        # Note: run_strategy.py currently doesn't support --config argument
        # So we temporarily replace the main config file
        # This is safe because we're using a copy
        backup_cfg = config_path.with_suffix('.yaml.backup')
        shutil.copy(config_path, backup_cfg)
        shutil.copy(tmp_cfg, config_path)
        
        try:
            run_cmd([
                "python", "run_strategy.py",
                "--strategy_profile", args.strategy_profile,
                "--start", args.start,
                "--end", args.end,
                "--run_id", scaled_run_id,
            ])
        finally:
            # Restore original config
            shutil.copy(backup_cfg, config_path)
            backup_cfg.unlink()
        
        print(f"\n[PASS 2 COMPLETE] Run ID: {scaled_run_id}")
    finally:
        # Cleanup temp config
        shutil.rmtree(tmp_cfg.parent, ignore_errors=True)

    # Compare
    if not args.skip_compare:
        print("\n" + "=" * 80)
        print("COMPARISON REPORT")
        print("=" * 80)
        
        run_cmd([
            "python", "scripts/diagnostics/compare_two_runs.py",
            "--baseline_run_id", base_run_id,
            "--scaled_run_id", scaled_run_id,
            "--runs_root", args.runs_root,
        ])

    print("\n" + "=" * 80)
    print("Two-Pass Audit Complete")
    print("=" * 80)
    print(f"Baseline run: {Path(args.runs_root) / base_run_id}")
    print(f"Scaled run:   {Path(args.runs_root) / scaled_run_id}")
    print(f"Comparison:   {Path(args.runs_root) / scaled_run_id / 'two_pass_comparison.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

