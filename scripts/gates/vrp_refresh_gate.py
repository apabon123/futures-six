#!/usr/bin/env python3
"""
VRP Smoke-Test Gate

Runs a short-window VRP backtest (2024-03-01 → 2024-04-30) using the VRP
baseline config, validates artifacts, and prints DB provenance at startup.

Usage:
    python scripts/gates/vrp_refresh_gate.py
    python scripts/gates/vrp_refresh_gate.py --profile phase4_vrp_baseline_v1
    python scripts/gates/vrp_refresh_gate.py --config_path configs/phase4_vrp_baseline_v1.yaml
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
import argparse

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SMOKE_START = "2024-03-01"
SMOKE_END = "2024-04-30"
DEFAULT_CONFIG = "configs/phase4_vrp_baseline_v1.yaml"
RUN_DIR_BASE = "reports/runs"


def _resolve_db_path() -> Path:
    """Resolve canonical DuckDB path from configs/data.yaml."""
    import yaml
    data_yaml = PROJECT_ROOT / "configs" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"configs/data.yaml not found at {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("db", {}).get("path", "")
    return (PROJECT_ROOT / raw).resolve()


def print_db_provenance():
    """Print DB provenance: path, size, mtime."""
    try:
        db_path = _resolve_db_path()
        # The DB directory may contain .duckdb files
        duckdb_files = list(db_path.glob("*.duckdb")) if db_path.is_dir() else [db_path]
        if not duckdb_files:
            print(f"[DB-PROVENANCE] WARNING: no .duckdb found under {db_path}")
            return

        for f in duckdb_files:
            stat = f.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
            print(f"[DB-PROVENANCE] path={f}  size={size_mb:.1f} MB  mtime={mtime}")
    except Exception as e:
        print(f"[DB-PROVENANCE] ERROR: {e}")


def print_git_info():
    """Print current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        is_dirty = bool(dirty.stdout.strip())
        print(f"[GIT] commit={commit}  dirty={is_dirty}")
    except Exception as e:
        print(f"[GIT] ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="VRP Smoke-Test Gate")
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Strategy profile name (from configs/strategies.yaml). "
             "If not set, uses --config_path instead.",
    )
    parser.add_argument(
        "--config_path", type=str, default=DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--start", type=str, default=SMOKE_START,
        help=f"Start date (default: {SMOKE_START})",
    )
    parser.add_argument(
        "--end", type=str, default=SMOKE_END,
        help=f"End date (default: {SMOKE_END})",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("VRP SMOKE-TEST GATE")
    print("=" * 80)

    # --- Provenance ---
    print_git_info()
    print_db_provenance()

    # --- Build run_id ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"vrp_smoke_{ts}"
    print(f"\n[GATE] run_id = {run_id}")
    print(f"[GATE] window  = {args.start} → {args.end}")
    print(f"[GATE] config  = {args.config_path}")
    if args.profile:
        print(f"[GATE] profile = {args.profile}")

    # --- Step 1: Run backtest ---
    print("\n" + "-" * 80)
    print("STEP 1: Running VRP smoke backtest …")
    print("-" * 80)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_strategy.py"),
        "--start", args.start,
        "--end", args.end,
        "--run_id", run_id,
        "--config_path", args.config_path,
        "--strict_universe",
    ]
    if args.profile:
        cmd.extend(["--strategy_profile", args.profile])

    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"\n✗ Backtest FAILED (exit code {rc})")
        return 1

    run_dir = PROJECT_ROOT / RUN_DIR_BASE / run_id
    if not run_dir.exists():
        print(f"\n✗ Run directory not found: {run_dir}")
        return 1

    print(f"\n✓ Backtest completed. Artifacts in {run_dir}")

    # --- Step 2: Validate artifacts ---
    print("\n" + "-" * 80)
    print("STEP 2: Validating Phase1C artifacts …")
    print("-" * 80)

    validator = PROJECT_ROOT / "scripts" / "diagnostics" / "validate_phase1c_artifacts.py"
    rc_val = subprocess.call(
        [sys.executable, str(validator), "--run_id", run_id],
        cwd=PROJECT_ROOT,
    )
    validator_pass = rc_val == 0
    print(f"\n{'✓' if validator_pass else '✗'} Validator {'PASS' if validator_pass else 'FAIL'}")

    # --- Step 3: Print summary ---
    print("\n" + "-" * 80)
    print("STEP 3: Run Summary")
    print("-" * 80)

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        metrics = meta.get("metrics_full", meta.get("metrics_eval", {}))
        print(f"  Total Return:  {metrics.get('cagr', 'N/A')}")
        print(f"  Volatility:    {metrics.get('vol', 'N/A')}")
        print(f"  Sharpe:        {metrics.get('sharpe', 'N/A')}")
        print(f"  Max Drawdown:  {metrics.get('max_drawdown', 'N/A')}")
        print(f"  Avg Turnover:  {metrics.get('avg_turnover', 'N/A')}")
    else:
        print("  (meta.json not found)")

    # --- Final verdict ---
    print("\n" + "=" * 80)
    if validator_pass:
        print("✓ VRP SMOKE-TEST GATE: PASS")
    else:
        print("✗ VRP SMOKE-TEST GATE: FAIL")
    print("=" * 80)

    return 0 if validator_pass else 1


if __name__ == "__main__":
    sys.exit(main())
