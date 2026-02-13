#!/usr/bin/env python3
"""
Canonical VRP Run: Jan 2020 → Oct 2024

One-command script that:
  1) Preflight coverage check
  2) Run backtest via run_strategy.py
  3) Validate artifacts
  4) Write PIN.md provenance file

Usage:
    python scripts/runs/run_vrp_canonical_2020_2024.py
    python scripts/runs/run_vrp_canonical_2020_2024.py --config_path configs/phase4_vrp_baseline_v1.yaml
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANONICAL_START = "2020-01-01"
CANONICAL_END = "2024-10-31"
DEFAULT_CONFIG = "configs/phase4_vrp_baseline_v1.yaml"
RUN_DIR_BASE = "reports/runs"


def _resolve_db_path() -> Path:
    """Resolve canonical DuckDB path from configs/data.yaml."""
    data_yaml = PROJECT_ROOT / "configs" / "data.yaml"
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("db", {}).get("path", "")
    return (PROJECT_ROOT / raw).resolve()


def _find_duckdb(db_dir: Path) -> Path:
    """Find the .duckdb file in the given directory."""
    if db_dir.is_file() and db_dir.suffix == ".duckdb":
        return db_dir
    candidates = sorted(db_dir.glob("*.duckdb"))
    if not candidates:
        raise FileNotFoundError(f"No .duckdb file found in {db_dir}")
    return candidates[0]


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_db_provenance() -> dict:
    try:
        db_dir = _resolve_db_path()
        db_file = _find_duckdb(db_dir)
        stat = db_file.stat()
        return {
            "path": str(db_file),
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


def _load_sleeve_weights(config_path: str) -> dict:
    """Extract sleeve weights from config."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        strategies = cfg.get("strategies", {})
        weights = {}
        for name, scfg in strategies.items():
            if isinstance(scfg, dict) and scfg.get("enabled"):
                weights[name] = scfg.get("weight", 0.0)
        return weights
    except Exception as e:
        return {"error": str(e)}


def write_pin(run_dir: Path, run_id: str, config_path: str, validator_pass: bool):
    """Write PIN.md provenance file."""
    git_commit = _get_git_commit()
    db_prov = _get_db_provenance()
    sleeve_weights = _load_sleeve_weights(config_path)

    lines = [
        "# Run Provenance PIN",
        "",
        f"**run_id**: `{run_id}`",
        f"**window**: {CANONICAL_START} → {CANONICAL_END}",
        f"**config**: `{config_path}`",
        f"**generated**: {datetime.now().isoformat()}",
        "",
        "## Git",
        f"- commit: `{git_commit}`",
        "",
        "## DB Provenance",
    ]
    if "error" in db_prov:
        lines.append(f"- error: {db_prov['error']}")
    else:
        lines.append(f"- path: `{db_prov['path']}`")
        lines.append(f"- size: {db_prov['size_mb']} MB")
        lines.append(f"- mtime: {db_prov['mtime']}")

    lines.extend(["", "## Sleeve Weights"])
    if isinstance(sleeve_weights, dict) and "error" not in sleeve_weights:
        for name, w in sorted(sleeve_weights.items()):
            lines.append(f"- {name}: {w}")
    else:
        lines.append(f"- error: {sleeve_weights.get('error', 'unknown')}")

    lines.extend([
        "",
        "## Validator",
        f"- status: {'PASS' if validator_pass else 'FAIL'}",
    ])

    # Append metrics from meta.json if available
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        metrics = meta.get("metrics_full", {})
        if metrics:
            lines.extend(["", "## Portfolio Metrics (Full Window)"])
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}")

    pin_path = run_dir / "PIN.md"
    pin_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PIN] Written to {pin_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Canonical VRP Run: 2020 → 2024")
    parser.add_argument(
        "--config_path", type=str, default=DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Strategy profile name (from configs/strategies.yaml).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CANONICAL VRP RUN: 2020 → 2024")
    print("=" * 80)

    # --- Step 1: Preflight ---
    print("\n" + "-" * 80)
    print("STEP 1: Preflight coverage check …")
    print("-" * 80)

    preflight = PROJECT_ROOT / "scripts" / "preflight" / "check_window_coverage.py"
    rc_pf = subprocess.call(
        [sys.executable, str(preflight), "--start", CANONICAL_START, "--end", CANONICAL_END],
        cwd=PROJECT_ROOT,
    )
    if rc_pf != 0:
        print("\n✗ Preflight FAILED. Cannot proceed — data coverage insufficient.")
        print("  Fix the missing series, then re-run.")
        return 1

    # --- Step 2: Run backtest ---
    print("\n" + "-" * 80)
    print("STEP 2: Running canonical VRP backtest …")
    print("-" * 80)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"vrp_canonical_2020_2024_{ts}"
    print(f"[RUN] run_id = {run_id}")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_strategy.py"),
        "--start", CANONICAL_START,
        "--end", CANONICAL_END,
        "--run_id", run_id,
        "--config_path", args.config_path,
        "--strict_universe",
    ]
    if args.profile:
        cmd.extend(["--strategy_profile", args.profile])

    rc_run = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if rc_run != 0:
        print(f"\n✗ Backtest FAILED (exit code {rc_run})")
        return 1

    run_dir = PROJECT_ROOT / RUN_DIR_BASE / run_id
    if not run_dir.exists():
        print(f"\n✗ Run directory not found: {run_dir}")
        return 1

    print(f"\n✓ Backtest completed. Artifacts in {run_dir}")

    # --- Step 3: Validate artifacts ---
    print("\n" + "-" * 80)
    print("STEP 3: Validating artifacts …")
    print("-" * 80)

    validator = PROJECT_ROOT / "scripts" / "diagnostics" / "validate_phase1c_artifacts.py"
    rc_val = subprocess.call(
        [sys.executable, str(validator), "--run_id", run_id],
        cwd=PROJECT_ROOT,
    )
    validator_pass = rc_val == 0
    print(f"\n{'✓' if validator_pass else '✗'} Validator {'PASS' if validator_pass else 'FAIL'}")

    # --- Step 4: Write PIN.md ---
    print("\n" + "-" * 80)
    print("STEP 4: Writing PIN.md …")
    print("-" * 80)

    write_pin(run_dir, run_id, args.config_path, validator_pass)

    # --- Final verdict ---
    print("\n" + "=" * 80)
    if validator_pass:
        print(f"✓ CANONICAL RUN COMPLETE: {run_id}")
    else:
        print(f"✗ CANONICAL RUN COMPLETE (validator failed): {run_id}")
    print(f"  Artifacts: {run_dir}")
    print(f"  PIN: {run_dir / 'PIN.md'}")
    print("=" * 80)

    return 0 if validator_pass else 1


if __name__ == "__main__":
    sys.exit(main())
