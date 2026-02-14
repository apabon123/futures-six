#!/usr/bin/env python3
"""
Backfill leverage_summary.json for runs that don't have it (e.g. pre-telemetry runs).

Reads weights.csv and meta.json from reports/runs/<run_id>/ and writes
reports/runs/<run_id>/analysis/leverage_summary.json.

Usage:
    python scripts/analysis/backfill_leverage_summary.py --run_id <run_id>
    python scripts/analysis/backfill_leverage_summary.py --run_id v1_frozen_baseline_20200106_20251031_20260213_190700
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def backfill_leverage_summary(run_id: str, run_dir_base: str = "reports/runs") -> bool:
    run_dir = PROJECT_ROOT / run_dir_base / run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return False

    out_path = run_dir / "analysis" / "leverage_summary.json"
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return True

    weights_path = run_dir / "weights.csv"
    if not weights_path.exists():
        print(f"No weights.csv in {run_dir}", file=sys.stderr)
        return False

    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    gross_series = weights.abs().sum(axis=1)

    meta = _load_json(run_dir / "meta.json")
    target_vol = None
    leverage_cap = None
    if meta and "config" in meta:
        rt = meta["config"].get("risk_targeting", {})
        if isinstance(rt, dict):
            target_vol = rt.get("target_vol")
            leverage_cap = rt.get("leverage_cap")

    metrics_eval = (meta or {}).get("metrics_eval", {})
    metrics_full = (meta or {}).get("metrics_full", {})
    realized_vol = metrics_eval.get("vol") or (metrics_full.get("vol") if metrics_full else None)

    leverage_summary = {
        "run_id": run_id,
        "gross_exposure_avg": float(gross_series.mean()) if len(gross_series) > 0 else None,
        "gross_exposure_median": float(gross_series.median()) if len(gross_series) > 0 else None,
        "gross_exposure_max": float(gross_series.max()) if len(gross_series) > 0 else None,
        "gross_exposure_p95": float(gross_series.quantile(0.95)) if len(gross_series) > 0 else None,
        "scaling_factor_avg": None,
        "target_vol": float(target_vol) if target_vol is not None else None,
        "realized_vol": float(realized_vol) if realized_vol is not None else None,
        "leverage_cap": float(leverage_cap) if leverage_cap is not None else None,
        "source": "backfill",
    }

    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(leverage_summary, f, indent=2)
    print(f"Wrote {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill leverage_summary.json for a run")
    parser.add_argument("--run_id", required=True, help="Run ID")
    parser.add_argument("--run_dir_base", default="reports/runs", help="Base directory for runs")
    args = parser.parse_args()
    ok = backfill_leverage_summary(args.run_id, args.run_dir_base)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
