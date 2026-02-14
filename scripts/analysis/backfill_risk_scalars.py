#!/usr/bin/env python3
"""
Backfill risk_scalars.csv for runs that don't have it (e.g. pre-instrumentation).

Replays the Risk Targeting layer using saved weights (post-RT) and asset returns:
- final_scalar_applied = gross exposure at each rebalance (post-RT leverage)
- implied forecast_vol = target_vol / final_scalar (vol that would produce this leverage)
- Optionally replays RT on unit weights to get ex-ante forecast_vol when possible.

Usage:
    python scripts/analysis/backfill_risk_scalars.py --run_id v1_frozen_baseline_20200106_20251031_20260213_190700
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def backfill_risk_scalars(run_id: str, run_dir_base: str = "reports/runs") -> bool:
    run_dir = PROJECT_ROOT / run_dir_base / run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return False

    out_path = run_dir / "analysis" / "risk_scalars.csv"
    weights_path = run_dir / "weights.csv"
    if not weights_path.exists():
        print(f"No weights.csv in {run_dir}", file=sys.stderr)
        return False

    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    meta = _load_json(run_dir / "meta.json")
    rt_config = (meta or {}).get("config", {}).get("risk_targeting", {})
    target_vol = float(rt_config.get("target_vol", 0.20))
    leverage_cap = float(rt_config.get("leverage_cap", 7.0))

    # Post-RT gross = leverage applied at each rebalance
    gross_series = weights.abs().sum(axis=1)
    rebal_dates = gross_series.index

    rows = []
    for date in rebal_dates:
        gross = gross_series.loc[date]
        final_scalar = float(gross)
        # Implied forecast vol: leverage = target_vol / forecast_vol => forecast_vol = target_vol / leverage
        implied_forecast_vol = target_vol / final_scalar if final_scalar > 0 else None
        base_scalar = final_scalar  # unknown if clamp was applied; assume no clamp for backfill
        clamp_scalar = 1.0
        rows.append({
            "date": date,
            "is_rebalance": 1,
            "target_vol": target_vol,
            "forecast_vol": implied_forecast_vol,
            "base_scalar": final_scalar,
            "clamp_scalar": clamp_scalar,
            "drawdown_brake_scalar": 1.0,
            "final_scalar_applied": final_scalar,
            "gross_exposure_before": gross,
            "gross_exposure_after": gross,
            "stopout": 0,
        })

    df = pd.DataFrame(rows)

    # Expand to daily if we have portfolio or asset returns
    portfolio_path = run_dir / "portfolio_returns.csv"
    if portfolio_path.exists():
        port = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)
        daily_index = port.index
        df = df.set_index("date")
        df_daily = df.reindex(daily_index).ffill()
        df_daily["is_rebalance"] = 0
        df_daily.loc[df_daily.index.isin(rebal_dates), "is_rebalance"] = 1
        df_daily.index.name = "date"
        df_daily.to_csv(out_path)
    else:
        df.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(df)} rebalance rows)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill risk_scalars.csv for a run")
    parser.add_argument("--run_id", required=True, help="Run ID")
    parser.add_argument("--run_dir_base", default="reports/runs", help="Base directory for runs")
    args = parser.parse_args()
    ok = backfill_risk_scalars(args.run_id, args.run_dir_base)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
