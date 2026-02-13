#!/usr/bin/env python3
"""
Backfill sleeve_returns.csv for an existing Phase-2 VRP run.

Reuses existing run (no market re-run). Computes atomic VRP sleeve returns
for the run's date range and writes reports/runs/<run_id>/sleeve_returns.csv
with columns date, vrp_core_meta, vrp_convergence_meta, vrp_alt_meta.

Usage:
    python scripts/diagnostics/backfill_phase2_vrp_sleeve_returns.py --run_id core_v5_vrp_core_phase2_20251209_220603
    python scripts/diagnostics/backfill_phase2_vrp_sleeve_returns.py --run_id core_v5_vrp_core_phase2_20251209_220603 --start 2020-01-01 --end 2025-10-31
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnostics.phase2_vrp_sleeve_io import write_vrp_sleeve_returns_csv


def _profile_and_loader_for_run_id(run_id: str):
    """Return (profile_name, compute_sleeve_returns_func) for a Phase-2 VRP run_id."""
    if "core_v5_vrp_core_phase2" in run_id:
        from scripts.diagnostics.run_core_v5_trend_csmom_vrp_core_phase2 import compute_sleeve_returns
        return "core_v5_trend_csmom_vrp_core_no_macro", compute_sleeve_returns
    if "core_v6_vrp_convergence_phase2" in run_id:
        from scripts.diagnostics.run_core_v6_trend_csmom_vrp_core_convergence_phase2 import compute_sleeve_returns
        return "core_v6_trend_csmom_vrp_core_convergence_no_macro", compute_sleeve_returns
    if "core_v6_vrp_alt_phase2" in run_id:
        from scripts.diagnostics.run_core_v6_trend_csmom_vrp_core_convergence_vrp_alt_phase2 import compute_sleeve_returns
        return "core_v6_trend_csmom_vrp_core_convergence_vrp_alt_no_macro", compute_sleeve_returns
    raise ValueError(f"Unknown Phase-2 VRP run_id pattern: {run_id}")


def main():
    ap = argparse.ArgumentParser(description="Backfill Phase-2 VRP sleeve_returns.csv")
    ap.add_argument("--run_id", required=True, help="Phase-2 VRP run_id (e.g. core_v5_vrp_core_phase2_20251209_220603)")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: from portfolio_returns or 2020-01-01)")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: from portfolio_returns or 2025-10-31)")
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / "reports" / "runs" / args.run_id
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1
    portfolio_path = run_dir / "portfolio_returns.csv"
    if not portfolio_path.exists():
        print(f"portfolio_returns.csv not found in {run_dir}", file=sys.stderr)
        return 1

    import pandas as pd
    from src.agents import MarketData

    port_returns = pd.read_csv(portfolio_path, parse_dates=["date"], index_col="date")
    date_index = port_returns.index.sort_values()
    start_date = args.start or str(date_index.min().date())
    end_date = args.end or str(date_index.max().date())

    profile, compute_sleeve_returns = _profile_and_loader_for_run_id(args.run_id)

    market = MarketData()
    try:
        sleeve_returns = compute_sleeve_returns(profile_name=profile, start_date=start_date, end_date=end_date, market=market)
    finally:
        market.close()

    # Align to run's date index (from portfolio_returns)
    write_vrp_sleeve_returns_csv(run_dir, sleeve_returns, date_index)
    print(f"Wrote {run_dir / 'sleeve_returns.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
