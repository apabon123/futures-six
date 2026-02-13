#!/usr/bin/env python3
"""
Sleeve-Level Attribution (READ-ONLY)

Reads existing run artifacts and produces:
- reports/runs/<run_id>/analysis/attribution_sleeves.csv
- reports/runs/<run_id>/ATTRIBUTION_SUMMARY.md

Uses sleeve_returns.csv (written by ExecSim). For fuller engine attribution,
run scripts/diagnostics/run_engine_attribution.py first.

Usage:
    python scripts/analysis/attribution_sleeves.py --run_id vrp_canonical_2020_2024_20260212_180537
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_DIR_BASE = PROJECT_ROOT / "reports" / "runs"


def load_csv(run_dir: Path, name: str, parse_dates: bool = True) -> pd.DataFrame:
    """Load CSV from run directory."""
    path = run_dir / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def run_attribution(run_id: str, run_dir: Path = None) -> bool:
    """Compute sleeve-level attribution and write outputs. Returns True on success."""
    if run_dir is None:
        run_dir = RUN_DIR_BASE / run_id

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return False

    sleeve = load_csv(run_dir, "sleeve_returns.csv")
    if sleeve.empty:
        print(f"ERROR: sleeve_returns.csv not found or empty in {run_dir}")
        return False

    portfolio = load_csv(run_dir, "portfolio_returns.csv")
    weights = load_csv(run_dir, "weights.csv")
    meta_path = run_dir / "meta.json"

    # Analysis output dir
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Cumulative PnL by sleeve
    cum_pnl = {}
    for col in sleeve.columns:
        s = sleeve[col].fillna(0)
        cum = (1 + s).cumprod()
        cum_pnl[col] = cum.iloc[-1] - 1.0 if len(cum) > 0 else 0.0

    # Total contribution % (sleeve PnL / portfolio PnL)
    if not portfolio.empty:
        port_col = "ret" if "ret" in portfolio.columns else portfolio.columns[0]
        port_rets = portfolio[port_col].fillna(0)
        total_port_pnl = port_rets.sum()
        contrib_pct = {}
        for col in sleeve.columns:
            s = sleeve[col].fillna(0)
            total_sleeve_pnl = s.sum()
            if abs(total_port_pnl) > 1e-12:
                contrib_pct[col] = total_sleeve_pnl / total_port_pnl
            else:
                contrib_pct[col] = np.nan
    else:
        contrib_pct = {col: np.nan for col in sleeve.columns}

    # Mean absolute daily contribution (proxy for activity/exposure magnitude)
    mean_abs = sleeve.abs().mean().to_dict()

    # Avg gross exposure: from weights we only have blended; use mean |sleeve_return| as proxy
    # (sleeve return magnitude ∝ exposure when return is nonzero)
    avg_gross_proxy = mean_abs  # same metric, labeled as activity

    # Turnover by sleeve: not available from artifacts (would need per-sleeve weight history)
    turnover = {col: np.nan for col in sleeve.columns}

    # Build attribution table
    rows = []
    for col in sleeve.columns:
        rows.append({
            "sleeve": col,
            "cumulative_pnl_pct": cum_pnl[col],
            "total_contribution_pct": contrib_pct[col],
            "mean_abs_daily_contrib": mean_abs[col],
            "avg_turnover_proxy": turnover[col],
        })

    attr_df = pd.DataFrame(rows)
    attr_path = analysis_dir / "attribution_sleeves.csv"
    attr_df.to_csv(attr_path, index=False)
    print(f"Saved: {attr_path}")

    # Cumulative contribution time series (for optional plot)
    cum_contrib = (1 + sleeve.fillna(0)).cumprod() - 1.0
    cum_contrib.to_csv(analysis_dir / "attribution_sleeves_cumulative.csv")
    print(f"Saved: {analysis_dir / 'attribution_sleeves_cumulative.csv'}")

    # ATTRIBUTION_SUMMARY.md
    lines = [
        "# Sleeve Attribution Summary",
        "",
        f"**Run ID:** `{run_id}`",
        "",
        "## Cumulative PnL by Sleeve",
        "",
        "| Sleeve | Cumulative PnL | Contribution % | Mean Abs Daily Contrib |",
        "|--------|----------------|----------------|------------------------|",
    ]
    for _, r in attr_df.iterrows():
        cp = r["cumulative_pnl_pct"]
        tc = r["total_contribution_pct"]
        ma = r["mean_abs_daily_contrib"]
        cp_str = f"{cp:.2%}" if not np.isnan(cp) else "N/A"
        tc_str = f"{tc:.1%}" if not np.isnan(tc) else "N/A"
        ma_str = f"{ma:.6f}" if not np.isnan(ma) else "N/A"
        lines.append(f"| {r['sleeve']} | {cp_str} | {tc_str} | {ma_str} |")

    lines.extend([
        "",
        "## Notes",
        "- Cumulative PnL: (1 + daily_sleeve_return).cumprod() - 1 over the run window",
        "- Contribution %: sleeve PnL / portfolio PnL (can exceed 100% due to diversification)",
        "- Mean |Abs| Daily Contrib: average absolute daily sleeve return (activity proxy)",
        "- Turnover by sleeve: N/A (requires per-sleeve weight history not in artifacts)",
        "",
        "For full engine attribution, run: `python scripts/diagnostics/run_engine_attribution.py --run_id <run_id>`",
    ])

    summary_path = run_dir / "ATTRIBUTION_SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {summary_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Sleeve-level attribution from run artifacts")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID to analyze")
    parser.add_argument("--run_dir", type=str, default=None, help="Optional path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    ok = run_attribution(args.run_id, run_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
