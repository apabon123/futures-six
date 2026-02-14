#!/usr/bin/env python3
"""
Produce risk_targeting_forecast_vs_realized.md for a run.

- Uses true forecast_vol from risk_scalars (forecast_vol_true or forecast_vol).
- Computes daily realized vol (rolling 20d and full sample) and rebalance-to-rebalance realized vol.
- Compares forecast to both to detect frequency mismatch vs estimator bias.
- Object consistency: compares rt_weights_snapshot (weights used in forecast) to weights.csv at each rebalance.

Usage:
    python scripts/analysis/risk_targeting_forecast_vs_realized.py --run_id trend_only_ewma_20200106_20251031_20260213_212931
    python scripts/analysis/risk_targeting_forecast_vs_realized.py --run_id v1_frozen_baseline_20200106_20251031_20260213_190700
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


def main():
    parser = argparse.ArgumentParser(description="Produce risk_targeting_forecast_vs_realized.md")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--run_dir_base", default="reports/runs")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir_base / args.run_id
    analysis_dir = run_dir / "analysis"
    csv_path = analysis_dir / "risk_scalars.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path}; run backfill_risk_scalars.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    meta = _load_json(run_dir / "meta.json")
    rt_config = (meta or {}).get("config", {}).get("risk_targeting", {})
    target_vol = float(rt_config.get("target_vol", 0.20))

    # Forecast vol: prefer true, else implied (use rebalance rows only for stats)
    if "forecast_vol_true" in df.columns:
        fvol_col = "forecast_vol_true"
        forecast_source = "true (runtime)"
    else:
        fvol_col = "forecast_vol"
        forecast_source = "implied (backfill)"
    fvol = df[fvol_col].dropna()
    if "is_rebalance" in df.columns:
        rebal_mask = df["is_rebalance"] == 1
        fvol_rebal = df.loc[rebal_mask, fvol_col].dropna()
    else:
        fvol_rebal = fvol

    # Restrict to rebalance rows for forecast stats (avoid ffill diluting)
    if len(fvol_rebal) > 0:
        forecast_avg = float(fvol_rebal.mean())
        forecast_median = float(fvol_rebal.median())
        forecast_min = float(fvol_rebal.min())
        forecast_max = float(fvol_rebal.max())
    else:
        forecast_avg = forecast_median = forecast_min = forecast_max = None

    # Portfolio returns
    port_path = run_dir / "portfolio_returns.csv"
    if not port_path.exists():
        print(f"Missing {port_path}", file=sys.stderr)
        sys.exit(1)
    port = pd.read_csv(port_path, index_col=0, parse_dates=True).squeeze()
    if isinstance(port, pd.DataFrame):
        port = port.iloc[:, 0]
    port = port.dropna()

    # Daily realized vol: full sample and rolling 20d
    daily_var = port.var()
    daily_vol_full = np.sqrt(daily_var * 252) if daily_var > 0 else 0.0
    roll20 = port.rolling(20).std() * np.sqrt(252)
    daily_vol_roll20_avg = float(roll20.mean()) if roll20.notna().any() else None
    daily_vol_roll20_median = float(roll20.median()) if roll20.notna().any() else None

    # Rebalance-to-rebalance realized vol: period returns between rebalance dates, annualized
    weights_path = run_dir / "weights.csv"
    rebal_dates = None
    rebal_vol_avg = rebal_vol_median = None
    if weights_path.exists():
        weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        rebal_dates = pd.to_datetime(weights.index)
        if len(rebal_dates) > 1:
            period_returns = []
            for i in range(len(rebal_dates) - 1):
                mask = (port.index >= rebal_dates[i]) & (port.index < rebal_dates[i + 1])
                if mask.sum() > 0:
                    period_returns.append((1 + port.loc[mask]).prod() - 1)
            if period_returns:
                period_returns = np.array(period_returns)
                n_periods_per_year = 52.0  # weekly rebalance
                rebal_vol_annual = np.std(period_returns) * np.sqrt(n_periods_per_year)
                rebal_vol_avg = float(rebal_vol_annual)
                rebal_vol_median = float(np.median(np.abs(period_returns))) * np.sqrt(n_periods_per_year)

    # Forecast / realized ratios (daily full sample as reference)
    ratio_daily = (forecast_avg / daily_vol_full) if (forecast_avg and daily_vol_full > 0) else None
    ratio_rebal = (forecast_avg / rebal_vol_avg) if (forecast_avg and rebal_vol_avg and rebal_vol_avg > 0) else None

    # Object consistency: rt_weights_snapshot (pre-RT) vs weights.csv (post-RT)
    snap_path = analysis_dir / "rt_weights_snapshot.csv"
    obj_consistent = None
    max_abs_diff_per_date = []
    if snap_path.exists() and weights_path.exists():
        snap = pd.read_csv(snap_path, parse_dates=["date"])
        weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        asset_cols = [c for c in snap.columns if c != "date"]
        for _, row in snap.iterrows():
            d = row.get("date")
            if pd.isna(d) or d not in weights.index:
                continue
            pre = pd.Series({c: row[c] for c in asset_cols if c in row.index}).astype(float).fillna(0)
            post = weights.loc[d].astype(float).fillna(0)
            common = pre.index.intersection(post.index)
            if len(common) == 0:
                continue
            pre = pre.reindex(common).fillna(0)
            post = post.reindex(common).fillna(0)
            gross_pre = pre.abs().sum()
            if gross_pre < 1e-12:
                continue
            scalars_at = df[df.index == d]
            lev = float(scalars_at["final_scalar_applied"].iloc[0]) if "final_scalar_applied" in df.columns and len(scalars_at) > 0 else (post.abs().sum() / gross_pre if gross_pre > 0 else 1.0)
            expected_post = pre / gross_pre * lev
            diff = (post - expected_post).abs()
            max_abs_diff_per_date.append({"date": d, "max_abs_diff": float(diff.max())})
        if max_abs_diff_per_date:
            obj_consistent = max(x["max_abs_diff"] for x in max_abs_diff_per_date) < 1e-5
        else:
            obj_consistent = None
    else:
        obj_consistent = None  # N/A

    # Build report
    lines = [
        "# Risk Targeting: Forecast vs Realized",
        "",
        f"**Run ID**: `{args.run_id}`",
        "",
        "## 1. Forecast vol (ex-ante)",
        "",
        f"- Source: {forecast_source}",
        f"- Avg forecast_vol (rebalance): {forecast_avg:.4f}" if forecast_avg is not None else "- Avg forecast_vol: N/A",
        f"- Median forecast_vol (rebalance): {forecast_median:.4f}" if forecast_median is not None else "- Median: N/A",
        f"- Min/Max forecast_vol: {forecast_min:.4f} / {forecast_max:.4f}" if (forecast_min is not None and forecast_max is not None) else "",
        f"- Target vol: {target_vol:.2%}",
        "",
        "## 2. Realized vol (ex-post)",
        "",
        "| Measure | Avg | Median |",
        "|---------|-----|--------|",
    ]
    if daily_vol_full is not None:
        lines.append(f"| Daily (full sample, annualized) | {daily_vol_full:.4f} | - |")
    if daily_vol_roll20_avg is not None:
        lines.append(f"| Daily rolling 20d vol | {daily_vol_roll20_avg:.4f} | {daily_vol_roll20_median or 0:.4f} |")
    if rebal_vol_avg is not None:
        lines.append(f"| Rebalance-to-rebalance vol | {rebal_vol_avg:.4f} | {rebal_vol_median or 0:.4f} |")
    lines.extend([
        "",
        "## 3. Forecast / Realized ratios",
        "",
    ])
    if ratio_daily is not None:
        lines.append(f"- Forecast / daily realized (full): **{ratio_daily:.3f}**")
    if ratio_rebal is not None:
        lines.append(f"- Forecast / rebalance realized: **{ratio_rebal:.3f}**")
    lines.extend([
        "",
        "## 4. Object consistency (weights used in forecast vs applied)",
        "",
    ])
    if obj_consistent is True:
        lines.append("Weights used in forecast (pre-RT snapshot) match applied weights (post-RT) up to scaling: **consistent**.")
    elif obj_consistent is False:
        lines.append("**Mismatch**: max abs diff between expected (pre*leverage/gross_pre) and applied weights exceeds tolerance.")
        if max_abs_diff_per_date:
            lines.append("")
            lines.append("| Date | Max abs diff |")
            lines.append("|------|--------------|")
            for x in max_abs_diff_per_date[:20]:
                lines.append(f"| {x['date']} | {x['max_abs_diff']:.6f} |")
            if len(max_abs_diff_per_date) > 20:
                lines.append(f"| ... | ({len(max_abs_diff_per_date) - 20} more) |")
    else:
        lines.append("N/A (no rt_weights_snapshot.csv or weights.csv).")
    lines.extend([
        "",
        "## 5. Conclusion",
        "",
    ])
    # Conclusion: estimator bias vs frequency mismatch vs object mismatch
    if ratio_daily and ratio_daily > 1.2:
        lines.append("- **Estimator bias (likely)**: Forecast vol is higher than daily realized (ratio > 1.2), so RT scales down; realized stays below target. Consider lower vol_floor, shorter lookback, or EWMA covariance.")
    elif ratio_daily and ratio_daily < 0.8:
        lines.append("- **Forecast low**: Forecast vol below realized; RT may over-lever.")
    else:
        lines.append("- Forecast and daily realized are in a similar range (ratio near 1).")
    if ratio_rebal is not None and rebal_vol_avg and daily_vol_full and abs(ratio_rebal - ratio_daily) > 0.2:
        lines.append("- **Frequency**: Rebalance-to-rebalance vol differs from daily vol; possible frequency mismatch between forecast (rebalance-date) and realized (daily).")
    if obj_consistent is False:
        lines.append("- **Object mismatch**: Weights used for forecast do not match applied weights; investigate pipeline.")
    lines.append("")

    out_md = analysis_dir / "risk_targeting_forecast_vs_realized.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")

    # Print key table
    print("\n--- Forecast vs Realized ---")
    print(f"forecast_vol (rebalance): avg={forecast_avg:.4f}" if forecast_avg is not None else "forecast_vol: N/A")
    print(f"daily realized (full):    {daily_vol_full:.4f}" if daily_vol_full else "daily realized: N/A")
    print(f"rebalance realized:      {rebal_vol_avg:.4f}" if rebal_vol_avg else "rebalance realized: N/A")
    print(f"forecast/daily ratio:    {ratio_daily:.3f}" if ratio_daily else "ratio: N/A")
    print(f"object consistent:       {obj_consistent}" if obj_consistent is not None else "object consistent: N/A")


if __name__ == "__main__":
    main()
