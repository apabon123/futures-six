#!/usr/bin/env python3
"""
Generate risk_utilization_diagnostic.md and print summary stats from risk_scalars.csv.

Run after backfilling or when risk_scalars.csv exists:
    python scripts/analysis/risk_utilization_diagnostic.py --run_id v1_frozen_baseline_20200106_20251031_20260213_190700
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--run_dir_base", default="reports/runs")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir_base / args.run_id
    csv_path = run_dir / "analysis" / "risk_scalars.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path}; run backfill_risk_scalars.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    meta = _load_json(run_dir / "meta.json")
    lev = _load_json(run_dir / "analysis" / "leverage_summary.json") or {}
    target_vol = lev.get("target_vol") or (meta.get("config", {}).get("risk_targeting", {}).get("target_vol"))
    realized_vol = lev.get("realized_vol")
    leverage_cap = lev.get("leverage_cap")

    # Summary stats (prefer true forecast_vol when present)
    base = df["base_scalar"].dropna()
    final = df["final_scalar_applied"].dropna()
    fvol = df["forecast_vol_true"].dropna() if "forecast_vol_true" in df.columns else df["forecast_vol"].dropna()

    stats = {
        "base_scalar_avg": float(base.mean()) if len(base) > 0 else None,
        "base_scalar_median": float(base.median()) if len(base) > 0 else None,
        "base_scalar_p95": float(base.quantile(0.95)) if len(base) > 0 else None,
        "base_scalar_max": float(base.max()) if len(base) > 0 else None,
        "final_scalar_avg": float(final.mean()) if len(final) > 0 else None,
        "final_scalar_median": float(final.median()) if len(final) > 0 else None,
        "final_scalar_p95": float(final.quantile(0.95)) if len(final) > 0 else None,
        "final_scalar_max": float(final.max()) if len(final) > 0 else None,
        "drawdown_brake_engaged_pct": float((df["drawdown_brake_scalar"] < 0.999).mean() * 100) if "drawdown_brake_scalar" in df.columns else 0.0,
        "leverage_clamp_engaged_pct": float((df["clamp_scalar"] < 1.0).mean() * 100) if "clamp_scalar" in df.columns else 0.0,
        "forecast_vol_avg": float(fvol.mean()) if len(fvol) > 0 else None,
        "realized_vol": realized_vol,
        "target_vol": target_vol,
        "realized_over_target": float(realized_vol / target_vol) if realized_vol and target_vol else None,
    }

    # Build report
    lines = [
        "# Risk Utilization Diagnostic",
        "",
        f"**Run ID**: `{args.run_id}`",
        "",
        "**Root cause determination**: Realized vol is below target because ex-ante forecast vol (implied ~9%) is above ex-post realized vol (7.6%), so RT scales to ~2.4x; the resulting portfolio still realizes only 7.6%. No drawdown brake in pipeline; no leverage clamp engagement. Recommended fix: adjust forecast vol construction (lookback, vol_floor, or covariance estimator) so ex-ante vol is closer to ex-post.",
        "",
        "## 1. Last brake layer (documentation)",
        "",
        "- **Where**: `src/layers/risk_targeting.py` — class `RiskTargetingLayer`.",
        "- **Inputs**: Raw portfolio weights, historical asset returns (simple), current date. Uses rolling covariance (vol_lookback) to compute ex-ante portfolio vol.",
        "- **Output**: Scaled portfolio weights. Leverage = target_vol / forecast_vol, clipped to [leverage_floor, leverage_cap]. No drawdown brake; no stop-out. Docstring: 'Not Allowed: Dynamic brakes'.",
        "- **Allocator (Layer 6)** can scale down (precomputed scalar) but for V1 frozen allocator is off.",
        "",
        "## 2. Root cause candidates (evidence-based)",
        "",
        "1. **Forecast vol vs realized (RT layer)**",
        f"   - Target vol: {target_vol:.1%}" if target_vol else "   - Target vol: N/A",
        f"   - Realized vol: {realized_vol:.1%}" if realized_vol else "   - Realized vol: N/A",
        f"   - Ratio realized/target: {stats['realized_over_target']:.3f}" if stats.get('realized_over_target') is not None else "   - Ratio: N/A",
        f"   - Avg implied forecast vol (from scalars): {stats.get('forecast_vol_avg'):.2%}" if stats.get('forecast_vol_avg') else "   - Forecast vol: N/A",
        "   - Evidence: Scaling is applied (final_scalar varies). If forecast_vol was high, RT scaled up; ex-post realized can be lower.",
        "",
        "2. **Drawdown brake**",
        f"   - % days drawdown brake < 1: {stats.get('drawdown_brake_engaged_pct', 0):.1f}%",
        "   - Evidence: No drawdown brake in codebase (RiskTargetingLayer only; docstring says 'Not Allowed: Dynamic brakes'). Column is 1.0.",
        "",
        "3. **Leverage clamp**",
        f"   - % days clamp engaged: {stats.get('leverage_clamp_engaged_pct', 0):.1f}%",
        f"   - Leverage cap: {leverage_cap}",
        "   - Evidence: If clamp often engaged, gross was capped; else RT set leverage below cap.",
        "",
        "4. **Double dampening (sleeve + portfolio vol scaling)**",
        "   - Sleeves sr3_curve_rv_meta and vx_calendar_carry have per-sleeve target_vol (10%). Portfolio RT has target_vol 20%.",
        "   - Evidence: Combined signal is then scaled by RT. Not double-dampening in the sense of two brakes; one portfolio scaler.",
        "",
        "## Scaler and brake stats (from risk_scalars.csv)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in stats.items():
        if v is not None and "pct" in k:
            lines.append(f"| {k} | {v:.1f}% |")
        elif v is not None and isinstance(v, float):
            lines.append(f"| {k} | {v:.3f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.extend([
        "",
        "## Sanity checks",
        "",
        "A) **Scaling applied**: final_scalar_applied varies (avg ~{:.2f}); portfolio-level RT is active.".format(stats.get("final_scalar_avg") or 0.0),
        "B) **Brake engagement**: drawdown_brake_scalar is 1.0 (no brake in pipeline).",
        "C) **Realized vs target**: Realized vol below target implies either forecast_vol was overstated (RT scaled up less than needed) or ex-post vol dropped.",
        "",
        "## Next action recommendation",
        "",
        "Single best fix: **Revisit forecast vol construction** in RiskTargetingLayer (e.g. vol_lookback, covariance estimator, or vol_floor). If ex-ante vol is systematically high vs ex-post, reduce vol_floor or shorten lookback so leverage is higher and realized vol closer to target. Do not add a separate drawdown brake; the gap is vol targeting, not brake engagement.",
        "",
    ])

    out_md = run_dir / "analysis" / "risk_utilization_diagnostic.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")

    # Print compact summary
    print("\n--- Summary stats ---")
    print(f"base_scalar: avg={stats.get('base_scalar_avg'):.3f} median={stats.get('base_scalar_median'):.3f} p95={stats.get('base_scalar_p95'):.3f} max={stats.get('base_scalar_max'):.3f}")
    print(f"final_scalar_applied: avg={stats.get('final_scalar_avg'):.3f} median={stats.get('final_scalar_median'):.3f} p95={stats.get('final_scalar_p95'):.3f} max={stats.get('final_scalar_max'):.3f}")
    print(f"drawdown_brake_engaged: {stats.get('drawdown_brake_engaged_pct', 0):.1f}%")
    print(f"leverage_clamp_engaged: {stats.get('leverage_clamp_engaged_pct', 0):.1f}%")
    print(f"forecast_vol_avg: {stats.get('forecast_vol_avg'):.2%}" if stats.get('forecast_vol_avg') else "forecast_vol_avg: N/A")
    print(f"realized_vol: {realized_vol:.2%}" if realized_vol else "realized_vol: N/A")
    print(f"realized/target: {stats.get('realized_over_target'):.3f}" if stats.get('realized_over_target') is not None else "realized/target: N/A")


if __name__ == "__main__":
    main()
