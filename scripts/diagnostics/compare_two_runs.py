#!/usr/bin/env python3
"""
Compare Two Runs - Allocator Audit Report

Generates comparison metrics and report for two-pass allocator audit.
Compares baseline (no scaling) vs scaled (with risk scalar applied).

Usage:
    python scripts/diagnostics/compare_two_runs.py \
        --baseline_run_id baseline_2024 \
        --scaled_run_id allocpass2_2024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def read_series_csv(path: Path, col: str) -> pd.Series:
    """Read a time series from CSV with flexible column handling."""
    # First try reading with index_col=0 (date as index)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # If col is in columns, use it
    if col in df.columns:
        return df[col].copy()
    
    # If df has only one column, assume it's the value we want
    if len(df.columns) == 1:
        return df.iloc[:, 0].copy()
    
    # Otherwise, try reading without index_col (date as a column)
    df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else [])
    if col not in df.columns:
        # allow 2-col format: date,value
        value_cols = [c for c in df.columns if c != "date"]
        if len(value_cols) != 1:
            raise ValueError(f"{path} expected column '{col}' or single value col; got {df.columns.tolist()}")
        col = value_cols[0]
    s = df.set_index("date")[col].astype(float).sort_index()
    return s


def to_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Convert equity curve to returns series."""
    rets = equity.pct_change().dropna()
    return rets


def cagr(equity: pd.Series) -> float:
    """Compute compound annual growth rate."""
    if len(equity) < 2:
        return np.nan
    start, end = equity.index[0], equity.index[-1]
    years = (end - start).days / 365.25
    if years <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def ann_vol(returns: pd.Series) -> float:
    """Compute annualized volatility."""
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series) -> float:
    """Compute Sharpe ratio."""
    if returns.empty:
        return np.nan
    vol = returns.std(ddof=0)
    if vol == 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (vol * np.sqrt(TRADING_DAYS)))


def max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown."""
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def worst_month(returns: pd.Series) -> Tuple[pd.Timestamp, float]:
    """Find worst monthly return."""
    if returns.empty:
        return (pd.NaT, np.nan)
    m = returns.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
    idx = m.idxmin()
    return (idx, float(m.loc[idx]))


def worst_quarter(returns: pd.Series) -> Tuple[pd.Timestamp, float]:
    """Find worst quarterly return."""
    if returns.empty:
        return (pd.NaT, np.nan)
    q = returns.resample("Q").apply(lambda x: (1.0 + x).prod() - 1.0)
    idx = q.idxmin()
    return (idx, float(q.loc[idx]))


def load_scalar_meta(run_dir: Path) -> Dict:
    """Load scalar metadata from run directory."""
    # Try a few likely paths
    candidates = [
        run_dir / "allocator_precomputed_meta.json",
        run_dir / "allocator_risk_v1_applied_meta.json",
        run_dir / "allocator_risk_v1_meta.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    return {}


def load_used_scalars(run_dir: Path) -> pd.Series | None:
    """Load actually-used scalars from run directory."""
    p = run_dir / "allocator_risk_v1_applied_used.csv"
    if not p.exists():
        return None
    s = read_series_csv(p, col="risk_scalar_used")
    return s


def summarize_scalars(used: pd.Series | None) -> Dict:
    """Summarize scalar usage statistics."""
    if used is None or used.empty:
        return {"available": False}
    return {
        "available": True,
        "n_rebalances": int(len(used)),
        "pct_scaled": float((used < 0.999999).mean()),
        "mean": float(used.mean()),
        "min": float(used.min()),
        "max": float(used.max()),
        "p05": float(np.quantile(used.values, 0.05)),
        "p50": float(np.quantile(used.values, 0.50)),
        "p95": float(np.quantile(used.values, 0.95)),
    }


def compute_metrics(run_dir: Path) -> Dict:
    """Compute performance metrics for a single run."""
    equity_path = run_dir / "equity_curve.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Missing {equity_path}")

    equity = read_series_csv(equity_path, col="equity")
    rets = to_returns_from_equity(equity)

    wm_dt, wm_val = worst_month(rets)
    wq_dt, wq_val = worst_quarter(rets)

    metrics = {
        "cagr": cagr(equity),
        "ann_vol": ann_vol(rets),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(equity),
        "worst_month": {"date": None if pd.isna(wm_dt) else wm_dt.strftime("%Y-%m-%d"), "return": wm_val},
        "worst_quarter": {"date": None if pd.isna(wq_dt) else wq_dt.strftime("%Y-%m-%d"), "return": wq_val},
        "start": equity.index[0].strftime("%Y-%m-%d"),
        "end": equity.index[-1].strftime("%Y-%m-%d"),
        "n_days": int(len(equity)),
    }

    scalars_used = load_used_scalars(run_dir)
    metrics["scalars"] = summarize_scalars(scalars_used)
    metrics["scalar_meta"] = load_scalar_meta(run_dir)

    return metrics


def diff_metrics(base: Dict, scaled: Dict) -> Dict:
    """Compute delta metrics between baseline and scaled runs."""
    def d(a, b):
        if a is None or b is None:
            return None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(b - a)
        return None

    return {
        "cagr_delta": d(base["cagr"], scaled["cagr"]),
        "ann_vol_delta": d(base["ann_vol"], scaled["ann_vol"]),
        "sharpe_delta": d(base["sharpe"], scaled["sharpe"]),
        "max_drawdown_delta": d(base["max_drawdown"], scaled["max_drawdown"]),
        "worst_month_delta": d(base["worst_month"]["return"], scaled["worst_month"]["return"]),
        "worst_quarter_delta": d(base["worst_quarter"]["return"], scaled["worst_quarter"]["return"]),
    }


def render_md(base_id: str, scaled_id: str, base: Dict, scaled: Dict, delta: Dict) -> str:
    """Render Markdown comparison report."""
    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x:.4f}"

    lines = []
    lines.append(f"# Two-Pass Allocator Comparison\n")
    lines.append(f"Baseline run: `{base_id}`  \nScaled run: `{scaled_id}`\n")

    lines.append("## Summary Metrics\n")
    lines.append("| Metric | Baseline | Scaled | Δ (Scaled - Baseline) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| CAGR | {fmt(base['cagr'])} | {fmt(scaled['cagr'])} | {fmt(delta['cagr_delta'])} |")
    lines.append(f"| Ann Vol | {fmt(base['ann_vol'])} | {fmt(scaled['ann_vol'])} | {fmt(delta['ann_vol_delta'])} |")
    lines.append(f"| Sharpe | {fmt(base['sharpe'])} | {fmt(scaled['sharpe'])} | {fmt(delta['sharpe_delta'])} |")
    lines.append(f"| MaxDD | {fmt(base['max_drawdown'])} | {fmt(scaled['max_drawdown'])} | {fmt(delta['max_drawdown_delta'])} |")
    lines.append(f"| Worst Month | {fmt(base['worst_month']['return'])} ({base['worst_month']['date']}) | {fmt(scaled['worst_month']['return'])} ({scaled['worst_month']['date']}) | {fmt(delta['worst_month_delta'])} |")
    lines.append(f"| Worst Quarter | {fmt(base['worst_quarter']['return'])} ({base['worst_quarter']['date']}) | {fmt(scaled['worst_quarter']['return'])} ({scaled['worst_quarter']['date']}) | {fmt(delta['worst_quarter_delta'])} |")

    lines.append("\n## Scalar Usage (Scaled Run)\n")
    sc = scaled.get("scalars", {})
    if sc.get("available"):
        lines.append(f"- Rebalances: {sc['n_rebalances']}")
        lines.append(f"- % scaled: {sc['pct_scaled']*100:.1f}%")
        lines.append(f"- Mean scalar: {sc['mean']:.4f}")
        lines.append(f"- Min scalar: {sc['min']:.4f}")
        lines.append(f"- Max scalar: {sc['max']:.4f}")
        lines.append(f"- P05/P50/P95: {sc['p05']:.4f} / {sc['p50']:.4f} / {sc['p95']:.4f}")
    else:
        lines.append("- Scalars not available (missing `allocator_risk_v1_applied_used.csv`).")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Generate comparison report for two-pass allocator audit"
    )
    ap.add_argument("--baseline_run_id", required=True, help="Baseline run ID (no scaling)")
    ap.add_argument("--scaled_run_id", required=True, help="Scaled run ID (with risk scalars)")
    ap.add_argument("--runs_root", default="reports/runs", help="Root directory for run artifacts")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    base_dir = runs_root / args.baseline_run_id
    scaled_dir = runs_root / args.scaled_run_id

    print(f"Computing metrics for baseline: {base_dir}")
    base_metrics = compute_metrics(base_dir)
    
    print(f"Computing metrics for scaled: {scaled_dir}")
    scaled_metrics = compute_metrics(scaled_dir)
    
    print("Computing deltas...")
    deltas = diff_metrics(base_metrics, scaled_metrics)

    out = {
        "baseline_run_id": args.baseline_run_id,
        "scaled_run_id": args.scaled_run_id,
        "baseline": base_metrics,
        "scaled": scaled_metrics,
        "delta": deltas,
    }

    out_json = scaled_dir / "two_pass_comparison.json"
    out_md = scaled_dir / "two_pass_comparison.md"

    out_json.write_text(json.dumps(out, indent=2), encoding='utf-8')
    out_md.write_text(render_md(args.baseline_run_id, args.scaled_run_id, base_metrics, scaled_metrics, deltas), encoding='utf-8')

    print(f"\n[SUCCESS] Wrote comparison reports:\n  - {out_json}\n  - {out_md}")


if __name__ == "__main__":
    main()

