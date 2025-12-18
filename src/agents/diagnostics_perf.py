"""
Performance Diagnostics Module

Lightweight performance diagnostics layer for backtest runs.

Provides:
- RunData container for loading run artifacts
- Core performance metrics (CAGR, vol, Sharpe, MaxDD, hit rate)
- Year-by-year performance breakdown
- Per-asset attribution summary
- Baseline comparison functionality
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os
import json
import pandas as pd
import numpy as np


@dataclass
class RunData:
    """Container for backtest run data."""
    run_id: str
    portfolio_returns: pd.Series        # index=DatetimeIndex, daily returns
    equity_curve: pd.Series             # index=DatetimeIndex, cumulative equity
    asset_returns: pd.DataFrame         # index=DatetimeIndex, columns=symbols, daily returns
    weights: pd.DataFrame               # index=DatetimeIndex (rebalance dates), columns=symbols
    meta: Dict


def load_run(run_id: str, base_dir: str = "reports/runs") -> RunData:
    """
    Load run artifacts from reports/runs/{run_id}/.
    
    Args:
        run_id: Run identifier
        base_dir: Base directory containing run folders
        
    Returns:
        RunData object with loaded data
        
    Raises:
        FileNotFoundError: If run directory or required files don't exist
    """
    run_dir = os.path.join(base_dir, run_id)
    
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load portfolio returns
    portfolio_returns_path = os.path.join(run_dir, "portfolio_returns.csv")
    if not os.path.exists(portfolio_returns_path):
        raise FileNotFoundError(f"portfolio_returns.csv not found in {run_dir}")
    
    portfolio_returns = pd.read_csv(
        portfolio_returns_path,
        parse_dates=["date"],
        index_col="date"
    )["ret"]
    
    # Load equity curve
    equity_curve_path = os.path.join(run_dir, "equity_curve.csv")
    if not os.path.exists(equity_curve_path):
        raise FileNotFoundError(f"equity_curve.csv not found in {run_dir}")
    
    equity_curve = pd.read_csv(
        equity_curve_path,
        parse_dates=["date"],
        index_col="date"
    )["equity"]
    
    # Load asset returns
    asset_returns_path = os.path.join(run_dir, "asset_returns.csv")
    if not os.path.exists(asset_returns_path):
        raise FileNotFoundError(f"asset_returns.csv not found in {run_dir}")
    
    asset_returns = pd.read_csv(
        asset_returns_path,
        index_col=0,
        parse_dates=True
    )
    
    # Load weights
    weights_path = os.path.join(run_dir, "weights.csv")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights.csv not found in {run_dir}")
    
    weights = pd.read_csv(
        weights_path,
        index_col=0,
        parse_dates=True
    )
    
    # Load meta
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {run_dir}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    return RunData(
        run_id=run_id,
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        asset_returns=asset_returns,
        weights=weights,
        meta=meta
    )


def compute_core_metrics(run: RunData) -> Dict[str, float]:
    """
    Compute core performance metrics for the given run.
    
    Assumes portfolio_returns is a daily return series (simple returns).
    
    Returns:
        Dict with keys: CAGR, Vol, Sharpe, MaxDD, HitRate
    """
    r = run.portfolio_returns
    
    if len(r) == 0:
        return {
            "CAGR": float('nan'),
            "Vol": float('nan'),
            "Sharpe": float('nan'),
            "MaxDD": float('nan'),
            "HitRate": float('nan')
        }
    
    # Number of trading days and years
    n_days = len(r)
    years = n_days / 252.0
    
    # CAGR
    equity = run.equity_curve
    if len(equity) < 2:
        cagr = float('nan')
    else:
        equity_start = equity.iloc[0]
        equity_end = equity.iloc[-1]
        if equity_start > 0 and years > 0:
            cagr = (equity_end / equity_start) ** (1 / years) - 1
        else:
            cagr = float('nan')
    
    # Annualized volatility
    vol = r.std() * (252 ** 0.5)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    if r.std() != 0:
        sharpe = r.mean() / r.std() * (252 ** 0.5)
    else:
        sharpe = float('nan')
    
    # Max drawdown
    if len(equity) < 2:
        max_dd = float('nan')
    else:
        running_max = equity.cummax()
        dd = (equity / running_max) - 1.0
        max_dd = dd.min()
    
    # Hit rate
    hit_rate = (r > 0).sum() / len(r) if len(r) > 0 else float('nan')
    
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "HitRate": hit_rate
    }


def compute_yearly_stats(run: RunData) -> pd.DataFrame:
    """
    Compute year-by-year performance stats.
    
    Returns:
        DataFrame with index=year (int), columns=['CAGR', 'Vol', 'Sharpe', 'MaxDD']
    """
    r = run.portfolio_returns
    equity = run.equity_curve
    
    if len(r) == 0 or len(equity) < 2:
        return pd.DataFrame(columns=['CAGR', 'Vol', 'Sharpe', 'MaxDD'])
    
    years = sorted(set(r.index.year))
    rows = []
    
    for y in years:
        mask = (r.index.year == y)
        r_y = r[mask]
        
        if len(r_y) == 0:
            continue
        
        # Equity for that year
        eq_y = equity[(equity.index.year == y)]
        
        if len(eq_y) < 2:
            continue
        
        # Calculate metrics for this year
        n_days = len(r_y)
        years_len = n_days / 252.0
        
        # CAGR for year
        if years_len > 0 and eq_y.iloc[0] > 0:
            cagr_y = (eq_y.iloc[-1] / eq_y.iloc[0]) ** (1 / years_len) - 1
        else:
            cagr_y = float('nan')
        
        # Volatility
        vol_y = r_y.std() * (252 ** 0.5)
        
        # Sharpe
        if r_y.std() != 0:
            sharpe_y = r_y.mean() / r_y.std() * (252 ** 0.5)
        else:
            sharpe_y = float('nan')
        
        # Max drawdown
        running_max_y = eq_y.cummax()
        dd_y = (eq_y / running_max_y) - 1.0
        maxdd_y = dd_y.min()
        
        rows.append({
            "Year": y,
            "CAGR": cagr_y,
            "Vol": vol_y,
            "Sharpe": sharpe_y,
            "MaxDD": maxdd_y
        })
    
    if len(rows) == 0:
        return pd.DataFrame(columns=['CAGR', 'Vol', 'Sharpe', 'MaxDD'])
    
    df = pd.DataFrame(rows).set_index("Year")
    return df


def compute_per_asset_stats(run: RunData) -> pd.DataFrame:
    """
    Compute per-asset attribution.
    
    For each symbol:
    - 'AnnRet': annualized return contribution proxy
    - 'AnnVol': annualized volatility of that asset's PnL contribution
    - 'Sharpe': AnnRet / AnnVol
    
    Returns:
        DataFrame with index=symbol, columns=['AnnRet', 'AnnVol', 'Sharpe']
    """
    # Align weights to daily index of asset_returns (forward-fill between rebalances)
    # Reindex weights to match asset_returns index, then forward-fill
    w = run.weights.reindex(run.asset_returns.index).ffill().fillna(0.0)
    
    # Compute per-asset daily PnL (approximate)
    asset_pnl = w * run.asset_returns
    
    # Compute stats per asset
    rows = []
    
    for sym in asset_pnl.columns:
        pnl_series = asset_pnl[sym].dropna()
        
        if len(pnl_series) == 0:
            continue
        
        # Annualized return contribution
        daily_mean = pnl_series.mean()
        ann_ret = daily_mean * 252
        
        # Annualized vol
        ann_vol = pnl_series.std() * (252 ** 0.5)
        
        # Sharpe contribution
        if ann_vol != 0:
            sharpe = ann_ret / ann_vol
        else:
            sharpe = float('nan')
        
        rows.append({
            "symbol": sym,
            "AnnRet": ann_ret,
            "AnnVol": ann_vol,
            "Sharpe": sharpe
        })
    
    if len(rows) == 0:
        return pd.DataFrame(columns=['AnnRet', 'AnnVol', 'Sharpe'])
    
    df = pd.DataFrame(rows).set_index("symbol")
    return df


def validate_comparison(current: RunData, baseline: RunData) -> None:
    """
    Validate that two runs can be compared fairly per Run Consistency Contract.
    
    Checks:
    - Row counts should match (within 5%)
    - Date indices should have substantial overlap (>= 95%)
    
    Raises:
        ValueError: If runs cannot be compared fairly
        
    See: docs/SOTs/PROCEDURES.md § 2 "Run Consistency Contract"
    """
    import logging
    logger = logging.getLogger(__name__)
    
    eq_curr = current.equity_curve
    eq_base = baseline.equity_curve
    
    # Check row counts
    n_curr = len(eq_curr)
    n_base = len(eq_base)
    
    logger.info("=" * 80)
    logger.info("RUN COMPARISON VALIDATION (Run Consistency Contract)")
    logger.info("=" * 80)
    logger.info(f"Current run:  {current.run_id}")
    logger.info(f"  Rows:       {n_curr}")
    logger.info(f"  Start:      {eq_curr.index[0]}")
    logger.info(f"  End:        {eq_curr.index[-1]}")
    logger.info(f"Baseline run: {baseline.run_id}")
    logger.info(f"  Rows:       {n_base}")
    logger.info(f"  Start:      {eq_base.index[0]}")
    logger.info(f"  End:        {eq_base.index[-1]}")
    
    if n_curr != n_base:
        row_diff_pct = abs(n_curr - n_base) / max(n_curr, n_base)
        logger.warning(f"⚠️  ROW COUNT MISMATCH: {n_curr} (current) vs {n_base} (baseline)")
        logger.warning(f"   Difference: {abs(n_curr - n_base)} rows ({row_diff_pct*100:.1f}%)")
        
        if row_diff_pct > 0.05:
            logger.error("❌ More than 5% row count difference — comparison invalid.")
            raise ValueError(
                f"Row count mismatch too large: {n_curr} vs {n_base} "
                f"({row_diff_pct*100:.1f}% difference). "
                "Cannot compare runs with different sample sizes. "
                "Ensure both runs use the same date range and alignment."
            )
    
    # Check date overlap
    idx_overlap = eq_curr.index.intersection(eq_base.index)
    overlap_ratio = len(idx_overlap) / max(n_curr, n_base)
    
    logger.info(f"Date overlap: {len(idx_overlap)} days ({overlap_ratio*100:.1f}%)")
    
    if not eq_curr.index.equals(eq_base.index):
        logger.warning("⚠️  Equity curves do not share identical indices — checking overlap.")
        
        if overlap_ratio < 0.95:
            logger.error(f"❌ Less than 95% overlap — cannot compare runs ({overlap_ratio*100:.1f}% overlap).")
            raise ValueError(
                f"Insufficient date overlap: {overlap_ratio*100:.1f}%. "
                "Cannot compare runs with different date ranges. "
                "Ensure both runs use the same start and end dates."
            )
        else:
            logger.warning(f"   Sufficient overlap ({overlap_ratio*100:.1f}%) — proceeding with comparison.")
    else:
        logger.info("✓ Runs have identical date indices — comparison valid.")
    
    logger.info("=" * 80)


def compare_to_baseline(current: RunData, baseline: RunData) -> Dict:
    """
    Compare current run to a baseline.
    
    Returns:
        Dict with:
        - 'metrics_current': core metrics for current run
        - 'metrics_baseline': core metrics for baseline run
        - 'metrics_delta': delta (current - baseline) for each metric
        - 'equity_ratio': pd.Series of equity_curr / equity_base over overlapping period
    """
    # Validate comparison per Run Consistency Contract
    validate_comparison(current, baseline)
    
    # Compute core metrics for both
    metrics_curr = compute_core_metrics(current)
    metrics_base = compute_core_metrics(baseline)
    
    # Compute deltas
    delta = {}
    for key in metrics_curr.keys():
        if key in metrics_base:
            curr_val = metrics_curr[key]
            base_val = metrics_base[key]
            if not (np.isnan(curr_val) or np.isnan(base_val)):
                delta[f"{key}_delta"] = curr_val - base_val
            else:
                delta[f"{key}_delta"] = float('nan')
    
    # Equity ratio (overlap period only)
    eq_curr = current.equity_curve
    eq_base = baseline.equity_curve
    
    idx = eq_curr.index.intersection(eq_base.index)
    
    if len(idx) > 0:
        # Normalize both to start at 1.0 for fair comparison
        eq_curr_norm = eq_curr.loc[idx] / eq_curr.loc[idx[0]]
        eq_base_norm = eq_base.loc[idx] / eq_base.loc[idx[0]]
        equity_ratio = eq_curr_norm / eq_base_norm
    else:
        equity_ratio = pd.Series(dtype=float)
    
    return {
        "metrics_current": metrics_curr,
        "metrics_baseline": metrics_base,
        "metrics_delta": delta,
        "equity_ratio": equity_ratio
    }

