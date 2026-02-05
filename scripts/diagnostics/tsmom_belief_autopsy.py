#!/usr/bin/env python3
"""
TSMOM Belief Autopsy — Phase 4 Engine Research Step 1

This script performs a comprehensive diagnostic of the current TSMOM engine
(Trend Meta-Sleeve) at Post-Construction. NO CODE CHANGES are made.

Research Protocol Compliance:
- Step 1: Belief Autopsy (No Code Changes)
- Baseline: phase4_tsmom_baseline_v1
- Purpose: Diagnose current promoted Trend Meta-Sleeve failure modes

Key Evaluation Questions (from DIAGNOSTICS.md & SYSTEM_CONSTRUCTION.md):
1. Is unconditional Sharpe ≥ ~0.4–0.5?
2. Where does it fail?
   - Short horizon?
   - Trend reversals?
   - Specific assets (bonds vs commodities vs FX)?
3. Is failure structural (bad economics) or interaction (horizon contamination, 
   crash behavior, whipsaw)?

Deliverable:
A diagnostic memo documenting observed structural failure modes of current TSMOM.

Author: Phase 4 Engine Research
Date: January 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# =============================================================================
# STEP 1: FREEZE THE BASELINE (Non-Negotiable)
# =============================================================================

# Default baseline run ID (to be updated after running baseline)
BASELINE_RUN_ID = "phase4_tsmom_baseline_v1"  # Will be updated with timestamp

# Canonical evaluation window (same as CSMOM Phase 4)
EVAL_START = "2020-03-20"
EVAL_END = "2025-10-31"

# Report paths
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "runs"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports" / "phase4_research" / "tsmom_autopsy"


def find_latest_tsmom_baseline() -> Optional[str]:
    """Find the latest TSMOM baseline run in reports directory."""
    if not REPORTS_DIR.exists():
        return None
    
    # Look for runs matching phase4_tsmom_baseline pattern
    candidates = []
    for run_dir in REPORTS_DIR.iterdir():
        if run_dir.is_dir() and "tsmom_baseline" in run_dir.name.lower():
            candidates.append(run_dir.name)
    
    if not candidates:
        return None
    
    # Sort by timestamp (assuming format includes timestamp)
    candidates.sort(reverse=True)
    return candidates[0]


def load_run_artifacts(run_id: str) -> dict:
    """Load all relevant artifacts from a run."""
    run_dir = REPORTS_DIR / run_id
    
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return {}
    
    artifacts = {}
    
    # Load sleeve returns
    sleeve_returns_path = run_dir / "sleeve_returns.csv"
    if sleeve_returns_path.exists():
        artifacts["sleeve_returns"] = pd.read_csv(sleeve_returns_path, index_col=0, parse_dates=True)
    
    # Load weights post-construction
    weights_path = run_dir / "weights_post_construction.csv"
    if weights_path.exists():
        artifacts["weights_post_construction"] = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    
    # Load asset returns
    asset_returns_path = run_dir / "asset_returns.csv"
    if asset_returns_path.exists():
        artifacts["asset_returns"] = pd.read_csv(asset_returns_path, index_col=0, parse_dates=True)
    
    # Load portfolio returns
    portfolio_returns_path = run_dir / "portfolio_returns.csv"
    if portfolio_returns_path.exists():
        artifacts["portfolio_returns"] = pd.read_csv(portfolio_returns_path, index_col=0, parse_dates=True)
    
    # Load equity curve
    equity_path = run_dir / "equity_curve.csv"
    if equity_path.exists():
        artifacts["equity_curve"] = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    
    # Load meta.json
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding='utf-8') as f:
            artifacts["meta"] = json.load(f)
    
    # Load canonical diagnostics
    diag_path = run_dir / "canonical_diagnostics.json"
    if diag_path.exists():
        with open(diag_path, encoding='utf-8') as f:
            artifacts["canonical_diagnostics"] = json.load(f)
    
    # Load engine attribution
    attr_path = run_dir / "engine_attribution_post_construction.json"
    if attr_path.exists():
        with open(attr_path, encoding='utf-8') as f:
            artifacts["engine_attribution"] = json.load(f)
    
    return artifacts


def compute_standalone_metrics(sleeve_returns: pd.DataFrame, eval_start: str) -> dict:
    """Compute standalone TSMOM metrics."""
    # For TSMOM baseline, use tsmom_multihorizon or portfolio returns
    if "tsmom_multihorizon" in sleeve_returns.columns:
        tsmom_rets = sleeve_returns["tsmom_multihorizon"].loc[eval_start:]
    elif "portfolio" in sleeve_returns.columns:
        tsmom_rets = sleeve_returns["portfolio"].loc[eval_start:]
    else:
        # Use first numeric column
        tsmom_rets = sleeve_returns.iloc[:, 0].loc[eval_start:]
    
    # Drop NaN and zero returns during warmup
    tsmom_rets = tsmom_rets.dropna()
    tsmom_rets = tsmom_rets[tsmom_rets != 0]
    
    if len(tsmom_rets) == 0:
        return {"error": "No valid TSMOM returns"}
    
    # Basic metrics
    ann_ret = tsmom_rets.mean() * 252
    ann_vol = tsmom_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Cumulative returns
    cum_ret = (1 + tsmom_rets).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    
    # CAGR calculation
    years = (tsmom_rets.index[-1] - tsmom_rets.index[0]).days / 365.25
    cagr = (cum_ret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    
    # Drawdown analysis
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Time under water
    underwater = drawdown < 0
    underwater_periods = []
    in_dd = False
    dd_start = None
    
    for date, is_uw in underwater.items():
        if is_uw and not in_dd:
            in_dd = True
            dd_start = date
        elif not is_uw and in_dd:
            in_dd = False
            underwater_periods.append((dd_start, date))
    
    if in_dd:
        underwater_periods.append((dd_start, underwater.index[-1]))
    
    # Longest underwater period
    if underwater_periods:
        longest_uw = max((end - start).days for start, end in underwater_periods)
    else:
        longest_uw = 0
    
    # Hit rate
    hit_rate = (tsmom_rets > 0).mean()
    
    # Skewness and kurtosis
    skew = tsmom_rets.skew()
    kurt = tsmom_rets.kurtosis()
    
    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "n_days": len(tsmom_rets),
        "annualized_return": ann_ret,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "calmar": calmar,
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "longest_underwater_days": longest_uw,
        "hit_rate": hit_rate,
        "skewness": skew,
        "kurtosis": kurt,
        "start_date": str(tsmom_rets.index[0].date()),
        "end_date": str(tsmom_rets.index[-1].date()),
    }


def compute_yearly_returns(returns: pd.DataFrame, eval_start: str) -> pd.DataFrame:
    """Compute yearly returns table for TSMOM."""
    # For TSMOM baseline, use tsmom_multihorizon or portfolio
    if "tsmom_multihorizon" in returns.columns:
        tsmom_rets = returns["tsmom_multihorizon"].loc[eval_start:]
    elif "portfolio" in returns.columns:
        tsmom_rets = returns["portfolio"].loc[eval_start:]
    else:
        tsmom_rets = returns.iloc[:, 0].loc[eval_start:]
    
    tsmom_rets = tsmom_rets.dropna()
    
    # Group by year
    yearly = tsmom_rets.groupby(tsmom_rets.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Also compute yearly Sharpe
    yearly_sharpe = tsmom_rets.groupby(tsmom_rets.index.year).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    
    # Yearly vol
    yearly_vol = tsmom_rets.groupby(tsmom_rets.index.year).apply(
        lambda x: x.std() * np.sqrt(252)
    )
    
    # Yearly max drawdown
    def compute_max_dd(x):
        cum = (1 + x).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        return dd.min()
    
    yearly_maxdd = tsmom_rets.groupby(tsmom_rets.index.year).apply(compute_max_dd)
    
    result = pd.DataFrame({
        "Return": yearly,
        "Vol": yearly_vol,
        "Sharpe": yearly_sharpe,
        "MaxDD": yearly_maxdd
    })
    
    return result


def analyze_asset_contribution(weights: pd.DataFrame, asset_returns: pd.DataFrame, eval_start: str) -> dict:
    """Analyze per-asset contribution to TSMOM."""
    # Filter to eval window
    weights_eval = weights.loc[eval_start:]
    returns_eval = asset_returns.loc[eval_start:]
    
    # Align indices
    common_idx = weights_eval.index.intersection(returns_eval.index)
    
    if len(common_idx) == 0:
        return {"error": "No overlapping dates between weights and returns"}
    
    weights_eval = weights_eval.loc[common_idx]
    returns_eval = returns_eval.loc[common_idx]
    
    # Align columns
    common_cols = weights_eval.columns.intersection(returns_eval.columns)
    weights_eval = weights_eval[common_cols]
    returns_eval = returns_eval[common_cols]
    
    # Compute contribution = weight * return for each asset
    contribution = weights_eval * returns_eval
    
    # Total contribution by asset
    total_contrib = contribution.sum()
    
    # Contribution Sharpe by asset
    contrib_sharpe = {}
    for col in contribution.columns:
        c = contribution[col].dropna()
        if len(c) > 0 and c.std() > 0:
            contrib_sharpe[col] = (c.mean() * 252) / (c.std() * np.sqrt(252))
        else:
            contrib_sharpe[col] = 0
    
    # Average absolute weight by asset
    avg_abs_weight = weights_eval.abs().mean()
    
    # Classify by asset class (simplified)
    asset_classes = {
        "equity_index": ["ES_", "NQ_", "RTY_", "YM_"],
        "rates": ["ZN_", "ZT_", "ZF_", "ZB_", "UB_", "SR3_"],
        "fx": ["6E_", "6B_", "6J_", "6A_", "6C_"],
        "commodities": ["CL_", "GC_", "SI_", "HG_", "NG_"],
    }
    
    class_contrib = {}
    for class_name, prefixes in asset_classes.items():
        class_assets = [col for col in contribution.columns 
                       if any(col.startswith(p) for p in prefixes)]
        if class_assets:
            class_contrib[class_name] = {
                "total_contribution": contribution[class_assets].sum().sum(),
                "n_assets": len(class_assets),
                "assets": class_assets
            }
    
    return {
        "total_contribution": total_contrib.to_dict(),
        "contribution_sharpe": contrib_sharpe,
        "avg_abs_weight": avg_abs_weight.to_dict(),
        "asset_class_contribution": class_contrib,
    }


def analyze_signal_stability(weights: pd.DataFrame, eval_start: str) -> dict:
    """Analyze signal stability and turnover."""
    weights_eval = weights.loc[eval_start:]
    
    if len(weights_eval) < 5:
        return {"error": "Insufficient data for stability analysis"}
    
    # Daily turnover (sum of absolute weight changes)
    weight_changes = weights_eval.diff().abs()
    daily_turnover = weight_changes.sum(axis=1)
    
    # Average daily turnover
    avg_turnover = daily_turnover.mean()
    
    # Weekly turnover (every Friday)
    weekly_idx = weights_eval.index[weights_eval.index.dayofweek == 4]  # Friday
    if len(weekly_idx) > 1:
        weekly_weights = weights_eval.loc[weekly_idx]
        weekly_changes = weekly_weights.diff().abs()
        weekly_turnover = weekly_changes.sum(axis=1)
        avg_weekly_turnover = weekly_turnover.mean()
    else:
        avg_weekly_turnover = np.nan
    
    # Signal persistence (autocorrelation of weights)
    autocorr = {}
    for col in weights_eval.columns:
        w = weights_eval[col].dropna()
        if len(w) > 5:
            autocorr[col] = w.autocorr(lag=5)  # 5-day autocorr
        else:
            autocorr[col] = np.nan
    
    avg_autocorr = np.nanmean(list(autocorr.values()))
    
    # Sign changes
    sign_changes = (weights_eval.shift(1) * weights_eval < 0).sum(axis=1)
    avg_sign_changes = sign_changes.mean()
    
    # Weight concentration (Herfindahl)
    weight_sq = weights_eval.abs() ** 2
    hhi = weight_sq.sum(axis=1) / (weights_eval.abs().sum(axis=1) ** 2)
    avg_hhi = hhi.mean()
    
    return {
        "avg_daily_turnover": avg_turnover,
        "avg_weekly_turnover": avg_weekly_turnover,
        "avg_5d_autocorrelation": avg_autocorr,
        "avg_daily_sign_changes": avg_sign_changes,
        "avg_weight_concentration_hhi": avg_hhi,
    }


def analyze_regime_behavior(returns: pd.DataFrame, eval_start: str) -> dict:
    """Analyze TSMOM behavior across market regimes."""
    # For TSMOM baseline
    if "tsmom_multihorizon" in returns.columns:
        tsmom_rets = returns["tsmom_multihorizon"].loc[eval_start:]
    elif "portfolio" in returns.columns:
        tsmom_rets = returns["portfolio"].loc[eval_start:]
    else:
        tsmom_rets = returns.iloc[:, 0].loc[eval_start:]
    
    tsmom_rets = tsmom_rets.dropna()
    
    if len(tsmom_rets) < 63:
        return {"error": "Insufficient data for regime analysis"}
    
    # Use rolling volatility as regime proxy
    rolling_vol = tsmom_rets.rolling(21).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    
    # Define regimes based on vol percentiles
    vol_25 = rolling_vol.quantile(0.25)
    vol_75 = rolling_vol.quantile(0.75)
    
    low_vol_mask = rolling_vol < vol_25
    high_vol_mask = rolling_vol > vol_75
    mid_vol_mask = ~low_vol_mask & ~high_vol_mask
    
    regimes = {}
    for name, mask in [("low_vol", low_vol_mask), ("mid_vol", mid_vol_mask), ("high_vol", high_vol_mask)]:
        # Align mask with returns
        aligned_mask = mask.reindex(tsmom_rets.index).fillna(False)
        regime_rets = tsmom_rets[aligned_mask]
        if len(regime_rets) > 20:
            ann_ret = regime_rets.mean() * 252
            ann_vol = regime_rets.std() * np.sqrt(252)
            regimes[name] = {
                "n_days": len(regime_rets),
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
                "hit_rate": (regime_rets > 0).mean()
            }
    
    # Crisis periods (specific known dates)
    crisis_periods = {
        "covid_crash_2020": ("2020-02-20", "2020-04-30"),
        "rates_shock_2022": ("2022-01-01", "2022-06-30"),
        "banking_stress_2023": ("2023-03-01", "2023-04-30"),
    }
    
    crisis_metrics = {}
    for crisis_name, (start, end) in crisis_periods.items():
        try:
            crisis_rets = tsmom_rets.loc[start:end]
            if len(crisis_rets) > 5:
                total_ret = (1 + crisis_rets).prod() - 1
                crisis_metrics[crisis_name] = {
                    "n_days": len(crisis_rets),
                    "total_return": total_ret,
                    "sharpe": (crisis_rets.mean() * 252) / (crisis_rets.std() * np.sqrt(252)) if crisis_rets.std() > 0 else 0,
                }
        except Exception:
            pass
    
    regimes["crisis_periods"] = crisis_metrics
    
    return regimes


def analyze_horizon_attribution(artifacts: dict, eval_start: str) -> dict:
    """Analyze contribution by horizon (if horizon-level data available)."""
    # This would require horizon-level return attribution
    # For now, return placeholder indicating need for additional instrumentation
    return {
        "note": "Horizon-level attribution requires additional instrumentation in feature service",
        "recommended_analysis": [
            "Long-term (252d) contribution",
            "Medium-term (84d) contribution", 
            "Short-term (21d) contribution",
            "Horizon correlation matrix",
            "Horizon-conditioned regime performance"
        ]
    }


def analyze_rolling_stability(returns: pd.DataFrame, eval_start: str) -> dict:
    """Analyze rolling stability metrics (6m/12m)."""
    if "tsmom_multihorizon" in returns.columns:
        tsmom_rets = returns["tsmom_multihorizon"].loc[eval_start:]
    elif "portfolio" in returns.columns:
        tsmom_rets = returns["portfolio"].loc[eval_start:]
    else:
        tsmom_rets = returns.iloc[:, 0].loc[eval_start:]
    
    tsmom_rets = tsmom_rets.dropna()
    
    if len(tsmom_rets) < 252:
        return {"error": "Insufficient data for rolling stability analysis"}
    
    # Rolling 6-month (126 trading days) Sharpe
    rolling_6m_sharpe = tsmom_rets.rolling(126).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    
    # Rolling 12-month (252 trading days) Sharpe
    rolling_12m_sharpe = tsmom_rets.rolling(252).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    
    # Statistics on rolling metrics
    return {
        "rolling_6m_sharpe": {
            "mean": rolling_6m_sharpe.mean(),
            "std": rolling_6m_sharpe.std(),
            "min": rolling_6m_sharpe.min(),
            "max": rolling_6m_sharpe.max(),
            "pct_positive": (rolling_6m_sharpe > 0).mean(),
        },
        "rolling_12m_sharpe": {
            "mean": rolling_12m_sharpe.mean(),
            "std": rolling_12m_sharpe.std(),
            "min": rolling_12m_sharpe.min(),
            "max": rolling_12m_sharpe.max(),
            "pct_positive": (rolling_12m_sharpe > 0).mean(),
        }
    }


def generate_memo(baseline_run_id: str, metrics: dict) -> str:
    """Generate the diagnostic memo."""
    memo = f"""# TSMOM Belief Autopsy — Structural Failure Mode Analysis

**Baseline Run ID:** `{baseline_run_id}`
**Generated:** {datetime.now().isoformat()}
**Research Protocol:** Phase 4 Engine Research, Step 1 (No Code Changes)

---

## Executive Summary

This memo documents the observed structural failure modes of the current TSMOM engine
(Trend Meta-Sleeve) at Post-Construction. NO hypotheses or fixes are proposed — this is 
pure measurement.

**Key Questions Being Answered:**
1. Is unconditional Sharpe ≥ ~0.4–0.5? (academic TSMOM benchmark)
2. Where does it fail? (horizon, asset class, regime)
3. Is failure structural or interaction-based?

---

## 1. Standalone TSMOM Metrics (Eval Window: {EVAL_START} to {EVAL_END})

| Metric | Value |
|--------|-------|
| N Days | {metrics['standalone']['n_days']} |
| CAGR | {metrics['standalone']['cagr']:.2%} |
| Annualized Return | {metrics['standalone']['annualized_return']:.2%} |
| Annualized Vol | {metrics['standalone']['annualized_vol']:.2%} |
| **Sharpe Ratio** | **{metrics['standalone']['sharpe']:.3f}** |
| Calmar Ratio | {metrics['standalone']['calmar']:.3f} |
| Total Return | {metrics['standalone']['total_return']:.2%} |
| Max Drawdown | {metrics['standalone']['max_drawdown']:.2%} |
| Longest Underwater | {metrics['standalone']['longest_underwater_days']} days |
| Hit Rate | {metrics['standalone']['hit_rate']:.2%} |
| Skewness | {metrics['standalone']['skewness']:.3f} |
| Kurtosis | {metrics['standalone']['kurtosis']:.3f} |

**Unconditional Sharpe Assessment:** 
- Target: ≥0.40–0.50 (academic TSMOM benchmark)
- Actual: {metrics['standalone']['sharpe']:.3f}
- Status: {"✅ MEETS BENCHMARK" if metrics['standalone']['sharpe'] >= 0.40 else "⚠️ BELOW BENCHMARK" if metrics['standalone']['sharpe'] > 0 else "❌ NEGATIVE SHARPE"}

---

## 2. Yearly Returns Table

| Year | Return | Vol | Sharpe | MaxDD |
|------|--------|-----|--------|-------|
"""
    
    for year, row in metrics['yearly'].iterrows():
        memo += f"| {year} | {row['Return']:.2%} | {row['Vol']:.2%} | {row['Sharpe']:.3f} | {row['MaxDD']:.2%} |\n"
    
    memo += f"""
**Year-by-Year Analysis:**
- Positive years: {(metrics['yearly']['Return'] > 0).sum()}
- Negative years: {(metrics['yearly']['Return'] < 0).sum()}
- Best year: {metrics['yearly']['Return'].idxmax()} ({metrics['yearly']['Return'].max():.2%})
- Worst year: {metrics['yearly']['Return'].idxmin()} ({metrics['yearly']['Return'].min():.2%})

---

## 3. Signal Stability & Turnover

| Metric | Value |
|--------|-------|
| Avg Daily Turnover | {metrics['stability']['avg_daily_turnover']:.4f} |
| Avg Weekly Turnover | {metrics['stability']['avg_weekly_turnover']:.4f} |
| Avg 5-Day Autocorrelation | {metrics['stability']['avg_5d_autocorrelation']:.3f} |
| Avg Daily Sign Changes | {metrics['stability']['avg_daily_sign_changes']:.2f} |
| Avg Weight Concentration (HHI) | {metrics['stability']['avg_weight_concentration_hhi']:.3f} |

**Observation:** {"High turnover suggests signal instability (whipsaw risk)." if metrics['stability']['avg_weekly_turnover'] > 0.5 else "Turnover is moderate."}
Signal autocorrelation of {metrics['stability']['avg_5d_autocorrelation']:.3f} indicates 
{"persistent signals (good for trend-following)" if metrics['stability']['avg_5d_autocorrelation'] > 0.7 else "moderate signal persistence (may indicate horizon mismatch)"}.

---

## 4. Regime-Conditioned Performance

"""
    
    for regime, stats in metrics['regimes'].items():
        if regime == "crisis_periods":
            memo += f"""### Crisis Period Analysis
"""
            for crisis_name, crisis_stats in stats.items():
                memo += f"""#### {crisis_name.replace('_', ' ').title()}
| Metric | Value |
|--------|-------|
| N Days | {crisis_stats['n_days']} |
| Total Return | {crisis_stats['total_return']:.2%} |
| Sharpe | {crisis_stats['sharpe']:.3f} |

"""
        else:
            memo += f"""### {regime.replace('_', ' ').title()} Regime
| Metric | Value |
|--------|-------|
| N Days | {stats['n_days']} |
| Ann Return | {stats['ann_return']:.2%} |
| Ann Vol | {stats['ann_vol']:.2%} |
| Sharpe | {stats['sharpe']:.3f} |
| Hit Rate | {stats['hit_rate']:.2%} |

"""
    
    memo += f"""---

## 5. Asset Class Contribution

"""
    
    if "asset_class_contribution" in metrics['asset_contribution']:
        for class_name, class_data in metrics['asset_contribution']['asset_class_contribution'].items():
            memo += f"""### {class_name.replace('_', ' ').title()}
- Total Contribution: {class_data['total_contribution']:.4f}
- Number of Assets: {class_data['n_assets']}
- Assets: {', '.join(class_data['assets'][:5])}{'...' if len(class_data['assets']) > 5 else ''}

"""
    
    memo += f"""---

## 6. Top 5 Asset Contributors (by total contribution)

| Asset | Total Contribution | Avg |Weight| | Contrib Sharpe |
|-------|-------------------|-------------|----------------|
"""
    
    # Sort by total contribution
    contrib = metrics['asset_contribution']['total_contribution']
    sorted_assets = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    for asset, total_c in sorted_assets:
        avg_w = metrics['asset_contribution']['avg_abs_weight'].get(asset, 0)
        c_sharpe = metrics['asset_contribution']['contribution_sharpe'].get(asset, 0)
        memo += f"| {asset} | {total_c:.4f} | {avg_w:.4f} | {c_sharpe:.3f} |\n"
    
    memo += f"""

---

## 7. Rolling Stability Analysis (6m/12m)

### 6-Month Rolling Sharpe
| Metric | Value |
|--------|-------|
| Mean | {metrics['rolling_stability']['rolling_6m_sharpe']['mean']:.3f} |
| Std | {metrics['rolling_stability']['rolling_6m_sharpe']['std']:.3f} |
| Min | {metrics['rolling_stability']['rolling_6m_sharpe']['min']:.3f} |
| Max | {metrics['rolling_stability']['rolling_6m_sharpe']['max']:.3f} |
| % Positive | {metrics['rolling_stability']['rolling_6m_sharpe']['pct_positive']:.1%} |

### 12-Month Rolling Sharpe
| Metric | Value |
|--------|-------|
| Mean | {metrics['rolling_stability']['rolling_12m_sharpe']['mean']:.3f} |
| Std | {metrics['rolling_stability']['rolling_12m_sharpe']['std']:.3f} |
| Min | {metrics['rolling_stability']['rolling_12m_sharpe']['min']:.3f} |
| Max | {metrics['rolling_stability']['rolling_12m_sharpe']['max']:.3f} |
| % Positive | {metrics['rolling_stability']['rolling_12m_sharpe']['pct_positive']:.1%} |

---

## 8. Observed Structural Failure Modes

Based on the above measurements:

"""
    
    # Diagnose failure modes
    failure_modes = []
    
    if metrics['standalone']['sharpe'] < 0.4:
        if metrics['standalone']['sharpe'] < 0:
            failure_modes.append(f"1. **Negative Standalone Sharpe ({metrics['standalone']['sharpe']:.3f})**: The TSMOM signal is "
                               f"destroying value on a risk-adjusted basis. This is the primary failure.")
        else:
            failure_modes.append(f"1. **Below-Benchmark Sharpe ({metrics['standalone']['sharpe']:.3f})**: The TSMOM signal "
                               f"is positive but below the academic benchmark of 0.4-0.5.")
    else:
        failure_modes.append(f"1. **Sharpe meets benchmark ({metrics['standalone']['sharpe']:.3f})**: Unconditional "
                           f"belief quality is acceptable.")
    
    # Check for year-by-year variability
    yearly_sharpe_std = metrics['yearly']['Sharpe'].std()
    if yearly_sharpe_std > 0.5:
        failure_modes.append(f"2. **High Year-to-Year Variability (σ={yearly_sharpe_std:.2f})**: "
                           f"Sharpe varies significantly across years, indicating regime-dependent behavior.")
    
    # Check for regime sensitivity
    regime_sharpes = {k: v['sharpe'] for k, v in metrics['regimes'].items() if k != 'crisis_periods'}
    if regime_sharpes:
        best_regime = max(regime_sharpes, key=regime_sharpes.get)
        worst_regime = min(regime_sharpes, key=regime_sharpes.get)
        if abs(regime_sharpes[best_regime] - regime_sharpes[worst_regime]) > 0.5:
            failure_modes.append(f"3. **Regime Sensitivity**: Best regime ({best_regime}: {regime_sharpes[best_regime]:.2f}) "
                               f"vs worst regime ({worst_regime}: {regime_sharpes[worst_regime]:.2f}). "
                               f"Gap = {abs(regime_sharpes[best_regime] - regime_sharpes[worst_regime]):.2f}")
    
    # Check for long underwater periods
    if metrics['standalone']['longest_underwater_days'] > 365:
        failure_modes.append(f"4. **Extended Bleed Periods**: {metrics['standalone']['longest_underwater_days']} days "
                           f"underwater indicates sustained drawdowns that damage compounding.")
    
    # Check for signal instability
    if metrics['stability']['avg_5d_autocorrelation'] < 0.5:
        failure_modes.append(f"5. **Signal Instability**: Low 5-day autocorrelation ({metrics['stability']['avg_5d_autocorrelation']:.3f}) "
                           f"suggests potential whipsaw behavior (horizon mismatch or trend reversal sensitivity).")
    
    for fm in failure_modes:
        memo += fm + "\n\n"
    
    memo += f"""---

## 9. Questions for Hypothesis Formation (Step 2)

These are questions to consider in Step 2 (Hypothesis Specification), NOT answers:

### Horizon-Related Questions
1. Is the 50/30/20 horizon weight optimal, or does one horizon contaminate others?
2. Is the short-term (21d) horizon causing whipsaw in trend reversals?
3. Would removing or reducing short-term weight improve stability?

### Feature-Related Questions
4. Are the legacy feature weights (0.5/0.3/0.2) optimal, or would equal-weight (canonical) perform better?
5. Is the persistence feature in medium-term adding value or noise?
6. Should breakout features be upweighted relative to return momentum?

### Vol Normalization Questions
7. Is EWMA vol normalization (halflife=63d) appropriate for all horizons?
8. Is the 5% vol floor too high or too low?
9. Should different horizons have different vol normalization parameters?

### Asset Class Questions
10. Which asset classes are driving positive/negative contribution?
11. Should certain asset classes be excluded or underweighted?
12. Is the signal quality consistent across asset classes?

### Structural vs Interaction Questions
13. Is the failure structural (bad economic belief) or interaction (horizon contamination)?
14. Does TSMOM fail in the same regimes as CSMOM, or differently?
15. Would simpler single-horizon TSMOM perform better?

---

## 10. Appendix: Data Lineage

- **Baseline Run**: `{baseline_run_id}`
- **Eval Window**: {EVAL_START} to {EVAL_END}
- **Artifacts Used**:
  - sleeve_returns.csv (or portfolio_returns.csv)
  - weights_post_construction.csv
  - asset_returns.csv
  - meta.json
  - canonical_diagnostics.json (if available)

**NO CODE WAS CHANGED. This is pure measurement.**

---

*End of TSMOM Belief Autopsy*
"""
    
    return memo


def main():
    """Run the TSMOM belief autopsy."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TSMOM Belief Autopsy — Phase 4 Engine Research")
    parser.add_argument("--run_id", type=str, default=None,
                       help="Run ID to analyze. If not provided, will search for latest tsmom baseline.")
    parser.add_argument("--eval_start", type=str, default=EVAL_START,
                       help=f"Evaluation start date (default: {EVAL_START})")
    parser.add_argument("--eval_end", type=str, default=EVAL_END,
                       help=f"Evaluation end date (default: {EVAL_END})")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TSMOM BELIEF AUTOPSY — Phase 4 Engine Research Step 1")
    print("=" * 70)
    
    # Determine run ID
    run_id = args.run_id
    if run_id is None:
        print("\nSearching for latest TSMOM baseline run...")
        run_id = find_latest_tsmom_baseline()
        if run_id is None:
            print("\nNo TSMOM baseline run found. Please run the baseline first:")
            print("  python run_strategy.py --config configs/phase4_tsmom_baseline_v1.yaml \\")
            print("      --start 2020-03-20 --end 2025-10-31 \\")
            print("      --output_dir reports/runs/phase4_tsmom_baseline_v1_$(date +%Y%m%d_%H%M%S)")
            return
    
    eval_start = args.eval_start
    eval_end = args.eval_end
    
    print(f"\nBaseline Run ID: {run_id}")
    print(f"Eval Window: {eval_start} to {eval_end}")
    print("\nLoading artifacts...")
    
    # Load baseline artifacts
    artifacts = load_run_artifacts(run_id)
    
    if not artifacts:
        print("ERROR: Could not load baseline artifacts")
        print(f"\nPlease verify the run exists at: {REPORTS_DIR / run_id}")
        return
    
    print(f"Loaded: {list(artifacts.keys())}")
    
    # Determine which returns to use
    # For standalone (single-sleeve) baseline runs, portfolio_returns IS the sleeve return
    # post risk-targeting. This is the canonical evaluation layer for Post-Construction.
    # sleeve_returns contains raw pre-scaling returns which have extreme volatility.
    if "portfolio_returns" in artifacts:
        returns_df = artifacts["portfolio_returns"]
        # Convert to DataFrame with column name if Series
        if isinstance(returns_df, pd.Series):
            returns_df = returns_df.to_frame("tsmom_multihorizon")
        # Rename 'ret' column to 'tsmom_multihorizon' for consistency
        if "ret" in returns_df.columns:
            returns_df = returns_df.rename(columns={"ret": "tsmom_multihorizon"})
        print(f"\nUsing portfolio_returns (Post-RT, canonical evaluation layer)")
        print(f"  Note: For single-sleeve baselines, portfolio_returns = sleeve after RT scaling")
    elif "sleeve_returns" in artifacts:
        returns_df = artifacts["sleeve_returns"]
        print(f"\nUsing sleeve_returns with columns: {list(returns_df.columns)}")
        print(f"  WARNING: sleeve_returns are RAW pre-scaling returns with extreme vol")
    else:
        print("ERROR: No returns data found")
        return
    
    # Compute metrics
    print("\nComputing standalone metrics...")
    standalone = compute_standalone_metrics(returns_df, eval_start)
    if "error" in standalone:
        print(f"ERROR: {standalone['error']}")
        return
    print(f"  Standalone Sharpe: {standalone['sharpe']:.3f}")
    print(f"  CAGR: {standalone['cagr']:.2%}")
    print(f"  Max DD: {standalone['max_drawdown']:.2%}")
    
    print("\nComputing yearly returns...")
    yearly = compute_yearly_returns(returns_df, eval_start)
    print(yearly.to_string())
    
    print("\nAnalyzing asset contribution...")
    if "weights_post_construction" in artifacts and "asset_returns" in artifacts:
        asset_contrib = analyze_asset_contribution(
            artifacts["weights_post_construction"],
            artifacts["asset_returns"],
            eval_start
        )
    else:
        asset_contrib = {"total_contribution": {}, "contribution_sharpe": {}, 
                        "avg_abs_weight": {}, "asset_class_contribution": {}}
        print("  (Skipped: missing weights or asset returns)")
    
    print("\nAnalyzing signal stability...")
    if "weights_post_construction" in artifacts:
        stability = analyze_signal_stability(artifacts["weights_post_construction"], eval_start)
        if "error" not in stability:
            print(f"  Avg Weekly Turnover: {stability['avg_weekly_turnover']:.4f}")
            print(f"  Avg 5-Day Autocorr: {stability['avg_5d_autocorrelation']:.3f}")
    else:
        stability = {"avg_daily_turnover": 0, "avg_weekly_turnover": 0, 
                    "avg_5d_autocorrelation": 0, "avg_daily_sign_changes": 0,
                    "avg_weight_concentration_hhi": 0}
        print("  (Skipped: missing weights)")
    
    print("\nAnalyzing regime behavior...")
    regimes = analyze_regime_behavior(returns_df, eval_start)
    if "error" not in regimes:
        for regime, stats in regimes.items():
            if regime != "crisis_periods" and isinstance(stats, dict) and "sharpe" in stats:
                print(f"  {regime}: Sharpe = {stats['sharpe']:.3f}")
    
    print("\nAnalyzing rolling stability...")
    rolling_stability = analyze_rolling_stability(returns_df, eval_start)
    if "error" not in rolling_stability:
        print(f"  6m rolling Sharpe mean: {rolling_stability['rolling_6m_sharpe']['mean']:.3f}")
        print(f"  12m rolling Sharpe mean: {rolling_stability['rolling_12m_sharpe']['mean']:.3f}")
    
    # Assemble metrics
    metrics = {
        "standalone": standalone,
        "yearly": yearly,
        "asset_contribution": asset_contrib,
        "stability": stability,
        "regimes": regimes,
        "rolling_stability": rolling_stability,
    }
    
    # Generate memo
    print("\nGenerating diagnostic memo...")
    memo = generate_memo(run_id, metrics)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write memo
    memo_path = OUTPUT_DIR / f"tsmom_belief_autopsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(memo_path, "w", encoding='utf-8') as f:
        f.write(memo)
    
    # Also write JSON metrics for future reference
    json_path = OUTPUT_DIR / f"tsmom_autopsy_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert yearly DataFrame to dict for JSON
    metrics_json = {
        "baseline_run_id": run_id,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "standalone": standalone,
        "yearly": yearly.to_dict(),
        "stability": {k: v if not isinstance(v, float) or not np.isnan(v) else None 
                     for k, v in stability.items()},
        "regimes": regimes,
        "rolling_stability": rolling_stability,
    }
    
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("AUTOPSY COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nMemo written to: {memo_path}")
    print(f"Metrics written to: {json_path}")
    print("\n" + "=" * 70)
    print("SUMMARY OF OBSERVATIONS")
    print("=" * 70)
    print(f"""
1. Standalone Sharpe: {standalone['sharpe']:.3f} {"(MEETS BENCHMARK >=0.40)" if standalone['sharpe'] >= 0.40 else "(BELOW BENCHMARK)" if standalone['sharpe'] > 0 else "(NEGATIVE)"}
2. CAGR: {standalone['cagr']:.2%}
3. Max Drawdown: {standalone['max_drawdown']:.2%}
4. Longest Underwater: {standalone['longest_underwater_days']} days
5. Yearly Sharpe Range: {yearly['Sharpe'].min():.2f} to {yearly['Sharpe'].max():.2f}
6. Signal Autocorrelation: {stability.get('avg_5d_autocorrelation', 'N/A'):.3f if isinstance(stability.get('avg_5d_autocorrelation'), float) else 'N/A'}
""")
    
    print("\nNEXT STEPS:")
    print("1. Review the diagnostic memo")
    print("2. Identify structural failure modes")
    print("3. If hypothesis needed, proceed to Step 2 (Hypothesis Specification)")
    print("4. DO NOT make any code changes until hypotheses are documented")


if __name__ == "__main__":
    main()
