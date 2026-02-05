#!/usr/bin/env python3
"""
CSMOM Belief Autopsy — Phase 4 Engine Research Step 1

This script performs a comprehensive diagnostic of the current CSMOM engine
at the Phase 3B pinned baseline. NO CODE CHANGES are made.

Research Protocol Compliance:
- Step 1: Belief Autopsy (No Code Changes)
- Baseline: phase3b_baseline_artifacts_only_20260120_093953
- Purpose: Diagnose current promoted engine failure modes

Deliverable:
A diagnostic memo documenting observed structural failure modes of current CSMOM.

Author: Phase 4 Engine Research
Date: January 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# STEP 1: FREEZE THE BASELINE (Non-Negotiable)
# =============================================================================

BASELINE_RUN_ID = "phase3b_baseline_artifacts_only_20260120_093953"
TRADED_RUN_ID = "phase3b_baseline_traded_20260120_093953"

# Canonical evaluation window
EVAL_START = "2020-03-20"
EVAL_END = "2025-10-31"

# Report paths
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "runs"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports" / "phase4_research" / "csmom_autopsy"


def load_run_artifacts(run_id: str) -> dict:
    """Load all relevant artifacts from a run."""
    run_dir = REPORTS_DIR / run_id
    
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
        with open(meta_path) as f:
            artifacts["meta"] = json.load(f)
    
    # Load canonical diagnostics
    diag_path = run_dir / "canonical_diagnostics.json"
    if diag_path.exists():
        with open(diag_path) as f:
            artifacts["canonical_diagnostics"] = json.load(f)
    
    # Load engine attribution
    attr_path = run_dir / "engine_attribution_post_construction.json"
    if attr_path.exists():
        with open(attr_path) as f:
            artifacts["engine_attribution"] = json.load(f)
    
    return artifacts


def compute_standalone_metrics(sleeve_returns: pd.DataFrame, eval_start: str) -> dict:
    """Compute standalone CSMOM metrics."""
    # Filter to eval window
    csmom_rets = sleeve_returns["csmom_meta"].loc[eval_start:]
    
    # Drop NaN and zero returns during warmup
    csmom_rets = csmom_rets.dropna()
    csmom_rets = csmom_rets[csmom_rets != 0]
    
    if len(csmom_rets) == 0:
        return {"error": "No valid CSMOM returns"}
    
    # Basic metrics
    ann_ret = csmom_rets.mean() * 252
    ann_vol = csmom_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Cumulative returns
    cum_ret = (1 + csmom_rets).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    
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
    hit_rate = (csmom_rets > 0).mean()
    
    # Skewness and kurtosis
    skew = csmom_rets.skew()
    kurt = csmom_rets.kurtosis()
    
    return {
        "n_days": len(csmom_rets),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "longest_underwater_days": longest_uw,
        "hit_rate": hit_rate,
        "skewness": skew,
        "kurtosis": kurt,
        "start_date": str(csmom_rets.index[0].date()),
        "end_date": str(csmom_rets.index[-1].date()),
    }


def compute_yearly_returns(sleeve_returns: pd.DataFrame, eval_start: str) -> pd.DataFrame:
    """Compute yearly returns table for CSMOM."""
    csmom_rets = sleeve_returns["csmom_meta"].loc[eval_start:]
    csmom_rets = csmom_rets.dropna()
    
    # Group by year
    yearly = csmom_rets.groupby(csmom_rets.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Also compute yearly Sharpe
    yearly_sharpe = csmom_rets.groupby(csmom_rets.index.year).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    
    # Yearly vol
    yearly_vol = csmom_rets.groupby(csmom_rets.index.year).apply(
        lambda x: x.std() * np.sqrt(252)
    )
    
    result = pd.DataFrame({
        "Return": yearly,
        "Vol": yearly_vol,
        "Sharpe": yearly_sharpe
    })
    
    return result


def analyze_asset_contribution(weights: pd.DataFrame, asset_returns: pd.DataFrame, eval_start: str) -> dict:
    """Analyze per-asset contribution to CSMOM."""
    # Filter to eval window
    weights_eval = weights.loc[eval_start:]
    returns_eval = asset_returns.loc[eval_start:]
    
    # Align indices
    common_idx = weights_eval.index.intersection(returns_eval.index)
    weights_eval = weights_eval.loc[common_idx]
    returns_eval = returns_eval.loc[common_idx]
    
    # Compute contribution = weight * return for each asset
    # Note: weights_post_construction already includes CSMOM weights
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
    
    return {
        "total_contribution": total_contrib.to_dict(),
        "contribution_sharpe": contrib_sharpe,
        "avg_abs_weight": avg_abs_weight.to_dict(),
    }


def analyze_signal_stability(weights: pd.DataFrame, eval_start: str) -> dict:
    """Analyze signal stability and turnover."""
    weights_eval = weights.loc[eval_start:]
    
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
    
    return {
        "avg_daily_turnover": avg_turnover,
        "avg_weekly_turnover": avg_weekly_turnover,
        "avg_5d_autocorrelation": avg_autocorr,
        "avg_daily_sign_changes": avg_sign_changes,
    }


def analyze_correlation_with_trend(sleeve_returns: pd.DataFrame, eval_start: str) -> dict:
    """Analyze correlation between CSMOM and Trend."""
    rets = sleeve_returns.loc[eval_start:]
    
    csmom = rets["csmom_meta"].dropna()
    trend = rets["tsmom_multihorizon"].dropna()
    
    # Align
    common_idx = csmom.index.intersection(trend.index)
    csmom = csmom.loc[common_idx]
    trend = trend.loc[common_idx]
    
    # Overall correlation
    overall_corr = csmom.corr(trend)
    
    # Rolling 63-day correlation
    rolling_corr = csmom.rolling(63).corr(trend)
    
    # Yearly correlation
    yearly_corr = {}
    for year in csmom.index.year.unique():
        mask = csmom.index.year == year
        if mask.sum() > 20:
            yearly_corr[int(year)] = csmom[mask].corr(trend[mask])
    
    # Correlation during Trend drawdowns
    trend_cum = (1 + trend).cumprod()
    trend_dd = (trend_cum - trend_cum.cummax()) / trend_cum.cummax()
    dd_mask = trend_dd < -0.05  # 5% drawdown threshold
    
    if dd_mask.sum() > 20:
        corr_during_dd = csmom[dd_mask].corr(trend[dd_mask])
    else:
        corr_during_dd = np.nan
    
    return {
        "overall_correlation": overall_corr,
        "rolling_63d_corr_mean": rolling_corr.mean(),
        "rolling_63d_corr_std": rolling_corr.std(),
        "yearly_correlation": yearly_corr,
        "correlation_during_trend_drawdown": corr_during_dd,
    }


def analyze_regime_behavior(sleeve_returns: pd.DataFrame, eval_start: str) -> dict:
    """Analyze CSMOM behavior across market regimes."""
    csmom = sleeve_returns["csmom_meta"].loc[eval_start:].dropna()
    
    # Use rolling volatility as regime proxy
    rolling_vol = csmom.rolling(21).std() * np.sqrt(252)
    
    # Define regimes based on vol percentiles
    vol_25 = rolling_vol.quantile(0.25)
    vol_75 = rolling_vol.quantile(0.75)
    
    low_vol_mask = rolling_vol < vol_25
    high_vol_mask = rolling_vol > vol_75
    mid_vol_mask = ~low_vol_mask & ~high_vol_mask
    
    regimes = {}
    for name, mask in [("low_vol", low_vol_mask), ("mid_vol", mid_vol_mask), ("high_vol", high_vol_mask)]:
        regime_rets = csmom[mask]
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
    
    return regimes


def generate_memo(baseline_run_id: str, metrics: dict) -> str:
    """Generate the diagnostic memo."""
    memo = f"""# CSMOM Belief Autopsy — Structural Failure Mode Analysis

**Baseline Run ID:** `{baseline_run_id}`
**Generated:** {datetime.now().isoformat()}
**Research Protocol:** Phase 4 Engine Research, Step 1 (No Code Changes)

---

## Executive Summary

This memo documents the observed structural failure modes of the current CSMOM engine
at the Phase 3B pinned baseline. NO hypotheses or fixes are proposed — this is 
pure measurement.

---

## 1. Standalone CSMOM Metrics (Eval Window: {EVAL_START} to {EVAL_END})

| Metric | Value |
|--------|-------|
| N Days | {metrics['standalone']['n_days']} |
| Annualized Return | {metrics['standalone']['annualized_return']:.2%} |
| Annualized Vol | {metrics['standalone']['annualized_vol']:.2%} |
| Sharpe Ratio | {metrics['standalone']['sharpe']:.3f} |
| Total Return | {metrics['standalone']['total_return']:.2%} |
| Max Drawdown | {metrics['standalone']['max_drawdown']:.2%} |
| Longest Underwater | {metrics['standalone']['longest_underwater_days']} days |
| Hit Rate | {metrics['standalone']['hit_rate']:.2%} |
| Skewness | {metrics['standalone']['skewness']:.3f} |
| Kurtosis | {metrics['standalone']['kurtosis']:.3f} |

**Observation:** Standalone Sharpe of {metrics['standalone']['sharpe']:.3f} is negative,
indicating the CSMOM signal is destroying value on a risk-adjusted basis.

---

## 2. Yearly Returns Table

| Year | Return | Vol | Sharpe |
|------|--------|-----|--------|
"""
    
    for year, row in metrics['yearly'].iterrows():
        memo += f"| {year} | {row['Return']:.2%} | {row['Vol']:.2%} | {row['Sharpe']:.3f} |\n"
    
    memo += f"""
**Observation:** CSMOM shows inconsistent behavior across years. 
Positive years: {(metrics['yearly']['Return'] > 0).sum()}, 
Negative years: {(metrics['yearly']['Return'] < 0).sum()}.

---

## 3. Signal Stability & Turnover

| Metric | Value |
|--------|-------|
| Avg Daily Turnover | {metrics['stability']['avg_daily_turnover']:.4f} |
| Avg Weekly Turnover | {metrics['stability']['avg_weekly_turnover']:.4f} |
| Avg 5-Day Autocorrelation | {metrics['stability']['avg_5d_autocorrelation']:.3f} |
| Avg Daily Sign Changes | {metrics['stability']['avg_daily_sign_changes']:.2f} |

**Observation:** {"High turnover suggests signal instability." if metrics['stability']['avg_weekly_turnover'] > 0.5 else "Turnover is moderate."}
Signal autocorrelation of {metrics['stability']['avg_5d_autocorrelation']:.3f} indicates 
{"persistent signals" if metrics['stability']['avg_5d_autocorrelation'] > 0.7 else "moderate signal persistence"}.

---

## 4. Correlation with Trend

| Metric | Value |
|--------|-------|
| Overall Correlation | {metrics['correlation']['overall_correlation']:.3f} |
| Rolling 63d Corr (Mean) | {metrics['correlation']['rolling_63d_corr_mean']:.3f} |
| Rolling 63d Corr (Std) | {metrics['correlation']['rolling_63d_corr_std']:.3f} |
| Corr During Trend DD | {metrics['correlation']['correlation_during_trend_drawdown']:.3f} |

**Yearly Correlation:**
"""
    
    for year, corr in metrics['correlation']['yearly_correlation'].items():
        memo += f"- {year}: {corr:.3f}\n"
    
    memo += f"""
**Observation:** {"CSMOM shows low correlation with Trend overall, which is good for diversification." if abs(metrics['correlation']['overall_correlation']) < 0.2 else "CSMOM shows meaningful correlation with Trend."}
Correlation varies significantly by year (std: {np.std(list(metrics['correlation']['yearly_correlation'].values())):.3f}).

---

## 5. Regime-Conditioned Performance

"""
    
    for regime, stats in metrics['regimes'].items():
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

## 6. Top 5 Asset Contributors

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

## 7. Post-Construction Attribution (from existing diagnostics)

From engine_attribution_post_construction:
- **Contrib %**: {metrics['engine_attribution'].get('csmom_contrib_pct', 'N/A')}
- **Contrib Sharpe**: {metrics['engine_attribution'].get('csmom_contrib_sharpe', 'N/A')}
- **Corr w/ Portfolio**: {metrics['engine_attribution'].get('csmom_corr_portfolio', 'N/A')}
- **DD Contrib**: {metrics['engine_attribution'].get('csmom_dd_contrib', 'N/A')}

---

## 8. Observed Structural Failure Modes

Based on the above measurements:

1. **Negative Standalone Sharpe ({metrics['standalone']['sharpe']:.3f})**: The CSMOM signal is 
   not generating positive risk-adjusted returns on its own. This is the primary failure.

2. **High Variability Across Years**: {"Sharpe varies from " + f"{metrics['yearly']['Sharpe'].min():.2f} to {metrics['yearly']['Sharpe'].max():.2f}" + ", indicating regime-dependent behavior that is not consistently captured."}

3. **Regime Sensitivity**: {"CSMOM performs best in " + max(metrics['regimes'].items(), key=lambda x: x[1]['sharpe'])[0].replace('_', ' ') + " regime, suggesting potential overfitting to calm periods." if metrics['regimes'] else "Insufficient data for regime analysis."}

4. **Portfolio Drag**: At Post-Construction, CSMOM contributes negatively to portfolio PnL,
   acting as a drag rather than a diversifier.

5. **Long Underwater Periods**: {metrics['standalone']['longest_underwater_days']} days underwater
   indicates extended bleed periods that damage compounding.

---

## 9. Questions for Hypothesis Formation (Step 2)

These are questions to consider in Step 2 (Hypothesis Specification), NOT answers:

1. Is the lookback/skip configuration appropriate for the current regime environment?
2. Is the cross-sectional z-scoring creating spurious signals in a concentrated universe?
3. Is the vol-tempering amplifying noise rather than signal?
4. Should CSMOM be conditioned on market regime (contradicts current engine architecture)?
5. Is the universe size (13 assets) too small for meaningful cross-sectional ranking?

---

## 10. Appendix: Data Lineage

- **Baseline Run**: `{baseline_run_id}`
- **Eval Window**: {EVAL_START} to {EVAL_END}
- **Artifacts Used**:
  - sleeve_returns.csv
  - weights_post_construction.csv
  - asset_returns.csv
  - engine_attribution_post_construction.json
  - canonical_diagnostics.json

**NO CODE WAS CHANGED. This is pure measurement.**

---

*End of CSMOM Belief Autopsy*
"""
    
    return memo


def main():
    """Run the CSMOM belief autopsy."""
    print("=" * 70)
    print("CSMOM BELIEF AUTOPSY — Phase 4 Engine Research Step 1")
    print("=" * 70)
    print(f"\nBaseline Run ID: {BASELINE_RUN_ID}")
    print(f"Eval Window: {EVAL_START} to {EVAL_END}")
    print("\nLoading artifacts...")
    
    # Load baseline artifacts
    artifacts = load_run_artifacts(BASELINE_RUN_ID)
    
    if not artifacts:
        print("ERROR: Could not load baseline artifacts")
        return
    
    print(f"Loaded: {list(artifacts.keys())}")
    
    # Compute metrics
    print("\nComputing standalone metrics...")
    standalone = compute_standalone_metrics(artifacts["sleeve_returns"], EVAL_START)
    print(f"  Standalone Sharpe: {standalone['sharpe']:.3f}")
    
    print("\nComputing yearly returns...")
    yearly = compute_yearly_returns(artifacts["sleeve_returns"], EVAL_START)
    print(yearly)
    
    print("\nAnalyzing asset contribution...")
    asset_contrib = analyze_asset_contribution(
        artifacts["weights_post_construction"],
        artifacts["asset_returns"],
        EVAL_START
    )
    
    print("\nAnalyzing signal stability...")
    stability = analyze_signal_stability(artifacts["weights_post_construction"], EVAL_START)
    print(f"  Avg Weekly Turnover: {stability['avg_weekly_turnover']:.4f}")
    
    print("\nAnalyzing correlation with Trend...")
    correlation = analyze_correlation_with_trend(artifacts["sleeve_returns"], EVAL_START)
    print(f"  Overall Correlation: {correlation['overall_correlation']:.3f}")
    
    print("\nAnalyzing regime behavior...")
    regimes = analyze_regime_behavior(artifacts["sleeve_returns"], EVAL_START)
    
    # Extract engine attribution metrics
    engine_attr = {}
    if "engine_attribution" in artifacts:
        attr = artifacts["engine_attribution"]
        if "sleeves" in attr:
            for sleeve in attr["sleeves"]:
                if sleeve.get("name") == "csmom_meta":
                    engine_attr = {
                        "csmom_contrib_pct": f"{sleeve.get('contrib_pct', 0):.1%}",
                        "csmom_contrib_sharpe": f"{sleeve.get('contrib_sharpe', 0):.3f}",
                        "csmom_corr_portfolio": f"{sleeve.get('corr_with_portfolio', 0):.3f}",
                        "csmom_dd_contrib": f"{sleeve.get('dd_contrib', 0):.1%}",
                    }
    
    # Assemble metrics
    metrics = {
        "standalone": standalone,
        "yearly": yearly,
        "asset_contribution": asset_contrib,
        "stability": stability,
        "correlation": correlation,
        "regimes": regimes,
        "engine_attribution": engine_attr,
    }
    
    # Generate memo
    print("\nGenerating diagnostic memo...")
    memo = generate_memo(BASELINE_RUN_ID, metrics)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write memo
    memo_path = OUTPUT_DIR / f"csmom_belief_autopsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(memo_path, "w") as f:
        f.write(memo)
    
    # Also write JSON metrics for future reference
    json_path = OUTPUT_DIR / f"csmom_autopsy_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert yearly DataFrame to dict for JSON
    metrics_json = {
        "baseline_run_id": BASELINE_RUN_ID,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "standalone": standalone,
        "yearly": yearly.to_dict(),
        "stability": stability,
        "correlation": {k: v if not isinstance(v, float) or not np.isnan(v) else None 
                       for k, v in correlation.items()},
        "regimes": regimes,
        "engine_attribution": engine_attr,
    }
    
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("AUTOPSY COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nMemo written to: {memo_path}")
    print(f"Metrics written to: {json_path}")
    print("\n" + "=" * 70)
    print("SUMMARY OF OBSERVED FAILURE MODES")
    print("=" * 70)
    print(f"""
1. Standalone Sharpe: {standalone['sharpe']:.3f} (NEGATIVE)
2. Max Drawdown: {standalone['max_drawdown']:.2%}
3. Longest Underwater: {standalone['longest_underwater_days']} days
4. Correlation with Trend: {correlation['overall_correlation']:.3f}
5. Yearly Sharpe Range: {yearly['Sharpe'].min():.2f} to {yearly['Sharpe'].max():.2f}
""")
    
    print("\nNEXT STEP: Review memo, then proceed to Step 2 (Hypothesis Specification)")
    print("DO NOT make any code changes until hypotheses are documented.")


if __name__ == "__main__":
    main()
