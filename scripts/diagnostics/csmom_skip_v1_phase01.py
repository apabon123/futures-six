#!/usr/bin/env python3
"""
CSMOM_skip_v1 Phase 0 and Phase 1 Test

Research Protocol: Phase 4 Engine Research
Hypothesis ID: CSMOM-H001
Variant: CSMOM_skip_v1

This script implements the Phase 0 sanity check and Phase 1 standalone test
for the CSMOM skip window hypothesis.

SINGLE CHANGE ONLY:
- Baseline: lookbacks (63, 126, 252) with skip (0, 0, 0)
- Variant: lookbacks (63, 126, 252) with skip (5, 10, 21)

All other parameters remain identical:
- weights: (0.4, 0.35, 0.25)
- vol_lookback: 63
- rebalance_freq: "D"
- neutralize_cross_section: True
- clip_score: 3.0

Author: Phase 4 Engine Research
Date: January 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_broker import MarketData

# =============================================================================
# CONFIGURATION (FROZEN - DO NOT MODIFY)
# =============================================================================

# Baseline configuration (current CSMOM)
BASELINE_CONFIG = {
    "lookbacks": (63, 126, 252),
    "skips": (0, 0, 0),  # No skip
    "weights": (0.4, 0.35, 0.25),
    "vol_lookback": 63,
    "rebalance_freq": "D",
    "neutralize_cross_section": True,
    "clip_score": 3.0,
}

# Variant configuration (CSMOM_skip_v1)
SKIP_V1_CONFIG = {
    "lookbacks": (63, 126, 252),
    "skips": (5, 10, 21),  # THE ONLY CHANGE
    "weights": (0.4, 0.35, 0.25),
    "vol_lookback": 63,
    "rebalance_freq": "D",
    "neutralize_cross_section": True,
    "clip_score": 3.0,
}

# Universe (same as baseline)
UNIVERSE = [
    "ES_FRONT_CALENDAR_2D",
    "NQ_FRONT_CALENDAR_2D",
    "RTY_FRONT_CALENDAR_2D",
    "ZT_FRONT_VOLUME",
    "ZF_FRONT_VOLUME",
    "ZN_FRONT_VOLUME",
    "UB_FRONT_VOLUME",
    "SR3_FRONT_CALENDAR",
    "CL_FRONT_VOLUME",
    "GC_FRONT_VOLUME",
    "6E_FRONT_CALENDAR",
    "6B_FRONT_CALENDAR",
    "6J_FRONT_CALENDAR",
]

# Evaluation window (same as baseline)
EVAL_START = "2020-03-20"
EVAL_END = "2025-10-31"

# Warmup buffer for longest lookback + skip
WARMUP_START = "2019-01-01"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase4_research" / "csmom_phase01"


# =============================================================================
# CSMOM SIGNAL COMPUTATION (WITH SKIP SUPPORT)
# =============================================================================

def compute_csmom_signals(
    md: MarketData,
    universe: Sequence[str],
    start: str,
    end: str,
    lookbacks: Sequence[int],
    skips: Sequence[int],
    weights: Sequence[float],
    vol_lookback: int,
    neutralize: bool,
    clip_score: float,
) -> pd.DataFrame:
    """
    Compute CSMOM signals with skip window support.
    
    This is a self-contained implementation for testing purposes.
    Does NOT modify production code.
    
    Args:
        md: MarketData instance
        universe: List of symbols
        start: Start date (with warmup buffer)
        end: End date
        lookbacks: Lookback periods for each horizon
        skips: Skip windows for each horizon (THE KEY PARAMETER)
        weights: Weights for each horizon
        vol_lookback: Lookback for volatility calculation
        neutralize: Whether to z-score cross-sectionally
        clip_score: Z-score clipping threshold
    
    Returns:
        DataFrame of signals [date x symbols] with values in [-1, 1]
    """
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Get log returns for momentum calculation
    rets = md.get_returns(tuple(universe), start=start, end=end, method="log")
    
    if rets.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=universe)
    
    # Compute k-day returns for each horizon WITH SKIP
    zscores = []
    for k, skip, w in zip(lookbacks, skips, weights):
        # For log returns: k-day log return = rolling sum
        # With skip: we want returns from [t-k-skip, t-skip]
        # This means: shift the rolling sum by skip days
        
        kret = rets.rolling(window=k, min_periods=k).sum()
        
        if skip > 0:
            # Shift by skip days: use returns ending skip days ago
            kret = kret.shift(skip)
        
        # Cross-sectional z-score per date
        mean_cs = kret.mean(axis=1, skipna=True)
        std_cs = kret.std(axis=1, skipna=True)
        std_cs = std_cs.replace(0.0, np.nan)
        
        z = kret.sub(mean_cs, axis=0).div(std_cs, axis=0)
        z = z.fillna(0.0)
        
        zscores.append(w * z)
    
    # Composite score (weighted sum)
    composite = sum(zscores)
    
    # Volatility tempering
    if vol_lookback > 0:
        rets_simple = md.get_returns(tuple(universe), start=start, end=end, method="simple")
        vol = rets_simple.rolling(window=vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252.0)
        vol = vol.replace(0.0, np.nan)
        
        vol_adjusted = composite.div(vol, axis=0)
        
        # Re-z-score cross-sectionally
        mean_cs = vol_adjusted.mean(axis=1, skipna=True)
        std_cs = vol_adjusted.std(axis=1, skipna=True)
        std_cs = std_cs.replace(0.0, np.nan)
        
        composite = vol_adjusted.sub(mean_cs, axis=0).div(std_cs, axis=0).fillna(0.0)
    
    # Cross-sectional neutralization
    if neutralize:
        mean_cs = composite.mean(axis=1, skipna=True)
        std_cs = composite.std(axis=1, skipna=True)
        std_cs = std_cs.replace(0.0, np.nan)
        
        composite = composite.sub(mean_cs, axis=0).div(std_cs, axis=0).fillna(0.0)
    
    # Clip and scale to [-1, 1]
    signals = composite.clip(lower=-clip_score, upper=clip_score) / clip_score
    
    return signals


def compute_sign_only_signals(
    md: MarketData,
    universe: Sequence[str],
    start: str,
    end: str,
    lookbacks: Sequence[int],
    skips: Sequence[int],
    weights: Sequence[float],
) -> pd.DataFrame:
    """
    Compute sign-only CSMOM signals for Phase 0.
    
    Simplified version: just sign of weighted k-day returns with skip.
    No vol tempering, no z-scoring, no clipping.
    """
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    rets = md.get_returns(tuple(universe), start=start, end=end, method="log")
    
    if rets.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=universe)
    
    # Compute weighted k-day returns with skip
    composite = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    
    for k, skip, w in zip(lookbacks, skips, weights):
        kret = rets.rolling(window=k, min_periods=k).sum()
        if skip > 0:
            kret = kret.shift(skip)
        composite += w * kret
    
    # Cross-sectional rank (simple: demean)
    mean_cs = composite.mean(axis=1, skipna=True)
    demeaned = composite.sub(mean_cs, axis=0)
    
    # Sign only
    signals = np.sign(demeaned)
    
    return signals


# =============================================================================
# BACKTEST ENGINE (MINIMAL)
# =============================================================================

def run_backtest(
    signals: pd.DataFrame,
    md: MarketData,
    eval_start: str,
    eval_end: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run minimal backtest: signals × returns.
    
    Returns:
        returns: Daily portfolio returns
        weights: Daily weights used
    """
    # Get simple returns for PnL
    asset_returns = md.get_returns(tuple(signals.columns), start=eval_start, end=eval_end, method="simple")
    
    # Align signals and returns
    common_idx = signals.index.intersection(asset_returns.index)
    signals_aligned = signals.loc[common_idx]
    returns_aligned = asset_returns.loc[common_idx]
    
    # Filter to eval window
    mask = (signals_aligned.index >= eval_start) & (signals_aligned.index <= eval_end)
    signals_eval = signals_aligned[mask]
    returns_eval = returns_aligned[mask]
    
    # Normalize weights to sum to 1 (absolute value)
    abs_sum = signals_eval.abs().sum(axis=1)
    abs_sum = abs_sum.replace(0.0, 1.0)  # Avoid division by zero
    weights = signals_eval.div(abs_sum, axis=0)
    
    # Portfolio returns = sum(weight * return)
    # Use previous day's weights (signal at t, return at t+1)
    weights_lagged = weights.shift(1)
    portfolio_returns = (weights_lagged * returns_eval).sum(axis=1)
    
    # Drop first row (NaN from lag)
    portfolio_returns = portfolio_returns.dropna()
    weights_lagged = weights_lagged.dropna()
    
    return portfolio_returns, weights_lagged


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(returns: pd.Series, weights: pd.DataFrame) -> dict:
    """Compute standard performance metrics."""
    if len(returns) == 0:
        return {"error": "No returns"}
    
    # Filter out zero returns at start (warmup)
    returns = returns[returns != 0]
    if len(returns) == 0:
        return {"error": "All zero returns"}
    
    # Basic metrics
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Cumulative returns
    cum_ret = (1 + returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    cagr = (cum_ret.iloc[-1]) ** (252 / len(returns)) - 1
    
    # Drawdown
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Time underwater
    underwater = drawdown < -0.001  # 0.1% threshold
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
    
    longest_uw = max((end - start).days for start, end in underwater_periods) if underwater_periods else 0
    
    # Hit rate
    hit_rate = (returns > 0).mean()
    
    # Asset contribution
    if weights is not None and len(weights) > 0:
        # Compute per-asset contribution
        asset_returns_df = weights.shift(1)  # This is wrong, need actual returns
        # Simplified: use average absolute weight as proxy
        avg_abs_weight = weights.abs().mean()
        top_assets = avg_abs_weight.nlargest(3)
        top_3_weight = top_assets.sum()
    else:
        top_3_weight = np.nan
    
    return {
        "n_days": len(returns),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "cagr": cagr,
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "longest_underwater_days": longest_uw,
        "hit_rate": hit_rate,
        "top_3_asset_weight": top_3_weight,
    }


def compute_regime_metrics(returns: pd.Series) -> dict:
    """Compute regime-conditioned metrics."""
    if len(returns) < 100:
        return {"error": "Insufficient data"}
    
    # Use rolling volatility as regime proxy
    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    
    vol_25 = rolling_vol.quantile(0.25)
    vol_75 = rolling_vol.quantile(0.75)
    
    regimes = {}
    for name, mask in [
        ("low_vol", rolling_vol < vol_25),
        ("mid_vol", (rolling_vol >= vol_25) & (rolling_vol <= vol_75)),
        ("high_vol", rolling_vol > vol_75),
    ]:
        regime_rets = returns[mask]
        if len(regime_rets) > 20:
            ann_ret = regime_rets.mean() * 252
            ann_vol = regime_rets.std() * np.sqrt(252)
            regimes[name] = {
                "n_days": len(regime_rets),
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
            }
    
    return regimes


def compute_yearly_metrics(returns: pd.Series) -> dict:
    """Compute yearly returns."""
    yearly = {}
    for year in returns.index.year.unique():
        year_rets = returns[returns.index.year == year]
        if len(year_rets) > 0:
            total_ret = (1 + year_rets).prod() - 1
            yearly[int(year)] = total_ret
    return yearly


def compute_correlation_with_trend(
    csmom_returns: pd.Series,
    md: MarketData,
    eval_start: str,
    eval_end: str,
) -> dict:
    """
    Compute correlation with a simple trend proxy.
    Using ES (S&P 500 futures) momentum as trend proxy.
    """
    # Get ES returns
    es_rets = md.get_returns(("ES_FRONT_CALENDAR_2D",), start=eval_start, end=eval_end, method="simple")
    
    if es_rets.empty:
        return {"error": "No ES data"}
    
    es_rets = es_rets["ES_FRONT_CALENDAR_2D"]
    
    # Simple trend proxy: sign of 63-day cumulative return
    es_log = md.get_returns(("ES_FRONT_CALENDAR_2D",), start=eval_start, end=eval_end, method="log")
    es_log = es_log["ES_FRONT_CALENDAR_2D"]
    trend_signal = np.sign(es_log.rolling(63).sum())
    
    # Trend returns (proxy for what a trend strategy would earn)
    trend_returns = trend_signal.shift(1) * es_rets
    trend_returns = trend_returns.dropna()
    
    # Align with CSMOM
    common_idx = csmom_returns.index.intersection(trend_returns.index)
    csmom_aligned = csmom_returns.loc[common_idx]
    trend_aligned = trend_returns.loc[common_idx]
    
    # Overall correlation
    overall_corr = csmom_aligned.corr(trend_aligned)
    
    # Correlation during trend drawdown
    trend_cum = (1 + trend_aligned).cumprod()
    trend_dd = (trend_cum - trend_cum.cummax()) / trend_cum.cummax()
    dd_mask = trend_dd < -0.05
    
    if dd_mask.sum() > 20:
        corr_during_dd = csmom_aligned[dd_mask].corr(trend_aligned[dd_mask])
    else:
        corr_during_dd = np.nan
    
    return {
        "overall_correlation": overall_corr,
        "correlation_during_trend_dd": corr_during_dd,
    }


# =============================================================================
# PHASE 0 TEST
# =============================================================================

def run_phase0(md: MarketData, config: dict, name: str) -> dict:
    """Run Phase 0 sanity check."""
    print(f"\n{'='*60}")
    print(f"PHASE 0: {name}")
    print(f"{'='*60}")
    
    # Compute sign-only signals
    signals = compute_sign_only_signals(
        md=md,
        universe=UNIVERSE,
        start=WARMUP_START,
        end=EVAL_END,
        lookbacks=config["lookbacks"],
        skips=config["skips"],
        weights=config["weights"],
    )
    
    print(f"Signals shape: {signals.shape}")
    
    # Run backtest
    returns, weights = run_backtest(signals, md, EVAL_START, EVAL_END)
    
    print(f"Returns shape: {returns.shape}")
    
    # Compute metrics
    metrics = compute_metrics(returns, weights)
    
    print(f"\nPhase 0 Results ({name}):")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  CAGR: {metrics['cagr']:.2%}")
    print(f"  MaxDD: {metrics['max_drawdown']:.2%}")
    print(f"  Top-3 Asset Weight: {metrics['top_3_asset_weight']:.2%}")
    
    # Phase 0 gate
    passed = metrics["sharpe"] >= 0.2
    print(f"\n  PHASE 0 GATE (Sharpe >= 0.2): {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": name,
        "config": config,
        "metrics": metrics,
        "passed": passed,
    }


# =============================================================================
# PHASE 1 TEST
# =============================================================================

def run_phase1(md: MarketData, config: dict, name: str) -> dict:
    """Run Phase 1 standalone test."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: {name}")
    print(f"{'='*60}")
    
    # Compute full signals
    signals = compute_csmom_signals(
        md=md,
        universe=UNIVERSE,
        start=WARMUP_START,
        end=EVAL_END,
        lookbacks=config["lookbacks"],
        skips=config["skips"],
        weights=config["weights"],
        vol_lookback=config["vol_lookback"],
        neutralize=config["neutralize_cross_section"],
        clip_score=config["clip_score"],
    )
    
    print(f"Signals shape: {signals.shape}")
    
    # Run backtest
    returns, weights = run_backtest(signals, md, EVAL_START, EVAL_END)
    
    print(f"Returns shape: {returns.shape}")
    
    # Compute all metrics
    metrics = compute_metrics(returns, weights)
    regime_metrics = compute_regime_metrics(returns)
    yearly_metrics = compute_yearly_metrics(returns)
    corr_metrics = compute_correlation_with_trend(returns, md, EVAL_START, EVAL_END)
    
    print(f"\nPhase 1 Results ({name}):")
    print(f"  Full-window Sharpe: {metrics['sharpe']:.3f}")
    print(f"  CAGR: {metrics['cagr']:.2%}")
    print(f"  MaxDD: {metrics['max_drawdown']:.2%}")
    print(f"  Longest Underwater: {metrics['longest_underwater_days']} days")
    
    print(f"\n  Regime Sharpe:")
    for regime, stats in regime_metrics.items():
        if isinstance(stats, dict) and "sharpe" in stats:
            print(f"    {regime}: {stats['sharpe']:.3f}")
    
    print(f"\n  Yearly Returns:")
    for year, ret in sorted(yearly_metrics.items()):
        print(f"    {year}: {ret:.2%}")
    
    print(f"\n  Correlation with Trend:")
    print(f"    Overall: {corr_metrics.get('overall_correlation', 'N/A'):.3f}")
    print(f"    During Trend DD: {corr_metrics.get('correlation_during_trend_dd', 'N/A'):.3f}")
    
    return {
        "name": name,
        "config": config,
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "yearly_metrics": yearly_metrics,
        "correlation_metrics": corr_metrics,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Phase 0 and Phase 1 tests for CSMOM_skip_v1."""
    print("=" * 70)
    print("CSMOM_skip_v1 PHASE 0 AND PHASE 1 TEST")
    print("Hypothesis ID: CSMOM-H001")
    print("=" * 70)
    print(f"\nEval Window: {EVAL_START} to {EVAL_END}")
    print(f"Universe: {len(UNIVERSE)} assets")
    
    # Initialize market data
    print("\nLoading market data...")
    md = MarketData()
    
    # ==========================================================================
    # PHASE 0
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("PHASE 0: SANITY CHECK")
    print("=" * 70)
    
    phase0_baseline = run_phase0(md, BASELINE_CONFIG, "Baseline (no skip)")
    phase0_skip = run_phase0(md, SKIP_V1_CONFIG, "CSMOM_skip_v1")
    
    # Check Phase 0 gate
    if not phase0_skip["passed"]:
        print("\n" + "=" * 70)
        print("PHASE 0 FAILED - EXPERIMENT TERMINATED")
        print("=" * 70)
        print("\nCSMOM_skip_v1 failed Phase 0 sanity check.")
        print("Per research protocol, this variant is REJECTED.")
        
        # Save results
        save_results("phase0_fail", phase0_baseline, phase0_skip, None, None)
        return
    
    print("\n" + "=" * 70)
    print("PHASE 0 PASSED - PROCEEDING TO PHASE 1")
    print("=" * 70)
    
    # ==========================================================================
    # PHASE 1
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("PHASE 1: STANDALONE TEST")
    print("=" * 70)
    
    phase1_baseline = run_phase1(md, BASELINE_CONFIG, "Baseline (no skip)")
    phase1_skip = run_phase1(md, SKIP_V1_CONFIG, "CSMOM_skip_v1")
    
    # ==========================================================================
    # COMPARISON SUMMARY
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n### PHASE 0 RESULTS ###")
    print(f"{'Metric':<25} {'Baseline':>12} {'Skip_v1':>12} {'Delta':>12}")
    print("-" * 61)
    for metric in ["sharpe", "cagr", "max_drawdown", "top_3_asset_weight"]:
        b = phase0_baseline["metrics"].get(metric, np.nan)
        s = phase0_skip["metrics"].get(metric, np.nan)
        d = s - b if not np.isnan(b) and not np.isnan(s) else np.nan
        if "weight" in metric or "cagr" in metric or "drawdown" in metric:
            print(f"{metric:<25} {b:>11.2%} {s:>11.2%} {d:>+11.2%}")
        else:
            print(f"{metric:<25} {b:>12.3f} {s:>12.3f} {d:>+12.3f}")
    
    print("\n### PHASE 1 RESULTS ###")
    print(f"{'Metric':<25} {'Baseline':>12} {'Skip_v1':>12} {'Delta':>12}")
    print("-" * 61)
    for metric in ["sharpe", "cagr", "max_drawdown", "longest_underwater_days"]:
        b = phase1_baseline["metrics"].get(metric, np.nan)
        s = phase1_skip["metrics"].get(metric, np.nan)
        d = s - b if not np.isnan(b) and not np.isnan(s) else np.nan
        if "cagr" in metric or "drawdown" in metric:
            print(f"{metric:<25} {b:>11.2%} {s:>11.2%} {d:>+11.2%}")
        elif "days" in metric:
            print(f"{metric:<25} {b:>12.0f} {s:>12.0f} {d:>+12.0f}")
        else:
            print(f"{metric:<25} {b:>12.3f} {s:>12.3f} {d:>+12.3f}")
    
    print("\n### REGIME SHARPE ###")
    print(f"{'Regime':<25} {'Baseline':>12} {'Skip_v1':>12} {'Delta':>12}")
    print("-" * 61)
    for regime in ["low_vol", "mid_vol", "high_vol"]:
        b = phase1_baseline["regime_metrics"].get(regime, {}).get("sharpe", np.nan)
        s = phase1_skip["regime_metrics"].get(regime, {}).get("sharpe", np.nan)
        d = s - b if not np.isnan(b) and not np.isnan(s) else np.nan
        print(f"{regime:<25} {b:>12.3f} {s:>12.3f} {d:>+12.3f}")
    
    print("\n### YEARLY RETURNS ###")
    print(f"{'Year':<25} {'Baseline':>12} {'Skip_v1':>12} {'Delta':>12}")
    print("-" * 61)
    all_years = sorted(set(phase1_baseline["yearly_metrics"].keys()) | set(phase1_skip["yearly_metrics"].keys()))
    for year in all_years:
        b = phase1_baseline["yearly_metrics"].get(year, np.nan)
        s = phase1_skip["yearly_metrics"].get(year, np.nan)
        d = s - b if not np.isnan(b) and not np.isnan(s) else np.nan
        print(f"{year:<25} {b:>11.2%} {s:>11.2%} {d:>+11.2%}")
    
    print("\n### CORRELATION WITH TREND ###")
    print(f"{'Metric':<25} {'Baseline':>12} {'Skip_v1':>12} {'Delta':>12}")
    print("-" * 61)
    for metric in ["overall_correlation", "correlation_during_trend_dd"]:
        b = phase1_baseline["correlation_metrics"].get(metric, np.nan)
        s = phase1_skip["correlation_metrics"].get(metric, np.nan)
        d = s - b if not np.isnan(b) and not np.isnan(s) else np.nan
        print(f"{metric:<25} {b:>12.3f} {s:>12.3f} {d:>+12.3f}")
    
    # Save results
    save_results("phase1_complete", phase0_baseline, phase0_skip, phase1_baseline, phase1_skip)
    
    print("\n" + "=" * 70)
    print("DECISION SUMMARY FOR PHASE 2 GATE")
    print("=" * 70)
    
    # Extract key metrics for decision
    high_vol_baseline = phase1_baseline["regime_metrics"].get("high_vol", {}).get("sharpe", np.nan)
    high_vol_skip = phase1_skip["regime_metrics"].get("high_vol", {}).get("sharpe", np.nan)
    
    y2020_baseline = phase1_baseline["yearly_metrics"].get(2020, np.nan)
    y2020_skip = phase1_skip["yearly_metrics"].get(2020, np.nan)
    
    corr_dd_baseline = phase1_baseline["correlation_metrics"].get("correlation_during_trend_dd", np.nan)
    corr_dd_skip = phase1_skip["correlation_metrics"].get("correlation_during_trend_dd", np.nan)
    
    print(f"""
| Metric                      | Baseline    | Skip_v1     | Improved? |
|-----------------------------|-------------|-------------|-----------|
| Phase 0 Sharpe              | {phase0_baseline['metrics']['sharpe']:>10.3f} | {phase0_skip['metrics']['sharpe']:>10.3f} | {'YES' if phase0_skip['metrics']['sharpe'] > phase0_baseline['metrics']['sharpe'] else 'NO':>9} |
| Full-window Sharpe          | {phase1_baseline['metrics']['sharpe']:>10.3f} | {phase1_skip['metrics']['sharpe']:>10.3f} | {'YES' if phase1_skip['metrics']['sharpe'] > phase1_baseline['metrics']['sharpe'] else 'NO':>9} |
| High-vol Regime Sharpe      | {high_vol_baseline:>10.3f} | {high_vol_skip:>10.3f} | {'YES' if high_vol_skip > high_vol_baseline else 'NO':>9} |
| 2020 Return                 | {y2020_baseline:>10.2%} | {y2020_skip:>10.2%} | {'YES' if y2020_skip > y2020_baseline else 'NO':>9} |
| Stress Corr w/ Trend        | {corr_dd_baseline:>10.3f} | {corr_dd_skip:>10.3f} | {'YES' if corr_dd_skip < corr_dd_baseline else 'NO':>9} |
| Longest Underwater (days)   | {phase1_baseline['metrics']['longest_underwater_days']:>10.0f} | {phase1_skip['metrics']['longest_underwater_days']:>10.0f} | {'YES' if phase1_skip['metrics']['longest_underwater_days'] < phase1_baseline['metrics']['longest_underwater_days'] else 'NO':>9} |
""")


def save_results(status: str, p0_base, p0_skip, p1_base, p1_skip):
    """Save results to output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"CSMOM_skip_v1_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "status": status,
        "timestamp": timestamp,
        "hypothesis_id": "CSMOM-H001",
        "variant_name": "CSMOM_skip_v1",
        "eval_window": {"start": EVAL_START, "end": EVAL_END},
        "phase0": {
            "baseline": p0_base,
            "variant": p0_skip,
        },
    }
    
    if p1_base and p1_skip:
        results["phase1"] = {
            "baseline": p1_base,
            "variant": p1_skip,
        }
    
    # Save JSON
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save markdown memo
    memo_path = out_dir / "CSMOM_skip_v1_phase01_results.md"
    with open(memo_path, "w") as f:
        f.write(f"# CSMOM_skip_v1 Phase 0/1 Results\n\n")
        f.write(f"**Hypothesis ID:** CSMOM-H001\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Status:** {status}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Baseline skips: {BASELINE_CONFIG['skips']}\n")
        f.write(f"- Variant skips: {SKIP_V1_CONFIG['skips']}\n\n")
        f.write(f"## Results saved to: {out_dir}\n")
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
