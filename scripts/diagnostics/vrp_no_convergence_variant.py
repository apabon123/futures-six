#!/usr/bin/env python3
"""
VRP No-Convergence Variant — Phase 4 Engine Research

Hypothesis: VRP-Convergence is a net-negative belief component
            and should be removed from the VRP Meta-Sleeve.

Baseline: VRP ensemble with 3 sub-sleeves (Sharpe 0.469)
Variant:  VRP ensemble with 2 sub-sleeves (Core + Alt only)

This script:
1. Loads existing Phase-1 timeseries for VRP-Core and VRP-Alt
2. Computes variant ensemble metrics (no Convergence)
3. Compares to baseline
4. Outputs promotion decision

Author: Phase 4 Engine Research
Date: January 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase4_research" / "vrp_autopsy"

# Phase-1 paths (same as baseline autopsy)
VRP_PHASE1_PATHS = {
    "vrp_core_meta": DIAGNOSTICS_DIR / "vrp_core_phase1" / "20251209_214729",
    "vrp_alt_meta": DIAGNOSTICS_DIR / "vrp_alt_phase1" / "20251213_123417",
}

# Baseline weights (3-sleeve)
BASELINE_WEIGHTS = {
    "vrp_core_meta": 0.30,
    "vrp_convergence_meta": 0.10,
    "vrp_alt_meta": 0.60,
}

# Variant weights (2-sleeve, renormalized)
# Core: 30 / 90 = 0.333, Alt: 60 / 90 = 0.667
VARIANT_WEIGHTS = {
    "vrp_core_meta": 0.333,
    "vrp_alt_meta": 0.667,
}

# Baseline metrics (from autopsy)
BASELINE_METRICS = {
    "sharpe": 0.469,
    "cagr": 0.0454,
    "max_drawdown": -0.1195,
    "calmar": 0.38,
}


def load_phase1_timeseries():
    """Load Phase-1 timeseries for Core and Alt only."""
    timeseries = {}
    for sleeve, path in VRP_PHASE1_PATHS.items():
        returns_path = path / "portfolio_returns.csv"
        if returns_path.exists():
            df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            timeseries[sleeve] = df
            print(f"  Loaded {sleeve}: {len(df)} rows")
    return timeseries


def compute_variant_ensemble(timeseries):
    """Compute variant ensemble returns (Core + Alt only)."""
    returns_dict = {}
    for sleeve, df in timeseries.items():
        if 'pnl' in df.columns:
            returns_dict[sleeve] = df['pnl']
        elif 'return' in df.columns:
            returns_dict[sleeve] = df['return']
        else:
            returns_dict[sleeve] = df.iloc[:, 0]
    
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()
    
    # Apply variant weights
    weighted_returns = pd.Series(0.0, index=returns_df.index)
    for sleeve in returns_df.columns:
        weight = VARIANT_WEIGHTS.get(sleeve, 0.0)
        weighted_returns += weight * returns_df[sleeve]
    
    return weighted_returns, returns_df


def compute_metrics(returns):
    """Compute standard metrics."""
    returns = returns.dropna()
    if len(returns) == 0:
        return {"error": "No valid returns"}
    
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum_ret = (1 + returns).cumprod()
    n_years = len(returns) / 252
    cagr = (cum_ret.iloc[-1]) ** (1/n_years) - 1 if n_years > 0 else 0
    
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "n_days": len(returns),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": (returns > 0).mean(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }


def compute_yearly(returns):
    """Compute yearly returns."""
    results = []
    for year in returns.index.year.unique():
        year_mask = returns.index.year == year
        year_rets = returns[year_mask]
        row = {"year": year}
        row["return"] = (1 + year_rets).prod() - 1
        row["vol"] = year_rets.std() * np.sqrt(252)
        row["sharpe"] = (year_rets.mean() * 252) / row["vol"] if row["vol"] > 0 else 0
        results.append(row)
    return pd.DataFrame(results)


def compute_regime_metrics(returns):
    """Compute regime-conditioned metrics."""
    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    vol_25 = rolling_vol.quantile(0.25)
    vol_75 = rolling_vol.quantile(0.75)
    
    regimes = {}
    for name, mask in [("low_vol", rolling_vol < vol_25), 
                       ("mid_vol", (rolling_vol >= vol_25) & (rolling_vol <= vol_75)),
                       ("high_vol", rolling_vol > vol_75)]:
        regime_rets = returns[mask]
        if len(regime_rets) > 20:
            ann_ret = regime_rets.mean() * 252
            ann_vol = regime_rets.std() * np.sqrt(252)
            regimes[name] = {
                "n_days": len(regime_rets),
                "ann_return": ann_ret,
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
            }
    return regimes


def compute_attribution(sub_returns, ensemble):
    """Compute attribution for each sub-sleeve."""
    total_pnl = ensemble.sum()
    attribution = {}
    for sleeve in sub_returns.columns:
        weight = VARIANT_WEIGHTS.get(sleeve, 0.0)
        weighted_rets = sub_returns[sleeve] * weight
        contrib = weighted_rets.sum()
        contrib_pct = contrib / total_pnl if abs(total_pnl) > 1e-10 else 0
        ann_ret = weighted_rets.mean() * 252
        ann_vol = weighted_rets.std() * np.sqrt(252)
        attribution[sleeve] = {
            "weight": weight,
            "contribution_pct": contrib_pct,
            "contribution_sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
            "correlation_with_ensemble": weighted_rets.corr(ensemble),
        }
    return attribution


def main():
    print("=" * 70)
    print("VRP NO-CONVERGENCE VARIANT — Phase 4 Engine Research")
    print("=" * 70)
    print("\nHypothesis: Removing VRP-Convergence improves ensemble Sharpe")
    print(f"Baseline Sharpe: {BASELINE_METRICS['sharpe']:.3f}")
    print(f"Baseline MaxDD: {BASELINE_METRICS['max_drawdown']:.2%}")
    
    # Load timeseries
    print("\n[1/5] Loading Phase-1 timeseries (Core + Alt only)...")
    timeseries = load_phase1_timeseries()
    
    # Compute variant ensemble
    print("\n[2/5] Computing variant ensemble...")
    variant_returns, sub_returns = compute_variant_ensemble(timeseries)
    variant_metrics = compute_metrics(variant_returns)
    
    print(f"  Variant Sharpe: {variant_metrics['sharpe']:.3f}")
    print(f"  Variant CAGR: {variant_metrics['cagr']:.2%}")
    print(f"  Variant MaxDD: {variant_metrics['max_drawdown']:.2%}")
    
    # Compute attribution
    print("\n[3/5] Computing attribution...")
    attribution = compute_attribution(sub_returns, variant_returns)
    for sleeve, attr in attribution.items():
        print(f"  {sleeve}: {attr['contribution_pct']:.1%} contrib, Sharpe {attr['contribution_sharpe']:.3f}")
    
    # Compute yearly
    print("\n[4/5] Computing yearly breakdown...")
    yearly = compute_yearly(variant_returns)
    
    # Compute regimes
    print("\n[5/5] Computing regime metrics...")
    regimes = compute_regime_metrics(variant_returns)
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: BASELINE vs VARIANT")
    print("=" * 70)
    
    sharpe_delta = variant_metrics['sharpe'] - BASELINE_METRICS['sharpe']
    cagr_delta = variant_metrics['cagr'] - BASELINE_METRICS['cagr']
    maxdd_delta = variant_metrics['max_drawdown'] - BASELINE_METRICS['max_drawdown']
    calmar_delta = variant_metrics['calmar'] - BASELINE_METRICS['calmar']
    
    print(f"""
| Metric | Baseline | Variant | Delta |
|--------|----------|---------|-------|
| Sharpe | {BASELINE_METRICS['sharpe']:.3f} | {variant_metrics['sharpe']:.3f} | {sharpe_delta:+.3f} |
| CAGR | {BASELINE_METRICS['cagr']:.2%} | {variant_metrics['cagr']:.2%} | {cagr_delta:+.2%} |
| MaxDD | {BASELINE_METRICS['max_drawdown']:.2%} | {variant_metrics['max_drawdown']:.2%} | {maxdd_delta:+.2%} |
| Calmar | {BASELINE_METRICS['calmar']:.3f} | {variant_metrics['calmar']:.3f} | {calmar_delta:+.3f} |
""")
    
    print("Yearly Performance (Variant):")
    for _, row in yearly.iterrows():
        print(f"  {int(row['year'])}: Return {row['return']:.2%}, Sharpe {row['sharpe']:.2f}")
    
    print("\nRegime Performance (Variant):")
    for regime, stats in regimes.items():
        print(f"  {regime}: Sharpe {stats['sharpe']:.3f}")
    
    # Promotion decision
    print("\n" + "=" * 70)
    print("PROMOTION DECISION")
    print("=" * 70)
    
    sharpe_improved = sharpe_delta > 0.03  # Meaningful improvement
    sharpe_target = variant_metrics['sharpe'] >= 0.50
    tail_ok = variant_metrics['max_drawdown'] >= BASELINE_METRICS['max_drawdown'] - 0.05  # No worse than 5% more DD
    
    print(f"""
Criteria:
1. Sharpe improved meaningfully (>+0.03): {"PASS" if sharpe_improved else "FAIL"} ({sharpe_delta:+.3f})
2. Sharpe meets target (>=0.50): {"PASS" if sharpe_target else "FAIL"} ({variant_metrics['sharpe']:.3f})
3. Tail behavior acceptable: {"PASS" if tail_ok else "FAIL"} (MaxDD {variant_metrics['max_drawdown']:.2%})

VERDICT: {"PROMOTE VRP_v2 (Core + Alt)" if sharpe_improved and tail_ok else "DO NOT PROMOTE"}
""")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "hypothesis": "Remove VRP-Convergence",
        "baseline_weights": BASELINE_WEIGHTS,
        "variant_weights": VARIANT_WEIGHTS,
        "baseline_metrics": BASELINE_METRICS,
        "variant_metrics": variant_metrics,
        "delta": {
            "sharpe": sharpe_delta,
            "cagr": cagr_delta,
            "max_drawdown": maxdd_delta,
            "calmar": calmar_delta,
        },
        "yearly": yearly.to_dict(),
        "regimes": regimes,
        "promotion_decision": {
            "sharpe_improved": sharpe_improved,
            "sharpe_target_met": sharpe_target,
            "tail_ok": tail_ok,
            "promote": sharpe_improved and tail_ok,
        },
    }
    
    json_path = OUTPUT_DIR / f"vrp_no_convergence_variant_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")
    
    return results


if __name__ == "__main__":
    main()
