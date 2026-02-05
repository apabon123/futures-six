#!/usr/bin/env python3
"""
Curve RV Belief Autopsy — Phase 4 Engine Research

This script performs a comprehensive diagnostic of the Curve RV Meta-Sleeve
as an ensemble belief object at Post-Construction.

Research Protocol Compliance:
- Step 1: Belief Autopsy (No Code Changes)
- Purpose: Diagnose Curve RV Meta-Sleeve structural properties

Curve RV Meta-Sleeve Composition (Core v9):
- Rank Fly Momentum: 62.5% (5% of 8% total)
- Pack Slope Momentum: 37.5% (3% of 8% total)
- Pack Curvature: PARKED (redundant with Rank Fly, 0.91 signal correlation)

Methodology:
- Load existing Phase-1 diagnostics for each atomic sleeve
- Compute ensemble metrics by combining atomic returns
- Analyze correlations and attribution
- Classify structural failure modes

Author: Phase 4 Engine Research
Date: January 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

EVAL_START = "2020-01-01"
EVAL_END = "2025-10-31"

PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE1_DIR = PROJECT_ROOT / "reports" / "runs" / "rates_curve_rv" / "sr3_curve_rv_momentum_phase1" / "20251217_134842"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase4_research" / "curve_rv_autopsy"

# Curve RV atomic sleeves (promoted in Core v9)
CURVE_RV_ATOMICS = ["rank_fly", "pack_slope"]

# Core v9 weights (within Curve RV Meta-Sleeve)
CORE_V9_WEIGHTS = {
    "rank_fly": 0.625,   # 5% / 8% = 62.5%
    "pack_slope": 0.375, # 3% / 8% = 37.5%
}

# Phase-1 standalone metrics (from meta.json)
PHASE1_METRICS = {
    "rank_fly": {
        "sharpe": 1.1908,
        "cagr": 0.1510,
        "max_dd": -0.2748,
        "vol": 0.1268,
        "hit_rate": 0.4544,
        "n_days": 1765,
    },
    "pack_slope": {
        "sharpe": 0.2837,
        "cagr": 0.0323,
        "max_dd": -0.2075,
        "vol": 0.1138,
        "hit_rate": 0.4811,
        "n_days": 1773,
    },
    "pack_curvature": {  # PARKED - for reference only
        "sharpe": 0.3925,
        "cagr": 0.0291,
        "max_dd": -0.2170,
        "vol": 0.0741,
        "hit_rate": 0.4555,
        "n_days": 1785,
        "status": "PARKED",
        "reason": "Redundant with Rank Fly (0.91 signal correlation)",
    },
}


def load_phase1_returns() -> Dict[str, pd.Series]:
    """Load Phase-1 portfolio returns for each atomic sleeve."""
    returns = {}
    
    for atomic in CURVE_RV_ATOMICS:
        path = PHASE1_DIR / f"{atomic}_portfolio_returns.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Get the returns column
            if 'portfolio_return' in df.columns:
                returns[atomic] = df['portfolio_return']
            elif 'pnl' in df.columns:
                returns[atomic] = df['pnl']
            else:
                returns[atomic] = df.iloc[:, 0]
            print(f"  Loaded {atomic}: {len(returns[atomic])} rows")
    
    return returns


def compute_ensemble_returns(atomic_returns: Dict[str, pd.Series]) -> pd.Series:
    """Compute weighted ensemble returns from atomic returns."""
    # Align to common dates
    returns_df = pd.DataFrame(atomic_returns)
    returns_df = returns_df.dropna()
    
    # Apply weights
    ensemble = pd.Series(0.0, index=returns_df.index)
    for atomic in returns_df.columns:
        weight = CORE_V9_WEIGHTS.get(atomic, 0.0)
        ensemble += weight * returns_df[atomic]
    
    return ensemble, returns_df


def compute_metrics(returns: pd.Series) -> dict:
    """Compute standard performance metrics."""
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
    
    # Longest underwater
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
    longest_uw = max((end - start).days for start, end in underwater_periods) if underwater_periods else 0
    
    return {
        "n_days": len(returns),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "longest_underwater_days": longest_uw,
        "hit_rate": (returns > 0).mean(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }


def compute_attribution(atomic_returns: pd.DataFrame, ensemble: pd.Series) -> dict:
    """Compute attribution metrics for each atomic sleeve."""
    attribution = {}
    total_pnl = ensemble.sum()
    
    for atomic in atomic_returns.columns:
        weight = CORE_V9_WEIGHTS.get(atomic, 0.0)
        weighted_rets = atomic_returns[atomic] * weight
        
        contrib = weighted_rets.sum()
        contrib_pct = contrib / total_pnl if abs(total_pnl) > 1e-10 else 0
        
        ann_ret = weighted_rets.mean() * 252
        ann_vol = weighted_rets.std() * np.sqrt(252)
        contrib_sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # Standalone metrics
        standalone_ret = atomic_returns[atomic].mean() * 252
        standalone_vol = atomic_returns[atomic].std() * np.sqrt(252)
        standalone_sharpe = standalone_ret / standalone_vol if standalone_vol > 0 else 0
        
        attribution[atomic] = {
            "weight": weight,
            "contribution_pct": contrib_pct,
            "contribution_sharpe": contrib_sharpe,
            "standalone_sharpe": standalone_sharpe,
            "correlation_with_ensemble": weighted_rets.corr(ensemble),
        }
    
    return attribution


def compute_yearly(ensemble: pd.Series) -> pd.DataFrame:
    """Compute yearly performance."""
    results = []
    for year in ensemble.index.year.unique():
        year_mask = ensemble.index.year == year
        year_rets = ensemble[year_mask]
        row = {"year": year}
        row["return"] = (1 + year_rets).prod() - 1
        row["vol"] = year_rets.std() * np.sqrt(252)
        row["sharpe"] = (year_rets.mean() * 252) / row["vol"] if row["vol"] > 0 else 0
        results.append(row)
    return pd.DataFrame(results)


def compute_regime_metrics(ensemble: pd.Series) -> dict:
    """Compute regime-conditioned metrics."""
    rolling_vol = ensemble.rolling(21).std() * np.sqrt(252)
    vol_25 = rolling_vol.quantile(0.25)
    vol_75 = rolling_vol.quantile(0.75)
    
    regimes = {}
    for name, mask in [("low_vol", rolling_vol < vol_25), 
                       ("mid_vol", (rolling_vol >= vol_25) & (rolling_vol <= vol_75)),
                       ("high_vol", rolling_vol > vol_75)]:
        regime_rets = ensemble[mask]
        if len(regime_rets) > 20:
            ann_ret = regime_rets.mean() * 252
            ann_vol = regime_rets.std() * np.sqrt(252)
            regimes[name] = {
                "n_days": len(regime_rets),
                "ann_return": ann_ret,
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
            }
    return regimes


def analyze_crisis_periods(ensemble: pd.Series) -> dict:
    """Analyze performance during specific crisis periods."""
    crisis_periods = {
        "2020_Q1_COVID": ("2020-02-19", "2020-03-23"),
        "2022_Rate_Hikes": ("2022-01-01", "2022-12-31"),
        "2023_Banking_Crisis": ("2023-03-01", "2023-05-31"),
    }
    
    results = {}
    for period_name, (start, end) in crisis_periods.items():
        try:
            period_mask = (ensemble.index >= start) & (ensemble.index <= end)
            period_rets = ensemble[period_mask]
            
            if len(period_rets) > 5:
                cum_ret = (1 + period_rets).prod() - 1
                ann_vol = period_rets.std() * np.sqrt(252)
                ann_ret = period_rets.mean() * 252
                
                results[period_name] = {
                    "n_days": len(period_rets),
                    "total_return": cum_ret,
                    "ann_vol": ann_vol,
                    "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
                    "hit_rate": (period_rets > 0).mean(),
                }
        except Exception as e:
            results[period_name] = {"error": str(e)}
    
    return results


def classify_failure_modes(metrics: dict) -> dict:
    """Classify structural failure modes."""
    ensemble = metrics['ensemble']
    attribution = metrics['attribution']
    
    failures = {}
    
    # 1. Negative Sharpe
    failures["negative_sharpe"] = {
        "status": "FAIL" if ensemble['sharpe'] < 0 else "PASS",
        "evidence": f"Sharpe = {ensemble['sharpe']:.3f}",
    }
    
    # 2. Near-zero Sharpe (< 0.2)
    failures["near_zero_sharpe"] = {
        "status": "FAIL" if ensemble['sharpe'] < 0.2 else "PASS",
        "evidence": f"Sharpe = {ensemble['sharpe']:.3f}",
    }
    
    # 3. Single-component dominance (>80% contribution)
    max_contrib = max(a['contribution_pct'] for a in attribution.values())
    max_contrib_name = max(attribution.keys(), key=lambda k: attribution[k]['contribution_pct'])
    failures["single_component_dominance"] = {
        "status": "FAIL" if max_contrib > 0.80 else "PASS",
        "evidence": f"{max_contrib_name} contributes {max_contrib:.1%}",
    }
    
    # 4. Negative contributor present
    negative_contribs = [k for k, v in attribution.items() if v['contribution_sharpe'] < 0]
    failures["negative_contributor"] = {
        "status": "FAIL" if negative_contribs else "PASS",
        "evidence": f"Negative contributors: {negative_contribs}" if negative_contribs else "None",
    }
    
    # 5. Regime fragility (negative Sharpe in any regime)
    regime_failures = [r for r, s in metrics['regimes'].items() if s['sharpe'] < 0]
    failures["regime_fragility"] = {
        "status": "FAIL" if regime_failures else "PASS",
        "evidence": f"Negative Sharpe in: {regime_failures}" if regime_failures else "All regimes positive",
    }
    
    # 6. Hidden tail risk (MaxDD > 30%)
    failures["hidden_tail_risk"] = {
        "status": "FAIL" if ensemble['max_drawdown'] < -0.30 else "PASS",
        "evidence": f"MaxDD = {ensemble['max_drawdown']:.2%}",
    }
    
    return failures


def generate_memo(metrics: dict) -> str:
    """Generate the Curve RV autopsy diagnostic memo."""
    memo = f"""# Curve RV Meta-Sleeve Belief Autopsy — Phase 4 Structural Analysis

**Generated:** {datetime.now().isoformat()}
**Research Protocol:** Phase 4 Engine Research, Step 1 (No Code Changes)

---

## Executive Summary

Curve RV Meta-Sleeve ensemble **{"PASSES" if metrics['ensemble']['sharpe'] >= 0.40 else "FAILS"}** 
the Phase 4 benchmark (Sharpe >= 0.40).

---

## 1. Individual Atomic Sleeve Phase-1 Results

| Atomic Sleeve | Sharpe | CAGR | MaxDD | Vol | Weight |
|---------------|--------|------|-------|-----|--------|
"""
    for atomic, pm in PHASE1_METRICS.items():
        if atomic in CURVE_RV_ATOMICS:
            weight = CORE_V9_WEIGHTS.get(atomic, 0)
            memo += f"| {atomic.replace('_', ' ').title()} | {pm['sharpe']:.3f} | {pm['cagr']:.2%} | {pm['max_dd']:.2%} | {pm['vol']:.2%} | {weight:.1%} |\n"
    
    # Add parked sleeve for reference
    pm = PHASE1_METRICS['pack_curvature']
    memo += f"| Pack Curvature (PARKED) | {pm['sharpe']:.3f} | {pm['cagr']:.2%} | {pm['max_dd']:.2%} | {pm['vol']:.2%} | 0% |\n"
    
    memo += f"""
**Key Observations:**
- **Rank Fly dominates** with Sharpe 1.19 — excellent standalone performer
- **Pack Slope is weak** with Sharpe 0.28 — borderline acceptable
- **Pack Curvature PARKED** — 0.91 signal correlation with Rank Fly (redundant)

---

## 2. Weighted Ensemble Metrics

| Metric | Value |
|--------|-------|
| N Days | {metrics['ensemble']['n_days']} |
| Annualized Return | {metrics['ensemble']['annualized_return']:.2%} |
| Annualized Vol | {metrics['ensemble']['annualized_vol']:.2%} |
| **Sharpe Ratio** | **{metrics['ensemble']['sharpe']:.3f}** |
| CAGR | {metrics['ensemble']['cagr']:.2%} |
| Max Drawdown | {metrics['ensemble']['max_drawdown']:.2%} |
| Calmar Ratio | {metrics['ensemble']['calmar']:.3f} |
| Longest Underwater | {metrics['ensemble']['longest_underwater_days']} days |
| Hit Rate | {metrics['ensemble']['hit_rate']:.2%} |
| Skewness | {metrics['ensemble']['skewness']:.3f} |
| Kurtosis | {metrics['ensemble']['kurtosis']:.3f} |

**Phase 4 Benchmark:**
- Sharpe >= 0.40: {"PASS" if metrics['ensemble']['sharpe'] >= 0.40 else "FAIL"}
- Sharpe >= 0.50: {"PASS" if metrics['ensemble']['sharpe'] >= 0.50 else "FAIL"}

---

## 3. Atomic Sleeve Attribution

| Atomic | Weight | Contrib % | Contrib Sharpe | Standalone Sharpe | Corr w/ Ensemble |
|--------|--------|----------:|---------------:|------------------:|-----------------:|
"""
    for atomic, attr in metrics['attribution'].items():
        memo += f"| {atomic.replace('_', ' ').title()} | {attr['weight']:.1%} | {attr['contribution_pct']:.1%} | {attr['contribution_sharpe']:.3f} | {attr['standalone_sharpe']:.3f} | {attr['correlation_with_ensemble']:.3f} |\n"
    
    # Dominance analysis
    contrib_pcts = {k: v['contribution_pct'] for k, v in metrics['attribution'].items()}
    max_contrib = max(contrib_pcts.values())
    max_contrib_name = max(contrib_pcts.keys(), key=lambda k: contrib_pcts[k])
    
    if max_contrib > 0.80:
        memo += f"\n**WARNING: DOMINANCE** — {max_contrib_name} contributes {max_contrib:.1%}\n"
    else:
        memo += f"\n**Diversified** — Max contribution: {max_contrib:.1%} ({max_contrib_name})\n"
    
    memo += f"""
---

## 4. Correlation Matrix

"""
    corr = metrics['correlation_matrix']
    memo += "| | " + " | ".join([c.replace("_", " ").upper() for c in corr.columns]) + " |\n"
    memo += "|" + "|".join(["---"] * (len(corr.columns) + 1)) + "|\n"
    for idx in corr.index:
        row_name = idx.replace("_", " ").upper()
        row_vals = " | ".join([f"{corr.loc[idx, c]:.3f}" for c in corr.columns])
        memo += f"| {row_name} | {row_vals} |\n"
    
    memo += f"""
---

## 5. Yearly Performance

| Year | Return | Vol | Sharpe |
|------|-------:|----:|-------:|
"""
    for _, row in metrics['yearly'].iterrows():
        memo += f"| {int(row['year'])} | {row['return']:.2%} | {row['vol']:.2%} | {row['sharpe']:.2f} |\n"
    
    memo += f"""
---

## 6. Regime-Conditioned Performance

"""
    for regime, stats in metrics['regimes'].items():
        regime_name = regime.replace('_', ' ').title()
        memo += f"**{regime_name}:** N={stats['n_days']}, Sharpe={stats['sharpe']:.3f}, Return={stats['ann_return']:.2%}\n\n"
    
    memo += f"""
---

## 7. Crisis Period Analysis

"""
    for period, stats in metrics['crisis'].items():
        if 'error' not in stats:
            period_name = period.replace('_', ' ')
            memo += f"**{period_name}:** Return={stats['total_return']:.2%}, Sharpe={stats['sharpe']:.3f}, N={stats['n_days']}\n\n"
    
    memo += f"""
---

## 8. Structural Failure Classification

| Failure Mode | Status | Evidence |
|--------------|--------|----------|
"""
    for mode, result in metrics['failures'].items():
        mode_name = mode.replace('_', ' ').title()
        status_icon = "PASS" if result['status'] == "PASS" else "FAIL"
        memo += f"| {mode_name} | {status_icon} | {result['evidence']} |\n"
    
    # Count failures
    n_failures = sum(1 for r in metrics['failures'].values() if r['status'] == "FAIL")
    
    memo += f"""
---

## 9. Overall Assessment

"""
    
    ensemble_sharpe = metrics['ensemble']['sharpe']
    
    if ensemble_sharpe >= 0.50 and n_failures == 0:
        memo += """**PASS — Curve RV Meta-Sleeve is structurally sound**

The Curve RV Meta-Sleeve demonstrates:
- Sharpe meeting target (>= 0.50)
- No structural failure modes detected
- Rank Fly is the dominant performer (as expected)

**Recommendation:** Curve RV passes Phase 4 structural assessment.
"""
    elif ensemble_sharpe >= 0.40:
        memo += f"""**ACCEPTABLE — Curve RV Meta-Sleeve meets minimum benchmark**

The Curve RV Meta-Sleeve demonstrates:
- Sharpe meeting minimum (>= 0.40)
- {n_failures} structural concern(s) detected

**Recommendation:** Log concerns for future research. Consider:
"""
        # Check for specific issues
        if metrics['failures']['single_component_dominance']['status'] == "FAIL":
            memo += "- Rank Fly dominates — Pack Slope may be adding noise\n"
        if metrics['failures']['negative_contributor']['status'] == "FAIL":
            memo += "- Negative contributor present — consider removal\n"
    else:
        memo += f"""**FAIL — Curve RV Meta-Sleeve has structural issues**

The Curve RV Meta-Sleeve demonstrates:
- Sharpe below minimum threshold
- {n_failures} structural failure(s) detected

**Recommendation:** Further investigation required.
"""
    
    memo += f"""
---

## 10. Appendix: Data Lineage

- **Phase-1 Data Source:** `{PHASE1_DIR}`
- **Eval Window:** {EVAL_START} to {EVAL_END}
- **Atomic Sleeves:**
  - Rank Fly Momentum: PROMOTED (62.5% weight)
  - Pack Slope Momentum: PROMOTED (37.5% weight)
  - Pack Curvature Momentum: PARKED (redundant)

**NO CODE WAS CHANGED. This is pure measurement.**

---

*End of Curve RV Belief Autopsy*
"""
    
    return memo


def main():
    print("=" * 70)
    print("CURVE RV BELIEF AUTOPSY — Phase 4 Engine Research")
    print("=" * 70)
    
    # Load Phase-1 returns
    print("\n[1/6] Loading Phase-1 returns...")
    atomic_returns = load_phase1_returns()
    
    if not atomic_returns:
        print("ERROR: No atomic returns found")
        return
    
    # Compute ensemble
    print("\n[2/6] Computing ensemble metrics...")
    ensemble, returns_df = compute_ensemble_returns(atomic_returns)
    ensemble_metrics = compute_metrics(ensemble)
    print(f"  Ensemble Sharpe: {ensemble_metrics['sharpe']:.3f}")
    print(f"  Ensemble CAGR: {ensemble_metrics['cagr']:.2%}")
    print(f"  Ensemble MaxDD: {ensemble_metrics['max_drawdown']:.2%}")
    
    # Attribution
    print("\n[3/6] Computing attribution...")
    attribution = compute_attribution(returns_df, ensemble)
    for atomic, attr in attribution.items():
        print(f"  {atomic}: {attr['contribution_pct']:.1%} contrib, Sharpe {attr['contribution_sharpe']:.3f}")
    
    # Correlation
    print("\n[4/6] Computing correlations...")
    corr_matrix = returns_df.corr()
    print(corr_matrix.to_string())
    
    # Yearly
    print("\n[5/6] Computing yearly performance...")
    yearly = compute_yearly(ensemble)
    
    # Regimes and crisis
    print("\n[6/6] Computing regime and crisis metrics...")
    regimes = compute_regime_metrics(ensemble)
    crisis = analyze_crisis_periods(ensemble)
    
    # Assemble metrics
    metrics = {
        "phase1_metrics": PHASE1_METRICS,
        "ensemble": ensemble_metrics,
        "attribution": attribution,
        "correlation_matrix": corr_matrix,
        "yearly": yearly,
        "regimes": regimes,
        "crisis": crisis,
        "weights": CORE_V9_WEIGHTS,
    }
    
    # Classify failures
    metrics['failures'] = classify_failure_modes(metrics)
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    memo = generate_memo(metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memo_path = OUTPUT_DIR / f"curve_rv_belief_autopsy_{timestamp}.md"
    with open(memo_path, "w", encoding="utf-8") as f:
        f.write(memo)
    
    # Save JSON metrics
    json_path = OUTPUT_DIR / f"curve_rv_autopsy_metrics_{timestamp}.json"
    metrics_json = {
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "phase1_metrics": PHASE1_METRICS,
        "ensemble": ensemble_metrics,
        "attribution": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                           for kk, vv in v.items()} for k, v in attribution.items()},
        "correlation_matrix": corr_matrix.to_dict(),
        "yearly": yearly.to_dict(),
        "regimes": regimes,
        "crisis": crisis,
        "weights": CORE_V9_WEIGHTS,
        "failures": metrics['failures'],
    }
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    
    print(f"\nReport: {memo_path}")
    print(f"Metrics: {json_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
INDIVIDUAL ATOMIC SHARPES (Phase-1):
  Rank Fly: {PHASE1_METRICS['rank_fly']['sharpe']:.3f}
  Pack Slope: {PHASE1_METRICS['pack_slope']['sharpe']:.3f}

WEIGHTED ENSEMBLE:
  Sharpe: {ensemble_metrics['sharpe']:.3f}
  CAGR: {ensemble_metrics['cagr']:.2%}
  MaxDD: {ensemble_metrics['max_drawdown']:.2%}
  Calmar: {ensemble_metrics['calmar']:.3f}

PHASE 4 BENCHMARK:
  Sharpe >= 0.40: {"PASS" if ensemble_metrics['sharpe'] >= 0.40 else "FAIL"}
  Sharpe >= 0.50: {"PASS" if ensemble_metrics['sharpe'] >= 0.50 else "FAIL"}

STRUCTURAL FAILURES: {sum(1 for r in metrics['failures'].values() if r['status'] == 'FAIL')}
""")
    
    return {
        "sharpe": ensemble_metrics['sharpe'],
        "cagr": ensemble_metrics['cagr'],
        "max_dd": ensemble_metrics['max_drawdown'],
        "pass_phase4": ensemble_metrics['sharpe'] >= 0.40,
        "n_failures": sum(1 for r in metrics['failures'].values() if r['status'] == 'FAIL'),
    }


if __name__ == "__main__":
    main()
