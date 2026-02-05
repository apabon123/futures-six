#!/usr/bin/env python3
"""
VRP Belief Autopsy — Phase 4 Engine Research

This script performs a comprehensive diagnostic of the VRP Meta-Sleeve
as an ensemble belief object at Post-Construction.

Research Protocol Compliance:
- Step 1: Belief Autopsy (No Code Changes)
- Purpose: Diagnose VRP Meta-Sleeve structural properties

Methodology:
- Load existing Phase-1 diagnostics for each VRP sub-sleeve
- Load timeseries data from Phase-1 runs
- Compute ensemble metrics by combining sub-sleeve returns
- Analyze correlations and attribution

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
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
PHASE_INDEX_DIR = PROJECT_ROOT / "reports" / "phase_index" / "vrp"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase4_research" / "vrp_autopsy"

# VRP sub-sleeves Phase-1 paths
VRP_PHASE1_PATHS = {
    "vrp_core_meta": DIAGNOSTICS_DIR / "vrp_core_phase1" / "20251209_214729",
    "vrp_convergence_meta": DIAGNOSTICS_DIR / "vrp_convergence_phase1" / "20251211_092647",
    "vrp_alt_meta": DIAGNOSTICS_DIR / "vrp_alt_phase1" / "20251213_123417",
}

VRP_PHASE1_METRICS = {
    "vrp_core_meta": PHASE_INDEX_DIR / "vrp_core_phase1.txt",
    "vrp_convergence_meta": PHASE_INDEX_DIR / "vrp_convergence_phase1.txt",
    "vrp_alt_meta": PHASE_INDEX_DIR / "vrp_alt" / "phase1.txt",
}

CORE_V9_VRP_WEIGHTS = {
    "vrp_core_meta": 0.06555,
    "vrp_convergence_meta": 0.02185,
    "vrp_alt_meta": 0.1311,
}

TOTAL_VRP_WEIGHT = sum(CORE_V9_VRP_WEIGHTS.values())
VRP_NORMALIZED_WEIGHTS = {k: v / TOTAL_VRP_WEIGHT for k, v in CORE_V9_VRP_WEIGHTS.items()}


def load_phase1_metrics() -> Dict[str, Dict]:
    """Load Phase-1 metrics from phase_index files."""
    metrics = {}
    for sleeve, path in VRP_PHASE1_METRICS.items():
        if path.exists():
            with open(path, 'r') as f:
                content = f.read()
            sleeve_metrics = {}
            for line in content.strip().split('\n'):
                if line.startswith('#'):
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        sleeve_metrics[key] = float(value) if '.' in value else value
                    except ValueError:
                        sleeve_metrics[key] = value
            metrics[sleeve] = sleeve_metrics
    return metrics


def load_phase1_timeseries() -> Dict[str, pd.DataFrame]:
    """Load Phase-1 timeseries data for each VRP sub-sleeve."""
    timeseries = {}
    for sleeve, path in VRP_PHASE1_PATHS.items():
        returns_path = path / "portfolio_returns.csv"
        if returns_path.exists():
            df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            timeseries[sleeve] = df
            print(f"  Loaded {sleeve}: {len(df)} rows")
    return timeseries


def compute_vrp_ensemble_returns(timeseries: Dict[str, pd.DataFrame]):
    """Compute weighted VRP ensemble returns from sub-sleeve returns."""
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
    
    weighted_returns = pd.Series(0.0, index=returns_df.index)
    for sleeve in returns_df.columns:
        weight = VRP_NORMALIZED_WEIGHTS.get(sleeve, 0.0)
        weighted_returns += weight * returns_df[sleeve]
    
    return weighted_returns, returns_df


def compute_ensemble_metrics(returns: pd.Series) -> dict:
    """Compute standard metrics for a returns series."""
    returns = returns.dropna()
    if len(returns) == 0:
        return {"error": "No valid returns"}
    
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum_ret = (1 + returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    n_years = len(returns) / 252
    cagr = (cum_ret.iloc[-1]) ** (1/n_years) - 1 if n_years > 0 else 0
    
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
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
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "longest_underwater_days": longest_uw,
        "hit_rate": (returns > 0).mean(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "start_date": str(returns.index[0].date()) if hasattr(returns.index[0], 'date') else str(returns.index[0]),
        "end_date": str(returns.index[-1].date()) if hasattr(returns.index[-1], 'date') else str(returns.index[-1]),
    }


def compute_sub_sleeve_attribution(sub_sleeve_returns: pd.DataFrame, ensemble: pd.Series) -> dict:
    """Compute attribution metrics for each sub-sleeve."""
    attribution = {}
    total_ensemble_pnl = ensemble.sum()
    
    for sleeve in sub_sleeve_returns.columns:
        sleeve_rets = sub_sleeve_returns[sleeve]
        weighted_rets = sleeve_rets * VRP_NORMALIZED_WEIGHTS.get(sleeve, 0.0)
        
        total_contrib = weighted_rets.sum()
        contrib_pct = total_contrib / total_ensemble_pnl if abs(total_ensemble_pnl) > 1e-10 else 0
        
        ann_ret = weighted_rets.mean() * 252
        ann_vol = weighted_rets.std() * np.sqrt(252)
        contrib_sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        standalone_ann_ret = sleeve_rets.mean() * 252
        standalone_ann_vol = sleeve_rets.std() * np.sqrt(252)
        standalone_sharpe = standalone_ann_ret / standalone_ann_vol if standalone_ann_vol > 0 else 0
        
        corr_with_ensemble = weighted_rets.corr(ensemble)
        
        attribution[sleeve] = {
            "weight": VRP_NORMALIZED_WEIGHTS.get(sleeve, 0.0),
            "contribution_pct": contrib_pct,
            "contribution_sharpe": contrib_sharpe,
            "standalone_sharpe": standalone_sharpe,
            "correlation_with_ensemble": corr_with_ensemble,
        }
    
    return attribution


def compute_yearly_from_returns(ensemble: pd.Series, sub_sleeve_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly returns table."""
    results = []
    for year in ensemble.index.year.unique():
        year_mask = ensemble.index.year == year
        year_ensemble = ensemble[year_mask]
        row = {"year": year}
        row["vrp_return"] = (1 + year_ensemble).prod() - 1
        row["vrp_vol"] = year_ensemble.std() * np.sqrt(252)
        row["vrp_sharpe"] = (year_ensemble.mean() * 252) / row["vrp_vol"] if row["vrp_vol"] > 0 else 0
        results.append(row)
    return pd.DataFrame(results)


def compute_regime_from_returns(ensemble: pd.Series, sub_sleeve_returns: pd.DataFrame) -> dict:
    """Analyze behavior across volatility regimes."""
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
                "ann_vol": ann_vol,
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
                "hit_rate": (regime_rets > 0).mean()
            }
    return regimes


def generate_vrp_memo(metrics: dict) -> str:
    """Generate the VRP autopsy diagnostic memo."""
    memo = f"""# VRP Meta-Sleeve Belief Autopsy — Phase 4 Structural Analysis

**Generated:** {datetime.now().isoformat()}
**Research Protocol:** Phase 4 Engine Research, Step 1 (No Code Changes)

---

## Executive Summary

VRP Meta-Sleeve ensemble **{"PASSES" if metrics['ensemble']['sharpe'] >= 0.40 else "FAILS"}** 
the Phase 4 benchmark (Sharpe ≥ 0.40).

---

## 1. Individual Sub-Sleeve Phase-1 Results

| Sub-Sleeve | Sharpe | CAGR | MaxDD | Weight |
|------------|--------|------|-------|--------|
"""
    for sleeve, pm in metrics['phase1_metrics'].items():
        sleeve_short = sleeve.replace("_meta", "").replace("vrp_", "VRP-").title()
        weight = metrics['weights'].get(sleeve, 0)
        cagr_val = pm.get('cagr', 0)
        cagr_str = f"{cagr_val:.2%}" if isinstance(cagr_val, (int, float)) else str(cagr_val)
        memo += f"| {sleeve_short} | {pm.get('sharpe', 'N/A')} | {cagr_str} | {pm.get('max_dd', 'N/A')} | {weight:.1%} |\n"
    
    sharpes = {s: pm.get('sharpe', 0) for s, pm in metrics['phase1_metrics'].items()}
    best_sleeve = max(sharpes, key=sharpes.get)
    worst_sleeve = min(sharpes, key=sharpes.get)
    
    memo += f"""
**Best performer:** {best_sleeve.replace('_meta', '')} (Sharpe: {sharpes[best_sleeve]})
**Worst performer:** {worst_sleeve.replace('_meta', '')} (Sharpe: {sharpes[worst_sleeve]})

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
| Hit Rate | {metrics['ensemble']['hit_rate']:.2%} |

**Phase 4 Benchmark:**
- Sharpe ≥ 0.40: {"✅ PASS" if metrics['ensemble']['sharpe'] >= 0.40 else "❌ FAIL"}
- Sharpe ≥ 0.50: {"✅ PASS" if metrics['ensemble']['sharpe'] >= 0.50 else "❌ FAIL"}

---

## 3. Sub-Sleeve Attribution

| Sub-Sleeve | Weight | Contrib % | Contrib Sharpe | Standalone Sharpe | Corr w/ Ensemble |
|------------|--------|----------:|---------------:|------------------:|-----------------:|
"""
    for sleeve, attr in metrics['attribution'].items():
        sleeve_short = sleeve.replace("_meta", "").replace("vrp_", "VRP-").title()
        memo += f"| {sleeve_short} | {attr['weight']:.1%} | {attr['contribution_pct']:.1%} | {attr['contribution_sharpe']:.3f} | {attr['standalone_sharpe']:.3f} | {attr['correlation_with_ensemble']:.3f} |\n"
    
    contrib_pcts = {s: a['contribution_pct'] for s, a in metrics['attribution'].items()}
    max_contrib_sleeve = max(contrib_pcts, key=contrib_pcts.get)
    max_contrib_pct = contrib_pcts[max_contrib_sleeve]
    
    if max_contrib_pct > 0.80:
        memo += f"\n⚠️ **DOMINANCE**: {max_contrib_sleeve.replace('_meta', '')} contributes {max_contrib_pct:.1%}\n"
    else:
        memo += f"\n✅ **DIVERSIFIED**: No single sleeve dominates (max: {max_contrib_pct:.1%})\n"
    
    memo += f"""
---

## 4. Correlation Matrix

"""
    corr = metrics['correlation_matrix']
    memo += "| | " + " | ".join([c.replace("_meta", "").replace("vrp_", "").upper() for c in corr.columns]) + " |\n"
    memo += "|" + "|".join(["---"] * (len(corr.columns) + 1)) + "|\n"
    for idx in corr.index:
        row_name = idx.replace("_meta", "").replace("vrp_", "").upper()
        row_vals = " | ".join([f"{corr.loc[idx, c]:.3f}" for c in corr.columns])
        memo += f"| {row_name} | {row_vals} |\n"
    
    memo += f"""
---

## 5. Yearly Performance

| Year | VRP Return | VRP Vol | VRP Sharpe |
|------|----------:|--------:|----------:|
"""
    for _, row in metrics['yearly'].iterrows():
        memo += f"| {int(row['year'])} | {row['vrp_return']:.2%} | {row['vrp_vol']:.2%} | {row['vrp_sharpe']:.2f} |\n"
    
    memo += f"""
---

## 6. Overall Assessment

"""
    ensemble_sharpe = metrics['ensemble']['sharpe']
    if ensemble_sharpe >= 0.50:
        memo += "**✅ VRP Meta-Sleeve PASSES Phase 4**\n\nStructurally sound with acceptable Sharpe.\n"
    elif ensemble_sharpe >= 0.40:
        memo += "**⚠️ VRP Meta-Sleeve ACCEPTABLE**\n\nMeets minimum benchmark but has room for improvement.\n"
    else:
        memo += "**❌ VRP Meta-Sleeve FAILS Phase 4**\n\nSharpe below 0.40 threshold.\n"
    
    memo += f"""
---

*End of VRP Belief Autopsy*
"""
    return memo


def main():
    """Run the VRP belief autopsy."""
    print("=" * 70)
    print("VRP BELIEF AUTOPSY — Phase 4 Engine Research")
    print("=" * 70)
    
    # Load Phase-1 metrics
    print("\n[1/5] Loading Phase-1 metrics...")
    phase1_metrics = load_phase1_metrics()
    for sleeve, m in phase1_metrics.items():
        print(f"  {sleeve}: Sharpe={m.get('sharpe', 'N/A')}")
    
    # Load timeseries
    print("\n[2/5] Loading Phase-1 timeseries...")
    timeseries = load_phase1_timeseries()
    
    if not timeseries:
        print("ERROR: No timeseries data found")
        return
    
    # Compute ensemble
    print("\n[3/5] Computing ensemble metrics...")
    vrp_ensemble, sub_sleeve_returns = compute_vrp_ensemble_returns(timeseries)
    ensemble_metrics = compute_ensemble_metrics(vrp_ensemble)
    print(f"  Ensemble Sharpe: {ensemble_metrics['sharpe']:.3f}")
    print(f"  Ensemble CAGR: {ensemble_metrics['cagr']:.2%}")
    
    # Attribution
    print("\n[4/5] Computing attribution...")
    attribution = compute_sub_sleeve_attribution(sub_sleeve_returns, vrp_ensemble)
    corr_matrix = sub_sleeve_returns.corr()
    yearly = compute_yearly_from_returns(vrp_ensemble, sub_sleeve_returns)
    regimes = compute_regime_from_returns(vrp_ensemble, sub_sleeve_returns)
    
    # Generate report
    print("\n[5/5] Generating report...")
    metrics = {
        "phase1_metrics": phase1_metrics,
        "ensemble": ensemble_metrics,
        "attribution": attribution,
        "correlation_matrix": corr_matrix,
        "yearly": yearly,
        "regimes": regimes,
        "weights": VRP_NORMALIZED_WEIGHTS,
    }
    
    memo = generate_vrp_memo(metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memo_path = OUTPUT_DIR / f"vrp_belief_autopsy_{timestamp}.md"
    with open(memo_path, "w", encoding="utf-8") as f:
        f.write(memo)
    
    json_path = OUTPUT_DIR / f"vrp_autopsy_metrics_{timestamp}.json"
    metrics_json = {
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "phase1_metrics": phase1_metrics,
        "ensemble": ensemble_metrics,
        "attribution": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                           for kk, vv in v.items()} for k, v in attribution.items()},
        "correlation_matrix": corr_matrix.to_dict(),
        "yearly": yearly.to_dict(),
        "regimes": regimes,
        "weights": VRP_NORMALIZED_WEIGHTS,
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
INDIVIDUAL SUB-SLEEVE SHARPES:
  VRP-Core: {phase1_metrics.get('vrp_core_meta', {}).get('sharpe', 'N/A')}
  VRP-Convergence: {phase1_metrics.get('vrp_convergence_meta', {}).get('sharpe', 'N/A')}
  VRP-Alt: {phase1_metrics.get('vrp_alt_meta', {}).get('sharpe', 'N/A')}

WEIGHTED ENSEMBLE:
  Sharpe: {ensemble_metrics['sharpe']:.3f}
  CAGR: {ensemble_metrics['cagr']:.2%}
  MaxDD: {ensemble_metrics['max_drawdown']:.2%}

PHASE 4 BENCHMARK:
  Sharpe >= 0.40: {"PASS" if ensemble_metrics['sharpe'] >= 0.40 else "FAIL"}
  Sharpe >= 0.50: {"PASS" if ensemble_metrics['sharpe'] >= 0.50 else "FAIL"}
""")
    
    return {
        "sharpe": ensemble_metrics['sharpe'],
        "cagr": ensemble_metrics['cagr'],
        "max_dd": ensemble_metrics['max_drawdown'],
        "pass_phase4": ensemble_metrics['sharpe'] >= 0.40,
    }


if __name__ == "__main__":
    main()
