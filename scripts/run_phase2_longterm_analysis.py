#!/usr/bin/env python3
"""
Phase-2 Analysis: Long-Term Canonical Trend Sleeve
Environmental and correlation stability checks for the canonical long-term (1/3, 1/3, 1/3) composite
vs the legacy (0.5, 0.3, 0.2) weighting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_equity_curve(run_id: str) -> pd.Series:
    """Load equity curve from run directory"""
    path = Path(f"reports/runs/{run_id}/equity_curve.csv")
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    return df['equity']

def compute_metrics_by_year(equity: pd.Series) -> pd.DataFrame:
    """Compute performance metrics by year"""
    returns = equity.pct_change().dropna()
    years = returns.index.year.unique()
    
    results = []
    for year in sorted(years):
        year_returns = returns[returns.index.year == year]
        if len(year_returns) == 0:
            continue
            
        # Compute metrics
        cagr = (1 + year_returns).prod() ** (252 / len(year_returns)) - 1
        vol = year_returns.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0
        maxdd = (equity[equity.index.year == year] / equity[equity.index.year == year].expanding().max() - 1).min()
        hit_rate = (year_returns > 0).sum() / len(year_returns)
        
        results.append({
            'year': year,
            'cagr': cagr,
            'vol': vol,
            'sharpe': sharpe,
            'maxdd': maxdd,
            'hit_rate': hit_rate,
            'n_days': len(year_returns)
        })
    
    return pd.DataFrame(results)

def compute_rolling_correlation(eq1: pd.Series, eq2: pd.Series, window: int = 63) -> pd.Series:
    """Compute rolling correlation between two equity curves"""
    ret1 = eq1.pct_change().dropna()
    ret2 = eq2.pct_change().dropna()
    
    # Align returns
    aligned = pd.DataFrame({'ret1': ret1, 'ret2': ret2}).dropna()
    
    return aligned['ret1'].rolling(window).corr(aligned['ret2'])

def main():
    print("="*80)
    print("PHASE-2 ANALYSIS: LONG-TERM CANONICAL TREND SLEEVE")
    print("="*80)
    print()
    
    # Load equity curves
    print("Loading equity curves...")
    canonical = load_equity_curve("core_v3_trend_long_canonical_v1")
    baseline = load_equity_curve("core_v3_no_macro_longterm_baseline")
    
    # Align equity curves
    aligned_idx = canonical.index.intersection(baseline.index)
    canonical = canonical[aligned_idx]
    baseline = baseline[aligned_idx]
    
    print(f"Overlapping period: {aligned_idx[0]} to {aligned_idx[-1]}")
    print(f"N days: {len(aligned_idx)}")
    print()
    
    # 1. Year-by-year comparison
    print("="*80)
    print("1. YEAR-BY-YEAR PERFORMANCE COMPARISON")
    print("="*80)
    print()
    
    canonical_yearly = compute_metrics_by_year(canonical)
    baseline_yearly = compute_metrics_by_year(baseline)
    
    # Merge and compute differences
    yearly_comp = canonical_yearly.merge(baseline_yearly, on='year', suffixes=('_canonical', '_baseline'))
    yearly_comp['sharpe_delta'] = yearly_comp['sharpe_canonical'] - yearly_comp['sharpe_baseline']
    yearly_comp['cagr_delta'] = yearly_comp['cagr_canonical'] - yearly_comp['cagr_baseline']
    yearly_comp['maxdd_delta'] = yearly_comp['maxdd_canonical'] - yearly_comp['maxdd_baseline']
    
    print("Year-by-Year Comparison:")
    print()
    print(f"{'Year':<6} {'Sharpe (Can)':<13} {'Sharpe (Base)':<14} {'Delta':<10} {'CAGR Delta':<12} {'MaxDD Delta':<12}")
    print("-" * 80)
    
    for _, row in yearly_comp.iterrows():
        year = int(row['year'])
        sharpe_c = row['sharpe_canonical']
        sharpe_b = row['sharpe_baseline']
        sharpe_d = row['sharpe_delta']
        cagr_d = row['cagr_delta']
        maxdd_d = row['maxdd_delta']
        
        marker = "✓" if sharpe_d > 0 else "✗"
        print(f"{year:<6} {sharpe_c:>+6.2f} {marker:<6} {sharpe_b:>+6.2f} {' '*8} {sharpe_d:>+6.3f} {' '*4} {cagr_d:>+6.1%} {' '*6} {maxdd_d:>+6.2%} {' '*6}")
    
    print()
    print(f"Years where Canonical outperformed: {(yearly_comp['sharpe_delta'] > 0).sum()} / {len(yearly_comp)}")
    print()
    
    # 2. Environmental checks
    print("="*80)
    print("2. ENVIRONMENTAL CHECKS")
    print("="*80)
    print()
    
    # 2a. Improvement consistency
    canonical_returns = canonical.pct_change().dropna()
    baseline_returns = baseline.pct_change().dropna()
    
    total_return_canonical = (1 + canonical_returns).prod() - 1
    total_return_baseline = (1 + baseline_returns).prod() - 1
    
    print(f"Total Return Canonical:  {total_return_canonical:>8.2%}")
    print(f"Total Return Baseline:   {total_return_baseline:>8.2%}")
    print(f"Delta:                   {total_return_canonical - total_return_baseline:>8.2%}")
    print()
    
    # 2b. Volatility profile
    vol_canonical = canonical_returns.std() * np.sqrt(252)
    vol_baseline = baseline_returns.std() * np.sqrt(252)
    
    print(f"Volatility Canonical:    {vol_canonical:>8.2%}")
    print(f"Volatility Baseline:     {vol_baseline:>8.2%}")
    print(f"Delta:                   {vol_canonical - vol_baseline:>8.2%}")
    print()
    
    # 2c. Drawdown analysis
    dd_canonical = (canonical / canonical.expanding().max() - 1).min()
    dd_baseline = (baseline / baseline.expanding().max() - 1).min()
    
    print(f"Max Drawdown Canonical:  {dd_canonical:>8.2%}")
    print(f"Max Drawdown Baseline:   {dd_baseline:>8.2%}")
    print(f"Delta:                   {dd_canonical - dd_baseline:>8.2%}")
    print()
    
    # 3. Correlation stability
    print("="*80)
    print("3. CORRELATION STABILITY")
    print("="*80)
    print()
    
    # Overall correlation
    aligned_returns = pd.DataFrame({
        'canonical': canonical_returns,
        'baseline': baseline_returns
    }).dropna()
    
    overall_corr = aligned_returns['canonical'].corr(aligned_returns['baseline'])
    print(f"Overall Correlation (Canonical vs Baseline): {overall_corr:.4f}")
    print()
    
    # Rolling correlation
    rolling_corr = compute_rolling_correlation(canonical, baseline, window=63)
    
    print("Rolling Correlation (63-day window):")
    print(f"  Mean:   {rolling_corr.mean():.4f}")
    print(f"  Median: {rolling_corr.median():.4f}")
    print(f"  Std:    {rolling_corr.std():.4f}")
    print(f"  Min:    {rolling_corr.min():.4f}")
    print(f"  Max:    {rolling_corr.max():.4f}")
    print()
    
    # Check for correlation stability (no dramatic shifts)
    corr_drift = rolling_corr.iloc[-63:].mean() - rolling_corr.iloc[:63].mean()
    print(f"Correlation Drift (first 63d vs last 63d): {corr_drift:+.4f}")
    
    if abs(corr_drift) > 0.1:
        print("  WARNING: Correlation drift > 0.1 detected!")
    else:
        print("  ✓ Correlation stable (drift < 0.1)")
    print()
    
    # 4. Hidden degradations check
    print("="*80)
    print("4. HIDDEN DEGRADATIONS CHECK")
    print("="*80)
    print()
    
    # 4a. Check for erratic behavior (large jumps)
    canonical_large_moves = (canonical_returns.abs() > 0.03).sum()
    baseline_large_moves = (baseline_returns.abs() > 0.03).sum()
    
    print(f"Days with |return| > 3%:")
    print(f"  Canonical: {canonical_large_moves} ({100 * canonical_large_moves / len(canonical_returns):.1f}%)")
    print(f"  Baseline:  {baseline_large_moves} ({100 * baseline_large_moves / len(baseline_returns):.1f}%)")
    
    if canonical_large_moves > baseline_large_moves * 1.2:
        print("  WARNING: Canonical has >20% more large moves!")
    else:
        print("  ✓ No excessive volatility spikes detected")
    print()
    
    # 4b. Check for concentration (look at equity curve smoothness)
    canonical_rolling_std = canonical_returns.rolling(21).std()
    baseline_rolling_std = baseline_returns.rolling(21).std()
    
    print(f"Rolling 21-day volatility:")
    print(f"  Canonical (mean): {canonical_rolling_std.mean() * np.sqrt(252):.2%}")
    print(f"  Baseline (mean):  {baseline_rolling_std.mean() * np.sqrt(252):.2%}")
    
    vol_stability_ratio = canonical_rolling_std.std() / baseline_rolling_std.std()
    print(f"  Volatility stability ratio (Can/Base): {vol_stability_ratio:.2f}")
    
    if vol_stability_ratio > 1.2:
        print("  WARNING: Canonical volatility more unstable!")
    else:
        print("  ✓ Volatility stability maintained")
    print()
    
    # 5. Phase-2 Pass Criteria Summary
    print("="*80)
    print("5. PHASE-2 PASS CRITERIA SUMMARY")
    print("="*80)
    print()
    
    sharpe_canonical = (canonical_returns.mean() / canonical_returns.std()) * np.sqrt(252)
    sharpe_baseline = (baseline_returns.mean() / baseline_returns.std()) * np.sqrt(252)
    sharpe_improvement = sharpe_canonical - sharpe_baseline
    
    checks = []
    
    # Check 1: Sharpe ≥ baseline
    check1_pass = sharpe_canonical >= sharpe_baseline
    checks.append(("✓" if check1_pass else "✗", "Sharpe ≥ baseline", 
                  f"{sharpe_canonical:.3f} vs {sharpe_baseline:.3f} (delta: {sharpe_improvement:+.3f})"))
    
    # Check 2: MaxDD ≤ baseline
    check2_pass = dd_canonical >= dd_baseline  # Less negative is better
    checks.append(("✓" if check2_pass else "✗", "MaxDD ≤ baseline", 
                  f"{dd_canonical:.2%} vs {dd_baseline:.2%}"))
    
    # Check 3: Improvements persist (similar shape)
    check3_pass = abs(corr_drift) < 0.1
    checks.append(("✓" if check3_pass else "✗", "Improvements persist (corr stable)", 
                  f"drift: {corr_drift:+.3f}"))
    
    # Check 4: Outperformance across years
    years_outperformed = (yearly_comp['sharpe_delta'] > 0).sum()
    check4_pass = years_outperformed >= len(yearly_comp) * 0.5
    checks.append(("✓" if check4_pass else "✗", "Outperformance across years", 
                  f"{years_outperformed}/{len(yearly_comp)} years"))
    
    # Check 5: No hidden degradation
    check5_pass = vol_stability_ratio <= 1.2 and canonical_large_moves <= baseline_large_moves * 1.2
    checks.append(("✓" if check5_pass else "✗", "No hidden degradation", 
                  f"vol stability: {vol_stability_ratio:.2f}, large moves: {canonical_large_moves} vs {baseline_large_moves}"))
    
    for marker, criterion, details in checks:
        print(f"{marker} {criterion:<40} {details}")
    
    print()
    
    all_passed = all(check[0] == "✓" for check in checks)
    
    if all_passed:
        print("="*80)
        print("✓✓✓ PHASE-2 PASSED! ✓✓✓")
        print("="*80)
        print()
        print("The canonical long-term (1/3, 1/3, 1/3) composite demonstrates:")
        print("  1. Consistent outperformance vs legacy (0.5, 0.3, 0.2) weighting")
        print("  2. Stable correlation profile (no dramatic shifts)")
        print("  3. Similar or better risk characteristics")
        print("  4. No hidden degradations or instabilities")
        print()
        print("RECOMMENDATION: PROMOTE to production as the canonical long-term atomic sleeve")
        print()
    else:
        print("="*80)
        print("PHASE-2 REVIEW REQUIRED")
        print("="*80)
        print()
        print("Some criteria not met. Review details above before promotion.")
        print()
    
    # Save summary
    summary = {
        "phase": "Phase-2",
        "sleeve": "Long-Term Canonical Trend (TSMOM-252)",
        "date": pd.Timestamp.now().isoformat(),
        "period": f"{aligned_idx[0].date()} to {aligned_idx[-1].date()}",
        "n_days": len(aligned_idx),
        "canonical_run_id": "core_v3_trend_long_canonical_v1",
        "baseline_run_id": "core_v3_no_macro_longterm_baseline",
        "metrics": {
            "sharpe_canonical": float(sharpe_canonical),
            "sharpe_baseline": float(sharpe_baseline),
            "sharpe_improvement": float(sharpe_improvement),
            "maxdd_canonical": float(dd_canonical),
            "maxdd_baseline": float(dd_baseline),
            "correlation": float(overall_corr),
            "correlation_drift": float(corr_drift),
            "vol_stability_ratio": float(vol_stability_ratio),
            "years_outperformed": int(years_outperformed),
            "total_years": int(len(yearly_comp))
        },
        "checks": {
            "sharpe_ge_baseline": bool(check1_pass),
            "maxdd_le_baseline": bool(check2_pass),
            "improvements_persist": bool(check3_pass),
            "outperformance_across_years": bool(check4_pass),
            "no_hidden_degradation": bool(check5_pass),
            "all_passed": bool(all_passed)
        },
        "recommendation": "PROMOTE" if all_passed else "REVIEW"
    }
    
    output_dir = Path("reports/phase2_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "long_term_canonical_phase2_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    print()

if __name__ == "__main__":
    main()

