#!/usr/bin/env python3
"""
Attribution Analysis Script (READ-ONLY)

Analyzes run artifacts from vrp_canonical_2020_2024_20260212_152540
to produce quantitative attribution data for the ATTRIBUTION_REPORT.

Does NOT modify any production code or SOT files.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_DIR = PROJECT_ROOT / "reports" / "runs" / "vrp_canonical_2020_2024_20260212_152540"
ANALYSIS_DIR = RUN_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name, parse_dates=True):
    """Load a CSV from the run directory."""
    path = RUN_DIR / name
    if not path.exists():
        print(f"  [SKIP] {name} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    return df


def section_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def analyze_meta():
    """Analyze run metadata and config."""
    section_header("1. RUN METADATA")
    meta_path = RUN_DIR / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"  Run ID: {meta['run_id']}")
    print(f"  Window: {meta['start_date']} to {meta['end_date']}")
    print(f"  Universe: {len(meta['universe'])} assets")
    print(f"  Rebalance: {meta['rebalance']} ({meta['n_rebalances']} rebalances)")
    print(f"  Trading days: {meta['n_trading_days']}")

    metrics = meta.get("metrics_full", {})
    print(f"\n  PERFORMANCE:")
    print(f"    CAGR:         {metrics.get('cagr', 0):.4%}")
    print(f"    Volatility:   {metrics.get('vol', 0):.4%}")
    print(f"    Sharpe:       {metrics.get('sharpe', 0):.4f}")
    print(f"    Max Drawdown: {metrics.get('max_drawdown', 0):.4%}")
    print(f"    Hit Rate:     {metrics.get('hit_rate', 0):.4%}")
    print(f"    Avg Turnover: {metrics.get('avg_turnover', 0):.4f}")
    print(f"    Avg Gross:    {metrics.get('avg_gross', 0):.4f}")
    print(f"    Avg Net:      {metrics.get('avg_net', 0):.6f}")

    # Risk targeting
    rt = meta.get("risk_targeting", {})
    print(f"\n  RISK TARGETING (Layer 5):")
    print(f"    Enabled:      {rt.get('enabled')}")
    print(f"    Effective:    {rt.get('effective')}")
    print(f"    Has Teeth:    {rt.get('has_teeth')}")
    mult_stats = rt.get("multiplier_stats", {})
    print(f"    Leverage p5:  {mult_stats.get('p5', 0):.4f}")
    print(f"    Leverage p50: {mult_stats.get('p50', 0):.4f}")
    print(f"    Leverage p95: {mult_stats.get('p95', 0):.4f}")
    print(f"    At cap:       {mult_stats.get('at_cap', 0):.2%}")
    print(f"    At floor:     {mult_stats.get('at_floor', 0):.2%}")
    vol_stats = rt.get("vol_stats", {})
    print(f"    Port vol p50: {vol_stats.get('p50', 0):.4%}")
    print(f"    Port vol p95: {vol_stats.get('p95', 0):.4%}")

    # Allocator v1
    alloc = meta.get("allocator_v1", {})
    print(f"\n  ALLOCATOR V1 (Layer 6):")
    print(f"    Enabled:      {alloc.get('enabled')}")
    print(f"    Mode:         {alloc.get('mode')}")
    print(f"    Profile:      {alloc.get('profile')}")
    print(f"    Effective:    {alloc.get('effective')}")
    print(f"    Has Teeth:    {alloc.get('has_teeth')}")

    # Policy
    policy = meta.get("policy_features", {})
    print(f"\n  ENGINE POLICY (Layer 2):")
    for feat_name, feat_info in policy.items():
        print(f"    {feat_name}: present={feat_info.get('present')}, "
              f"valid={feat_info.get('n_valid')}/{feat_info.get('n_dates')}")

    return meta


def analyze_equity_curve():
    """Analyze equity curve and drawdowns."""
    section_header("2. EQUITY CURVE ANALYSIS")
    eq = load_csv("equity_curve.csv")
    if eq.empty:
        print("  [SKIP] No equity curve")
        return None

    equity = eq['equity'] if 'equity' in eq.columns else eq.iloc[:, 0]

    # Compute drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    print(f"  Start equity: {equity.iloc[0]:.6f}")
    print(f"  End equity:   {equity.iloc[-1]:.6f}")
    print(f"  Max equity:   {equity.max():.6f} on {equity.idxmax()}")
    print(f"  Min equity:   {equity.min():.6f} on {equity.idxmin()}")
    print(f"  Max drawdown: {drawdown.min():.4%} on {drawdown.idxmin()}")

    # Yearly returns
    print(f"\n  YEARLY RETURNS:")
    equity_ts = equity.copy()
    equity_ts.index = pd.to_datetime(equity_ts.index)
    yearly = equity_ts.resample('YE').last()
    yearly_ret = yearly.pct_change().dropna()
    for dt, ret in yearly_ret.items():
        print(f"    {dt.year}: {ret:.4%}")

    # Save drawdown data
    dd_df = pd.DataFrame({
        'equity': equity,
        'peak': peak,
        'drawdown': drawdown
    })
    dd_df.to_csv(ANALYSIS_DIR / "drawdown_analysis.csv")
    print(f"\n  Saved: drawdown_analysis.csv")

    return equity


def analyze_weights():
    """Analyze portfolio weights."""
    section_header("3. PORTFOLIO WEIGHTS ANALYSIS")
    weights = load_csv("weights.csv")
    if weights.empty:
        print("  [SKIP] No weights data")
        return None

    # If date is a column, set as index
    if 'date' in weights.columns:
        weights = weights.set_index('date')

    print(f"  Shape: {weights.shape}")
    print(f"  Date range: {weights.index[0]} to {weights.index[-1]}")

    # Gross and net exposure
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)

    print(f"\n  GROSS EXPOSURE:")
    print(f"    Mean:   {gross.mean():.4f}")
    print(f"    Median: {gross.median():.4f}")
    print(f"    Std:    {gross.std():.4f}")
    print(f"    Min:    {gross.min():.4f}")
    print(f"    Max:    {gross.max():.4f}")
    print(f"    Zero:   {(gross < 0.001).sum()}/{len(gross)} ({(gross < 0.001).sum()/len(gross):.1%})")

    print(f"\n  NET EXPOSURE:")
    print(f"    Mean:   {net.mean():.6f}")
    print(f"    Median: {net.median():.6f}")
    print(f"    Std:    {net.std():.4f}")
    print(f"    Min:    {net.min():.4f}")
    print(f"    Max:    {net.max():.4f}")

    # Per-asset average absolute weight
    print(f"\n  PER-ASSET AVERAGE |WEIGHT|:")
    avg_abs_w = weights.abs().mean()
    for asset, w in avg_abs_w.sort_values(ascending=False).items():
        print(f"    {asset:30s}: {w:.6f}")

    # Check how many dates have ALL zero weights
    all_zero_rows = (weights.abs().sum(axis=1) < 1e-10).sum()
    print(f"\n  Dates with ALL zero weights: {all_zero_rows}/{len(weights)} ({all_zero_rows/len(weights):.1%})")

    # Turnover
    weight_diff = weights.diff().abs().sum(axis=1)
    print(f"\n  TURNOVER (sum of absolute weight changes):")
    print(f"    Mean:   {weight_diff.mean():.4f}")
    print(f"    Median: {weight_diff.median():.4f}")

    # Save summary
    summary = pd.DataFrame({
        'gross': gross,
        'net': net,
        'turnover': weight_diff
    })
    summary.to_csv(ANALYSIS_DIR / "exposure_summary.csv")
    print(f"\n  Saved: exposure_summary.csv")

    return weights


def analyze_weights_raw_vs_scaled():
    """Compare raw weights vs scaled weights to understand allocator impact."""
    section_header("4. RAW vs SCALED WEIGHTS")
    raw = load_csv("weights_raw.csv")
    scaled = load_csv("weights_scaled.csv")

    if raw.empty and scaled.empty:
        print("  [SKIP] No raw/scaled weight data")
        return

    for label, df in [("Raw", raw), ("Scaled", scaled)]:
        if df.empty:
            print(f"  [SKIP] {label} weights not found")
            continue
        if 'date' in df.columns:
            df = df.set_index('date')
        gross = df.abs().sum(axis=1)
        print(f"\n  {label.upper()} WEIGHTS:")
        print(f"    Shape:  {df.shape}")
        print(f"    Gross mean: {gross.mean():.4f}")
        print(f"    Gross max:  {gross.max():.4f}")
        print(f"    All zero:   {(gross < 1e-10).sum()}/{len(gross)}")


def analyze_risk_targeting():
    """Analyze risk targeting artifacts."""
    section_header("5. RISK TARGETING (Layer 5)")

    # Load params
    params_path = RUN_DIR / "risk_targeting" / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        print(f"  PARAMS:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    # Leverage series
    lev = load_csv("risk_targeting/leverage_series.csv")
    if not lev.empty:
        leverage = lev['leverage'] if 'leverage' in lev.columns else lev.iloc[:, 0]
        print(f"\n  LEVERAGE SERIES:")
        print(f"    Mean:   {leverage.mean():.4f}")
        print(f"    Median: {leverage.median():.4f}")
        print(f"    Std:    {leverage.std():.4f}")
        print(f"    Min:    {leverage.min():.4f}")
        print(f"    Max:    {leverage.max():.4f}")
        print(f"    At cap (7.0): {(leverage >= 6.99).sum()}/{len(leverage)}")
        print(f"    At floor (1.0): {(leverage <= 1.01).sum()}/{len(leverage)}")

    # Realized vol
    vol = load_csv("risk_targeting/realized_vol.csv")
    if not vol.empty:
        rvol = vol['realized_vol'] if 'realized_vol' in vol.columns else vol.iloc[:, 0]
        print(f"\n  REALIZED VOL:")
        print(f"    Mean:   {rvol.mean():.4%}")
        print(f"    Median: {rvol.median():.4%}")
        print(f"    Min:    {rvol.min():.4%}")
        print(f"    Max:    {rvol.max():.4%}")

    # Pre and post RT weights
    pre = load_csv("risk_targeting/weights_pre_risk_targeting.csv", parse_dates=False)
    post = load_csv("risk_targeting/weights_post_risk_targeting.csv", parse_dates=False)

    for label, df in [("Pre-RT", pre), ("Post-RT", post)]:
        if df.empty:
            continue
        # Pivot to wide format
        if 'instrument' in df.columns and 'weight' in df.columns and 'date' in df.columns:
            pivoted = df.pivot_table(index='date', columns='instrument', values='weight', aggfunc='first')
            gross = pivoted.abs().sum(axis=1)
            print(f"\n  {label.upper()} WEIGHTS GROSS:")
            print(f"    Mean: {gross.mean():.4f}")
            print(f"    Max:  {gross.max():.4f}")
            print(f"    Zero: {(gross < 1e-10).sum()}/{len(gross)}")


def analyze_allocator_v1():
    """Analyze allocator v1 artifacts."""
    section_header("6. ALLOCATOR V1 (Layer 6)")

    # Scalars
    scalars = load_csv("allocator_scalars_at_rebalances.csv", parse_dates=False)
    if not scalars.empty:
        if 'rebalance_date' in scalars.columns:
            scalars['rebalance_date'] = pd.to_datetime(scalars['rebalance_date'])
        print(f"  SCALARS AT REBALANCES:")
        print(f"    Shape: {scalars.shape}")
        for col in ['risk_scalar_computed', 'risk_scalar_applied']:
            if col in scalars.columns:
                s = scalars[col]
                print(f"    {col}:")
                print(f"      Mean:   {s.mean():.4f}")
                print(f"      Median: {s.median():.4f}")
                print(f"      Min:    {s.min():.4f}")
                print(f"      Max:    {s.max():.4f}")
                print(f"      Always 1.0: {(s == 1.0).all()}")

    # Regime
    regime = load_csv("allocator_regime_v1.csv")
    if not regime.empty:
        if 'regime' in regime.columns:
            print(f"\n  REGIME DISTRIBUTION:")
            dist = regime['regime'].value_counts(normalize=True)
            for r, pct in dist.items():
                print(f"    {r}: {pct:.2%} ({regime['regime'].value_counts()[r]} days)")


def analyze_engine_policy():
    """Analyze engine policy artifacts."""
    section_header("7. ENGINE POLICY (Layer 2)")

    state = load_csv("engine_policy_state_v1.csv")
    if not state.empty:
        print(f"  STATE shape: {state.shape}")
        print(f"  Columns: {list(state.columns)}")
        for col in state.columns:
            if state[col].dtype in [np.float64, np.int64, float, int]:
                print(f"    {col}: mean={state[col].mean():.4f}, "
                      f"std={state[col].std():.4f}, "
                      f"NaN={state[col].isna().sum()}")

    applied = load_csv("engine_policy_applied_v1.csv", parse_dates=False)
    if not applied.empty:
        print(f"\n  APPLIED MULTIPLIERS shape: {applied.shape}")
        print(f"  Columns: {list(applied.columns)}")
        for col in applied.columns:
            if applied[col].dtype in [np.float64, np.int64, float, int]:
                print(f"    {col}: mean={applied[col].mean():.4f}, "
                      f"zero_pct={(applied[col] == 0).sum()/len(applied[col]):.2%}")


def analyze_portfolio_returns():
    """Analyze portfolio returns distribution."""
    section_header("8. PORTFOLIO RETURNS")
    rets = load_csv("portfolio_returns.csv")
    if rets.empty:
        print("  [SKIP] No returns data")
        return None

    ret = rets['ret'] if 'ret' in rets.columns else rets.iloc[:, 0]

    print(f"  Count: {len(ret)}")
    print(f"  Mean:  {ret.mean():.6f} (daily)")
    print(f"  Std:   {ret.std():.6f} (daily)")
    print(f"  Skew:  {ret.skew():.4f}")
    print(f"  Kurt:  {ret.kurtosis():.4f}")
    print(f"  Min:   {ret.min():.6f}")
    print(f"  Max:   {ret.max():.6f}")

    # Non-zero returns
    nonzero = (ret.abs() > 1e-10).sum()
    print(f"  Non-zero: {nonzero}/{len(ret)} ({nonzero/len(ret):.1%})")

    # Positive vs negative days
    pos = (ret > 0).sum()
    neg = (ret < 0).sum()
    zero = (ret.abs() <= 1e-10).sum()
    print(f"  Positive: {pos}/{len(ret)} ({pos/len(ret):.1%})")
    print(f"  Negative: {neg}/{len(ret)} ({neg/len(ret):.1%})")
    print(f"  Zero:     {zero}/{len(ret)} ({zero/len(ret):.1%})")

    # Monthly returns
    print(f"\n  MONTHLY RETURNS (first 12 months):")
    ret.index = pd.to_datetime(ret.index)
    monthly = ret.resample('ME').sum()
    for dt, r in monthly.head(12).items():
        print(f"    {dt.strftime('%Y-%m')}: {r:.4%}")

    return ret


def analyze_sleeve_returns():
    """Analyze per-sleeve return attribution."""
    section_header("9. SLEEVE RETURNS (Attribution)")
    sleeve = load_csv("sleeve_returns.csv")
    if sleeve.empty:
        print("  [SKIP] No sleeve returns data")
        return None

    print(f"  Shape: {sleeve.shape}")
    print(f"  Columns: {list(sleeve.columns)}")

    for col in sleeve.columns:
        if sleeve[col].dtype in [np.float64, float]:
            s = sleeve[col]
            cum = (1 + s).cumprod().iloc[-1] - 1
            print(f"\n  {col}:")
            print(f"    Daily mean: {s.mean():.6f}")
            print(f"    Daily std:  {s.std():.6f}")
            print(f"    Cumulative: {cum:.4%}")
            print(f"    Non-zero:   {(s.abs() > 1e-10).sum()}/{len(s)}")

    return sleeve


def analyze_asset_returns():
    """Analyze asset-level returns."""
    section_header("10. ASSET RETURNS")
    aret = load_csv("asset_returns.csv")
    if aret.empty:
        print("  [SKIP] No asset returns data")
        return None

    print(f"  Shape: {aret.shape}")
    print(f"  Date range: {aret.index[0]} to {aret.index[-1]}")

    # NaN analysis
    print(f"\n  NaN RATES:")
    for col in aret.columns:
        nan_pct = aret[col].isna().sum() / len(aret)
        print(f"    {col:30s}: {nan_pct:.2%}")

    # Annualized returns per asset
    print(f"\n  ANNUALIZED RETURNS (approx):")
    for col in aret.columns:
        mean_daily = aret[col].mean()
        ann_ret = mean_daily * 252
        ann_vol = aret[col].std() * np.sqrt(252)
        print(f"    {col:30s}: ret={ann_ret:.2%}, vol={ann_vol:.2%}")

    return aret


def compute_weight_x_return_attribution():
    """Compute weight * return attribution per asset."""
    section_header("11. WEIGHT × RETURN ATTRIBUTION")
    weights = load_csv("weights.csv")
    aret = load_csv("asset_returns.csv")

    if weights.empty or aret.empty:
        print("  [SKIP] Missing weights or asset returns")
        return

    if 'date' in weights.columns:
        weights = weights.set_index('date')

    weights.index = pd.to_datetime(weights.index)
    aret.index = pd.to_datetime(aret.index)

    # Forward-fill weights to daily frequency
    all_dates = aret.index
    weights_daily = weights.reindex(all_dates).ffill()

    # Align columns
    common_cols = weights_daily.columns.intersection(aret.columns)
    if len(common_cols) == 0:
        print("  [SKIP] No common assets between weights and returns")
        return

    # Per-asset contribution = weight_t * return_t+1
    # Using previous-day weights (weights are set on rebalance, returns are next-day)
    w = weights_daily[common_cols].shift(1).fillna(0)
    r = aret[common_cols].fillna(0)

    contrib = w * r

    print(f"  Common assets: {len(common_cols)}")
    print(f"  Date range: {contrib.index[1]} to {contrib.index[-1]}")

    # Per-asset cumulative contribution
    cum_contrib = contrib.sum()
    print(f"\n  CUMULATIVE CONTRIBUTION BY ASSET:")
    for asset, c in cum_contrib.sort_values().items():
        print(f"    {asset:30s}: {c:.6f}")

    total_contrib = cum_contrib.sum()
    print(f"    {'TOTAL':30s}: {total_contrib:.6f}")

    # Save attribution
    contrib.to_csv(ANALYSIS_DIR / "asset_attribution.csv")
    cum_df = pd.DataFrame({
        'asset': cum_contrib.index,
        'cumulative_contribution': cum_contrib.values
    })
    cum_df.to_csv(ANALYSIS_DIR / "asset_cumulative_attribution.csv", index=False)
    print(f"\n  Saved: asset_attribution.csv, asset_cumulative_attribution.csv")


def analyze_signal_coverage():
    """Analyze signal coverage and NaN rates."""
    section_header("12. SIGNAL COVERAGE & NaN ANALYSIS")

    # Check sleeve_returns for NaN
    sleeve = load_csv("sleeve_returns.csv")
    if not sleeve.empty:
        print(f"  SLEEVE RETURNS NaN rates:")
        for col in sleeve.columns:
            nan_pct = sleeve[col].isna().sum() / len(sleeve)
            print(f"    {col}: {nan_pct:.2%}")

    # Check weights for NaN
    weights = load_csv("weights.csv")
    if not weights.empty:
        if 'date' in weights.columns:
            weights = weights.set_index('date')
        nan_total = weights.isna().sum().sum()
        print(f"\n  WEIGHTS NaN total: {nan_total}")


def critical_finding_vrp_signals():
    """
    CRITICAL ANALYSIS: VRP strategies return VX1 signals.
    VX1 is NOT in the universe. Investigate how this affects the portfolio.
    """
    section_header("13. CRITICAL: VRP SIGNAL → UNIVERSE MAPPING")

    print("  VRP STRATEGY SIGNAL ANALYSIS:")
    print("  ================================")
    print("  VRP-Core, VRP-Alt, VRP-Convergence all produce signals for 'VX1'")
    print("  Universe contains 13 assets (ES, NQ, RTY, ZT, ZF, ZN, UB, SR3, CL, GC, 6E, 6B, 6J)")
    print("  VX1 is NOT in the universe.")
    print()

    # Check the actual config
    config_path = PROJECT_ROOT / "configs" / "phase4_vrp_baseline_v1.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        strategies = cfg.get("strategies", {})
        enabled = {k: v for k, v in strategies.items()
                   if isinstance(v, dict) and v.get("enabled")}
        print(f"  ENABLED STRATEGIES IN CONFIG:")
        for name, scfg in enabled.items():
            print(f"    {name}: weight={scfg.get('weight', 'N/A')}")

        rt_cfg = cfg.get("risk_targeting", {})
        print(f"\n  RISK TARGETING CONFIG:")
        print(f"    target_vol: {rt_cfg.get('target_vol')}")
        print(f"    leverage_cap: {rt_cfg.get('leverage_cap')}")
        print(f"    leverage_floor: {rt_cfg.get('leverage_floor')}")

    # Check weights - are they actually non-zero?
    weights = load_csv("weights.csv")
    if not weights.empty:
        if 'date' in weights.columns:
            weights = weights.set_index('date')
        gross = weights.abs().sum(axis=1)
        print(f"\n  WEIGHT ANALYSIS:")
        print(f"    All-zero weight rows: {(gross < 1e-10).sum()}/{len(gross)}")
        print(f"    Non-zero weight rows: {(gross >= 1e-10).sum()}/{len(gross)}")
        print(f"    Average gross exposure: {gross.mean():.6f}")

        # Show first few non-zero weight rows
        nonzero_mask = gross >= 1e-10
        if nonzero_mask.any():
            print(f"\n  FIRST 3 NON-ZERO WEIGHT ROWS:")
            first_nonzero = weights[nonzero_mask].head(3)
            for idx, row in first_nonzero.iterrows():
                active = row[row.abs() > 1e-10]
                print(f"    {idx}: gross={row.abs().sum():.4f}, "
                      f"active_assets={len(active)}")
                for asset, w in active.items():
                    print(f"      {asset}: {w:.6f}")


def analyze_rt_config_mismatch():
    """
    CRITICAL: Check target_vol mismatch between VRP config and default.
    """
    section_header("14. CRITICAL: TARGET VOL MISMATCH")

    config_path = PROJECT_ROOT / "configs" / "phase4_vrp_baseline_v1.yaml"
    default_config = PROJECT_ROOT / "configs" / "strategies.yaml"

    import yaml

    print("  CONFIG COMPARISON:")
    for label, path in [("VRP Config", config_path), ("Default Config", default_config)]:
        if path.exists():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            rt = cfg.get("risk_targeting", {})
            print(f"\n  {label} ({path.name}):")
            print(f"    target_vol:      {rt.get('target_vol', 'NOT SET')}")
            print(f"    leverage_cap:    {rt.get('leverage_cap', 'NOT SET')}")
            print(f"    leverage_floor:  {rt.get('leverage_floor', 'NOT SET')}")
            print(f"    vol_floor:       {rt.get('vol_floor', 'NOT SET')}")

    # Check actual params used in run
    params_path = RUN_DIR / "risk_targeting" / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        print(f"\n  ACTUAL PARAMS USED IN RUN:")
        print(f"    target_vol:      {params.get('target_vol')}")
        print(f"    leverage_cap:    {params.get('leverage_cap')}")
        print(f"    leverage_floor:  {params.get('leverage_floor')}")
        print(f"    vol_floor:       {params.get('vol_floor')}")


def main():
    print("=" * 80)
    print("  ATTRIBUTION ANALYSIS")
    print(f"  Run: vrp_canonical_2020_2024_20260212_152540")
    print(f"  Run Dir: {RUN_DIR}")
    print("=" * 80)

    # Check run directory exists
    if not RUN_DIR.exists():
        print(f"ERROR: Run directory not found: {RUN_DIR}")
        return 1

    # List all artifacts
    print(f"\n  ARTIFACTS FOUND:")
    for p in sorted(RUN_DIR.rglob("*")):
        if p.is_file() and p.suffix in ['.csv', '.json', '.md']:
            rel = p.relative_to(RUN_DIR)
            size = p.stat().st_size
            print(f"    {str(rel):50s} ({size:>10,d} bytes)")

    # Run all analyses
    meta = analyze_meta()
    equity = analyze_equity_curve()
    weights = analyze_weights()
    analyze_weights_raw_vs_scaled()
    analyze_risk_targeting()
    analyze_allocator_v1()
    analyze_engine_policy()
    ret = analyze_portfolio_returns()
    sleeve = analyze_sleeve_returns()
    aret = analyze_asset_returns()
    compute_weight_x_return_attribution()
    analyze_signal_coverage()
    critical_finding_vrp_signals()
    analyze_rt_config_mismatch()

    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print(f"  Output saved to: {ANALYSIS_DIR}")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
