"""
Engine Attribution at Post-Construction

Computes engine-level contribution analysis at the Post-Construction stage,
which is the canonical system belief evaluation layer.

This is the "production Phase 2" bridge - it allows engine attribution to be
computed using the same Construction v1 rules as production.

Outputs:
- engine_attribution_post_construction.json
- engine_attribution_post_construction.md

Reference:
- SYSTEM_CONSTRUCTION.md § "Portfolio Construction v1 (Canonical)"
- PROCEDURES.md § "Construction Harness Contract"
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from .canonical_diagnostics import load_run_artifacts
from .waterfall_attribution import compute_stage_metrics, compute_regime_conditioned_metrics

logger = logging.getLogger(__name__)


def compute_sleeve_contribution_series(
    sleeve_returns: pd.DataFrame,
    sleeve_weights: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute per-sleeve contribution series.
    
    If sleeve_weights provided: contrib_s(t) = w_s(t) * r_s(t)
    Otherwise: use sleeve_returns directly (already weighted in exec_sim)
    
    Args:
        sleeve_returns: DataFrame with sleeve returns (date x sleeve)
        sleeve_weights: Optional DataFrame with sleeve weights (date x sleeve)
        
    Returns:
        DataFrame with contribution series (date x sleeve)
    """
    if sleeve_weights is not None and not sleeve_weights.empty:
        # Align weights to returns dates (forward-fill)
        weights_aligned = sleeve_weights.reindex(sleeve_returns.index).ffill().fillna(0.0)
        # Contribution = weight * return
        contribution = weights_aligned * sleeve_returns
    else:
        # Sleeve returns are already weighted in exec_sim pipeline
        contribution = sleeve_returns.copy()
    
    return contribution


def compute_contribution_metrics(
    contribution_series: pd.Series,
    portfolio_returns: pd.Series,
    regime_series: Optional[pd.Series] = None
) -> Dict:
    """
    Compute contribution metrics for a single sleeve.
    
    Args:
        contribution_series: Time series of sleeve contribution
        portfolio_returns: Time series of total portfolio returns
        regime_series: Optional regime classification series
        
    Returns:
        Dict with contribution metrics
    """
    # Align series
    common_dates = contribution_series.index.intersection(portfolio_returns.index)
    if len(common_dates) == 0:
        return {
            'total_contribution_pct': np.nan,
            'contribution_sharpe': np.nan,
            'contribution_vol': np.nan,
            'correlation_with_portfolio': np.nan,
            'drawdown_contribution_proxy': np.nan,
            'rolling_sharpe_6m': np.nan,
            'rolling_sharpe_12m': np.nan,
            'n_days': 0
        }
    
    contrib = contribution_series.loc[common_dates]
    portfolio = portfolio_returns.loc[common_dates]
    
    # Total contribution as % of total PnL
    total_portfolio_pnl = portfolio.sum()
    total_sleeve_pnl = contrib.sum()
    if abs(total_portfolio_pnl) > 1e-10:
        total_contribution_pct = total_sleeve_pnl / total_portfolio_pnl
    else:
        total_contribution_pct = np.nan
    
    # Contribution Sharpe
    contrib_mean = contrib.mean() * 252
    contrib_std = contrib.std() * np.sqrt(252)
    contribution_sharpe = contrib_mean / contrib_std if contrib_std > 1e-10 else 0.0
    
    # Contribution volatility
    contribution_vol = contrib_std
    
    # Correlation with portfolio
    if len(contrib) > 10:
        correlation = contrib.corr(portfolio)
    else:
        correlation = np.nan
    
    # Drawdown contribution proxy: sum of contributions during portfolio drawdowns
    portfolio_equity = (1 + portfolio).cumprod()
    portfolio_hwm = portfolio_equity.cummax()
    portfolio_dd = (portfolio_equity - portfolio_hwm) / portfolio_hwm
    dd_mask = portfolio_dd < -0.05  # During drawdowns > 5%
    if dd_mask.sum() > 0:
        dd_contrib = contrib[dd_mask].sum()
        dd_portfolio = portfolio[dd_mask].sum()
        if abs(dd_portfolio) > 1e-10:
            drawdown_contribution_proxy = dd_contrib / dd_portfolio
        else:
            drawdown_contribution_proxy = np.nan
    else:
        drawdown_contribution_proxy = np.nan
    
    # Rolling Sharpe stability (6m and 12m)
    rolling_sharpe_6m = np.nan
    rolling_sharpe_12m = np.nan
    if len(contrib) > 126:  # ~6 months
        rolling_mean_6m = contrib.rolling(126).mean() * 252
        rolling_std_6m = contrib.rolling(126).std() * np.sqrt(252)
        rolling_sharpe_6m_series = rolling_mean_6m / rolling_std_6m.replace(0, np.nan)
        rolling_sharpe_6m = rolling_sharpe_6m_series.std()  # Stability = std of rolling Sharpe
    
    if len(contrib) > 252:  # ~12 months
        rolling_mean_12m = contrib.rolling(252).mean() * 252
        rolling_std_12m = contrib.rolling(252).std() * np.sqrt(252)
        rolling_sharpe_12m_series = rolling_mean_12m / rolling_std_12m.replace(0, np.nan)
        rolling_sharpe_12m = rolling_sharpe_12m_series.std()  # Stability = std of rolling Sharpe
    
    return {
        'total_contribution_pct': float(total_contribution_pct) if not np.isnan(total_contribution_pct) else None,
        'contribution_sharpe': float(contribution_sharpe) if not np.isnan(contribution_sharpe) else None,
        'contribution_vol': float(contribution_vol) if not np.isnan(contribution_vol) else None,
        'correlation_with_portfolio': float(correlation) if not np.isnan(correlation) else None,
        'drawdown_contribution_proxy': float(drawdown_contribution_proxy) if not np.isnan(drawdown_contribution_proxy) else None,
        'rolling_sharpe_6m_stability': float(rolling_sharpe_6m) if not np.isnan(rolling_sharpe_6m) else None,
        'rolling_sharpe_12m_stability': float(rolling_sharpe_12m) if not np.isnan(rolling_sharpe_12m) else None,
        'n_days': len(contrib)
    }


def compute_regime_contribution(
    contribution_series: pd.Series,
    regime_series: Optional[pd.Series] = None,
    vol_series: Optional[pd.Series] = None
) -> Dict:
    """
    Compute regime-conditioned contribution metrics.
    
    Args:
        contribution_series: Time series of sleeve contribution
        regime_series: Optional regime classification (NORMAL/ELEVATED/STRESS/CRISIS)
        vol_series: Optional realized volatility for high-vol/low-vol split
        
    Returns:
        Dict with regime-conditioned metrics
    """
    result = {
        'crisis': {'contribution': np.nan, 'sharpe': np.nan, 'n_days': 0},
        'calm': {'contribution': np.nan, 'sharpe': np.nan, 'n_days': 0},
        'high_vol': {'contribution': np.nan, 'sharpe': np.nan, 'n_days': 0},
        'low_vol': {'contribution': np.nan, 'sharpe': np.nan, 'n_days': 0},
    }
    
    # Crisis vs Calm (using regime classification if available)
    if regime_series is not None:
        crisis_mask = regime_series.isin(['CRISIS', 'STRESS'])
        calm_mask = regime_series.isin(['NORMAL', 'ELEVATED'])
        
        crisis_dates = contribution_series.index.intersection(regime_series[crisis_mask].index)
        calm_dates = contribution_series.index.intersection(regime_series[calm_mask].index)
        
        if len(crisis_dates) > 0:
            crisis_contrib = contribution_series.loc[crisis_dates]
            result['crisis'] = {
                'contribution': float(crisis_contrib.sum()),
                'sharpe': float(crisis_contrib.mean() * 252 / (crisis_contrib.std() * np.sqrt(252))) if crisis_contrib.std() > 0 else 0.0,
                'n_days': len(crisis_contrib)
            }
        
        if len(calm_dates) > 0:
            calm_contrib = contribution_series.loc[calm_dates]
            result['calm'] = {
                'contribution': float(calm_contrib.sum()),
                'sharpe': float(calm_contrib.mean() * 252 / (calm_contrib.std() * np.sqrt(252))) if calm_contrib.std() > 0 else 0.0,
                'n_days': len(calm_contrib)
            }
    
    # High-vol vs Low-vol
    if vol_series is not None:
        vol_aligned = vol_series.reindex(contribution_series.index)
        vol_median = vol_aligned.median()
        
        high_vol_mask = vol_aligned >= vol_median
        low_vol_mask = vol_aligned < vol_median
        
        high_vol_dates = contribution_series.index[high_vol_mask]
        low_vol_dates = contribution_series.index[low_vol_mask]
        
        if len(high_vol_dates) > 0:
            high_vol_contrib = contribution_series.loc[high_vol_dates]
            result['high_vol'] = {
                'contribution': float(high_vol_contrib.sum()),
                'sharpe': float(high_vol_contrib.mean() * 252 / (high_vol_contrib.std() * np.sqrt(252))) if high_vol_contrib.std() > 0 else 0.0,
                'n_days': len(high_vol_contrib)
            }
        
        if len(low_vol_dates) > 0:
            low_vol_contrib = contribution_series.loc[low_vol_dates]
            result['low_vol'] = {
                'contribution': float(low_vol_contrib.sum()),
                'sharpe': float(low_vol_contrib.mean() * 252 / (low_vol_contrib.std() * np.sqrt(252))) if low_vol_contrib.std() > 0 else 0.0,
                'n_days': len(low_vol_contrib)
            }
    
    return result


def classify_sleeve_roles(sleeve_metrics: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Classify sleeves into roles based on their contribution characteristics.
    
    Roles:
    - positive_contributors: Positive total contribution
    - negative_contributors: Negative total contribution
    - diversifiers: Low correlation with portfolio + positive contribution
    - red_flags: Negative contribution + high correlation
    
    Args:
        sleeve_metrics: Dict mapping sleeve name to metrics dict
        
    Returns:
        Dict mapping role to list of sleeve names
    """
    roles = {
        'top_positive_contributors': [],
        'top_negative_contributors': [],
        'diversifiers': [],
        'red_flags': []
    }
    
    # Sort by total contribution
    sorted_sleeves = sorted(
        sleeve_metrics.items(),
        key=lambda x: x[1].get('total_contribution_pct', 0) or 0,
        reverse=True
    )
    
    for sleeve_name, metrics in sorted_sleeves:
        contrib_pct = metrics.get('total_contribution_pct')
        correlation = metrics.get('correlation_with_portfolio')
        
        if contrib_pct is None:
            continue
        
        # Positive vs negative contributors
        if contrib_pct > 0:
            roles['top_positive_contributors'].append(sleeve_name)
        else:
            roles['top_negative_contributors'].append(sleeve_name)
        
        # Diversifiers: low correlation (< 0.3) AND positive contribution
        if correlation is not None and correlation < 0.3 and contrib_pct > 0:
            roles['diversifiers'].append(sleeve_name)
        
        # Red flags: negative contribution AND high correlation (> 0.5)
        if correlation is not None and correlation > 0.5 and contrib_pct < 0:
            roles['red_flags'].append(sleeve_name)
    
    return roles


def compute_engine_attribution_post_construction(
    run_id: str,
    run_dir: Optional[Path] = None
) -> Dict:
    """
    Compute engine attribution at Post-Construction stage.
    
    This is the canonical system belief evaluation layer. Engine attribution
    computed here reflects engine quality before RT and Allocator modifications.
    
    Args:
        run_id: Run identifier
        run_dir: Optional path to run directory
        
    Returns:
        Dict with engine attribution results
    """
    if run_dir is None:
        run_dir = Path(f"reports/runs/{run_id}")
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    logger.info(f"Computing engine attribution at Post-Construction for {run_id}")
    artifacts = load_run_artifacts(run_dir)
    
    # Load sleeve returns (post-policy)
    sleeve_returns = artifacts.get('sleeve_returns')
    if sleeve_returns is None or sleeve_returns.empty:
        logger.warning("sleeve_returns.csv not available - cannot compute engine attribution")
        return {
            'run_id': run_id,
            'generated_at': datetime.now().isoformat(),
            'error': 'sleeve_returns.csv not available',
            'sleeve_metrics': {},
            'sleeve_roles': {},
            'regime_contribution': {}
        }
    
    # Load portfolio returns at Post-Construction
    weights_post_construction = artifacts.get('weights_post_construction')
    asset_returns = artifacts.get('asset_returns')
    
    # Reconstruct Post-Construction returns
    if weights_post_construction is not None and not weights_post_construction.empty and asset_returns is not None:
        weights_daily = weights_post_construction.reindex(asset_returns.index).ffill().fillna(0.0)
        common_symbols = weights_daily.columns.intersection(asset_returns.columns)
        if len(common_symbols) > 0:
            weights_aligned = weights_daily[common_symbols]
            returns_aligned = asset_returns[common_symbols]
            # Use log/simple conversion (Phase 3B contract)
            returns_log = np.log(1 + returns_aligned)
            portfolio_log = (weights_aligned * returns_log).sum(axis=1)
            portfolio_returns_post_construction = np.exp(portfolio_log) - 1.0
        else:
            # Fallback to simple sum
            portfolio_returns_post_construction = sleeve_returns.sum(axis=1)
    else:
        # Fallback: sum of sleeve returns
        portfolio_returns_post_construction = sleeve_returns.sum(axis=1)
    
    # Load regime classification if available
    regime_file = run_dir / "allocator_regime_v1.csv"
    regime_series = None
    if regime_file.exists():
        try:
            regime_df = pd.read_csv(regime_file, parse_dates=['rebalance_date'], index_col='rebalance_date')
            if 'regime' in regime_df.columns:
                regime_series = regime_df['regime']
        except Exception as e:
            logger.warning(f"Failed to load regime classification: {e}")
    
    # Compute realized vol for regime conditioning
    vol_series = None
    if len(portfolio_returns_post_construction) > 21:
        vol_series = portfolio_returns_post_construction.rolling(21).std() * np.sqrt(252)
    
    # Load meta for evaluation window
    meta = artifacts.get('meta', {})
    evaluation_start = meta.get('evaluation_start_date')
    
    # Filter to evaluation window
    if evaluation_start:
        eval_start_dt = pd.Timestamp(evaluation_start)
        sleeve_returns = sleeve_returns[sleeve_returns.index >= eval_start_dt]
        portfolio_returns_post_construction = portfolio_returns_post_construction[
            portfolio_returns_post_construction.index >= eval_start_dt
        ]
    
    # Compute contribution series (sleeve_returns are already weighted)
    contribution_series = compute_sleeve_contribution_series(sleeve_returns)
    
    # Compute metrics for each sleeve
    sleeve_metrics = {}
    regime_contribution = {}
    
    for sleeve_name in sleeve_returns.columns:
        sleeve_contrib = contribution_series[sleeve_name].dropna()
        
        # Contribution metrics
        metrics = compute_contribution_metrics(
            sleeve_contrib,
            portfolio_returns_post_construction,
            regime_series
        )
        sleeve_metrics[sleeve_name] = metrics
        
        # Regime-conditioned contribution
        regime_contrib = compute_regime_contribution(
            sleeve_contrib,
            regime_series,
            vol_series
        )
        regime_contribution[sleeve_name] = regime_contrib
    
    # Classify sleeve roles
    sleeve_roles = classify_sleeve_roles(sleeve_metrics)
    
    # Build report
    report = {
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'evaluation_start': evaluation_start,
        'stage': 'Post-Construction (Pre-RT)',
        'note': 'Engine attribution computed at Post-Construction, the canonical system belief evaluation layer.',
        'sleeve_metrics': sleeve_metrics,
        'sleeve_roles': sleeve_roles,
        'regime_contribution': regime_contribution
    }
    
    return report


def format_engine_attribution_report(report: Dict) -> str:
    """
    Format engine attribution report as Markdown.
    
    Args:
        report: Engine attribution report dict
        
    Returns:
        Markdown-formatted report string
    """
    lines = []
    lines.append("# Engine Attribution at Post-Construction")
    lines.append("")
    lines.append(f"**Run ID:** `{report['run_id']}`")
    lines.append(f"**Generated:** {report['generated_at']}")
    lines.append(f"**Stage:** {report.get('stage', 'Post-Construction (Pre-RT)')}")
    if report.get('evaluation_start'):
        lines.append(f"**Evaluation Window Start:** {report['evaluation_start']}")
    lines.append("")
    lines.append(f"> {report.get('note', '')}")
    lines.append("")
    
    if 'error' in report:
        lines.append(f"**Error:** {report['error']}")
        return "\n".join(lines)
    
    lines.append("---")
    lines.append("")
    
    # Sleeve Metrics Table
    lines.append("## Sleeve Contribution Metrics")
    lines.append("")
    lines.append("| Sleeve | Contrib % | Contrib Sharpe | Vol | Corr w/ Port | DD Contrib | N Days |")
    lines.append("|--------|----------:|---------------:|----:|-------------:|-----------:|-------:|")
    
    sleeve_metrics = report.get('sleeve_metrics', {})
    
    # Sort by contribution percentage (descending)
    sorted_sleeves = sorted(
        sleeve_metrics.items(),
        key=lambda x: x[1].get('total_contribution_pct', 0) or 0,
        reverse=True
    )
    
    for sleeve_name, metrics in sorted_sleeves:
        contrib_pct = metrics.get('total_contribution_pct')
        contrib_sharpe = metrics.get('contribution_sharpe')
        contrib_vol = metrics.get('contribution_vol')
        correlation = metrics.get('correlation_with_portfolio')
        dd_contrib = metrics.get('drawdown_contribution_proxy')
        n_days = metrics.get('n_days', 0)
        
        contrib_pct_str = f"{contrib_pct:.1%}" if contrib_pct is not None else "N/A"
        contrib_sharpe_str = f"{contrib_sharpe:.2f}" if contrib_sharpe is not None else "N/A"
        contrib_vol_str = f"{contrib_vol:.1%}" if contrib_vol is not None else "N/A"
        corr_str = f"{correlation:.2f}" if correlation is not None else "N/A"
        dd_str = f"{dd_contrib:.1%}" if dd_contrib is not None else "N/A"
        
        lines.append(f"| {sleeve_name} | {contrib_pct_str} | {contrib_sharpe_str} | {contrib_vol_str} | {corr_str} | {dd_str} | {n_days} |")
    
    lines.append("")
    
    # Sleeve Roles
    lines.append("## Sleeve Role Classification")
    lines.append("")
    
    sleeve_roles = report.get('sleeve_roles', {})
    
    top_positive = sleeve_roles.get('top_positive_contributors', [])
    if top_positive:
        lines.append("### Top Positive Contributors")
        lines.append("")
        for sleeve in top_positive[:5]:  # Top 5
            metrics = sleeve_metrics.get(sleeve, {})
            contrib_pct = metrics.get('total_contribution_pct')
            contrib_str = f"{contrib_pct:.1%}" if contrib_pct is not None else "N/A"
            lines.append(f"- **{sleeve}**: {contrib_str} of total PnL")
        lines.append("")
    
    top_negative = sleeve_roles.get('top_negative_contributors', [])
    if top_negative:
        lines.append("### Top Negative Contributors")
        lines.append("")
        for sleeve in top_negative[:5]:  # Top 5
            metrics = sleeve_metrics.get(sleeve, {})
            contrib_pct = metrics.get('total_contribution_pct')
            contrib_str = f"{contrib_pct:.1%}" if contrib_pct is not None else "N/A"
            lines.append(f"- **{sleeve}**: {contrib_str} of total PnL")
        lines.append("")
    
    diversifiers = sleeve_roles.get('diversifiers', [])
    if diversifiers:
        lines.append("### Diversifiers (Low Corr + Positive Contrib)")
        lines.append("")
        for sleeve in diversifiers:
            metrics = sleeve_metrics.get(sleeve, {})
            correlation = metrics.get('correlation_with_portfolio')
            contrib_pct = metrics.get('total_contribution_pct')
            corr_str = f"{correlation:.2f}" if correlation is not None else "N/A"
            contrib_str = f"{contrib_pct:.1%}" if contrib_pct is not None else "N/A"
            lines.append(f"- **{sleeve}**: Corr {corr_str}, Contrib {contrib_str}")
        lines.append("")
    
    red_flags = sleeve_roles.get('red_flags', [])
    if red_flags:
        lines.append("### Red Flags (Negative Contrib + High Corr)")
        lines.append("")
        for sleeve in red_flags:
            metrics = sleeve_metrics.get(sleeve, {})
            correlation = metrics.get('correlation_with_portfolio')
            contrib_pct = metrics.get('total_contribution_pct')
            corr_str = f"{correlation:.2f}" if correlation is not None else "N/A"
            contrib_str = f"{contrib_pct:.1%}" if contrib_pct is not None else "N/A"
            lines.append(f"- **{sleeve}**: Corr {corr_str}, Contrib {contrib_str} **[INVESTIGATE]**")
        lines.append("")
    
    # Regime-Conditioned Contribution
    lines.append("## Regime-Conditioned Contribution")
    lines.append("")
    
    regime_contribution = report.get('regime_contribution', {})
    
    if regime_contribution:
        lines.append("| Sleeve | Crisis Contrib | Crisis Sharpe | Calm Contrib | Calm Sharpe |")
        lines.append("|--------|---------------:|--------------:|-------------:|------------:|")
        
        for sleeve_name in sorted_sleeves:
            sleeve = sleeve_name[0]  # tuple from sorted
            regime_data = regime_contribution.get(sleeve, {})
            crisis = regime_data.get('crisis', {})
            calm = regime_data.get('calm', {})
            
            crisis_contrib = crisis.get('contribution')
            crisis_sharpe = crisis.get('sharpe')
            calm_contrib = calm.get('contribution')
            calm_sharpe = calm.get('sharpe')
            
            crisis_contrib_str = f"{crisis_contrib:.4f}" if crisis_contrib is not None and not np.isnan(crisis_contrib) else "N/A"
            crisis_sharpe_str = f"{crisis_sharpe:.2f}" if crisis_sharpe is not None and not np.isnan(crisis_sharpe) else "N/A"
            calm_contrib_str = f"{calm_contrib:.4f}" if calm_contrib is not None and not np.isnan(calm_contrib) else "N/A"
            calm_sharpe_str = f"{calm_sharpe:.2f}" if calm_sharpe is not None and not np.isnan(calm_sharpe) else "N/A"
            
            lines.append(f"| {sleeve} | {crisis_contrib_str} | {crisis_sharpe_str} | {calm_contrib_str} | {calm_sharpe_str} |")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append("")
    lines.append("- **Contrib %**: Sleeve's share of total portfolio PnL (should sum to ~100%)")
    lines.append("- **Contrib Sharpe**: Risk-adjusted return of the sleeve's contribution stream")
    lines.append("- **Corr w/ Port**: How much the sleeve moves with the portfolio (lower = better diversification)")
    lines.append("- **DD Contrib**: Sleeve's share of PnL during portfolio drawdowns > 5%")
    lines.append("- **Diversifiers**: Low correlation + positive contribution = valuable for portfolio")
    lines.append("- **Red Flags**: High correlation + negative contribution = investigate for removal/reweight")
    lines.append("")
    
    return "\n".join(lines)
