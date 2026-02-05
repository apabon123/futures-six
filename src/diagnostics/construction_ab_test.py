"""
Portfolio Construction A/B Test Harness

Utility for Phase 2 promotion testing inside production rules.
Uses Construction v1 rules (static sleeve weighting).

This harness allows comparing:
- Base portfolio (existing sleeves)
- Candidate portfolio (base + new sleeve or modified weights)

The comparison is done at the Post-Construction stage, which is the
canonical system belief evaluation layer.

Important: This harness uses the same Construction v1 rules as production.

Reference:
- SYSTEM_CONSTRUCTION.md § "Portfolio Construction v1 (Canonical)"
- PROCEDURES.md § "Construction Harness Contract"
- ROADMAP.md § "Construction v2"
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from .canonical_diagnostics import load_run_artifacts
from .waterfall_attribution import compute_stage_metrics, compute_regime_conditioned_metrics, reconstruct_stage_returns
from .engine_attribution import compute_contribution_metrics

logger = logging.getLogger(__name__)


def portfolio_construction_ab_test(
    base_run_id: str,
    candidate_run_id: str,
    base_run_dir: Optional[Path] = None,
    candidate_run_dir: Optional[Path] = None
) -> Dict:
    """
    Compare base portfolio vs candidate portfolio at Post-Construction stage.
    
    This is the "Phase 2 test inside production rules" - it compares two runs
    at the Post-Construction layer using the same Construction v1 semantics.
    
    Args:
        base_run_id: Run ID of the base portfolio
        candidate_run_id: Run ID of the candidate portfolio
        base_run_dir: Optional path to base run directory
        candidate_run_dir: Optional path to candidate run directory
        
    Returns:
        Dict with A/B comparison results:
        - Delta Post-Construction Sharpe / CAGR / MaxDD
        - Delta regime performance
        - Contribution deltas
    """
    if base_run_dir is None:
        base_run_dir = Path(f"reports/runs/{base_run_id}")
    if candidate_run_dir is None:
        candidate_run_dir = Path(f"reports/runs/{candidate_run_id}")
    
    if not base_run_dir.exists():
        raise FileNotFoundError(f"Base run directory not found: {base_run_dir}")
    if not candidate_run_dir.exists():
        raise FileNotFoundError(f"Candidate run directory not found: {candidate_run_dir}")
    
    logger.info(f"Running A/B test: {base_run_id} vs {candidate_run_id}")
    
    # Load artifacts
    base_artifacts = load_run_artifacts(base_run_dir)
    candidate_artifacts = load_run_artifacts(candidate_run_dir)
    
    # Store run_dir in artifacts for reconstruct_stage_returns
    base_artifacts['_run_dir'] = base_run_dir
    candidate_artifacts['_run_dir'] = candidate_run_dir
    
    # Reconstruct Post-Construction returns for both
    base_returns = reconstruct_stage_returns(base_artifacts, 'post_construction')
    candidate_returns = reconstruct_stage_returns(candidate_artifacts, 'post_construction')
    
    if base_returns is None or candidate_returns is None:
        return {
            'base_run_id': base_run_id,
            'candidate_run_id': candidate_run_id,
            'generated_at': datetime.now().isoformat(),
            'error': 'Could not reconstruct Post-Construction returns for one or both runs',
            'base_available': base_returns is not None,
            'candidate_available': candidate_returns is not None
        }
    
    # Align to common dates
    common_dates = base_returns.index.intersection(candidate_returns.index)
    if len(common_dates) == 0:
        return {
            'base_run_id': base_run_id,
            'candidate_run_id': candidate_run_id,
            'generated_at': datetime.now().isoformat(),
            'error': 'No common dates between base and candidate runs'
        }
    
    base_returns = base_returns.loc[common_dates]
    candidate_returns = candidate_returns.loc[common_dates]
    
    # Compute metrics for both
    base_equity = (1 + base_returns).cumprod()
    base_equity.iloc[0] = 1.0
    candidate_equity = (1 + candidate_returns).cumprod()
    candidate_equity.iloc[0] = 1.0
    
    base_metrics = compute_stage_metrics(base_returns, base_equity)
    candidate_metrics = compute_stage_metrics(candidate_returns, candidate_equity)
    
    # Compute deltas
    deltas = {}
    for metric in ['cagr', 'vol', 'sharpe', 'max_drawdown', 'time_under_water']:
        base_val = base_metrics.get(metric)
        cand_val = candidate_metrics.get(metric)
        if base_val is not None and cand_val is not None:
            if not np.isnan(base_val) and not np.isnan(cand_val):
                deltas[f'delta_{metric}'] = cand_val - base_val
            else:
                deltas[f'delta_{metric}'] = None
        else:
            deltas[f'delta_{metric}'] = None
    
    # Regime-conditioned comparison
    # Load regime series if available (use base run's regime classification)
    regime_file = base_run_dir / "allocator_regime_v1.csv"
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
    if len(base_returns) > 21:
        vol_series = base_returns.rolling(21).std() * np.sqrt(252)
    
    base_regime = compute_regime_conditioned_metrics(base_returns, regime_series, vol_series)
    candidate_regime = compute_regime_conditioned_metrics(candidate_returns, regime_series, vol_series)
    
    # Regime deltas
    regime_deltas = {}
    for regime in ['crisis', 'calm', 'high_vol', 'low_vol']:
        base_regime_data = base_regime.get(regime, {})
        cand_regime_data = candidate_regime.get(regime, {})
        
        base_sharpe = base_regime_data.get('sharpe')
        cand_sharpe = cand_regime_data.get('sharpe')
        
        if base_sharpe is not None and cand_sharpe is not None:
            if not np.isnan(base_sharpe) and not np.isnan(cand_sharpe):
                regime_deltas[f'{regime}_delta_sharpe'] = cand_sharpe - base_sharpe
            else:
                regime_deltas[f'{regime}_delta_sharpe'] = None
        else:
            regime_deltas[f'{regime}_delta_sharpe'] = None
    
    # Sleeve-level contribution comparison
    base_sleeve_returns = base_artifacts.get('sleeve_returns')
    candidate_sleeve_returns = candidate_artifacts.get('sleeve_returns')
    
    sleeve_comparison = {}
    if base_sleeve_returns is not None and candidate_sleeve_returns is not None:
        # Find sleeves in both
        base_sleeves = set(base_sleeve_returns.columns)
        candidate_sleeves = set(candidate_sleeve_returns.columns)
        
        common_sleeves = base_sleeves.intersection(candidate_sleeves)
        new_sleeves = candidate_sleeves - base_sleeves
        removed_sleeves = base_sleeves - candidate_sleeves
        
        sleeve_comparison['common_sleeves'] = list(common_sleeves)
        sleeve_comparison['new_sleeves'] = list(new_sleeves)
        sleeve_comparison['removed_sleeves'] = list(removed_sleeves)
        
        # Contribution comparison for common sleeves
        contribution_deltas = {}
        for sleeve in common_sleeves:
            base_contrib = base_sleeve_returns[sleeve].loc[common_dates]
            cand_contrib = candidate_sleeve_returns[sleeve].loc[common_dates]
            
            base_contrib_metrics = compute_contribution_metrics(base_contrib, base_returns)
            cand_contrib_metrics = compute_contribution_metrics(cand_contrib, candidate_returns)
            
            base_pct = base_contrib_metrics.get('total_contribution_pct')
            cand_pct = cand_contrib_metrics.get('total_contribution_pct')
            
            if base_pct is not None and cand_pct is not None:
                contribution_deltas[sleeve] = {
                    'base_contribution_pct': base_pct,
                    'candidate_contribution_pct': cand_pct,
                    'delta_contribution_pct': cand_pct - base_pct
                }
        
        sleeve_comparison['contribution_deltas'] = contribution_deltas
        
        # New sleeve contributions
        new_sleeve_contributions = {}
        for sleeve in new_sleeves:
            cand_contrib = candidate_sleeve_returns[sleeve].loc[common_dates]
            cand_contrib_metrics = compute_contribution_metrics(cand_contrib, candidate_returns)
            new_sleeve_contributions[sleeve] = {
                'contribution_pct': cand_contrib_metrics.get('total_contribution_pct'),
                'contribution_sharpe': cand_contrib_metrics.get('contribution_sharpe'),
                'correlation_with_portfolio': cand_contrib_metrics.get('correlation_with_portfolio')
            }
        
        sleeve_comparison['new_sleeve_contributions'] = new_sleeve_contributions
    
    # Promotion recommendation
    recommendation = compute_promotion_recommendation(
        deltas,
        regime_deltas,
        base_metrics,
        candidate_metrics
    )
    
    report = {
        'base_run_id': base_run_id,
        'candidate_run_id': candidate_run_id,
        'generated_at': datetime.now().isoformat(),
        'stage': 'Post-Construction (Pre-RT)',
        'n_common_dates': len(common_dates),
        'date_range': {
            'start': common_dates[0].strftime('%Y-%m-%d'),
            'end': common_dates[-1].strftime('%Y-%m-%d')
        },
        'base_metrics': base_metrics,
        'candidate_metrics': candidate_metrics,
        'deltas': deltas,
        'regime_deltas': regime_deltas,
        'base_regime': base_regime,
        'candidate_regime': candidate_regime,
        'sleeve_comparison': sleeve_comparison,
        'recommendation': recommendation
    }
    
    return report


def compute_promotion_recommendation(
    deltas: Dict,
    regime_deltas: Dict,
    base_metrics: Dict,
    candidate_metrics: Dict
) -> Dict:
    """
    Compute promotion recommendation based on A/B test results.
    
    Promotion criteria:
    - Sharpe improves or stays similar (delta >= -0.05)
    - MaxDD doesn't worsen significantly (delta <= 0.02, i.e., no more than 2% worse)
    - Crisis regime Sharpe doesn't collapse
    
    Args:
        deltas: Metric deltas
        regime_deltas: Regime-conditioned deltas
        base_metrics: Base portfolio metrics
        candidate_metrics: Candidate portfolio metrics
        
    Returns:
        Dict with recommendation and reasoning
    """
    issues = []
    positives = []
    
    # Check Sharpe
    delta_sharpe = deltas.get('delta_sharpe')
    if delta_sharpe is not None:
        if delta_sharpe >= 0.05:
            positives.append(f"Sharpe improves by {delta_sharpe:.3f}")
        elif delta_sharpe >= -0.05:
            positives.append(f"Sharpe similar (delta {delta_sharpe:.3f})")
        else:
            issues.append(f"Sharpe degrades by {abs(delta_sharpe):.3f}")
    
    # Check MaxDD
    delta_maxdd = deltas.get('delta_max_drawdown')
    if delta_maxdd is not None:
        # Note: max_drawdown is negative, so "worse" means more negative
        if delta_maxdd <= 0.0:  # Candidate has less negative (better) or same MaxDD
            positives.append(f"MaxDD improves or same (delta {delta_maxdd:.2%})")
        elif delta_maxdd <= 0.02:  # Up to 2% worse
            positives.append(f"MaxDD slightly worse (delta {delta_maxdd:.2%})")
        else:
            issues.append(f"MaxDD worsens significantly (delta {delta_maxdd:.2%})")
    
    # Check crisis regime
    crisis_delta = regime_deltas.get('crisis_delta_sharpe')
    if crisis_delta is not None:
        if crisis_delta >= 0:
            positives.append(f"Crisis Sharpe improves (delta {crisis_delta:.3f})")
        elif crisis_delta >= -0.2:
            positives.append(f"Crisis Sharpe similar (delta {crisis_delta:.3f})")
        else:
            issues.append(f"Crisis Sharpe collapses (delta {crisis_delta:.3f})")
    
    # Overall recommendation
    if len(issues) == 0:
        recommendation = "PROMOTE"
        reason = "All promotion criteria met"
    elif len(issues) <= 1 and len(positives) >= 2:
        recommendation = "CONDITIONAL"
        reason = "Minor issues, may promote with caution"
    else:
        recommendation = "DO NOT PROMOTE"
        reason = "Significant issues detected"
    
    return {
        'recommendation': recommendation,
        'reason': reason,
        'positives': positives,
        'issues': issues
    }


def format_ab_test_report(report: Dict) -> str:
    """
    Format A/B test report as Markdown.
    
    Args:
        report: A/B test report dict
        
    Returns:
        Markdown-formatted report string
    """
    lines = []
    lines.append("# Portfolio Construction A/B Test Report")
    lines.append("")
    lines.append(f"**Base Run:** `{report['base_run_id']}`")
    lines.append(f"**Candidate Run:** `{report['candidate_run_id']}`")
    lines.append(f"**Generated:** {report['generated_at']}")
    lines.append(f"**Stage:** {report.get('stage', 'Post-Construction (Pre-RT)')}")
    lines.append("")
    
    if 'error' in report:
        lines.append(f"**Error:** {report['error']}")
        return "\n".join(lines)
    
    date_range = report.get('date_range', {})
    lines.append(f"**Date Range:** {date_range.get('start')} to {date_range.get('end')}")
    lines.append(f"**Common Trading Days:** {report.get('n_common_dates', 0)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Recommendation
    recommendation = report.get('recommendation', {})
    rec_text = recommendation.get('recommendation', 'UNKNOWN')
    rec_reason = recommendation.get('reason', '')
    
    if rec_text == 'PROMOTE':
        lines.append(f"## Recommendation: **{rec_text}**")
    elif rec_text == 'CONDITIONAL':
        lines.append(f"## Recommendation: **{rec_text}**")
    else:
        lines.append(f"## Recommendation: **{rec_text}**")
    
    lines.append(f"*{rec_reason}*")
    lines.append("")
    
    positives = recommendation.get('positives', [])
    issues = recommendation.get('issues', [])
    
    if positives:
        lines.append("**Positives:**")
        for p in positives:
            lines.append(f"- {p}")
        lines.append("")
    
    if issues:
        lines.append("**Issues:**")
        for i in issues:
            lines.append(f"- {i}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Metrics Comparison
    lines.append("## Post-Construction Metrics Comparison")
    lines.append("")
    lines.append("| Metric | Base | Candidate | Delta |")
    lines.append("|--------|-----:|----------:|------:|")
    
    base_metrics = report.get('base_metrics', {})
    candidate_metrics = report.get('candidate_metrics', {})
    deltas = report.get('deltas', {})
    
    metrics_display = [
        ('CAGR', 'cagr', '{:.2%}'),
        ('Volatility', 'vol', '{:.2%}'),
        ('Sharpe', 'sharpe', '{:.3f}'),
        ('Max Drawdown', 'max_drawdown', '{:.2%}'),
        ('Time Under Water', 'time_under_water', '{:.1%}')
    ]
    
    for display_name, metric_key, fmt in metrics_display:
        base_val = base_metrics.get(metric_key)
        cand_val = candidate_metrics.get(metric_key)
        delta_val = deltas.get(f'delta_{metric_key}')
        
        base_str = fmt.format(base_val) if base_val is not None and not np.isnan(base_val) else "N/A"
        cand_str = fmt.format(cand_val) if cand_val is not None and not np.isnan(cand_val) else "N/A"
        delta_str = fmt.format(delta_val) if delta_val is not None else "N/A"
        
        lines.append(f"| {display_name} | {base_str} | {cand_str} | {delta_str} |")
    
    lines.append("")
    
    # Regime Comparison
    lines.append("## Regime-Conditioned Performance")
    lines.append("")
    lines.append("| Regime | Base Sharpe | Candidate Sharpe | Delta |")
    lines.append("|--------|------------:|-----------------:|------:|")
    
    base_regime = report.get('base_regime', {})
    candidate_regime = report.get('candidate_regime', {})
    regime_deltas = report.get('regime_deltas', {})
    
    for regime in ['crisis', 'calm', 'high_vol', 'low_vol']:
        base_sharpe = base_regime.get(regime, {}).get('sharpe')
        cand_sharpe = candidate_regime.get(regime, {}).get('sharpe')
        delta_sharpe = regime_deltas.get(f'{regime}_delta_sharpe')
        
        base_str = f"{base_sharpe:.3f}" if base_sharpe is not None and not np.isnan(base_sharpe) else "N/A"
        cand_str = f"{cand_sharpe:.3f}" if cand_sharpe is not None and not np.isnan(cand_sharpe) else "N/A"
        delta_str = f"{delta_sharpe:.3f}" if delta_sharpe is not None else "N/A"
        
        lines.append(f"| {regime.replace('_', ' ').title()} | {base_str} | {cand_str} | {delta_str} |")
    
    lines.append("")
    
    # Sleeve Comparison
    sleeve_comparison = report.get('sleeve_comparison', {})
    if sleeve_comparison:
        lines.append("## Sleeve Comparison")
        lines.append("")
        
        new_sleeves = sleeve_comparison.get('new_sleeves', [])
        removed_sleeves = sleeve_comparison.get('removed_sleeves', [])
        
        if new_sleeves:
            lines.append("### New Sleeves in Candidate")
            lines.append("")
            new_contributions = sleeve_comparison.get('new_sleeve_contributions', {})
            for sleeve in new_sleeves:
                contrib = new_contributions.get(sleeve, {})
                contrib_pct = contrib.get('contribution_pct')
                contrib_sharpe = contrib.get('contribution_sharpe')
                correlation = contrib.get('correlation_with_portfolio')
                
                contrib_str = f"{contrib_pct:.1%}" if contrib_pct is not None else "N/A"
                sharpe_str = f"{contrib_sharpe:.2f}" if contrib_sharpe is not None else "N/A"
                corr_str = f"{correlation:.2f}" if correlation is not None else "N/A"
                
                lines.append(f"- **{sleeve}**: Contrib {contrib_str}, Sharpe {sharpe_str}, Corr {corr_str}")
            lines.append("")
        
        if removed_sleeves:
            lines.append("### Removed Sleeves from Base")
            lines.append("")
            for sleeve in removed_sleeves:
                lines.append(f"- {sleeve}")
            lines.append("")
        
        contribution_deltas = sleeve_comparison.get('contribution_deltas', {})
        if contribution_deltas:
            lines.append("### Common Sleeve Contribution Deltas")
            lines.append("")
            lines.append("| Sleeve | Base Contrib | Cand Contrib | Delta |")
            lines.append("|--------|-------------:|-------------:|------:|")
            
            for sleeve, data in sorted(contribution_deltas.items(), 
                                        key=lambda x: abs(x[1].get('delta_contribution_pct', 0) or 0),
                                        reverse=True):
                base_pct = data.get('base_contribution_pct')
                cand_pct = data.get('candidate_contribution_pct')
                delta_pct = data.get('delta_contribution_pct')
                
                base_str = f"{base_pct:.1%}" if base_pct is not None else "N/A"
                cand_str = f"{cand_pct:.1%}" if cand_pct is not None else "N/A"
                delta_str = f"{delta_pct:.1%}" if delta_pct is not None else "N/A"
                
                lines.append(f"| {sleeve} | {base_str} | {cand_str} | {delta_str} |")
            
            lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("**Note:** This A/B test uses Construction v1 rules (static sleeve weighting). "
                 "The comparison is at Post-Construction stage, the canonical system belief evaluation layer.")
    lines.append("")
    
    return "\n".join(lines)
