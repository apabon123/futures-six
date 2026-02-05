"""
Allocator v1 Baseline Analysis

Extracts Allocator v1 facts from the pinned RT-frozen run and compares to distribution targets.
Implements the "what good looks like" contract for Allocator v1.

Usage:
    python scripts/diagnostics/analyze_allocator_v1_baseline.py <run_id>
    
Example:
    python scripts/diagnostics/analyze_allocator_v1_baseline.py canonical_frozen_stack_compute_phase3a_governed_20260114_230316
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_allocator_artifacts(run_dir: Path) -> Dict:
    """Load all Allocator v1-related artifacts from run directory."""
    artifacts = {}
    
    # Load meta.json
    meta_file = run_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json not found in {run_dir}")
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        artifacts['meta'] = json.load(f)
    
    # Load allocator regime series
    regime_file = run_dir / "allocator" / "regime_series.csv"
    if regime_file.exists():
        regime_df = pd.read_csv(regime_file, parse_dates=['date'], index_col='date')
        artifacts['regime_series'] = regime_df['regime'] if 'regime' in regime_df.columns else None
    else:
        raise FileNotFoundError(f"regime_series.csv not found in {run_dir / 'allocator'}")
    
    # Load allocator multiplier series
    multiplier_file = run_dir / "allocator" / "multiplier_series.csv"
    if multiplier_file.exists():
        multiplier_df = pd.read_csv(multiplier_file, parse_dates=['date'], index_col='date')
        artifacts['multiplier_series'] = multiplier_df['multiplier'] if 'multiplier' in multiplier_df.columns else None
    else:
        raise FileNotFoundError(f"multiplier_series.csv not found in {run_dir / 'allocator'}")
    
    # Load RT leverage series for joint analysis
    rt_leverage_file = run_dir / "risk_targeting" / "leverage_series.csv"
    if rt_leverage_file.exists():
        leverage_df = pd.read_csv(rt_leverage_file, parse_dates=['date'], index_col='date')
        artifacts['rt_leverage_series'] = leverage_df['leverage']
    else:
        print(f"Warning: RT leverage_series.csv not found, joint analysis will be limited")
        artifacts['rt_leverage_series'] = None
    
    return artifacts


def compute_allocator_governance(artifacts: Dict) -> Dict:
    """Extract Allocator v1 governance facts from meta.json and artifacts."""
    meta = artifacts['meta']
    alloc_meta = meta.get('allocator_v1', {})
    multiplier_series = artifacts['multiplier_series']
    
    governance = {
        'alloc_enabled': alloc_meta.get('enabled', False),
        'alloc_effective': alloc_meta.get('effective', False),
        'alloc_has_teeth': alloc_meta.get('has_teeth', False),
        'scalar_stats': alloc_meta.get('scalar_stats', {}),
        'regime_distribution': alloc_meta.get('regime_distribution', {})
    }
    
    # Recompute scalar stats if missing or NaN
    if multiplier_series is not None and len(multiplier_series) > 0:
        multiplier_values = multiplier_series.dropna()
        if len(multiplier_values) > 0:
            governance['scalar_stats'] = {
                'p5': float(np.percentile(multiplier_values, 5)),
                'p50': float(np.percentile(multiplier_values, 50)),
                'p95': float(np.percentile(multiplier_values, 95)),
                'p10': float(np.percentile(multiplier_values, 10)),
                'p90': float(np.percentile(multiplier_values, 90)),
                'at_min': float(np.sum(multiplier_values <= 0.25 + 1e-6) / len(multiplier_values) * 100)
            }
            
            # Active percentage (scalar != 1.0)
            governance['active_pct'] = float((np.abs(multiplier_values - 1.0) > 1e-6).sum() / len(multiplier_values) * 100)
    
    # Recompute regime distribution if missing
    regime_series = artifacts.get('regime_series')
    if regime_series is not None and len(regime_series) > 0:
        regime_counts = Counter(regime_series.dropna())
        total = len(regime_series.dropna())
        if total > 0:
            governance['regime_distribution'] = {
                k: float(v / total * 100) for k, v in regime_counts.items()
            }
    
    return governance


def analyze_allocator_regime_distribution(regime_series: pd.Series) -> Dict:
    """Analyze allocator regime distribution."""
    if regime_series is None or len(regime_series) == 0:
        return {}
    
    regime = regime_series.dropna()
    if len(regime) == 0:
        return {}
    
    regime_counts = Counter(regime)
    total = len(regime)
    
    # Map to canonical regime names
    canonical_regimes = ['NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']
    distribution = {}
    for regime_name in canonical_regimes:
        count = regime_counts.get(regime_name, 0)
        distribution[regime_name] = {
            'count': count,
            'pct': float(count / total * 100) if total > 0 else 0.0
        }
    
    # Add any non-canonical regimes
    for regime_name, count in regime_counts.items():
        if regime_name not in canonical_regimes:
            distribution[regime_name] = {
                'count': count,
                'pct': float(count / total * 100) if total > 0 else 0.0
            }
    
    # Compute regime transitions
    transitions = {}
    regime_list = regime.tolist()
    for i in range(len(regime_list) - 1):
        from_regime = regime_list[i]
        to_regime = regime_list[i + 1]
        if from_regime != to_regime:
            key = f"{from_regime}->{to_regime}"
            transitions[key] = transitions.get(key, 0) + 1
    
    # Compute max consecutive days per regime
    max_consecutive = {}
    current_regime = None
    current_count = 0
    for regime_name in regime_list:
        if regime_name == current_regime:
            current_count += 1
        else:
            if current_regime is not None:
                if current_regime not in max_consecutive or current_count > max_consecutive[current_regime]:
                    max_consecutive[current_regime] = current_count
            current_regime = regime_name
            current_count = 1
    # Handle last regime
    if current_regime is not None:
        if current_regime not in max_consecutive or current_count > max_consecutive[current_regime]:
            max_consecutive[current_regime] = current_count
    
    return {
        'distribution': distribution,
        'total_days': total,
        'transitions': transitions,
        'max_consecutive': max_consecutive
    }


def analyze_allocator_scalar_distribution(multiplier_series: pd.Series) -> Dict:
    """Analyze allocator scalar distribution."""
    if multiplier_series is None or len(multiplier_series) == 0:
        return {}
    
    multiplier = multiplier_series.dropna()
    if len(multiplier) == 0:
        return {}
    
    return {
        'p5': float(np.percentile(multiplier, 5)),
        'p10': float(np.percentile(multiplier, 10)),
        'p50': float(np.percentile(multiplier, 50)),
        'p90': float(np.percentile(multiplier, 90)),
        'p95': float(np.percentile(multiplier, 95)),
        'min': float(multiplier.min()),
        'max': float(multiplier.max()),
        'mean': float(multiplier.mean()),
        'std': float(multiplier.std()),
        'active_pct': float((np.abs(multiplier - 1.0) > 1e-6).sum() / len(multiplier) * 100),
        'pct_below_0.9': float((multiplier < 0.9).sum() / len(multiplier) * 100),
        'pct_below_0.8': float((multiplier < 0.8).sum() / len(multiplier) * 100),
        'pct_below_0.7': float((multiplier < 0.7).sum() / len(multiplier) * 100),
        'pct_at_min': float((multiplier <= 0.25 + 1e-6).sum() / len(multiplier) * 100)
    }


def analyze_joint_rt_allocator_behavior(allocator_multiplier: pd.Series, rt_leverage: pd.Series) -> Dict:
    """Analyze joint behavior of RT and Allocator."""
    if allocator_multiplier is None or rt_leverage is None:
        return {}
    
    # Align series
    common_dates = allocator_multiplier.index.intersection(rt_leverage.index)
    if len(common_dates) == 0:
        return {}
    
    alloc_aligned = allocator_multiplier.loc[common_dates]
    rt_aligned = rt_leverage.loc[common_dates]
    
    # When allocator clips (multiplier < 1.0), what was RT multiplier?
    alloc_clipped_mask = (alloc_aligned < 1.0 - 1e-6)
    rt_when_alloc_clipped = rt_aligned[alloc_clipped_mask]
    
    # When RT is high (e.g., > p90), is allocator active?
    rt_p90 = rt_aligned.quantile(0.90)
    rt_high_mask = (rt_aligned >= rt_p90)
    alloc_when_rt_high = alloc_aligned[rt_high_mask]
    
    # When RT is low (e.g., < p10), is allocator inactive?
    rt_p10 = rt_aligned.quantile(0.10)
    rt_low_mask = (rt_aligned <= rt_p10)
    alloc_when_rt_low = alloc_aligned[rt_low_mask]
    
    joint_analysis = {
        'rt_when_alloc_clipped': {
            'count': len(rt_when_alloc_clipped),
            'median': float(rt_when_alloc_clipped.median()) if len(rt_when_alloc_clipped) > 0 else None,
            'p25': float(rt_when_alloc_clipped.quantile(0.25)) if len(rt_when_alloc_clipped) > 0 else None,
            'p75': float(rt_when_alloc_clipped.quantile(0.75)) if len(rt_when_alloc_clipped) > 0 else None,
            'mean': float(rt_when_alloc_clipped.mean()) if len(rt_when_alloc_clipped) > 0 else None
        },
        'alloc_when_rt_high': {
            'count': len(alloc_when_rt_high),
            'median': float(alloc_when_rt_high.median()) if len(alloc_when_rt_high) > 0 else None,
            'p25': float(alloc_when_rt_high.quantile(0.25)) if len(alloc_when_rt_high) > 0 else None,
            'p75': float(alloc_when_rt_high.quantile(0.75)) if len(alloc_when_rt_high) > 0 else None,
            'active_pct': float((np.abs(alloc_when_rt_high - 1.0) > 1e-6).sum() / len(alloc_when_rt_high) * 100) if len(alloc_when_rt_high) > 0 else 0.0
        },
        'alloc_when_rt_low': {
            'count': len(alloc_when_rt_low),
            'median': float(alloc_when_rt_low.median()) if len(alloc_when_rt_low) > 0 else None,
            'active_pct': float((np.abs(alloc_when_rt_low - 1.0) > 1e-6).sum() / len(alloc_when_rt_low) * 100) if len(alloc_when_rt_low) > 0 else 0.0
        }
    }
    
    return joint_analysis


def analyze_allocator_time_clustering(multiplier_series: pd.Series) -> Dict:
    """Analyze when allocator is active, especially during known stress periods."""
    multiplier = multiplier_series.dropna()
    
    # Define periods of interest (known stress periods)
    periods = {
        '2020_Q1': (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-03-31')),
        '2020_Q2': (pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30')),
        '2022_Q1': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-03-31')),
        '2022_Q2': (pd.Timestamp('2022-04-01'), pd.Timestamp('2022-06-30')),
        '2022_H1': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-06-30')),
    }
    
    clustering = {}
    overall_active_pct = float((np.abs(multiplier - 1.0) > 1e-6).sum() / len(multiplier) * 100) if len(multiplier) > 0 else 0.0
    overall_median = float(multiplier.median()) if len(multiplier) > 0 else 1.0
    
    for period_name, (start, end) in periods.items():
        period_mult = multiplier[(multiplier.index >= start) & (multiplier.index <= end)]
        if len(period_mult) > 0:
            clustering[period_name] = {
                'count': len(period_mult),
                'median': float(period_mult.median()),
                'p25': float(period_mult.quantile(0.25)),
                'p75': float(period_mult.quantile(0.75)),
                'min': float(period_mult.min()),
                'max': float(period_mult.max()),
                'active_pct': float((np.abs(period_mult - 1.0) > 1e-6).sum() / len(period_mult) * 100),
                'vs_overall_median': float(period_mult.median() - overall_median)
            }
    
    return clustering


def compare_to_targets(governance: Dict, regime_analysis: Dict, scalar_analysis: Dict) -> Dict:
    """Compare Allocator outputs to distribution targets."""
    targets = {
        'regime_full_risk_pct': (70.0, 85.0),      # NORMAL regime
        'regime_clipped_risk_pct': (10.0, 25.0),   # ELEVATED/STRESS combined
        'regime_off_pct': (0.0, 5.0),              # CRISIS or near-off
        'scalar_p50_target': 1.0,                  # Should be ~1.0 (mostly inactive)
        'scalar_p90_max': 1.0,                     # p90 should be <= 1.0
        'scalar_p5_min': 0.3,                      # p5 should be meaningfully < 1.0
        'scalar_p5_max': 0.6,
        'active_pct_max': 30.0                     # Should be rarely active (<30%)
    }
    
    comparison = {
        'targets': targets,
        'actuals': {},
        'status': {}
    }
    
    # Regime distribution targets
    regime_dist = governance.get('regime_distribution', {})
    normal_pct = regime_dist.get('NORMAL', 0.0)
    elevated_pct = regime_dist.get('ELEVATED', 0.0)
    stress_pct = regime_dist.get('STRESS', 0.0)
    crisis_pct = regime_dist.get('CRISIS', 0.0)
    
    clipped_pct = elevated_pct + stress_pct
    off_pct = crisis_pct
    
    comparison['actuals']['regime_full_risk_pct'] = normal_pct
    comparison['actuals']['regime_clipped_risk_pct'] = clipped_pct
    comparison['actuals']['regime_off_pct'] = off_pct
    
    # Scalar distribution targets
    scalar_stats = governance.get('scalar_stats', {})
    p50_scalar = scalar_stats.get('p50')
    p90_scalar = scalar_stats.get('p90')
    p5_scalar = scalar_stats.get('p5')
    active_pct = governance.get('active_pct', 0.0)
    
    comparison['actuals']['scalar_p50'] = p50_scalar
    comparison['actuals']['scalar_p90'] = p90_scalar
    comparison['actuals']['scalar_p5'] = p5_scalar
    comparison['actuals']['active_pct'] = active_pct
    
    # Check targets
    comparison['status']['regime_full_risk'] = (
        targets['regime_full_risk_pct'][0] <= normal_pct <= targets['regime_full_risk_pct'][1]
    )
    comparison['status']['regime_clipped_risk'] = (
        targets['regime_clipped_risk_pct'][0] <= clipped_pct <= targets['regime_clipped_risk_pct'][1]
    )
    comparison['status']['regime_off'] = (
        targets['regime_off_pct'][0] <= off_pct <= targets['regime_off_pct'][1]
    )
    
    if p50_scalar is not None:
        comparison['status']['scalar_p50'] = abs(p50_scalar - 1.0) < 0.05  # Within 5% of 1.0
    else:
        comparison['status']['scalar_p50'] = False
    
    if p90_scalar is not None:
        comparison['status']['scalar_p90'] = p90_scalar <= targets['scalar_p90_max']
    else:
        comparison['status']['scalar_p90'] = False
    
    if p5_scalar is not None:
        comparison['status']['scalar_p5'] = (
            targets['scalar_p5_min'] <= p5_scalar <= targets['scalar_p5_max']
        )
    else:
        comparison['status']['scalar_p5'] = False
    
    comparison['status']['active_pct'] = active_pct <= targets['active_pct_max']
    
    return comparison


def diagnose_failure_modes(governance: Dict, regime_analysis: Dict, scalar_analysis: Dict,
                           clustering: Dict, joint_analysis: Dict) -> Dict:
    """Diagnose common Allocator v1 failure modes."""
    diagnostics = {
        'failure_modes': [],
        'recommendations': []
    }
    
    scalar_stats = governance.get('scalar_stats', {})
    active_pct = governance.get('active_pct', 0.0)
    p50_scalar = scalar_stats.get('p50')
    p5_scalar = scalar_stats.get('p5')
    
    # Failure Mode A: Allocator inert
    if active_pct < 1.0 or (p50_scalar is not None and abs(p50_scalar - 1.0) < 1e-6):
        diagnostics['failure_modes'].append({
            'mode': 'A',
            'description': 'Allocator inert (not active enough)',
            'actual': f'Active % = {active_pct:.1f}%, p50 scalar = {p50_scalar:.3f}' if p50_scalar else f'Active % = {active_pct:.1f}%',
            'target': 'Active % > 1%, p50 scalar ≈ 1.0 (but some variation)',
            'likely_causes': [
                'Regime thresholds too conservative',
                'Regime classifier not detecting stress',
                'Risk scalar mapping too lenient'
            ],
            'fix': 'Tune regime thresholds or risk scalar mapping'
        })
    
    # Failure Mode B: Allocator always-on
    if active_pct > 50.0 or (p50_scalar is not None and p50_scalar < 0.9):
        diagnostics['failure_modes'].append({
            'mode': 'B',
            'description': 'Allocator always-on (doing RT\'s job)',
            'actual': f'Active % = {active_pct:.1f}%, p50 scalar = {p50_scalar:.3f}' if p50_scalar else f'Active % = {active_pct:.1f}%',
            'target': 'Active % < 30%, p50 scalar ≈ 1.0',
            'likely_causes': [
                'Regime thresholds too aggressive',
                'Allocator scaling too frequently',
                'Overlapping with RT responsibility'
            ],
            'fix': 'Raise regime thresholds or reduce allocator sensitivity'
        })
    
    # Failure Mode C: Allocator mistimed
    # Check if allocator is active in calm periods but not in known stress
    mistimed = False
    if clustering:
        # Check known stress periods (2020 Q1, 2022 H1)
        stress_periods = ['2020_Q1', '2022_H1']
        overall_active_pct = active_pct
        
        for period in stress_periods:
            if period in clustering:
                period_active = clustering[period].get('active_pct', 0.0)
                period_median = clustering[period].get('median', 1.0)
                
                # Allocator should be MORE active in stress periods
                if period_active < overall_active_pct * 0.8 or period_median > 0.95:
                    mistimed = True
                    break
    
    # Also check if allocator is active when RT is low (shouldn't be)
    if joint_analysis:
        alloc_when_rt_low = joint_analysis.get('alloc_when_rt_low', {})
        if alloc_when_rt_low.get('active_pct', 0.0) > 10.0:
            mistimed = True
    
    if mistimed:
        diagnostics['failure_modes'].append({
            'mode': 'C',
            'description': 'Allocator mistimed (active in calm, inactive in stress)',
            'actual': 'Allocator firing pattern doesn\'t align with known stress periods',
            'target': 'Active during known stress (2020 Q1, 2022 H1), inactive in calm',
            'likely_causes': [
                'Regime signal wrong (not detecting actual stress)',
                'State features not capturing true risk',
                'Regime thresholds misaligned with market conditions'
            ],
            'fix': 'Review regime classification logic and state feature definitions'
        })
    
    # Failure Mode D: Allocator too violent
    # Check for sharp on/off toggling and large scalars without persistence
    if regime_analysis:
        transitions = regime_analysis.get('transitions', {})
        total_days = regime_analysis.get('total_days', 1)
        transition_rate = sum(transitions.values()) / total_days * 100 if total_days > 0 else 0
        
        # Check max consecutive days (should have some persistence)
        max_consecutive = regime_analysis.get('max_consecutive', {})
        min_persistence = min(max_consecutive.values()) if max_consecutive else 0
        
        # Check scalar volatility
        scalar_std = scalar_analysis.get('std', 0.0)
        
        if transition_rate > 5.0 or min_persistence < 3 or scalar_std > 0.2:
            diagnostics['failure_modes'].append({
                'mode': 'D',
                'description': 'Allocator too violent (sharp toggling, no persistence)',
                'actual': f'Transition rate = {transition_rate:.1f}%/day, min persistence = {min_persistence}d, scalar std = {scalar_std:.3f}',
                'target': 'Transition rate < 5%/day, min persistence ≥ 3d, scalar std < 0.2',
                'likely_causes': [
                    'Regime hysteresis insufficient',
                    'Anti-thrash mechanism too weak',
                    'Risk scalar smoothing too reactive'
                ],
                'fix': 'Increase regime hysteresis, strengthen anti-thrash, increase EWMA smoothing'
            })
    
    return diagnostics


def generate_report(governance: Dict, regime_analysis: Dict, scalar_analysis: Dict,
                   clustering: Dict, joint_analysis: Dict, comparison: Dict,
                   diagnostics: Dict, run_id: str) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Allocator v1 Baseline Analysis")
    lines.append("")
    lines.append(f"**Run ID:** `{run_id}`")
    lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Allocator Contract Reminder
    lines.append("## Allocator v1 Contract")
    lines.append("")
    lines.append("**Role:** survival + regime brake")
    lines.append("")
    lines.append("**NOT:** alpha, volatility targeting, return optimization")
    lines.append("")
    lines.append("**Question Allocator Answers:**")
    lines.append("> \"Given the current regime, should we run the RT-sized book, clip it, or stand down?\"")
    lines.append("")
    lines.append("**Properties:**")
    lines.append("- Must be stateful")
    lines.append("- Must be discrete or low-dimensional")
    lines.append("- Must be rarely active")
    lines.append("- When active, must be obviously justified")
    lines.append("- If allocator is 'busy' every week, it's doing too much")
    lines.append("")
    
    # Step 1: Allocator Facts
    lines.append("## Step 1: Allocator Facts from Baseline")
    lines.append("")
    lines.append("### Governance/Telemetry")
    lines.append("")
    lines.append(f"- **alloc_enabled:** {governance.get('alloc_enabled', False)}")
    lines.append(f"- **alloc_effective:** {governance.get('alloc_effective', False)}")
    lines.append(f"- **alloc_has_teeth:** {governance.get('alloc_has_teeth', False)}")
    lines.append(f"- **active_pct:** {governance.get('active_pct', 0.0):.1f}%")
    lines.append("")
    
    # Regime distribution
    lines.append("### Regime Distribution")
    lines.append("")
    regime_dist = governance.get('regime_distribution', {})
    for regime_name in ['NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']:
        pct = regime_dist.get(regime_name, 0.0)
        lines.append(f"- **{regime_name}:** {pct:.1f}%")
    lines.append("")
    
    # Regime analysis details
    if regime_analysis:
        lines.append("### Regime Details")
        lines.append("")
        transitions = regime_analysis.get('transitions', {})
        if transitions:
            lines.append("**Regime Transitions:**")
            lines.append("")
            for transition, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"- {transition}: {count} times")
            lines.append("")
        
        max_consecutive = regime_analysis.get('max_consecutive', {})
        if max_consecutive:
            lines.append("**Max Consecutive Days per Regime:**")
            lines.append("")
            for regime, days in max_consecutive.items():
                lines.append(f"- {regime}: {days} days")
            lines.append("")
    
    # Scalar statistics
    lines.append("### Scalar Distribution")
    lines.append("")
    scalar_stats = governance.get('scalar_stats', {})
    lines.append(f"- **p5:** {scalar_stats.get('p5', 'N/A'):.3f}" if scalar_stats.get('p5') is not None else "- **p5:** N/A")
    lines.append(f"- **p50:** {scalar_stats.get('p50', 'N/A'):.3f}" if scalar_stats.get('p50') is not None else "- **p50:** N/A")
    lines.append(f"- **p95:** {scalar_stats.get('p95', 'N/A'):.3f}" if scalar_stats.get('p95') is not None else "- **p95:** N/A")
    lines.append(f"- **Active % (scalar ≠ 1.0):** {governance.get('active_pct', 0.0):.1f}%")
    lines.append("")
    
    if scalar_analysis:
        lines.append("**Additional Scalar Statistics:**")
        lines.append("")
        lines.append(f"- **Min:** {scalar_analysis.get('min', 'N/A'):.3f}" if scalar_analysis.get('min') is not None else "- **Min:** N/A")
        lines.append(f"- **Max:** {scalar_analysis.get('max', 'N/A'):.3f}" if scalar_analysis.get('max') is not None else "- **Max:** N/A")
        lines.append(f"- **Std:** {scalar_analysis.get('std', 'N/A'):.3f}" if scalar_analysis.get('std') is not None else "- **Std:** N/A")
        lines.append(f"- **% < 0.9:** {scalar_analysis.get('pct_below_0.9', 0.0):.1f}%")
        lines.append(f"- **% < 0.8:** {scalar_analysis.get('pct_below_0.8', 0.0):.1f}%")
        lines.append(f"- **% at min (≤0.25):** {scalar_analysis.get('pct_at_min', 0.0):.1f}%")
        lines.append("")
    
    # Joint RT-Allocator analysis
    if joint_analysis:
        lines.append("### Joint Behavior with RT")
        lines.append("")
        
        rt_when_clipped = joint_analysis.get('rt_when_alloc_clipped', {})
        if rt_when_clipped.get('count', 0) > 0:
            lines.append("**When Allocator Clips (multiplier < 1.0):**")
            lines.append("")
            lines.append(f"- RT multiplier median: {rt_when_clipped.get('median', 'N/A'):.3f}" if rt_when_clipped.get('median') is not None else "- RT multiplier median: N/A")
            lines.append(f"- RT multiplier mean: {rt_when_clipped.get('mean', 'N/A'):.3f}" if rt_when_clipped.get('mean') is not None else "- RT multiplier mean: N/A")
            lines.append(f"- Count: {rt_when_clipped.get('count', 0)}")
            lines.append("")
        
        alloc_when_rt_high = joint_analysis.get('alloc_when_rt_high', {})
        if alloc_when_rt_high.get('count', 0) > 0:
            lines.append("**When RT is High (≥p90):**")
            lines.append("")
            lines.append(f"- Allocator active %: {alloc_when_rt_high.get('active_pct', 0.0):.1f}%")
            lines.append(f"- Allocator scalar median: {alloc_when_rt_high.get('median', 'N/A'):.3f}" if alloc_when_rt_high.get('median') is not None else "- Allocator scalar median: N/A")
            lines.append("")
        
        alloc_when_rt_low = joint_analysis.get('alloc_when_rt_low', {})
        if alloc_when_rt_low.get('count', 0) > 0:
            lines.append("**When RT is Low (≤p10):**")
            lines.append("")
            lines.append(f"- Allocator active %: {alloc_when_rt_low.get('active_pct', 0.0):.1f}%")
            lines.append(f"- Allocator scalar median: {alloc_when_rt_low.get('median', 'N/A'):.3f}" if alloc_when_rt_low.get('median') is not None else "- Allocator scalar median: N/A")
            lines.append("")
    
    # Time clustering
    if clustering:
        lines.append("### Time Clustering: Allocator Activity by Period")
        lines.append("")
        lines.append("| Period | Count | Median Scalar | Active % | vs Overall Median |")
        lines.append("|--------|-------|---------------|----------|-------------------|")
        overall_median = scalar_stats.get('p50', 1.0)
        for period_name, period_data in clustering.items():
            lines.append(
                f"| {period_name} | {period_data['count']} | {period_data['median']:.3f} | "
                f"{period_data['active_pct']:.1f}% | {period_data['vs_overall_median']:+.3f} |"
            )
        lines.append("")
    
    # Step 2: Compare to Targets
    lines.append("## Step 2: Compare Allocator Outputs to Distribution Targets")
    lines.append("")
    lines.append("### Targets")
    lines.append("")
    targets = comparison.get('targets', {})
    lines.append(f"- **Full risk (NORMAL) %:** {targets.get('regime_full_risk_pct', (0, 0))[0]:.0f}% - {targets.get('regime_full_risk_pct', (0, 0))[1]:.0f}%")
    lines.append(f"- **Clipped risk (ELEVATED/STRESS) %:** {targets.get('regime_clipped_risk_pct', (0, 0))[0]:.0f}% - {targets.get('regime_clipped_risk_pct', (0, 0))[1]:.0f}%")
    lines.append(f"- **Off (CRISIS) %:** {targets.get('regime_off_pct', (0, 0))[0]:.0f}% - {targets.get('regime_off_pct', (0, 0))[1]:.0f}%")
    lines.append(f"- **Scalar p50:** ≈ {targets.get('scalar_p50_target', 1.0):.2f}")
    lines.append(f"- **Scalar p90:** ≤ {targets.get('scalar_p90_max', 1.0):.2f}")
    lines.append(f"- **Scalar p5:** {targets.get('scalar_p5_min', 0.3):.2f} - {targets.get('scalar_p5_max', 0.6):.2f}")
    lines.append(f"- **Active %:** < {targets.get('active_pct_max', 30.0):.0f}%")
    lines.append("")
    
    lines.append("### Actuals")
    lines.append("")
    actuals = comparison.get('actuals', {})
    lines.append(f"- **Full risk (NORMAL) %:** {actuals.get('regime_full_risk_pct', 0.0):.1f}%")
    lines.append(f"- **Clipped risk (ELEVATED/STRESS) %:** {actuals.get('regime_clipped_risk_pct', 0.0):.1f}%")
    lines.append(f"- **Off (CRISIS) %:** {actuals.get('regime_off_pct', 0.0):.1f}%")
    if actuals.get('scalar_p50') is not None:
        lines.append(f"- **Scalar p50:** {actuals['scalar_p50']:.3f}")
    if actuals.get('scalar_p90') is not None:
        lines.append(f"- **Scalar p90:** {actuals['scalar_p90']:.3f}")
    if actuals.get('scalar_p5') is not None:
        lines.append(f"- **Scalar p5:** {actuals['scalar_p5']:.3f}")
    lines.append(f"- **Active %:** {actuals.get('active_pct', 0.0):.1f}%")
    lines.append("")
    
    lines.append("### Status")
    lines.append("")
    status = comparison.get('status', {})
    for key, value in status.items():
        status_symbol = "✅" if value else "❌"
        lines.append(f"- {status_symbol} **{key}:** {value}")
    lines.append("")
    
    # Step 3: Diagnose Failure Modes
    lines.append("## Step 3: Diagnose Failure Modes")
    lines.append("")
    
    failure_modes = diagnostics.get('failure_modes', [])
    if not failure_modes:
        lines.append("✅ **No failure modes detected.** Allocator appears to be functioning within expected parameters.")
        lines.append("")
    else:
        for mode_info in failure_modes:
            lines.append(f"### Failure Mode {mode_info['mode']}: {mode_info['description']}")
            lines.append("")
            lines.append(f"- **Actual:** {mode_info['actual']}")
            lines.append(f"- **Target:** {mode_info['target']}")
            lines.append(f"- **Likely Causes:**")
            for cause in mode_info['likely_causes']:
                lines.append(f"  - {cause}")
            lines.append(f"- **Fix:** {mode_info['fix']}")
            lines.append("")
    
    # Acceptance Criteria
    lines.append("## Step 4: Acceptance Criteria to 'Freeze Allocator'")
    lines.append("")
    
    acceptance_criteria = []
    
    # Governance passes
    gov_passes = (
        governance.get('alloc_enabled', False) and
        governance.get('alloc_effective', False) and
        governance.get('alloc_has_teeth', False)
    )
    acceptance_criteria.append(("Governance passes", gov_passes))
    
    # Distribution targets hit
    all_targets_pass = all(comparison.get('status', {}).values())
    acceptance_criteria.append(("Distribution targets hit", all_targets_pass))
    
    # No failure modes
    acceptance_criteria.append(("No failure modes detected", len(failure_modes) == 0))
    
    # Temporal clustering (should be active in known stress)
    interpretability_ok = True
    overall_active_pct = governance.get('active_pct', 0.0)
    if clustering:
        stress_periods = ['2020_Q1', '2022_H1']
        for period in stress_periods:
            if period in clustering:
                period_active = clustering[period].get('active_pct', 0.0)
                period_median = clustering[period].get('median', 1.0)
                # Should be more active in stress, or at least not less active
                if period_active < overall_active_pct * 0.8 and period_median > 0.95:
                    interpretability_ok = False
                    break
    
    acceptance_criteria.append(("Interpretability sanity (active in known stress)", interpretability_ok))
    
    lines.append("| Criterion | Status |")
    lines.append("|-----------|-------|")
    for criterion, passed in acceptance_criteria:
        status_symbol = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"| {criterion} | {status_symbol} |")
    lines.append("")
    
    all_passed = all(criteria[1] for criteria in acceptance_criteria)
    if all_passed:
        lines.append("**✅ Allocator is ready to freeze.**")
    else:
        lines.append("**❌ Allocator is NOT ready to freeze.** Address failure modes above.")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnostics/analyze_allocator_v1_baseline.py <run_id>")
        print("\nExample:")
        print("  python scripts/diagnostics/analyze_allocator_v1_baseline.py canonical_frozen_stack_compute_phase3a_governed_20260114_230316")
        sys.exit(1)
    
    run_id = sys.argv[1]
    run_dir = project_root / "reports" / "runs" / run_id
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    print(f"Loading Allocator artifacts from {run_dir}...")
    artifacts = load_allocator_artifacts(run_dir)
    
    print("Computing Allocator governance...")
    governance = compute_allocator_governance(artifacts)
    
    print("Analyzing regime distribution...")
    regime_analysis = analyze_allocator_regime_distribution(artifacts.get('regime_series'))
    
    print("Analyzing scalar distribution...")
    scalar_analysis = analyze_allocator_scalar_distribution(artifacts.get('multiplier_series'))
    
    print("Analyzing time clustering...")
    clustering = analyze_allocator_time_clustering(artifacts.get('multiplier_series'))
    
    print("Analyzing joint RT-Allocator behavior...")
    joint_analysis = analyze_joint_rt_allocator_behavior(
        artifacts.get('multiplier_series'),
        artifacts.get('rt_leverage_series')
    )
    
    print("Comparing to targets...")
    comparison = compare_to_targets(governance, regime_analysis, scalar_analysis)
    
    print("Diagnosing failure modes...")
    diagnostics = diagnose_failure_modes(governance, regime_analysis, scalar_analysis,
                                        clustering, joint_analysis)
    
    print("Generating report...")
    report = generate_report(governance, regime_analysis, scalar_analysis, clustering,
                           joint_analysis, comparison, diagnostics, run_id)
    
    # Save report
    output_file = run_dir / "allocator_v1_analysis.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    print("\n" + "="*80)
    try:
        print(report)
    except UnicodeEncodeError:
        print("(Report contains Unicode characters - view file directly)")
    print("="*80)


if __name__ == "__main__":
    main()
