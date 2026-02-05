"""
Risk Targeting v1 Baseline Analysis

Extracts RT facts from the pinned baseline run and compares to distribution targets.
Implements the "what good looks like" contract for RT v1.

Usage:
    python scripts/diagnostics/analyze_rt_v1_baseline.py <run_id>
    
Example:
    python scripts/diagnostics/analyze_rt_v1_baseline.py canonical_frozen_stack_compute_phase3a_governed_20260114_230316
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_rt_artifacts(run_dir: Path) -> Dict:
    """Load all RT-related artifacts from run directory."""
    artifacts = {}
    
    # Load meta.json
    meta_file = run_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json not found in {run_dir}")
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        artifacts['meta'] = json.load(f)
    
    # Load RT leverage series
    leverage_file = run_dir / "risk_targeting" / "leverage_series.csv"
    if leverage_file.exists():
        leverage_df = pd.read_csv(leverage_file, parse_dates=['date'], index_col='date')
        artifacts['leverage_series'] = leverage_df['leverage']
    else:
        raise FileNotFoundError(f"leverage_series.csv not found in {run_dir / 'risk_targeting'}")
    
    # Load RT realized vol series
    vol_file = run_dir / "risk_targeting" / "realized_vol.csv"
    if vol_file.exists():
        vol_df = pd.read_csv(vol_file, parse_dates=['date'], index_col='date')
        artifacts['realized_vol'] = vol_df['realized_vol']
    else:
        print(f"Warning: realized_vol.csv not found, vol stats will be missing")
        artifacts['realized_vol'] = None
    
    # Load weights (pre-RT, post-RT, post-allocator)
    weights_pre_rt_file = run_dir / "risk_targeting" / "weights_pre_risk_targeting.csv"
    weights_post_rt_file = run_dir / "risk_targeting" / "weights_post_risk_targeting.csv"
    weights_raw_file = run_dir / "weights_raw.csv"
    weights_scaled_file = run_dir / "weights_scaled.csv"
    
    if weights_pre_rt_file.exists():
        # Long format: date, instrument, weight
        weights_pre_rt_long = pd.read_csv(weights_pre_rt_file, parse_dates=['date'])
        weights_pre_rt = weights_pre_rt_long.pivot(index='date', columns='instrument', values='weight')
        artifacts['weights_pre_rt'] = weights_pre_rt
    else:
        artifacts['weights_pre_rt'] = None
    
    if weights_post_rt_file.exists():
        weights_post_rt_long = pd.read_csv(weights_post_rt_file, parse_dates=['date'])
        weights_post_rt = weights_post_rt_long.pivot(index='date', columns='instrument', values='weight')
        artifacts['weights_post_rt'] = weights_post_rt
    else:
        artifacts['weights_post_rt'] = None
    
    if weights_raw_file.exists():
        weights_raw = pd.read_csv(weights_raw_file, index_col=0, parse_dates=True)
        artifacts['weights_raw'] = weights_raw
    else:
        artifacts['weights_raw'] = None
    
    if weights_scaled_file.exists():
        weights_scaled = pd.read_csv(weights_scaled_file, index_col=0, parse_dates=True)
        artifacts['weights_scaled'] = weights_scaled
    else:
        artifacts['weights_scaled'] = None
    
    return artifacts


def compute_rt_governance(artifacts: Dict) -> Dict:
    """Extract RT governance facts from meta.json and artifacts."""
    meta = artifacts['meta']
    rt_meta = meta.get('risk_targeting', {})
    leverage_series = artifacts['leverage_series']
    
    governance = {
        'rt_enabled': rt_meta.get('enabled', False),
        'rt_effective': rt_meta.get('effective', False),
        'rt_has_teeth': rt_meta.get('has_teeth', False),
        'rt_multiplier_stats': rt_meta.get('multiplier_stats', {}),
        'effective_start_date': meta.get('effective_start_date'),
        'evaluation_start_date': meta.get('evaluation_start_date'),
        'per_stage_effective_starts': meta.get('per_stage_effective_starts', {})
    }
    
    # Recompute multiplier stats if missing or NaN
    if leverage_series is not None and len(leverage_series) > 0:
        leverage_values = leverage_series.dropna()
        if len(leverage_values) > 0:
            governance['rt_multiplier_stats'] = {
                'p5': float(np.percentile(leverage_values, 5)),
                'p50': float(np.percentile(leverage_values, 50)),
                'p95': float(np.percentile(leverage_values, 95)),
                'at_cap': float(np.sum(leverage_values >= 7.0 - 1e-6) / len(leverage_values) * 100),
                'at_floor': float(np.sum(leverage_values <= 1.0 + 1e-6) / len(leverage_values) * 100)
            }
    
    return governance


def compute_rt_vol_stats(artifacts: Dict) -> Dict:
    """Compute realized vol statistics used by RT."""
    vol_series = artifacts.get('realized_vol')
    
    if vol_series is None or len(vol_series) == 0:
        return {}
    
    vol_values = vol_series.dropna()
    if len(vol_values) == 0:
        return {}
    
    return {
        'p5': float(np.percentile(vol_values, 5)),
        'p50': float(np.percentile(vol_values, 50)),
        'p95': float(np.percentile(vol_values, 95)),
        'min': float(vol_values.min()),
        'max': float(vol_values.max())
    }


def compute_gross_exposure_by_stage(artifacts: Dict) -> Optional[pd.DataFrame]:
    """Compute gross exposure at each stage: pre-RT, post-RT, post-allocator."""
    weights_pre_rt = artifacts.get('weights_pre_rt')
    weights_post_rt = artifacts.get('weights_post_rt')
    weights_raw = artifacts.get('weights_raw')
    weights_scaled = artifacts.get('weights_scaled')
    
    # Determine which weights we have
    # weights_raw is post-RT, pre-allocator
    # weights_scaled is post-allocator
    
    exposures = {}
    
    if weights_pre_rt is not None:
        exposures['pre_rt'] = weights_pre_rt.abs().sum(axis=1)
    
    if weights_post_rt is not None:
        exposures['post_rt'] = weights_post_rt.abs().sum(axis=1)
    elif weights_raw is not None:
        # weights_raw is post-RT, pre-allocator
        exposures['post_rt'] = weights_raw.abs().sum(axis=1)
    
    if weights_scaled is not None:
        exposures['post_allocator'] = weights_scaled.abs().sum(axis=1)
    elif weights_raw is not None:
        # If no allocator, post-allocator = post-RT
        exposures['post_allocator'] = weights_raw.abs().sum(axis=1)
    
    if not exposures:
        return None
    
    # Align all series to common dates
    common_dates = None
    for exp in exposures.values():
        if common_dates is None:
            common_dates = exp.index
        else:
            common_dates = common_dates.intersection(exp.index)
    
    if common_dates is None or len(common_dates) == 0:
        return None
    
    # Create DataFrame
    result = pd.DataFrame(index=common_dates)
    for name, exp in exposures.items():
        result[name] = exp.loc[common_dates]
    
    return result


def analyze_rt_time_clustering(leverage_series: pd.Series) -> Dict:
    """Analyze when RT is high vs low, especially during known vol spikes (2020Q1, 2022)."""
    leverage = leverage_series.dropna()
    
    # Define periods of interest
    periods = {
        '2020_Q1': (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-03-31')),
        '2020_Q2': (pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30')),
        '2022_Q1': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-03-31')),
        '2022_Q2': (pd.Timestamp('2022-04-01'), pd.Timestamp('2022-06-30')),
        '2022_H1': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-06-30')),
        '2025_Q2': (pd.Timestamp('2025-04-01'), pd.Timestamp('2025-06-30')),
    }
    
    clustering = {}
    overall_median = leverage.median()
    
    for period_name, (start, end) in periods.items():
        period_leverage = leverage[(leverage.index >= start) & (leverage.index <= end)]
        if len(period_leverage) > 0:
            clustering[period_name] = {
                'count': len(period_leverage),
                'median': float(period_leverage.median()),
                'p25': float(period_leverage.quantile(0.25)),
                'p75': float(period_leverage.quantile(0.75)),
                'min': float(period_leverage.min()),
                'max': float(period_leverage.max()),
                'vs_overall_median': float(period_leverage.median() - overall_median)
            }
    
    return clustering


def compare_to_targets(governance: Dict, exposure_df: Optional[pd.DataFrame]) -> Dict:
    """Compare RT outputs to distribution targets."""
    targets = {
        'p50_gross_target': (4.0, 4.5),
        'p90_gross_target': (5.5, 6.5),
        'cap_binding_target_pct': 5.0  # Single-digit % unless markets persistently ultra-low vol
    }
    
    comparison = {
        'targets': targets,
        'actuals': {},
        'status': {}
    }
    
    # Get multiplier stats
    multiplier_stats = governance.get('rt_multiplier_stats', {})
    p50_mult = multiplier_stats.get('p50')
    p90_mult = multiplier_stats.get('p95')  # p95 ≈ p90
    cap_binding_pct = multiplier_stats.get('at_cap', 0.0)
    
    comparison['actuals']['rt_multiplier_p50'] = p50_mult
    comparison['actuals']['rt_multiplier_p90'] = p90_mult
    comparison['actuals']['cap_binding_pct'] = cap_binding_pct
    
    # Check if we have exposure data
    if exposure_df is not None and 'post_rt' in exposure_df.columns:
        post_rt_exposure = exposure_df['post_rt'].dropna()
        if len(post_rt_exposure) > 0:
            p50_gross = float(np.percentile(post_rt_exposure, 50))
            p90_gross = float(np.percentile(post_rt_exposure, 90))
            
            comparison['actuals']['post_rt_gross_p50'] = p50_gross
            comparison['actuals']['post_rt_gross_p90'] = p90_gross
            
            # Check targets
            comparison['status']['p50_gross'] = (
                targets['p50_gross_target'][0] <= p50_gross <= targets['p50_gross_target'][1]
            )
            comparison['status']['p90_gross'] = (
                targets['p90_gross_target'][0] <= p90_gross <= targets['p90_gross_target'][1]
            )
    
    # Check multiplier targets (if base gross is ~1x, then RT multiplier should match gross targets)
    if p50_mult is not None:
        comparison['status']['p50_multiplier'] = (
            targets['p50_gross_target'][0] <= p50_mult <= targets['p50_gross_target'][1]
        )
    
    if p90_mult is not None:
        comparison['status']['p90_multiplier'] = (
            targets['p90_gross_target'][0] <= p90_mult <= targets['p90_gross_target'][1]
        )
    
    # Check cap binding
    comparison['status']['cap_binding'] = cap_binding_pct <= targets['cap_binding_target_pct']
    
    return comparison


def diagnose_failure_modes(governance: Dict, vol_stats: Dict, comparison: Dict) -> Dict:
    """Diagnose common RT failure modes."""
    diagnostics = {
        'failure_modes': [],
        'recommendations': []
    }
    
    multiplier_stats = governance.get('rt_multiplier_stats', {})
    p5_mult = multiplier_stats.get('p5')
    p50_mult = multiplier_stats.get('p50')
    p95_mult = multiplier_stats.get('p95')
    cap_binding_pct = multiplier_stats.get('at_cap', 0.0)
    
    # Failure Mode A: p50 too low (e.g., 2×–3×)
    if p50_mult is not None and p50_mult < 3.5:
        diagnostics['failure_modes'].append({
            'mode': 'A',
            'description': 'p50 too low (RT not reaching intended normal sizing)',
            'actual': p50_mult,
            'target': '4.0-4.5x',
            'likely_causes': [
                'target vol too low',
                'realized vol estimator too high',
                'conservative floor/cap structure'
            ],
            'fix': 'RT calibration (target vol, vol estimator, floor/cap structure)'
        })
    
    # Failure Mode B: cap binds often
    if cap_binding_pct > 5.0:
        diagnostics['failure_modes'].append({
            'mode': 'B',
            'description': 'Cap binds too often',
            'actual': f'{cap_binding_pct:.1f}%',
            'target': '<5%',
            'likely_causes': [
                'target vol too high',
                'realized vol estimator collapsing too low (too reactive / too short window)'
            ],
            'fix': 'Raise vol floor, lengthen vol window, or reduce target'
        })
    
    # Failure Mode C: p5 multiplier is still huge (e.g., p5 = 3×)
    if p5_mult is not None and p5_mult > 2.5:
        diagnostics['failure_modes'].append({
            'mode': 'C',
            'description': 'p5 multiplier too high (RT not de-risking enough in high-vol conditions)',
            'actual': p5_mult,
            'target': '<2.5x (should naturally come down when vol spikes)',
            'likely_causes': [
                'estimator not responsive enough',
                'vol regime adjustments needed',
                'minimum-vol clamp too conservative'
            ],
            'fix': 'Improve estimator responsiveness, vol regime adjustments, or set more conservative minimum-vol clamp'
        })
    
    # Failure Mode D: RT "has teeth" but basically constant
    if governance.get('rt_has_teeth', False) and p5_mult is not None and p95_mult is not None:
        spread = p95_mult - p5_mult
        if spread < 1.0:
            diagnostics['failure_modes'].append({
                'mode': 'D',
                'description': 'RT has teeth but basically constant (not meaningful variation)',
                'actual': f'p95-p5 spread = {spread:.2f}x',
                'target': '>1.0x spread',
                'likely_causes': [
                    'realized vol too stable (window too long)',
                    'target too close to realized'
                ],
                'fix': 'Shorten vol window or adjust target'
            })
    
    return diagnostics


def generate_report(governance: Dict, vol_stats: Dict, exposure_df: Optional[pd.DataFrame],
                   clustering: Dict, comparison: Dict, diagnostics: Dict, run_id: str) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Risk Targeting v1 Baseline Analysis")
    lines.append("")
    lines.append(f"**Run ID:** `{run_id}`")
    lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Step 1: RT Facts
    lines.append("## Step 1: RT Facts from Baseline")
    lines.append("")
    lines.append("### Governance/Telemetry")
    lines.append("")
    lines.append(f"- **rt_enabled:** {governance.get('rt_enabled', False)}")
    lines.append(f"- **rt_effective:** {governance.get('rt_effective', False)}")
    lines.append(f"- **rt_has_teeth:** {governance.get('rt_has_teeth', False)}")
    lines.append(f"- **effective_start_date:** {governance.get('effective_start_date', 'N/A')}")
    lines.append(f"- **evaluation_start_date:** {governance.get('evaluation_start_date', 'N/A')}")
    lines.append("")
    
    multiplier_stats = governance.get('rt_multiplier_stats', {})
    lines.append("### Multiplier Statistics")
    lines.append("")
    lines.append(f"- **p5:** {multiplier_stats.get('p5', 'N/A'):.3f}" if multiplier_stats.get('p5') is not None else "- **p5:** N/A")
    lines.append(f"- **p50:** {multiplier_stats.get('p50', 'N/A'):.3f}" if multiplier_stats.get('p50') is not None else "- **p50:** N/A")
    lines.append(f"- **p95:** {multiplier_stats.get('p95', 'N/A'):.3f}" if multiplier_stats.get('p95') is not None else "- **p95:** N/A")
    lines.append(f"- **% time at cap:** {multiplier_stats.get('at_cap', 0.0):.1f}%")
    lines.append(f"- **% time at floor:** {multiplier_stats.get('at_floor', 0.0):.1f}%")
    lines.append("")
    
    if vol_stats:
        lines.append("### Realized Vol Statistics (used by RT)")
        lines.append("")
        lines.append(f"- **p5:** {vol_stats.get('p5', 'N/A'):.4f}" if vol_stats.get('p5') is not None else "- **p5:** N/A")
        lines.append(f"- **p50:** {vol_stats.get('p50', 'N/A'):.4f}" if vol_stats.get('p50') is not None else "- **p50:** N/A")
        lines.append(f"- **p95:** {vol_stats.get('p95', 'N/A'):.4f}" if vol_stats.get('p95') is not None else "- **p95:** N/A")
        lines.append(f"- **min:** {vol_stats.get('min', 'N/A'):.4f}" if vol_stats.get('min') is not None else "- **min:** N/A")
        lines.append(f"- **max:** {vol_stats.get('max', 'N/A'):.4f}" if vol_stats.get('max') is not None else "- **max:** N/A")
        lines.append("")
    
    # Time clustering
    if clustering:
        lines.append("### Time Clustering: RT High vs Low")
        lines.append("")
        lines.append("| Period | Count | Median | p25 | p75 | Min | Max | vs Overall Median |")
        lines.append("|--------|-------|--------|-----|-----|-----|-----|-------------------|")
        overall_median = multiplier_stats.get('p50', 0)
        for period_name, period_data in clustering.items():
            lines.append(
                f"| {period_name} | {period_data['count']} | {period_data['median']:.3f} | "
                f"{period_data['p25']:.3f} | {period_data['p75']:.3f} | {period_data['min']:.3f} | "
                f"{period_data['max']:.3f} | {period_data['vs_overall_median']:+.3f} |"
            )
        lines.append("")
    
    # Step 2: Compare to Targets
    lines.append("## Step 2: Compare RT Outputs to Distribution Targets")
    lines.append("")
    lines.append("### Targets")
    lines.append("")
    targets = comparison.get('targets', {})
    lines.append(f"- **p50 gross:** {targets.get('p50_gross_target', (0, 0))[0]:.1f}× - {targets.get('p50_gross_target', (0, 0))[1]:.1f}×")
    lines.append(f"- **p90 gross:** {targets.get('p90_gross_target', (0, 0))[0]:.1f}× - {targets.get('p90_gross_target', (0, 0))[1]:.1f}×")
    lines.append(f"- **cap binding:** <{targets.get('cap_binding_target_pct', 0):.1f}%")
    lines.append("")
    
    lines.append("### Actuals")
    lines.append("")
    actuals = comparison.get('actuals', {})
    if 'rt_multiplier_p50' in actuals and actuals['rt_multiplier_p50'] is not None:
        lines.append(f"- **RT multiplier p50:** {actuals['rt_multiplier_p50']:.3f}×")
    if 'rt_multiplier_p90' in actuals and actuals['rt_multiplier_p90'] is not None:
        lines.append(f"- **RT multiplier p90:** {actuals['rt_multiplier_p90']:.3f}×")
    if 'post_rt_gross_p50' in actuals:
        lines.append(f"- **Post-RT gross p50:** {actuals['post_rt_gross_p50']:.3f}×")
    if 'post_rt_gross_p90' in actuals:
        lines.append(f"- **Post-RT gross p90:** {actuals['post_rt_gross_p90']:.3f}×")
    lines.append(f"- **Cap binding:** {actuals.get('cap_binding_pct', 0.0):.1f}%")
    lines.append("")
    
    lines.append("### Status")
    lines.append("")
    status = comparison.get('status', {})
    for key, value in status.items():
        status_symbol = "✅" if value else "❌"
        lines.append(f"- {status_symbol} **{key}:** {value}")
    lines.append("")
    
    # Step 3: Diagnose Failure Modes
    lines.append("## Step 3: Diagnose Which Knob to Turn (RT, not Allocator)")
    lines.append("")
    
    failure_modes = diagnostics.get('failure_modes', [])
    if not failure_modes:
        lines.append("✅ **No failure modes detected.** RT appears to be functioning within expected parameters.")
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
    
    # Step 4: Acceptance Criteria
    lines.append("## Step 4: Acceptance Criteria to 'Freeze RT'")
    lines.append("")
    
    acceptance_criteria = []
    
    # Governance passes
    gov_passes = (
        governance.get('rt_enabled', False) and
        governance.get('rt_effective', False) and
        governance.get('rt_has_teeth', False) and
        multiplier_stats.get('p50') is not None and
        np.isfinite(multiplier_stats.get('p50', np.nan))
    )
    acceptance_criteria.append(("Governance passes", gov_passes))
    
    # Distribution targets hit
    p50_ok = status.get('p50_multiplier', False) or status.get('p50_gross', False)
    p90_ok = status.get('p90_multiplier', False) or status.get('p90_gross', False)
    cap_ok = status.get('cap_binding', False)
    acceptance_criteria.append(("Distribution targets hit", p50_ok and p90_ok and cap_ok))
    
    # Interpretability sanity (RT downshifts in vol spikes)
    interpretability_ok = True
    if clustering:
        # Check that RT is lower in 2020 Q1 and 2022 periods
        if '2020_Q1' in clustering:
            q1_2020_median = clustering['2020_Q1']['median']
            if q1_2020_median > overall_median + 0.5:  # Should be lower, not higher
                interpretability_ok = False
        if '2022_Q1' in clustering or '2022_H1' in clustering:
            period_key = '2022_H1' if '2022_H1' in clustering else '2022_Q1'
            q1_2022_median = clustering[period_key]['median']
            if q1_2022_median > overall_median + 0.5:
                interpretability_ok = False
    
    acceptance_criteria.append(("Interpretability sanity (RT downshifts in vol spikes)", interpretability_ok))
    
    lines.append("| Criterion | Status |")
    lines.append("|-----------|-------|")
    for criterion, passed in acceptance_criteria:
        status_symbol = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"| {criterion} | {status_symbol} |")
    lines.append("")
    
    all_passed = all(criteria[1] for criteria in acceptance_criteria)
    if all_passed:
        lines.append("**✅ RT is ready to freeze.**")
    else:
        lines.append("**❌ RT is NOT ready to freeze.** Address failure modes above.")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnostics/analyze_rt_v1_baseline.py <run_id>")
        print("\nExample:")
        print("  python scripts/diagnostics/analyze_rt_v1_baseline.py canonical_frozen_stack_compute_phase3a_governed_20260114_230316")
        sys.exit(1)
    
    run_id = sys.argv[1]
    run_dir = project_root / "reports" / "runs" / run_id
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    print(f"Loading RT artifacts from {run_dir}...")
    artifacts = load_rt_artifacts(run_dir)
    
    print("Computing RT governance...")
    governance = compute_rt_governance(artifacts)
    
    print("Computing RT vol stats...")
    vol_stats = compute_rt_vol_stats(artifacts)
    
    print("Computing gross exposure by stage...")
    exposure_df = compute_gross_exposure_by_stage(artifacts)
    
    print("Analyzing RT time clustering...")
    clustering = analyze_rt_time_clustering(artifacts['leverage_series'])
    
    print("Comparing to targets...")
    comparison = compare_to_targets(governance, exposure_df)
    
    print("Diagnosing failure modes...")
    diagnostics = diagnose_failure_modes(governance, vol_stats, comparison)
    
    print("Generating report...")
    report = generate_report(governance, vol_stats, exposure_df, clustering, comparison, diagnostics, run_id)
    
    # Save report
    output_file = run_dir / "rt_v1_analysis.md"
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
