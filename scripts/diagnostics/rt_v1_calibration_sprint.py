"""
Risk Targeting v1 Calibration Sprint

Runs 3 deterministic calibration runs with different target_vol values:
- Run D: target_vol = 0.40
- Run E: target_vol = 0.42
- Run F: target_vol = 0.44

All other RT parameters remain fixed (cap/floor unchanged).

After each run, runs RT analysis and checks acceptance criteria.

Usage:
    python scripts/diagnostics/rt_v1_calibration_sprint.py
"""

import sys
import json
import yaml
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.diagnostics.analyze_rt_v1_baseline import (
    load_rt_artifacts,
    compute_rt_governance,
    compute_rt_vol_stats,
    compute_gross_exposure_by_stage,
    compare_to_targets,
    diagnose_failure_modes,
    analyze_rt_time_clustering
)


def load_baseline_config() -> dict:
    """Load the baseline config file and merge with strategies.yaml for profiles."""
    config_path = project_root / "configs" / "canonical_frozen_stack_compute.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Baseline config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Merge with strategies.yaml to get strategy_profiles
    strategies_path = project_root / "configs" / "strategies.yaml"
    if strategies_path.exists():
        with open(strategies_path, 'r', encoding='utf-8') as f:
            strategies_config = yaml.safe_load(f) or {}
            if 'strategy_profiles' in strategies_config:
                config['strategy_profiles'] = strategies_config['strategy_profiles']
    
    return config


def create_calibration_config(target_vol: float, run_label: str) -> Path:
    """Create a temporary config file with modified target_vol."""
    baseline_config = load_baseline_config()
    
    # Update target_vol in risk_targeting section
    if 'risk_targeting' not in baseline_config:
        baseline_config['risk_targeting'] = {}
    
    baseline_config['risk_targeting']['target_vol'] = target_vol
    
    # Save to temporary config file
    temp_config_dir = project_root / "configs" / "temp_rt_calibration"
    temp_config_dir.mkdir(exist_ok=True)
    
    temp_config_path = temp_config_dir / f"rt_calibration_{run_label}.yaml"
    
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(baseline_config, f, default_flow_style=False, sort_keys=False)
    
    return temp_config_path


def run_backtest(target_vol: float, run_label: str) -> str:
    """Run a single backtest with given target_vol."""
    print(f"\n{'='*80}")
    print(f"Running calibration {run_label}: target_vol = {target_vol}")
    print(f"{'='*80}\n")
    
    # Create temporary config
    config_path = create_calibration_config(target_vol, run_label)
    
    # Generate run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"rt_calibration_{run_label}_{target_vol}_{timestamp}"
    
    # Build command
    cmd = [
        sys.executable,
        str(project_root / "run_strategy.py"),
        "--strategy_profile", "core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro",
        "--start", "2020-01-06",
        "--end", "2025-10-31",
        "--run_id", run_id,
        "--config_path", str(config_path)
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run backtest
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Backtest failed for {run_label}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Backtest failed for {run_label}")
    
    print(f"Backtest completed: {run_id}")
    return run_id


def analyze_run(run_id: str) -> Dict:
    """Run RT analysis on a completed run."""
    print(f"\nAnalyzing run: {run_id}")
    
    run_dir = project_root / "reports" / "runs" / run_id
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load artifacts
    artifacts = load_rt_artifacts(run_dir)
    
    # Compute metrics
    governance = compute_rt_governance(artifacts)
    vol_stats = compute_rt_vol_stats(artifacts)
    exposure_df = compute_gross_exposure_by_stage(artifacts)
    clustering = analyze_rt_time_clustering(artifacts['leverage_series'])
    comparison = compare_to_targets(governance, exposure_df)
    diagnostics = diagnose_failure_modes(governance, vol_stats, comparison)
    
    return {
        'run_id': run_id,
        'governance': governance,
        'vol_stats': vol_stats,
        'exposure_df': exposure_df,
        'clustering': clustering,
        'comparison': comparison,
        'diagnostics': diagnostics
    }


def check_acceptance_criteria(analysis: Dict) -> Dict:
    """Check acceptance criteria for a run."""
    governance = analysis['governance']
    comparison = analysis['comparison']
    clustering = analysis['clustering']
    
    multiplier_stats = governance.get('rt_multiplier_stats', {})
    p50_mult = multiplier_stats.get('p50')
    p95_mult = multiplier_stats.get('p95')
    cap_binding_pct = multiplier_stats.get('at_cap', 0.0)
    
    # Acceptance criteria
    criteria = {
        'governance_passes': (
            governance.get('rt_enabled', False) and
            governance.get('rt_effective', False) and
            governance.get('rt_has_teeth', False) and
            p50_mult is not None and
            np.isfinite(p50_mult)
        ),
        'p50_in_target': (
            p50_mult is not None and
            4.0 <= p50_mult <= 4.5
        ),
        'p90_in_target': (
            p95_mult is not None and
            5.5 <= p95_mult <= 6.5
        ),
        'cap_binding_rare': cap_binding_pct <= 5.0,
        'interpretability_ok': True  # Will check clustering
    }
    
    # Check interpretability (RT downshifts in vol spikes)
    if clustering:
        overall_median = p50_mult if p50_mult else 0
        if '2020_Q1' in clustering:
            q1_2020_median = clustering['2020_Q1']['median']
            if q1_2020_median > overall_median + 0.5:
                criteria['interpretability_ok'] = False
        if '2022_Q1' in clustering or '2022_H1' in clustering:
            period_key = '2022_H1' if '2022_H1' in clustering else '2022_Q1'
            q1_2022_median = clustering[period_key]['median']
            if q1_2022_median > overall_median + 0.5:
                criteria['interpretability_ok'] = False
    
    criteria['all_passed'] = all(criteria.values())
    
    return criteria


def generate_summary_report(analyses: List[Dict]) -> str:
    """Generate summary report comparing all calibration runs."""
    lines = []
    lines.append("# Risk Targeting v1 Calibration Sprint Summary")
    lines.append("")
    lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## Calibration Runs")
    lines.append("")
    lines.append("| Run | target_vol | Run ID | Status |")
    lines.append("|-----|------------|--------|--------|")
    
    for i, analysis in enumerate(analyses, 1):
        run_label = ['D', 'E', 'F'][i-1]
        target_vol = [0.40, 0.42, 0.44][i-1]
        run_id = analysis['run_id']
        criteria = check_acceptance_criteria(analysis)
        status = "[PASS]" if criteria['all_passed'] else "[FAIL]"
        lines.append(f"| {run_label} | {target_vol} | `{run_id}` | {status} |")
    
    lines.append("")
    
    # Detailed comparison
    lines.append("## Detailed Comparison")
    lines.append("")
    
    for i, analysis in enumerate(analyses, 1):
        run_label = ['D', 'E', 'F'][i-1]
        target_vol = [0.40, 0.42, 0.44][i-1]
        
        lines.append(f"### Run {run_label}: target_vol = {target_vol}")
        lines.append("")
        
        governance = analysis['governance']
        multiplier_stats = governance.get('rt_multiplier_stats', {})
        comparison = analysis['comparison']
        criteria = check_acceptance_criteria(analysis)
        
        lines.append("#### Multiplier Statistics")
        lines.append("")
        lines.append(f"- **p5:** {multiplier_stats.get('p5', 'N/A'):.3f}" if multiplier_stats.get('p5') is not None else "- **p5:** N/A")
        lines.append(f"- **p50:** {multiplier_stats.get('p50', 'N/A'):.3f}" if multiplier_stats.get('p50') is not None else "- **p50:** N/A")
        lines.append(f"- **p95:** {multiplier_stats.get('p95', 'N/A'):.3f}" if multiplier_stats.get('p95') is not None else "- **p95:** N/A")
        lines.append(f"- **% time at cap:** {multiplier_stats.get('at_cap', 0.0):.1f}%")
        lines.append(f"- **% time at floor:** {multiplier_stats.get('at_floor', 0.0):.1f}%")
        lines.append("")
        
        lines.append("#### Acceptance Criteria")
        lines.append("")
        lines.append("| Criterion | Status |")
        lines.append("|-----------|-------|")
        for criterion, passed in criteria.items():
            if criterion == 'all_passed':
                continue
            status_symbol = "[PASS]" if passed else "[FAIL]"
            lines.append(f"| {criterion} | {status_symbol} |")
        lines.append("")
        
        # Failure modes if any
        failure_modes = analysis['diagnostics'].get('failure_modes', [])
        if failure_modes:
            lines.append("#### Failure Modes")
            lines.append("")
            for mode_info in failure_modes:
                lines.append(f"- **Mode {mode_info['mode']}:** {mode_info['description']}")
            lines.append("")
    
    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    
    passed_runs = [i for i, a in enumerate(analyses) if check_acceptance_criteria(a)['all_passed']]
    
    if passed_runs:
        best_run_idx = passed_runs[0]  # First passing run
        best_run_label = ['D', 'E', 'F'][best_run_idx]
        best_target_vol = [0.40, 0.42, 0.44][best_run_idx]
        lines.append(f"**[PASS] Run {best_run_label} (target_vol = {best_target_vol}) passes all acceptance criteria.**")
        lines.append("")
        lines.append("**Action:** Freeze RT at this target_vol and proceed to allocator calibration.")
    else:
        lines.append("**[FAIL] No runs passed all acceptance criteria.**")
        lines.append("")
        lines.append("**Next Steps:**")
        lines.append("1. Review failure modes above")
        lines.append("2. Consider secondary knobs:")
        lines.append("   - vol_floor (if realized vol too low)")
        lines.append("   - vol_lookback (if distribution shape wrong)")
        lines.append("   - leverage_floor (if floor binding is non-trivial)")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Run calibration sprint."""
    
    print("="*80)
    print("Risk Targeting v1 Calibration Sprint (Higher target_vol Grid)")
    print("="*80)
    print("\nThis will run 3 backtests with different target_vol values:")
    print("  Run D: target_vol = 0.40")
    print("  Run E: target_vol = 0.42")
    print("  Run F: target_vol = 0.44")
    print("\nAll other RT parameters remain fixed.")
    print("\nThis may take a while...")
    
    analyses = []
    
    target_vols = [0.40, 0.42, 0.44]
    run_labels = ['D', 'E', 'F']
    
    for target_vol, run_label in zip(target_vols, run_labels):
        try:
            # Run backtest
            run_id = run_backtest(target_vol, run_label)
            
            # Analyze
            analysis = analyze_run(run_id)
            analyses.append(analysis)
            
            # Check acceptance
            criteria = check_acceptance_criteria(analysis)
            print(f"\n{'='*80}")
            print(f"Run {run_label} Acceptance Criteria:")
            print(f"{'='*80}")
            for criterion, passed in criteria.items():
                if criterion == 'all_passed':
                    continue
                status = "[PASS]" if passed else "[FAIL]"
                print(f"  {status} {criterion}")
            
            if criteria['all_passed']:
                print(f"\n[PASS] Run {run_label} PASSES all acceptance criteria!")
                print(f"   Consider freezing RT at target_vol = {target_vol}")
            else:
                print(f"\n[FAIL] Run {run_label} does NOT pass all acceptance criteria.")
            
        except Exception as e:
            print(f"\nERROR in Run {run_label}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary report
    if analyses:
        print(f"\n{'='*80}")
        print("Generating summary report...")
        print(f"{'='*80}\n")
        
        summary = generate_summary_report(analyses)
        
        # Save summary
        summary_file = project_root / "reports" / "runs" / "rt_calibration_sprint_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Summary report saved to: {summary_file}")
        print("\n" + "="*80)
        print(summary)
        print("="*80)
    
    print("\nCalibration sprint complete!")


if __name__ == "__main__":
    main()
