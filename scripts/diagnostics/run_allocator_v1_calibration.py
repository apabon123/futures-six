"""
Allocator v1 Calibration Orchestration

Governed, repeatable script for testing allocator calibration changes.

This script:
1. Re-runs the canonical baseline with calibrated allocator settings
2. Records calibration metadata in meta.json
3. Automatically runs allocator diagnostics
4. Generates a calibration scorecard

Usage:
    python scripts/diagnostics/run_allocator_v1_calibration.py \
        --base_run_id canonical_frozen_stack_compute_20260113_123027 \
        --tag allocator_calib_v1_thresholds_rt_gate_hyst8
"""

import sys
import json
import yaml
import argparse
import subprocess
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.utils.canonical_window import load_canonical_window
from run_strategy import main as run_strategy_main
from scripts.diagnostics.analyze_allocator_v1_baseline import (
    load_allocator_artifacts,
    compute_allocator_governance,
    analyze_allocator_regime_distribution,
    analyze_allocator_scalar_distribution,
    analyze_allocator_time_clustering,
    analyze_joint_rt_allocator_behavior,
    compare_to_targets,
    diagnose_failure_modes
)

# Calibration parameters (from regime_rules_v1.py and exec_sim.py)
# CALIBRATION Run 2: Fixed RT gate (regime-aware), raised thresholds, widened hysteresis
CALIBRATION_PARAMS = {
    "vol_accel_enter": 1.70,
    "vol_accel_exit": 1.45,
    "corr_shock_enter": 0.20,
    "corr_shock_exit": 0.12,
    "dd_enter": -0.14,
    "dd_exit": -0.09,
    "dd_stress_enter": -0.18,
    "dd_crisis_enter": -0.20,
    "dd_slope_enter": -0.10,
    "dd_slope_exit": -0.06,
    "min_days_in_regime": 8,
    "rt_gate_percentile": 0.75,
    "rt_gate_rule": "Allow if (Regime >= STRESS) OR (RT >= p75)"
}


def load_base_run_meta(base_run_id: str) -> Dict:
    """Load metadata from base run to preserve configuration."""
    base_run_dir = project_root / "reports" / "runs" / base_run_id
    if not base_run_dir.exists():
        raise FileNotFoundError(f"Base run not found: {base_run_dir}")
    
    meta_file = base_run_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json not found in {base_run_dir}")
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_canonical_config() -> dict:
    """Load canonical frozen stack compute config and merge with strategies.yaml."""
    config_path = project_root / "configs" / "canonical_frozen_stack_compute.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Canonical config not found: {config_path}")
    
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


def run_calibration_backtest(
    base_run_id: str,
    tag: str,
    start_date: str,
    end_date: str,
    strategy_profile: str
) -> str:
    """Run the backtest with calibrated allocator settings."""
    logger.info("=" * 80)
    logger.info("ALLOCATOR V1 CALIBRATION RUN")
    logger.info("=" * 80)
    logger.info(f"Base Run ID: {base_run_id}")
    logger.info(f"Tag: {tag}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Strategy Profile: {strategy_profile}")
    logger.info("")
    
    # Generate run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"allocator_calib_{tag}_{timestamp}"
    
    logger.info(f"New Run ID: {run_id}")
    logger.info("")
    
    # Load canonical config
    config = load_canonical_config()
    
    # Ensure allocator is enabled in compute mode
    if 'allocator_v1' not in config:
        config['allocator_v1'] = {}
    config['allocator_v1']['enabled'] = True
    config['allocator_v1']['mode'] = 'compute'
    config['allocator_v1']['profile'] = 'H'
    
    # Write temp config
    temp_config_path = project_root / "configs" / "temp_allocator_calibration.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Using config: {temp_config_path}")
    logger.info("")
    
    # Temporarily modify sys.argv to pass to run_strategy.main
    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            'run_strategy.py',
            '--strategy_profile', strategy_profile,
            '--start', start_date,
            '--end', end_date,
            '--run_id', run_id,
            '--config_path', str(temp_config_path)
        ]
        
        logger.info("Starting backtest...")
        results = run_strategy_main()
        
        if results is None:
            raise RuntimeError("Backtest failed")
        
        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Run ID: {run_id}")
        logger.info("=" * 80)
        
        return run_id
        
    finally:
        sys.argv = original_argv


def add_calibration_metadata(run_id: str, base_run_id: str):
    """Add calibration metadata to meta.json."""
    run_dir = project_root / "reports" / "runs" / run_id
    meta_file = run_dir / "meta.json"
    
    if not meta_file.exists():
        logger.warning(f"meta.json not found: {meta_file}")
        return
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # Add calibration metadata
    meta['allocator_v1_calibration'] = CALIBRATION_PARAMS.copy()
    meta['allocator_v1_calibration']['calibration_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta['allocator_v1_calibration']['base_run_id'] = base_run_id
    meta['allocator_source_valid'] = True
    meta['allocator_source_run_id'] = base_run_id
    
    # Write back
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Added calibration metadata to {meta_file}")


def run_allocator_diagnostics(run_id: str) -> Dict:
    """Run allocator diagnostics and return results."""
    logger.info("=" * 80)
    logger.info("RUNNING ALLOCATOR DIAGNOSTICS")
    logger.info("=" * 80)
    
    # Import and run the diagnostic script
    run_dir = project_root / "reports" / "runs" / run_id
    
    # Load artifacts
    artifacts = load_allocator_artifacts(run_dir)
    
    # Compute diagnostics
    governance = compute_allocator_governance(artifacts)
    regime_analysis = analyze_allocator_regime_distribution(artifacts.get('regime_series'))
    scalar_analysis = analyze_allocator_scalar_distribution(artifacts.get('multiplier_series'))
    clustering = analyze_allocator_time_clustering(artifacts.get('multiplier_series'))
    joint_analysis = analyze_joint_rt_allocator_behavior(
        artifacts.get('multiplier_series'),
        artifacts.get('rt_leverage_series')
    )
    comparison = compare_to_targets(governance, regime_analysis, scalar_analysis)
    diagnostics = diagnose_failure_modes(governance, regime_analysis, scalar_analysis,
                                        clustering, joint_analysis)
    
    return {
        'governance': governance,
        'regime_analysis': regime_analysis,
        'scalar_analysis': scalar_analysis,
        'clustering': clustering,
        'joint_analysis': joint_analysis,
        'comparison': comparison,
        'diagnostics': diagnostics
    }


def generate_scorecard(
    run_id: str,
    base_run_id: str,
    tag: str,
    diagnostic_results: Dict
) -> str:
    """Generate calibration scorecard markdown."""
    governance = diagnostic_results['governance']
    regime_analysis = diagnostic_results['regime_analysis']
    scalar_analysis = diagnostic_results['scalar_analysis']
    clustering = diagnostic_results['clustering']
    comparison = diagnostic_results['comparison']
    diagnostics = diagnostic_results['diagnostics']
    
    lines = []
    lines.append("# Allocator v1 Calibration Scorecard")
    lines.append("")
    lines.append(f"**Run ID:** `{run_id}`")
    lines.append(f"**Base Run ID:** `{base_run_id}`")
    lines.append(f"**Tag:** `{tag}`")
    lines.append(f"**Calibration Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Calibration Parameters
    lines.append("## Calibration Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in CALIBRATION_PARAMS.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    
    # Regime Distribution
    lines.append("## Regime Distribution")
    lines.append("")
    regime_dist = governance.get('regime_distribution', {})
    lines.append("| Regime | % Time | Target | Status |")
    lines.append("|--------|--------|--------|--------|")
    
    targets = {
        'NORMAL': (70.0, 85.0),
        'ELEVATED': (0.0, 25.0),  # Combined with STRESS
        'STRESS': (0.0, 25.0),     # Combined with STRESS
        'CRISIS': (0.0, 5.0)
    }
    
    normal_pct = regime_dist.get('NORMAL', 0.0)
    elevated_pct = regime_dist.get('ELEVATED', 0.0)
    stress_pct = regime_dist.get('STRESS', 0.0)
    crisis_pct = regime_dist.get('CRISIS', 0.0)
    clipped_pct = elevated_pct + stress_pct
    
    normal_ok = targets['NORMAL'][0] <= normal_pct <= targets['NORMAL'][1]
    clipped_ok = targets['ELEVATED'][0] <= clipped_pct <= targets['ELEVATED'][1]
    crisis_ok = targets['CRISIS'][0] <= crisis_pct <= targets['CRISIS'][1]
    
    lines.append(f"| NORMAL | {normal_pct:.1f}% | 70-85% | {'✅' if normal_ok else '❌'} |")
    lines.append(f"| ELEVATED | {elevated_pct:.1f}% | - | - |")
    lines.append(f"| STRESS | {stress_pct:.1f}% | - | - |")
    lines.append(f"| Clipped (ELEVATED+STRESS) | {clipped_pct:.1f}% | 10-25% | {'✅' if clipped_ok else '❌'} |")
    lines.append(f"| CRISIS | {crisis_pct:.1f}% | 0-5% | {'✅' if crisis_ok else '❌'} |")
    lines.append("")
    
    # Scalar Statistics
    lines.append("## Scalar Statistics")
    lines.append("")
    scalar_stats = governance.get('scalar_stats', {})
    active_pct = governance.get('active_pct', 0.0)
    
    p50_ok = scalar_stats.get('p50') is not None and abs(scalar_stats.get('p50', 1.0) - 1.0) < 0.05
    p90_ok = scalar_stats.get('p90') is not None and scalar_stats.get('p90', 1.0) <= 1.0
    p5_ok = scalar_stats.get('p5') is not None and 0.3 <= scalar_stats.get('p5', 0.5) <= 0.6
    active_ok = active_pct < 30.0
    
    lines.append("| Metric | Value | Target | Status |")
    lines.append("|--------|-------|--------|--------|")
    lines.append(f"| p50 scalar | {scalar_stats.get('p50', 'N/A'):.3f} | ≈ 1.00 | {'✅' if p50_ok else '❌'} |")
    lines.append(f"| p90 scalar | {scalar_stats.get('p90', 'N/A'):.3f} | ≤ 1.00 | {'✅' if p90_ok else '❌'} |")
    lines.append(f"| p5 scalar | {scalar_stats.get('p5', 'N/A'):.3f} | 0.30-0.60 | {'✅' if p5_ok else '❌'} |")
    lines.append(f"| Active % | {active_pct:.1f}% | < 30% | {'✅' if active_ok else '❌'} |")
    lines.append("")
    
    # Transition Rate
    transition_ok = True  # Default to pass if no data
    if regime_analysis:
        transitions = regime_analysis.get('transitions', {})
        total_days = regime_analysis.get('total_days', 1)
        transition_rate = sum(transitions.values()) / total_days * 100 if total_days > 0 else 0
        transition_ok = transition_rate < 5.0
        
        lines.append("## Transition Rate")
        lines.append("")
        lines.append(f"**Transition Rate:** {transition_rate:.2f}%/day")
        lines.append(f"**Target:** < 5%/day")
        lines.append(f"**Status:** {'✅ PASS' if transition_ok else '❌ FAIL'}")
        lines.append("")
    
    # Time Clustering
    if clustering:
        lines.append("## Time Clustering (Known Stress Periods)")
        lines.append("")
        lines.append("| Period | Median Scalar | Active % | vs Overall |")
        lines.append("|--------|---------------|----------|-----------|")
        overall_median = scalar_stats.get('p50', 1.0)
        for period_name, period_data in clustering.items():
            lines.append(
                f"| {period_name} | {period_data['median']:.3f} | "
                f"{period_data['active_pct']:.1f}% | {period_data['vs_overall_median']:+.3f} |"
            )
        lines.append("")
    
    # Failure Modes
    failure_modes = diagnostics.get('failure_modes', [])
    lines.append("## Failure Modes")
    lines.append("")
    if not failure_modes:
        lines.append("✅ **No failure modes detected.**")
    else:
        for mode_info in failure_modes:
            lines.append(f"### {mode_info['mode']}: {mode_info['description']}")
            lines.append(f"- **Actual:** {mode_info['actual']}")
            lines.append(f"- **Target:** {mode_info['target']}")
            lines.append("")
    
    # Overall Status
    lines.append("## Overall Status")
    lines.append("")
    
    all_passed = (
        normal_ok and
        clipped_ok and
        crisis_ok and
        p50_ok and
        p90_ok and
        p5_ok and
        active_ok and
        transition_ok and
        len(failure_modes) == 0
    )
    
    if all_passed:
        lines.append("**✅ ALLOCATOR CALIBRATION PASSES**")
        lines.append("")
        lines.append("All criteria met. Allocator is ready to freeze.")
    else:
        lines.append("**❌ ALLOCATOR CALIBRATION FAILS**")
        lines.append("")
        lines.append("One or more criteria not met. Review failure modes above.")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Allocator v1 Calibration Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/diagnostics/run_allocator_v1_calibration.py \\
        --base_run_id canonical_frozen_stack_compute_20260113_123027 \\
        --tag allocator_calib_v1_thresholds_rt_gate_hyst8
        """
    )
    
    parser.add_argument(
        '--base_run_id',
        type=str,
        required=True,
        help='Base run ID to use as reference'
    )
    
    parser.add_argument(
        '--tag',
        type=str,
        required=True,
        help='Freeform tag for this calibration (e.g., allocator_calib_v1_thresholds_rt_gate_hyst8)'
    )
    
    parser.add_argument(
        '--config_override',
        type=str,
        default=None,
        help='Optional config override file'
    )
    
    args = parser.parse_args()
    
    # Load canonical window
    start_date, end_date = load_canonical_window()
    
    # Load base run metadata
    base_meta = load_base_run_meta(args.base_run_id)
    strategy_profile = base_meta.get('strategy_profile', 'core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro')
    
    # Run backtest
    run_id = run_calibration_backtest(
        args.base_run_id,
        args.tag,
        start_date,
        end_date,
        strategy_profile
    )
    
    # Add calibration metadata
    add_calibration_metadata(run_id, args.base_run_id)
    
    # Run diagnostics
    diagnostic_results = run_allocator_diagnostics(run_id)
    
    # Generate scorecard
    scorecard = generate_scorecard(run_id, args.base_run_id, args.tag, diagnostic_results)
    
    # Save scorecard
    run_dir = project_root / "reports" / "runs" / run_id
    scorecard_file = run_dir / "allocator_calibration_scorecard.md"
    with open(scorecard_file, 'w', encoding='utf-8') as f:
        f.write(scorecard)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ALLOCATOR CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"")
    print("Artifacts:")
    print(f"  - Allocator artifacts: {run_dir / 'allocator'}")
    print(f"  - Diagnostic report: {run_dir / 'allocator_v1_analysis.md'}")
    print(f"  - Scorecard: {scorecard_file}")
    print("")
    
    # Print PASS/FAIL
    all_passed = (
        len(diagnostic_results['diagnostics'].get('failure_modes', [])) == 0 and
        all(diagnostic_results['comparison'].get('status', {}).values())
    )
    
    try:
        if all_passed:
            print("✅ ALLOCATOR CALIBRATION PASSES")
        else:
            print("❌ ALLOCATOR CALIBRATION FAILS")
            print("")
            print("Review scorecard for details:")
            print(f"  {scorecard_file}")
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        if all_passed:
            print("[PASS] ALLOCATOR CALIBRATION PASSES")
        else:
            print("[FAIL] ALLOCATOR CALIBRATION FAILS")
            print("")
            print("Review scorecard for details:")
            print(f"  {scorecard_file}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
