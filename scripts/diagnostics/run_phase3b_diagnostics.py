"""
Run Phase 3B Diagnostics Pack

Runs the complete Phase 3B diagnostic suite in one command:
1. Phase 3B checkpoint verification
2. Waterfall attribution
3. Engine attribution at Post-Construction

This makes Phase 3B diagnostics repeatable with a single command.

Usage:
    # Run on a single run
    python scripts/diagnostics/run_phase3b_diagnostics.py --run_id <run_id>
    
    # Run on the pinned baseline pair (default)
    python scripts/diagnostics/run_phase3b_diagnostics.py --pinned
    
    # Skip checkpoint verification (faster)
    python scripts/diagnostics/run_phase3b_diagnostics.py --run_id <run_id> --skip-checkpoints

Reference:
- PROCEDURES.md § "Purpose of Phase 3B Attribution"
- PINNED/README.md § "Phase 3B Close-Out Summary"
"""

import sys
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.waterfall_attribution import (
    compute_waterfall_attribution,
    format_waterfall_report
)
from src.diagnostics.engine_attribution import (
    compute_engine_attribution_post_construction,
    format_engine_attribution_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default pinned baseline pair
PINNED_ARTIFACTS_ONLY = "phase3b_baseline_artifacts_only_20260120_093953"
PINNED_TRADED = "phase3b_baseline_traded_20260120_093953"


def run_checkpoints(run_id: str) -> bool:
    """Run Phase 3B checkpoint verification."""
    import subprocess
    
    logger.info(f"Running Phase 3B checkpoints for {run_id}...")
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/diagnostics/verify_phase3b_baseline_checkpoints.py", 
             "--run_id", run_id],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=120
        )
        
        if result.returncode == 0:
            logger.info(f"Checkpoints PASSED for {run_id}")
            return True
        else:
            logger.warning(f"Checkpoints FAILED for {run_id}")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Checkpoint verification timed out for {run_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to run checkpoints: {e}")
        return False


def run_waterfall(run_id: str, run_dir: Path) -> dict:
    """Run waterfall attribution."""
    logger.info(f"Computing waterfall attribution for {run_id}...")
    
    report = compute_waterfall_attribution(run_id, run_dir)
    
    # Save JSON
    json_path = run_dir / 'waterfall_attribution.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save Markdown
    md_report = format_waterfall_report(report)
    md_path = run_dir / 'waterfall_attribution.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    logger.info(f"Saved waterfall attribution: {md_path}")
    return report


def run_engine_attribution(run_id: str, run_dir: Path) -> dict:
    """Run engine attribution at Post-Construction."""
    logger.info(f"Computing engine attribution at Post-Construction for {run_id}...")
    
    report = compute_engine_attribution_post_construction(run_id, run_dir)
    
    # Save JSON
    json_path = run_dir / 'engine_attribution_post_construction.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save Markdown
    md_report = format_engine_attribution_report(report)
    md_path = run_dir / 'engine_attribution_post_construction.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    logger.info(f"Saved engine attribution: {md_path}")
    return report


def run_diagnostics_for_run(run_id: str, skip_checkpoints: bool = False) -> dict:
    """Run full Phase 3B diagnostic suite for a single run."""
    run_dir = Path(f"reports/runs/{run_id}")
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    results = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'checkpoints': None,
        'waterfall': None,
        'engine_attribution': None
    }
    
    # 1. Checkpoints (optional)
    if not skip_checkpoints:
        results['checkpoints'] = run_checkpoints(run_id)
    else:
        logger.info("Skipping checkpoint verification")
    
    # 2. Waterfall attribution
    try:
        waterfall = run_waterfall(run_id, run_dir)
        results['waterfall'] = {
            'success': True,
            'post_construction_sharpe': waterfall.get('portfolio_waterfall', {})
                .get('post_construction', {}).get('metrics', {}).get('sharpe'),
            'post_allocator_sharpe': waterfall.get('portfolio_waterfall', {})
                .get('post_allocator', {}).get('metrics', {}).get('sharpe')
        }
    except Exception as e:
        logger.error(f"Waterfall attribution failed: {e}")
        results['waterfall'] = {'success': False, 'error': str(e)}
    
    # 3. Engine attribution
    try:
        engine_attr = run_engine_attribution(run_id, run_dir)
        results['engine_attribution'] = {
            'success': True,
            'sleeves_analyzed': len(engine_attr.get('sleeve_metrics', {})),
            'top_positive': engine_attr.get('sleeve_roles', {}).get('top_positive_contributors', [])[:3],
            'top_negative': engine_attr.get('sleeve_roles', {}).get('top_negative_contributors', [])[:3],
            'red_flags': engine_attr.get('sleeve_roles', {}).get('red_flags', [])
        }
    except Exception as e:
        logger.error(f"Engine attribution failed: {e}")
        results['engine_attribution'] = {'success': False, 'error': str(e)}
    
    return results


def print_summary(results: dict):
    """Print diagnostic summary."""
    print("\n" + "=" * 70)
    print("PHASE 3B DIAGNOSTICS SUMMARY")
    print("=" * 70)
    print(f"Run ID: {results['run_id']}")
    print(f"Timestamp: {results['timestamp']}")
    print()
    
    # Checkpoints
    if results['checkpoints'] is not None:
        status = "PASSED" if results['checkpoints'] else "FAILED"
        print(f"Checkpoints: {status}")
    else:
        print("Checkpoints: SKIPPED")
    
    # Waterfall
    waterfall = results.get('waterfall', {})
    if waterfall.get('success'):
        pc_sharpe = waterfall.get('post_construction_sharpe')
        pa_sharpe = waterfall.get('post_allocator_sharpe')
        pc_str = f"{pc_sharpe:.2f}" if pc_sharpe is not None else "N/A"
        pa_str = f"{pa_sharpe:.2f}" if pa_sharpe is not None else "N/A"
        print(f"Waterfall: OK (Post-Construction Sharpe: {pc_str}, Post-Allocator Sharpe: {pa_str})")
    else:
        print(f"Waterfall: FAILED ({waterfall.get('error', 'Unknown error')})")
    
    # Engine attribution
    engine = results.get('engine_attribution', {})
    if engine.get('success'):
        n_sleeves = engine.get('sleeves_analyzed', 0)
        top_pos = engine.get('top_positive', [])
        top_neg = engine.get('top_negative', [])
        red_flags = engine.get('red_flags', [])
        
        print(f"Engine Attribution: OK ({n_sleeves} sleeves)")
        if top_pos:
            print(f"  Top Positive: {', '.join(top_pos)}")
        if top_neg:
            print(f"  Top Negative: {', '.join(top_neg)}")
        if red_flags:
            print(f"  RED FLAGS: {', '.join(red_flags)}")
    else:
        print(f"Engine Attribution: FAILED ({engine.get('error', 'Unknown error')})")
    
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Run Phase 3B Diagnostics Pack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on pinned baseline pair
    python scripts/diagnostics/run_phase3b_diagnostics.py --pinned
    
    # Run on a specific run
    python scripts/diagnostics/run_phase3b_diagnostics.py --run_id my_run_id
    
    # Skip checkpoint verification (faster)
    python scripts/diagnostics/run_phase3b_diagnostics.py --run_id my_run_id --skip-checkpoints
"""
    )
    
    parser.add_argument('--run_id', type=str, help='Run ID to analyze')
    parser.add_argument('--pinned', action='store_true', 
                        help='Run on pinned baseline pair (artifacts-only and traded)')
    parser.add_argument('--skip-checkpoints', action='store_true',
                        help='Skip checkpoint verification')
    
    args = parser.parse_args()
    
    if args.pinned:
        # Run on both pinned baselines
        print("\n" + "=" * 70)
        print("PHASE 3B DIAGNOSTICS - PINNED BASELINE PAIR")
        print("=" * 70)
        
        all_results = []
        
        for run_id in [PINNED_ARTIFACTS_ONLY, PINNED_TRADED]:
            print(f"\n--- Processing: {run_id} ---\n")
            try:
                results = run_diagnostics_for_run(run_id, args.skip_checkpoints)
                all_results.append(results)
                print_summary(results)
            except Exception as e:
                logger.error(f"Failed to process {run_id}: {e}")
        
        # Overall summary
        print("\n" + "=" * 70)
        print("PINNED PAIR SUMMARY")
        print("=" * 70)
        print(f"Artifacts-only: {PINNED_ARTIFACTS_ONLY}")
        print(f"Traded: {PINNED_TRADED}")
        print()
        
        all_ok = all(
            r.get('checkpoints', True) is not False and
            r.get('waterfall', {}).get('success', False) and
            r.get('engine_attribution', {}).get('success', False)
            for r in all_results
        )
        
        if all_ok:
            print("Status: ALL DIAGNOSTICS PASSED")
        else:
            print("Status: SOME DIAGNOSTICS FAILED - review output above")
        
        print("=" * 70)
        
    elif args.run_id:
        # Run on single run
        try:
            results = run_diagnostics_for_run(args.run_id, args.skip_checkpoints)
            print_summary(results)
        except Exception as e:
            logger.error(f"Failed: {e}")
            sys.exit(1)
    else:
        parser.error("Must provide either --run_id or --pinned")


if __name__ == '__main__':
    main()
