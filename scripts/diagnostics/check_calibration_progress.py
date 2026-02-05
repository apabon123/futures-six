"""
Quick script to check progress of RT calibration sprint and summarize results.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

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

def check_runs():
    """Check which calibration runs have completed."""
    runs_dir = project_root / "reports" / "runs"
    
    # Check both grids
    target_vols = [0.28, 0.32, 0.36, 0.40, 0.42, 0.44]
    run_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    results = []
    
    for target_vol, run_label in zip(target_vols, run_labels):
        # Find runs matching this calibration
        pattern = f"rt_calibration_{run_label}_*"
        matching_runs = list(runs_dir.glob(pattern))
        
        if matching_runs:
            # Get most recent run
            latest_run = max(matching_runs, key=lambda p: p.stat().st_mtime)
            run_id = latest_run.name
            
            # Check if it has RT artifacts
            rt_dir = latest_run / "risk_targeting"
            if rt_dir.exists() and (rt_dir / "leverage_series.csv").exists():
                try:
                    artifacts = load_rt_artifacts(latest_run)
                    governance = compute_rt_governance(artifacts)
                    vol_stats = compute_rt_vol_stats(artifacts)
                    exposure_df = compute_gross_exposure_by_stage(artifacts)
                    clustering = analyze_rt_time_clustering(artifacts['leverage_series'])
                    comparison = compare_to_targets(governance, exposure_df)
                    
                    multiplier_stats = governance.get('rt_multiplier_stats', {})
                    
                    results.append({
                        'run_label': run_label,
                        'target_vol': target_vol,
                        'run_id': run_id,
                        'status': 'complete',
                        'p50': multiplier_stats.get('p50'),
                        'p95': multiplier_stats.get('p95'),
                        'cap_binding_pct': multiplier_stats.get('at_cap', 0.0),
                        'floor_binding_pct': multiplier_stats.get('at_floor', 0.0)
                    })
                except Exception as e:
                    results.append({
                        'run_label': run_label,
                        'target_vol': target_vol,
                        'run_id': run_id,
                        'status': f'error: {e}'
                    })
            else:
                results.append({
                    'run_label': run_label,
                    'target_vol': target_vol,
                    'run_id': run_id,
                    'status': 'in_progress'
                })
        else:
            results.append({
                'run_label': run_label,
                'target_vol': target_vol,
                'run_id': None,
                'status': 'not_started'
            })
    
    return results

if __name__ == "__main__":
    results = check_runs()
    
    print("\n" + "="*80)
    print("RT Calibration Sprint Progress")
    print("="*80 + "\n")
    
    for r in results:
        print(f"Run {r['run_label']} (target_vol = {r['target_vol']}): {r['status']}")
        if r['status'] == 'complete':
            print(f"  Run ID: {r['run_id']}")
            if r['p50'] is not None:
                print(f"  p50 multiplier: {r['p50']:.3f}x")
                print(f"  p95 multiplier: {r['p95']:.3f}x")
                print(f"  Cap binding: {r['cap_binding_pct']:.1f}%")
                print(f"  Floor binding: {r['floor_binding_pct']:.1f}%")
                
                # Check targets
                p50_ok = 4.0 <= r['p50'] <= 4.5 if r['p50'] else False
                p95_ok = 5.5 <= r['p95'] <= 6.5 if r['p95'] else False
                cap_ok = r['cap_binding_pct'] <= 5.0
                
                print(f"  Targets: p50={'[PASS]' if p50_ok else '[FAIL]'}, p95={'[PASS]' if p95_ok else '[FAIL]'}, cap={'[PASS]' if cap_ok else '[FAIL]'}")
        print()
