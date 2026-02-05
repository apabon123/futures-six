"""
Wait for calibration runs to complete, then analyze all results.
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.diagnostics.check_calibration_progress import check_runs
from scripts.diagnostics.rt_v1_calibration_sprint import (
    analyze_run,
    check_acceptance_criteria,
    generate_summary_report
)

def wait_for_completion(max_wait_minutes=60, check_interval_seconds=30):
    """Wait for all calibration runs to complete."""
    print("Waiting for calibration runs to complete...")
    print(f"Will check every {check_interval_seconds} seconds, max wait: {max_wait_minutes} minutes\n")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        results = check_runs()
        completed = [r for r in results if r['status'] == 'complete']
        in_progress = [r for r in results if r['status'] == 'in_progress']
        errors = [r for r in results if 'error' in r['status']]
        not_started = [r for r in results if r['status'] == 'not_started']
        
        elapsed_minutes = (time.time() - start_time) / 60
        
        print(f"[{elapsed_minutes:.1f} min] Status: {len(completed)} complete, {len(in_progress)} in progress, {len(errors)} errors, {len(not_started)} not started")
        
        if len(completed) == 3:
            print("\nAll runs completed!")
            return results
        elif time.time() - start_time > max_wait_seconds:
            print(f"\nMax wait time ({max_wait_minutes} min) reached. Analyzing completed runs...")
            return results
        elif len(in_progress) == 0 and len(not_started) == 0:
            # All that can complete have completed
            print("\nNo more runs in progress. Analyzing completed runs...")
            return results
        
        time.sleep(check_interval_seconds)

def main():
    # Wait for completion
    results = wait_for_completion(max_wait_minutes=90, check_interval_seconds=45)
    
    # Analyze completed runs
    completed = [r for r in results if r['status'] == 'complete']
    
    if not completed:
        print("\nNo completed runs to analyze.")
        return
    
    print(f"\n{'='*80}")
    print("Analyzing completed runs...")
    print(f"{'='*80}\n")
    
    analyses = []
    for r in completed:
        try:
            analysis = analyze_run(r['run_id'])
            analyses.append(analysis)
            print(f"✓ Analyzed {r['run_label']}: {r['run_id']}")
        except Exception as e:
            print(f"✗ Error analyzing {r['run_label']}: {e}")
    
    if analyses:
        # Generate summary
        summary = generate_summary_report(analyses)
        
        # Save summary
        summary_file = project_root / "reports" / "runs" / "rt_calibration_sprint_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}\n")
        print(summary)
        print(f"\n{'='*80}")
        print(f"Summary saved to: {summary_file}")
    else:
        print("\nNo analyses generated.")

if __name__ == "__main__":
    main()
