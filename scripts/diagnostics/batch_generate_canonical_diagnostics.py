"""
Batch Generate Canonical Diagnostics

Generates canonical diagnostics (canonical_diagnostics.json/.md) for multiple runs.

Features:
- Skip runs that already have canonical_diagnostics.json
- Support --latest N to process N most recent runs
- Support --run_ids to process specific runs
- Generate summary table with status and reasons

Usage:
    # Generate for 25 most recent runs
    python scripts/diagnostics/batch_generate_canonical_diagnostics.py --latest 25
    
    # Generate for specific runs
    python scripts/diagnostics/batch_generate_canonical_diagnostics.py --run_ids run1 run2 run3
    
    # Generate for all runs (use with caution)
    python scripts/diagnostics/batch_generate_canonical_diagnostics.py --all
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_run_timestamp(run_dir: Path) -> Optional[float]:
    """
    Get timestamp for sorting runs (prefer meta.json timestamp, fallback to mtime).
    
    Returns:
        Timestamp as float, or None if unavailable
    """
    # Try meta.json first
    meta_file = run_dir / 'meta.json'
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                # Check for run_id with timestamp pattern YYYYMMDD_HHMMSS
                run_id = meta.get('run_id', '')
                if '_' in run_id:
                    try:
                        # Extract timestamp from run_id if it follows pattern
                        timestamp_str = run_id.split('_')[-2] + '_' + run_id.split('_')[-1]
                        dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        return dt.timestamp()
                    except (ValueError, IndexError):
                        pass
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Fallback to directory mtime
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return None


def find_run_directories(base_dir: Path, latest: Optional[int] = None) -> List[Tuple[Path, str]]:
    """
    Find run directories to process.
    
    Args:
        base_dir: Base directory containing runs (reports/runs)
        latest: If specified, return only the N most recent runs
        
    Returns:
        List of (run_dir, run_id) tuples, sorted by timestamp (newest first)
    """
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return []
    
    runs = []
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_id = run_dir.name
        timestamp = get_run_timestamp(run_dir)
        
        if timestamp is not None:
            runs.append((run_dir, run_id, timestamp))
        else:
            # Include runs without timestamps at the end
            runs.append((run_dir, run_id, 0.0))
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x[2], reverse=True)
    
    # Apply latest limit
    if latest is not None:
        runs = runs[:latest]
    
    # Return (run_dir, run_id) tuples
    return [(run_dir, run_id) for run_dir, run_id, _ in runs]


def run_id_exists_in_directory(run_id: str, base_dir: Path) -> bool:
    """Check if a specific run_id exists in the base directory."""
    run_dir = base_dir / run_id
    return run_dir.exists() and run_dir.is_dir()


def check_required_artifacts(run_dir: Path) -> Tuple[bool, List[str]]:
    """
    Check if required artifacts exist.
    
    Required artifacts:
    - portfolio_returns.csv
    - equity_curve.csv
    - weights.csv (or weights_scaled.csv or weights_raw.csv)
    - meta.json
    
    Returns:
        (all_present: bool, missing: List[str])
    """
    required = {
        'portfolio_returns.csv': (run_dir / 'portfolio_returns.csv').exists(),
        'equity_curve.csv': (run_dir / 'equity_curve.csv').exists(),
        'meta.json': (run_dir / 'meta.json').exists(),
    }
    
    # Check for weights (any variant)
    has_weights = (
        (run_dir / 'weights.csv').exists() or
        (run_dir / 'weights_scaled.csv').exists() or
        (run_dir / 'weights_raw.csv').exists()
    )
    required['weights*.csv'] = has_weights
    
    missing = [name for name, exists in required.items() if not exists]
    all_present = len(missing) == 0
    
    return all_present, missing


def generate_diagnostics_for_run(run_id: str, base_dir: Path) -> Tuple[bool, str, str, List[str]]:
    """
    Generate canonical diagnostics for a single run.
    
    Args:
        run_id: Run identifier
        base_dir: Base directory containing runs
        
    Returns:
        (success: bool, error_type: str, error_message: str, missing_artifacts: List[str])
    """
    run_dir = base_dir / run_id
    diagnostics_file = run_dir / 'canonical_diagnostics.json'
    
    # Check if already exists
    if diagnostics_file.exists():
        return (True, "", "", [])
    
    # Check if run directory exists
    if not run_dir.exists():
        return (False, "run_directory_not_found", "Run directory does not exist", [])
    
    # Check required artifacts
    all_present, missing_artifacts = check_required_artifacts(run_dir)
    if not all_present:
        missing_str = ", ".join(missing_artifacts)
        return (False, "missing_required_artifacts", f"Missing: {missing_str}", missing_artifacts)
    
    # Generate diagnostics
    script_path = Path(__file__).parent / 'generate_canonical_diagnostics.py'
    cmd = [
        sys.executable,
        str(script_path),
        '--run_id',
        run_id
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Verify file was created
            if diagnostics_file.exists():
                return (True, "", "", [])
            else:
                return (False, "script_succeeded_but_file_missing", "Script succeeded but output file missing", [])
        else:
            # Extract error message from stderr
            error_lines = result.stderr.strip().split('\n') if result.stderr else []
            error_msg = error_lines[-1] if error_lines else "unknown_error"
            
            # Clean up error message
            if error_msg.startswith("Error: "):
                error_msg = error_msg[7:]
            if len(error_msg) > 80:
                error_msg = error_msg[:77] + "..."
            
            return (False, "generation_error", error_msg, [])
            
    except subprocess.TimeoutExpired:
        return (False, "timeout", "Generation timed out after 5 minutes", [])
    except Exception as e:
        return (False, "exception", str(e)[:80], [])


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate canonical diagnostics for multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for 25 most recent runs
  python scripts/diagnostics/batch_generate_canonical_diagnostics.py --latest 25
  
  # Generate for specific runs
  python scripts/diagnostics/batch_generate_canonical_diagnostics.py --run_ids run1 run2
  
  # Generate for all runs (use with caution)
  python scripts/diagnostics/batch_generate_canonical_diagnostics.py --all
        """
    )
    
    parser.add_argument(
        '--latest',
        type=int,
        default=None,
        help='Process N most recent runs (sorted by timestamp)'
    )
    
    parser.add_argument(
        '--run_ids',
        nargs='+',
        default=None,
        help='Process specific run IDs'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all runs (use with caution)'
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default='reports/runs',
        help='Base directory containing run folders (default: reports/runs)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually generating'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Determine which runs to process
    if args.run_ids:
        # Specific run IDs
        runs_to_process = []
        for run_id in args.run_ids:
            if run_id_exists_in_directory(run_id, base_dir):
                run_dir = base_dir / run_id
                runs_to_process.append((run_dir, run_id))
            else:
                logger.warning(f"Run ID not found: {run_id}")
                runs_to_process.append((None, run_id))  # Track missing runs
    elif args.all:
        # All runs
        runs_to_process = find_run_directories(base_dir)
    elif args.latest:
        # Latest N runs
        runs_to_process = find_run_directories(base_dir, latest=args.latest)
    else:
        parser.error("Must specify one of: --latest N, --run_ids <ids>, or --all")
    
    if not runs_to_process:
        logger.info("No runs found to process.")
        return 0
    
    logger.info(f"Found {len(runs_to_process)} run(s) to process")
    
    if args.dry_run:
        logger.info("\nDry-run mode: Would process the following runs:")
        for run_dir, run_id in runs_to_process:
            if run_dir:
                exists = (run_dir / 'canonical_diagnostics.json').exists()
                status = "SKIP (exists)" if exists else "GENERATE"
                logger.info(f"  {run_id}: {status}")
            else:
                logger.info(f"  {run_id}: NOT FOUND")
        return 0
    
    # Process runs
    results = []
    for run_dir, run_id in runs_to_process:
        if run_dir is None:
            results.append((run_id, False, "run_directory_not_found", "", []))
            continue
        
        # Quick check if already exists (before calling function)
        diagnostics_file = run_dir / 'canonical_diagnostics.json'
        if diagnostics_file.exists():
            results.append((run_id, True, "already_exists", "", []))
            logger.info(f"\n{run_id}: ✓ Skipped (already exists)")
            continue
        
        logger.info(f"\nProcessing: {run_id}")
        success, error_type, error_msg, missing_artifacts = generate_diagnostics_for_run(run_id, base_dir)
        results.append((run_id, success, error_type, error_msg, missing_artifacts))
        
        if success:
            logger.info(f"  ✓ Generated successfully")
        else:
            if error_type == "missing_required_artifacts":
                missing_str = ", ".join(missing_artifacts)
                logger.warning(f"  ✗ Failed: Missing artifacts: {missing_str}")
            else:
                logger.warning(f"  ✗ Failed ({error_type}): {error_msg}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("BATCH GENERATION SUMMARY")
    print("=" * 120)
    print(f"{'Run ID':<50} {'Status':<12} {'Error Type':<25} {'Missing Artifacts/Error':<30}")
    print("-" * 120)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for run_id, success, error_type, error_msg, missing_artifacts in results:
        if success:
            if error_type == "already_exists" or error_type == "":
                status = "SKIPPED" if error_type == "already_exists" else "SUCCESS"
                if status == "SKIPPED":
                    skip_count += 1
                else:
                    success_count += 1
                display_error = ""
                display_missing = ""
            else:
                status = "SUCCESS"
                success_count += 1
                display_error = ""
                display_missing = ""
        else:
            status = "FAILED"
            fail_count += 1
            display_error = error_type if error_type else "unknown"
            
            if missing_artifacts:
                display_missing = ", ".join(missing_artifacts)
                if len(display_missing) > 28:
                    display_missing = display_missing[:25] + "..."
            else:
                display_missing = error_msg[:28] if error_msg else ""
        
        # Truncate long run IDs
        display_id = run_id if len(run_id) <= 48 else run_id[:45] + "..."
        print(f"{display_id:<50} {status:<12} {display_error:<25} {display_missing:<30}")
    
    print("-" * 120)
    print(f"Total: {len(results)} | Success: {success_count} | Skipped: {skip_count} | Failed: {fail_count}")
    print("=" * 120)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
