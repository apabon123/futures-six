"""
Migrate Allocator Scalars CSV Format to Canonical Format

One-time migration script to convert legacy allocator_scalars_at_rebalances.csv files
to canonical format (with rebalance_date column).

This script:
1. Scans reports/runs/*/allocator_scalars_at_rebalances.csv
2. Detects legacy format (missing rebalance_date column)
3. Rewrites in canonical format (with rebalance_date column)
4. Adds migration note to meta.json

Usage:
    python scripts/diagnostics/migrate_allocator_scalars_format.py
    python scripts/diagnostics/migrate_allocator_scalars_format.py --dry-run
    python scripts/diagnostics/migrate_allocator_scalars_format.py --run_id <specific_run>
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def needs_migration(csv_path: Path) -> bool:
    """
    Check if CSV file needs migration (legacy format).
    
    Returns True if:
    - File exists
    - First column is 'Unnamed: 0' (legacy format without rebalance_date)
    """
    if not csv_path.exists():
        return False
    
    try:
        df = pd.read_csv(csv_path, nrows=0)  # Read header only
        first_col = df.columns[0]
        return first_col.startswith('Unnamed:') or first_col != 'rebalance_date'
    except Exception as e:
        logger.warning(f"Error checking {csv_path}: {e}")
        return False


def migrate_csv(csv_path: Path) -> bool:
    """
    Migrate CSV from legacy format to canonical format.
    
    Returns True if migration was successful.
    """
    try:
        # Read the CSV (handles legacy format)
        df = pd.read_csv(csv_path)
        
        # Identify date column
        first_col = df.columns[0]
        if first_col.startswith('Unnamed:'):
            # Legacy format: first column is the date index
            date_col = first_col
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.index.name = 'rebalance_date'
        elif first_col == 'date':
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.index.name = 'rebalance_date'
        else:
            # Try reading with index_col=0
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = 'rebalance_date'
        
        # Write in canonical format
        df.to_csv(csv_path, index_label='rebalance_date')
        return True
        
    except Exception as e:
        logger.error(f"Error migrating {csv_path}: {e}")
        return False


def update_meta_json(meta_path: Path, migration_note: str = "allocator_scalars_at_rebalances_v1") -> bool:
    """
    Add migration note to meta.json.
    
    Returns True if update was successful.
    """
    try:
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        else:
            meta = {}
        
        # Add artifact_migrations field if it doesn't exist
        if 'artifact_migrations' not in meta:
            meta['artifact_migrations'] = []
        
        # Add migration note if not already present
        if migration_note not in meta['artifact_migrations']:
            meta['artifact_migrations'].append(migration_note)
        
        # Write back
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating {meta_path}: {e}")
        return False


def find_runs_to_migrate(base_dir: Path, run_id: Optional[str] = None) -> List[Path]:
    """Find all run directories that need migration."""
    runs_to_migrate = []
    
    if run_id:
        # Single run
        run_dir = base_dir / run_id
        csv_path = run_dir / 'allocator_scalars_at_rebalances.csv'
        if needs_migration(csv_path):
            runs_to_migrate.append(run_dir)
    else:
        # Scan all runs
        for run_dir in sorted(base_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            csv_path = run_dir / 'allocator_scalars_at_rebalances.csv'
            if needs_migration(csv_path):
                runs_to_migrate.append(run_dir)
    
    return runs_to_migrate


def main():
    parser = argparse.ArgumentParser(
        description="Migrate allocator_scalars_at_rebalances.csv to canonical format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes'
    )
    parser.add_argument(
        '--run_id',
        type=str,
        default=None,
        help='Migrate specific run_id only (default: scan all runs)'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='reports/runs',
        help='Base directory containing run folders (default: reports/runs)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return 1
    
    # Find runs to migrate
    runs_to_migrate = find_runs_to_migrate(base_dir, args.run_id)
    
    if not runs_to_migrate:
        logger.info("No runs found that need migration.")
        return 0
    
    logger.info(f"Found {len(runs_to_migrate)} run(s) that need migration:")
    for run_dir in runs_to_migrate:
        logger.info(f"  - {run_dir.name}")
    
    if args.dry_run:
        logger.info("\nDry-run mode: No changes will be made.")
        return 0
    
    # Perform migration
    logger.info("\nStarting migration...")
    migrated = 0
    failed = 0
    
    for run_dir in runs_to_migrate:
        csv_path = run_dir / 'allocator_scalars_at_rebalances.csv'
        meta_path = run_dir / 'meta.json'
        
        logger.info(f"\nMigrating: {run_dir.name}")
        
        # Migrate CSV
        if migrate_csv(csv_path):
            logger.info(f"  ✓ Migrated CSV: {csv_path.name}")
            
            # Update meta.json
            if update_meta_json(meta_path):
                logger.info(f"  ✓ Updated meta.json")
                migrated += 1
            else:
                logger.warning(f"  ⚠ CSV migrated but meta.json update failed")
                migrated += 1
        else:
            logger.error(f"  ✗ Failed to migrate CSV")
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successfully migrated: {migrated}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {len(runs_to_migrate)}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
