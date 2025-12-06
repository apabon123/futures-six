"""
Migration script to reorganize existing Phase-0 runs into the new canonical structure.

This script:
1. Moves existing timestamped runs to archive/
2. Copies the most recent passing run to latest/
3. Creates phase_index entries

Usage:
    python scripts/migrate_phase_structure.py
"""

import sys
from pathlib import Path
from datetime import datetime
import shutil
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.phase_index import (
    get_sleeve_dirs,
    copy_to_latest,
    update_phase_index,
    set_sleeve_status
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_sleeve(meta_sleeve: str, sleeve_name: str, existing_timestamps: list, is_passing: bool = True):
    """
    Migrate a sleeve's existing runs to the new structure.
    
    Args:
        meta_sleeve: Meta-sleeve name (e.g., "trend")
        sleeve_name: Atomic sleeve name (e.g., "breakout_mid_50_100")
        existing_timestamps: List of timestamp strings (e.g., ["20251118_225521"])
        is_passing: Whether this sleeve passed Phase-0
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Migrating {meta_sleeve}/{sleeve_name}")
    logger.info(f"{'='*80}")
    
    sleeve_dirs = get_sleeve_dirs(meta_sleeve, sleeve_name)
    base_dir = sleeve_dirs["base"]
    archive_dir = sleeve_dirs["archive"]
    latest_dir = sleeve_dirs["latest"]
    
    # Create directories
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the most recent timestamp
    if not existing_timestamps:
        logger.warning(f"No existing timestamps found for {sleeve_name}")
        return
    
    # Sort timestamps (most recent first)
    existing_timestamps.sort(reverse=True)
    most_recent = existing_timestamps[0]
    
    logger.info(f"Found {len(existing_timestamps)} existing runs")
    logger.info(f"Most recent: {most_recent}")
    
    # Move all timestamped runs to archive/
    old_base = base_dir
    for timestamp in existing_timestamps:
        old_dir = old_base / timestamp
        if old_dir.exists():
            new_dir = archive_dir / timestamp
            if new_dir.exists():
                logger.warning(f"  {timestamp} already in archive, skipping")
            else:
                logger.info(f"  Moving {timestamp} -> archive/")
                shutil.move(str(old_dir), str(new_dir))
        else:
            logger.warning(f"  {timestamp} not found, skipping")
    
    # Copy most recent to latest/ if passing
    if is_passing:
        most_recent_dir = archive_dir / most_recent
        if most_recent_dir.exists():
            logger.info(f"Copying {most_recent} -> latest/")
            copy_to_latest(most_recent_dir, latest_dir)
            update_phase_index(meta_sleeve, sleeve_name, "phase0")
            logger.info(f"✓ Migration complete: {latest_dir}")
        else:
            logger.error(f"Most recent directory not found: {most_recent_dir}")
    else:
        # For failed sleeves, create latest_failed/
        latest_failed_dir = base_dir / "latest_failed"
        most_recent_dir = archive_dir / most_recent
        if most_recent_dir.exists():
            logger.info(f"Copying {most_recent} -> latest_failed/")
            copy_to_latest(most_recent_dir, latest_failed_dir)
            set_sleeve_status(meta_sleeve, sleeve_name, "PARKED after Phase-0/1 fail")
            logger.info(f"✓ Migration complete: {latest_failed_dir}")


def main():
    """Migrate all existing trend sleeve runs."""
    
    # Define sleeves to migrate
    trend_sleeves = [
        {
            "name": "breakout_mid_50_100",
            "timestamps": ["20251118_225521"],
            "is_passing": True
        },
        {
            "name": "residual_trend",
            "timestamps": ["20251118_135058", "20251118_135440"],
            "is_passing": True
        },
        {
            "name": "persistence",
            "timestamps": ["20251118_180756", "20251118_180815", "20251118_181245"],
            "is_passing": False  # Failed Phase-1
        }
    ]
    
    logger.info("="*80)
    logger.info("PHASE STRUCTURE MIGRATION")
    logger.info("="*80)
    logger.info("This script will reorganize existing Phase-0 runs into the new structure:")
    logger.info("  - Move timestamped runs to archive/")
    logger.info("  - Copy most recent passing run to latest/")
    logger.info("  - Create phase_index entries")
    logger.info("")
    
    for sleeve_info in trend_sleeves:
        migrate_sleeve(
            meta_sleeve="trend",
            sleeve_name=sleeve_info["name"],
            existing_timestamps=sleeve_info["timestamps"],
            is_passing=sleeve_info["is_passing"]
        )
    
    logger.info("\n" + "="*80)
    logger.info("MIGRATION COMPLETE")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("1. Verify latest/ directories contain expected files")
    logger.info("2. Check phase_index/trend/{sleeve_name}/phase0.txt entries")
    logger.info("3. Update Phase-1/2 scripts to use update_phase_index() after runs")


if __name__ == "__main__":
    main()

