"""
Phase Indexing Utilities

Helper functions for managing canonical Phase-0/1/2 results in a clean directory structure.

Structure:
  reports/
    sanity_checks/
      trend/
        {sleeve_name}/
          archive/
            {timestamp}/
          latest/          # Canonical Phase-0 results
    phase_index/
      trend/
        {sleeve_name}/
          phase0.txt      # Points to latest Phase-0
          phase1.txt      # Points to Phase-1 run_id
          phase2.txt      # Points to Phase-2 run_id
          status.txt      # Optional: status for parked sleeves
"""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Base directories
REPORTS_DIR = Path("reports")
SANITY_CHECKS_DIR = REPORTS_DIR / "sanity_checks"
PHASE_INDEX_DIR = REPORTS_DIR / "phase_index"
RUNS_DIR = REPORTS_DIR / "runs"


def get_sleeve_dirs(meta_sleeve: str, sleeve_name: str) -> dict:
    """
    Get directory paths for a sleeve's Phase-0 structure.
    
    Args:
        meta_sleeve: Meta-sleeve name (e.g., "trend")
        sleeve_name: Atomic sleeve name (e.g., "breakout_mid_50_100")
        
    Returns:
        Dict with keys: base, archive, latest
    """
    base_dir = SANITY_CHECKS_DIR / meta_sleeve / sleeve_name
    return {
        "base": base_dir,
        "archive": base_dir / "archive",
        "latest": base_dir / "latest"
    }


def get_phase_index_dir(meta_sleeve: str, sleeve_name: str) -> Path:
    """
    Get the phase index directory for a sleeve.
    
    Args:
        meta_sleeve: Meta-sleeve name (e.g., "trend")
        sleeve_name: Atomic sleeve name (e.g., "breakout_mid_50_100")
        
    Returns:
        Path to phase_index/{meta_sleeve}/{sleeve_name}/
    """
    return PHASE_INDEX_DIR / meta_sleeve / sleeve_name


def update_phase_index(
    meta_sleeve: str,
    sleeve_name: str,
    phase: str,
    run_id: Optional[str] = None,
    path: Optional[Path] = None
) -> None:
    """
    Update the phase index to point to a canonical run.
    
    Args:
        meta_sleeve: Meta-sleeve name (e.g., "trend")
        sleeve_name: Atomic sleeve name (e.g., "breakout_mid_50_100")
        phase: Phase name ("phase0", "phase1", "phase2")
        run_id: For Phase-1/2, the run_id in reports/runs/
        path: For Phase-0, the path to latest/ directory (relative to reports/)
    """
    index_dir = get_phase_index_dir(meta_sleeve, sleeve_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    phase_file = index_dir / f"{phase}.txt"
    
    if phase == "phase0":
        # Phase-0 points to latest/ directory
        if path is None:
            path = f"sanity_checks/{meta_sleeve}/{sleeve_name}/latest"
        with open(phase_file, "w") as f:
            f.write(f"{path}\n")
        logger.info(f"Updated {phase_file} -> {path}")
    else:
        # Phase-1/2 point to run_id
        if run_id is None:
            raise ValueError(f"run_id required for {phase}")
        with open(phase_file, "w") as f:
            f.write(f"{run_id}\n")
        logger.info(f"Updated {phase_file} -> {run_id}")


def get_phase_path(meta_sleeve: str, sleeve_name: str, phase: str) -> Optional[Path]:
    """
    Get the canonical path for a phase.
    
    Args:
        meta_sleeve: Meta-sleeve name (e.g., "trend")
        sleeve_name: Atomic sleeve name (e.g., "breakout_mid_50_100")
        phase: Phase name ("phase0", "phase1", "phase2")
        
    Returns:
        Path to the canonical run, or None if not set
    """
    index_dir = get_phase_index_dir(meta_sleeve, sleeve_name)
    phase_file = index_dir / f"{phase}.txt"
    
    if not phase_file.exists():
        return None
    
    with open(phase_file, "r") as f:
        content = f.read().strip()
    
    if phase == "phase0":
        # Content is a relative path to latest/
        return REPORTS_DIR / content
    else:
        # Content is a run_id
        return RUNS_DIR / content


def set_sleeve_status(meta_sleeve: str, sleeve_name: str, status: str) -> None:
    """
    Set status for a sleeve (e.g., "PARKED", "PRODUCTION").
    
    Args:
        meta_sleeve: Meta-sleeve name (e.g., "trend")
        sleeve_name: Atomic sleeve name (e.g., "persistence")
        status: Status string
    """
    index_dir = get_phase_index_dir(meta_sleeve, sleeve_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    status_file = index_dir / "status.txt"
    with open(status_file, "w") as f:
        f.write(f"{status}\n")
    logger.info(f"Updated {status_file} -> {status}")


def copy_to_latest(
    source_dir: Path,
    latest_dir: Path,
    files_to_copy: Optional[list] = None
) -> None:
    """
    Copy selected files from archive run to latest/ directory.
    
    Args:
        source_dir: Source directory (archive/{timestamp}/)
        latest_dir: Destination directory (latest/)
        files_to_copy: List of filenames to copy. If None, uses default set.
    """
    import shutil
    
    if files_to_copy is None:
        # Default files to keep in latest/
        files_to_copy = [
            "portfolio_returns.csv",
            "equity_curve.csv",
            "equity_curve.png",
            "per_asset_stats.csv",
            "meta.json",
            "return_histogram.png"
        ]
    
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing latest/ directory
    if latest_dir.exists():
        for item in latest_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    # Copy selected files
    copied = []
    for filename in files_to_copy:
        source_file = source_dir / filename
        if source_file.exists():
            dest_file = latest_dir / filename
            shutil.copy2(source_file, dest_file)
            copied.append(filename)
        else:
            logger.warning(f"File not found in source: {filename}")
    
    logger.info(f"Copied {len(copied)} files to {latest_dir}")
    return copied

