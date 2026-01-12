"""
Artifact Writer for System Layers

Provides deterministic, auditable artifact writing for Risk Targeting and Allocator layers.

Key Principles:
- Deterministic: Stable column order, stable sorting, ISO dates
- Append mode: One row per date (for time series)
- Once mode: Write once per run (for params/metadata)
- No lookahead: Only write what's been computed so far
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ArtifactWriter:
    """
    Writes artifacts for system layers in a deterministic, auditable format.
    
    Supports two modes:
    - append: Add one row per date (for time series)
    - once: Write once per run (for params/metadata)
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize ArtifactWriter.
        
        Args:
            base_dir: Base directory for artifacts (e.g., reports/runs/{run_id}/)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Track which files have been written in "once" mode
        self._once_written = set()
        
        logger.info(f"[ArtifactWriter] Initialized with base_dir: {self.base_dir}")
    
    def write_csv(
        self,
        relative_path: str,
        df: pd.DataFrame,
        mode: str = "append",
        dedupe_subset: Optional[list] = None
    ) -> None:
        """
        Write DataFrame to CSV with deterministic formatting.
        
        Args:
            relative_path: Path relative to base_dir (e.g., "risk_targeting/leverage_series.csv")
            df: DataFrame to write
            mode: "append" (add rows) or "overwrite" (replace file)
            dedupe_subset: Columns to use for deduplication. If None, auto-detects:
                          - For panel data (with 'instrument'): ['date', 'instrument']
                          - For time series (no 'instrument'): ['date']
        """
        file_path = self.base_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure deterministic formatting
        if isinstance(df.index, pd.DatetimeIndex):
            # Reset index to column for CSV
            df_to_write = df.reset_index()
            if 'index' in df_to_write.columns:
                df_to_write = df_to_write.rename(columns={'index': 'date'})
            # Ensure date column is ISO format
            if 'date' in df_to_write.columns:
                df_to_write['date'] = pd.to_datetime(df_to_write['date']).dt.strftime('%Y-%m-%d')
        else:
            df_to_write = df.copy()
        
        # Sort columns for deterministic output
        df_to_write = df_to_write.reindex(sorted(df_to_write.columns), axis=1)
        
        # Auto-detect dedupe_subset if not provided
        if dedupe_subset is None:
            if 'instrument' in df_to_write.columns:
                # Panel data: dedupe by (date, instrument)
                dedupe_subset = ['date', 'instrument']
            elif 'date' in df_to_write.columns:
                # Time series: dedupe by date only
                dedupe_subset = ['date']
        
        # Sort rows
        if 'date' in df_to_write.columns:
            df_to_write['date'] = pd.to_datetime(df_to_write['date']).dt.strftime('%Y-%m-%d')
            sort_cols = dedupe_subset if dedupe_subset else ['date']
            df_to_write = df_to_write.sort_values(sort_cols)
        elif df.index.name == 'date' or (isinstance(df.index, pd.DatetimeIndex) and len(df) > 0):
            # If index is date, sort by index
            df_to_write = df_to_write.sort_index()
        
        # Write CSV
        if mode == "append" and file_path.exists():
            # Append mode: read existing, append new, write back
            existing = pd.read_csv(file_path)
            combined = pd.concat([existing, df_to_write], ignore_index=True)
            
            # Remove duplicates (keep last)
            if dedupe_subset is not None:
                # Normalize date column for comparison
                if 'date' in combined.columns:
                    combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
                combined = combined.drop_duplicates(subset=dedupe_subset, keep='last')
                combined = combined.sort_values(dedupe_subset)
            
            combined.to_csv(file_path, index=False, float_format='%.8f')
            logger.debug(f"[ArtifactWriter] Appended to {relative_path}: {len(df_to_write)} rows (dedupe_subset={dedupe_subset})")
        else:
            # Overwrite mode or new file
            df_to_write.to_csv(file_path, index=False, float_format='%.8f')
            logger.debug(f"[ArtifactWriter] Wrote {relative_path}: {len(df_to_write)} rows")
    
    def write_json(
        self,
        relative_path: str,
        obj: Dict[str, Any],
        mode: str = "once"
    ) -> None:
        """
        Write dictionary to JSON file.
        
        Args:
            relative_path: Path relative to base_dir (e.g., "risk_targeting/params.json")
            obj: Dictionary to write
            mode: "once" (write only if not exists) or "overwrite"
        """
        file_path = self.base_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if mode == "once":
            if file_path.exists() or relative_path in self._once_written:
                logger.debug(f"[ArtifactWriter] Skipping {relative_path} (already written)")
                return
            self._once_written.add(relative_path)
        
        # Write JSON with deterministic formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        logger.debug(f"[ArtifactWriter] Wrote {relative_path}")
    
    def get_path(self, relative_path: str) -> Path:
        """Get full path for a relative path."""
        return self.base_dir / relative_path


def create_artifact_writer(run_dir: Optional[Path] = None) -> Optional[ArtifactWriter]:
    """
    Factory function to create ArtifactWriter.
    
    Args:
        run_dir: Directory for artifacts (if None, returns None for no-op)
    
    Returns:
        ArtifactWriter instance or None
    """
    if run_dir is None:
        return None
    
    return ArtifactWriter(Path(run_dir))

