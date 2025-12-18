"""
Canonical Evaluation Window Utilities

Loads and provides the canonical evaluation window used for all performance reporting.
"""

import yaml
from pathlib import Path
from typing import Tuple


def load_canonical_window() -> Tuple[str, str]:
    """
    Load canonical evaluation window from config.
    
    Returns:
        Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
    """
    path = Path("configs/canonical_window.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical window config not found: {path}\n"
            "Please create configs/canonical_window.yaml"
        )
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    w = cfg["canonical_window"]
    return w["start_date"], w["end_date"]

