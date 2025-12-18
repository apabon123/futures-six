"""
Canonical Backtest Window Configuration

NOTE: The canonical evaluation window (2020-01-06 to 2025-10-31) is now defined in
configs/canonical_window.yaml. Use src.utils.canonical_window.load_canonical_window()
to load it.

These defaults are kept for backward compatibility but should be replaced with
canonical window in new code.

See: docs/SOTs/PROCEDURES.md § 2 "Run Consistency Contract"
"""

from typing import Tuple

# Legacy defaults (use canonical_window.yaml for canonical metrics)
# The canonical window is 2020-01-06 to 2025-10-31
CANONICAL_START = "2020-01-01"
CANONICAL_END = "2025-10-31"

# Import canonical window loader
try:
    from src.utils.canonical_window import load_canonical_window as _load_canonical
    
    def get_canonical_window() -> Tuple[str, str]:
        """Get canonical evaluation window from config."""
        return _load_canonical()
except ImportError:
    # Fallback if canonical_window.py doesn't exist yet
    def get_canonical_window() -> Tuple[str, str]:
        """Get canonical evaluation window (legacy fallback)."""
        return ("2020-01-06", "2025-10-31")

