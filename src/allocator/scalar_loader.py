"""
Allocator Scalar Loader

Utility for loading precomputed risk scalars from prior runs.
Single source of truth for scalar loading logic.
"""

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_precomputed_applied_scalars(
    run_dir: Union[str, Path],
    filename: str = "allocator_risk_v1_applied.csv"
) -> pd.Series:
    """
    Load precomputed applied risk scalars from a prior run.
    
    Args:
        run_dir: Path to run directory (e.g., reports/runs/<run_id>)
        filename: Filename of scalar CSV (default: allocator_risk_v1_applied.csv)
    
    Returns:
        pd.Series: Risk scalars indexed by date, named 'risk_scalar_applied'
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    run_dir = Path(run_dir)
    scalar_file = run_dir / filename
    
    if not scalar_file.exists():
        raise FileNotFoundError(
            f"Precomputed scalar file not found: {scalar_file}\n"
            f"Make sure the baseline run has allocator artifacts generated."
        )
    
    logger.info(f"[ScalarLoader] Loading precomputed scalars from: {scalar_file}")
    
    # Read CSV with date index
    try:
        df = pd.read_csv(scalar_file, index_col=0, parse_dates=True)
    except Exception as e:
        raise ValueError(f"Failed to read scalar file {scalar_file}: {e}")
    
    # Extract scalar column
    if 'risk_scalar_applied' in df.columns:
        scalars = df['risk_scalar_applied']
    elif len(df.columns) == 1:
        # Single column, use it
        scalars = df.iloc[:, 0]
        scalars.name = 'risk_scalar_applied'
    else:
        raise ValueError(
            f"Expected 'risk_scalar_applied' column in {scalar_file}, "
            f"found: {df.columns.tolist()}"
        )
    
    # Validate data
    if not isinstance(scalars.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be DatetimeIndex, got {type(scalars.index)}")
    
    if not scalars.index.is_monotonic_increasing:
        logger.warning("[ScalarLoader] Index is not monotonic, sorting...")
        scalars = scalars.sort_index()
    
    # Check for NaNs
    n_nan = scalars.isna().sum()
    if n_nan > 0:
        logger.warning(
            f"[ScalarLoader] Found {n_nan} NaN values in scalar series "
            f"({n_nan / len(scalars) * 100:.1f}%). These will be forward-filled."
        )
        scalars = scalars.fillna(method='ffill')
    
    # Log summary
    logger.info(
        f"[ScalarLoader] Loaded {len(scalars)} scalars from "
        f"{scalars.index[0].strftime('%Y-%m-%d')} to "
        f"{scalars.index[-1].strftime('%Y-%m-%d')}"
    )
    logger.info(
        f"[ScalarLoader] Scalar stats: mean={scalars.mean():.3f}, "
        f"min={scalars.min():.3f}, max={scalars.max():.3f}, "
        f"n<1.0={(scalars < 1.0).sum()}"
    )
    
    return scalars


def validate_scalar_series(scalars: pd.Series) -> None:
    """
    Validate that a scalar series is well-formed.
    
    Args:
        scalars: Series of risk scalars
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(scalars, pd.Series):
        raise ValueError(f"Expected pd.Series, got {type(scalars)}")
    
    if not isinstance(scalars.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex, got {type(scalars.index)}")
    
    if not scalars.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing")
    
    if scalars.isna().any():
        raise ValueError(f"Scalar series contains {scalars.isna().sum()} NaN values")
    
    if (scalars < 0).any() or (scalars > 1.5).any():
        logger.warning(
            f"[ScalarLoader] Unusual scalar values detected: "
            f"min={scalars.min():.3f}, max={scalars.max():.3f}"
        )
    
    logger.info("[ScalarLoader] ✓ Scalar series validation passed")

