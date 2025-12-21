"""
Allocator State Validator

Validates allocator state artifacts for correctness and consistency.
Implements canonical sanity checks from PROCEDURES.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

from .state_v1 import REQUIRED_FEATURES, OPTIONAL_FEATURES, ALL_FEATURES

logger = logging.getLogger(__name__)


def validate_allocator_state_v1(
    state_df: pd.DataFrame,
    meta: Optional[Dict] = None,
    warn_threshold_pct: float = 0.05
) -> None:
    """
    Validate allocator state v1 DataFrame and metadata.
    
    Performs the following checks:
    1. Required columns are present
    2. Date index is monotonic
    3. No NaN values in REQUIRED_FEATURES
    4. Warns if rows_dropped > warn_threshold_pct of sample
    
    Args:
        state_df: Allocator state DataFrame to validate
        meta: Optional metadata dict (if available)
        warn_threshold_pct: Threshold for warning about dropped rows (default: 5%)
    
    Raises:
        AssertionError: If validation fails
        
    Logs:
        Warning if rows_dropped exceeds threshold
    """
    logger.debug("[Validator] Validating allocator state v1...")
    
    # Check 1: Required columns present
    missing_required = [col for col in REQUIRED_FEATURES if col not in state_df.columns]
    if missing_required:
        raise AssertionError(
            f"[Validator] FAIL: Missing required features: {missing_required}. "
            f"Expected: {REQUIRED_FEATURES}"
        )
    logger.debug(f"[Validator] ✓ All required features present ({len(REQUIRED_FEATURES)} columns)")
    
    # Check 2: Monotonic date index
    if not isinstance(state_df.index, pd.DatetimeIndex):
        raise AssertionError(
            f"[Validator] FAIL: Index must be DatetimeIndex, got {type(state_df.index)}"
        )
    
    if not state_df.index.is_monotonic_increasing:
        raise AssertionError(
            "[Validator] FAIL: Date index is not monotonic increasing"
        )
    logger.debug("[Validator] ✓ Date index is monotonic increasing")
    
    # Check 3: No NaN in REQUIRED_FEATURES
    required_cols_present = [col for col in REQUIRED_FEATURES if col in state_df.columns]
    nan_counts = state_df[required_cols_present].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    
    if len(nan_features) > 0:
        raise AssertionError(
            f"[Validator] FAIL: NaN values found in required features:\n{nan_features}"
        )
    logger.debug("[Validator] ✓ No NaN values in required features")
    
    # Check 4: Optional features may have NaN (but shouldn't if properly computed)
    optional_cols_present = [col for col in OPTIONAL_FEATURES if col in state_df.columns]
    if optional_cols_present:
        optional_nan_counts = state_df[optional_cols_present].isna().sum()
        optional_nan_features = optional_nan_counts[optional_nan_counts > 0]
        
        if len(optional_nan_features) > 0:
            logger.warning(
                f"[Validator] Optional features contain NaN values:\n{optional_nan_features}"
            )
        else:
            logger.debug(f"[Validator] ✓ No NaN in optional features ({len(optional_cols_present)} present)")
    
    # Check 5: Warn if too many rows dropped (from metadata)
    if meta is not None:
        rows_dropped = meta.get('rows_dropped', 0)
        rows_requested = meta.get('rows_requested', len(state_df))
        
        if rows_requested > 0:
            drop_pct = rows_dropped / rows_requested
            
            if drop_pct > warn_threshold_pct:
                logger.warning(
                    f"[Validator] ⚠️  Large number of rows dropped: {rows_dropped}/{rows_requested} "
                    f"({drop_pct*100:.1f}% > {warn_threshold_pct*100:.1f}% threshold). "
                    "This may indicate data quality issues or misaligned inputs."
                )
            else:
                logger.debug(
                    f"[Validator] ✓ Rows dropped: {rows_dropped}/{rows_requested} "
                    f"({drop_pct*100:.1f}% ≤ {warn_threshold_pct*100:.1f}% threshold)"
                )
    
    # Check 6: Feature coverage
    features_present = [col for col in ALL_FEATURES if col in state_df.columns]
    features_missing = [col for col in ALL_FEATURES if col not in state_df.columns]
    
    logger.info(
        f"[Validator] ✓ Validation passed: {len(features_present)}/{len(ALL_FEATURES)} features present"
    )
    
    if features_missing:
        # Missing features should only be optional
        unexpected_missing = [f for f in features_missing if f in REQUIRED_FEATURES]
        if unexpected_missing:
            raise AssertionError(
                f"[Validator] FAIL: Required features missing: {unexpected_missing}"
            )
        logger.debug(f"[Validator] Optional features missing (expected): {features_missing}")
    
    logger.debug("[Validator] ✓ Allocator state v1 validation complete")


def validate_inputs_aligned(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_returns: pd.DataFrame
) -> None:
    """
    Validate that input time series are properly aligned.
    
    Args:
        portfolio_returns: Portfolio returns series
        equity_curve: Equity curve series
        asset_returns: Asset returns DataFrame
    
    Raises:
        AssertionError: If inputs are not aligned
    """
    logger.debug("[Validator] Validating input alignment...")
    
    # Check that all inputs have DatetimeIndex
    for name, data in [
        ('portfolio_returns', portfolio_returns),
        ('equity_curve', equity_curve),
        ('asset_returns', asset_returns)
    ]:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise AssertionError(
                f"[Validator] FAIL: {name} must have DatetimeIndex, got {type(data.index)}"
            )
    
    # Check overlap
    common_dates = portfolio_returns.index.intersection(equity_curve.index)
    common_dates = common_dates.intersection(asset_returns.index)
    
    if len(common_dates) == 0:
        raise AssertionError(
            "[Validator] FAIL: No overlapping dates between portfolio_returns, "
            "equity_curve, and asset_returns"
        )
    
    overlap_pct = len(common_dates) / max(
        len(portfolio_returns),
        len(equity_curve),
        len(asset_returns)
    )
    
    if overlap_pct < 0.9:
        logger.warning(
            f"[Validator] ⚠️  Low overlap between inputs: {overlap_pct*100:.1f}% "
            f"({len(common_dates)} common dates)"
        )
    else:
        logger.debug(
            f"[Validator] ✓ Input alignment: {len(common_dates)} common dates "
            f"({overlap_pct*100:.1f}% overlap)"
        )

