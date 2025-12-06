"""
FX/Commodity Carry Features: Roll Yield Features for FX and Commodities.

Computes Carver-style carry features from continuous futures (rank 0 & rank 1)
for FX (6E, 6B, 6J) and commodities (CL, GC). Features are standardized using
rolling z-scores.
"""

import logging
from typing import Optional, Union, List
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _zscore_rolling(s: pd.Series, window: int = 252, clip: float = 3.0, min_periods: Optional[int] = None) -> pd.Series:
    """
    Compute rolling z-score with clipping.
    
    Args:
        s: Input series
        window: Rolling window size in days
        clip: Z-score clipping bounds
        min_periods: Minimum periods for rolling calculation (default: window // 2)
        
    Returns:
        Z-scored and clipped series
    """
    if min_periods is None:
        min_periods = max(window // 2, 126)  # At least 126 days (half year)
    
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sigma
    return z.clip(-clip, clip)


class FxCommodCarryFeatures:
    """
    Builds carry features from roll yield between front (rank 0) and next (rank 1)
    contracts for FX and commodities.
    
    Features computed (3 per root):
    - carry_ts_z_<root>: Time-series roll yield standardized per root
      - Positive = backwardation (attractive long carry)
      - Negative = contango (attractive short carry)
    - carry_xs_z_<root>: Cross-sectional carry strength (relative to other assets on same day)
    - carry_mom_63_z_<root>: 63-day carry momentum (change in roll yield over 3 months)
    """
    
    def __init__(
        self,
        roots: Optional[List[str]] = None,
        window: int = 252,
        clip: float = 3.0
    ):
        """
        Initialize FX/Commodity carry features calculator.
        
        Args:
            roots: List of root symbols (default: ["CL", "GC", "6E", "6B", "6J"])
            window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
        """
        self.roots = roots if roots is not None else ["CL", "GC", "6E", "6B", "6J"]
        self.window = window
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute FX/Commodity carry features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - carry_ts_z_<root>: Time-series roll yield standardized per root
            - carry_xs_z_<root>: Cross-sectional carry strength standardized
            - carry_mom_63_z_<root>: 63-day carry momentum standardized
        """
        # Query ALL available data (don't limit start_date) to build full rolling window history
        # The rolling window will handle the lookback requirement
        features_dict = {}
        all_dates = None
        roll_yield_raw_dict = {}  # Store raw roll yields for cross-sectional and momentum features
        
        for root in self.roots:
            try:
                # Get rank 0 and rank 1 close prices using get_contracts_by_root
                # This requires rank 0 and rank 1 continuous series to exist in the database.
                # For SR3: SR3_FRONT_CALENDAR (rank 0), SR3_RANK_1_CALENDAR (rank 1) - ✅ exists
                # For CL/GC: CL_FRONT_VOLUME (rank 0), CL_RANK_1_VOLUME (rank 1) - ❌ rank 1 missing
                # For 6E/6B/6J: 6E_FRONT_CALENDAR (rank 0), 6E_RANK_1_CALENDAR (rank 1) - ✅ exists (note: no _2D suffix)
                #
                # SOLUTION: Load rank 1 continuous series from Databento for CL, GC, 6E, 6B, 6J
                # These should follow the same roll rule as rank 0 (volume or calendar)
                close = market.get_contracts_by_root(
                    root=root,
                    ranks=[0, 1],
                    fields=("close",),
                    start=None,  # Get all available data from the beginning
                    end=end_date
                )
                
                if close.empty:
                    logger.warning(
                        f"[FxCommodCarry] No data found for root {root}. "
                        f"This requires rank 0 AND rank 1 continuous series to exist in the database. "
                        f"For example, CL needs both CL_FRONT_VOLUME (rank 0) and CL_RANK_1_VOLUME (rank 1). "
                        f"Currently only rank 0 exists. Please load rank 1 continuous series from Databento."
                    )
                    continue
                
                # Ensure we have both ranks
                if 0 not in close.columns or 1 not in close.columns:
                    logger.warning(
                        f"[FxCommodCarry] Missing required ranks for {root}. "
                        f"Available: {list(close.columns)}"
                    )
                    continue
                
                # Extract F0 and F1 (front and next contracts)
                F0 = close[0]
                F1 = close[1]
                
                # Handle missing data: forward-fill and then backward-fill
                # This is reasonable for futures contracts where missing data is usually
                # due to contract roll timing or data gaps
                F0_filled = F0.ffill().bfill()
                F1_filled = F1.ffill().bfill()
                
                # Check if we have enough complete data
                complete_mask = (F0_filled.notna() & F1_filled.notna())
                complete_dates = complete_mask.sum()
                
                if complete_dates < self.window:
                    logger.warning(
                        f"[FxCommodCarry] Insufficient complete data for {root}: "
                        f"{complete_dates} days with both ranks, need {self.window}. "
                        f"Using forward-fill to handle gaps."
                    )
                    # Continue anyway - we'll compute what we can
                
                # Compute roll yield: roll_yield_raw = -(ln(F1) - ln(F0))
                # This ensures:
                # - Backwardation (F1 < F0) => log(F1/F0) < 0 => roll_yield_raw > 0
                # - Contango (F1 > F0) => log(F1/F0) > 0 => roll_yield_raw < 0
                roll_yield_raw = -(np.log(F1_filled) - np.log(F0_filled))
                
                # Handle any remaining NaN or inf values
                roll_yield_raw = roll_yield_raw.replace([np.inf, -np.inf], np.nan)
                
                # Store raw roll yield for cross-sectional and momentum features
                roll_yield_raw_dict[root] = roll_yield_raw
                
                # Standardize with rolling z-score (time-series feature)
                carry_ts_z = _zscore_rolling(
                    roll_yield_raw,
                    window=self.window,
                    clip=self.clip,
                    min_periods=max(self.window // 2, 126)
                )
                
                # Store time-series feature
                feature_name_ts = f"carry_ts_z_{root}"
                features_dict[feature_name_ts] = carry_ts_z
                
                # Track dates for alignment
                if all_dates is None:
                    all_dates = carry_ts_z.index
                else:
                    # Union of all dates
                    all_dates = all_dates.union(carry_ts_z.index)
                
                logger.debug(
                    f"[FxCommodCarry] Computed {feature_name_ts} for {root}: "
                    f"{len(carry_ts_z.dropna())} non-null values"
                )
                
            except Exception as e:
                logger.warning(
                    f"[FxCommodCarry] Error computing features for {root}: {e}"
                )
                continue
        
        if not features_dict:
            logger.warning("[FxCommodCarry] No features could be computed")
            return pd.DataFrame()
        
        # Combine raw roll yields into DataFrame for cross-sectional features
        roll_yield_df = pd.DataFrame(roll_yield_raw_dict, index=all_dates)
        roll_yield_df = roll_yield_df.sort_index()
        
        # Compute cross-sectional features (daily z-score across roots)
        # Only use roots that are actually in the DataFrame
        available_roots = [r for r in self.roots if r in roll_yield_df.columns]
        
        if len(available_roots) > 1:  # Need at least 2 roots for cross-sectional comparison
            for root in available_roots:
                # Cross-sectional z-score: compare each root's roll yield to the mean across all available roots on that day
                # Use a small floor for std to avoid dividing by ~0
                xs_mean = roll_yield_df[available_roots].mean(axis=1)
                xs_std = roll_yield_df[available_roots].std(axis=1)
                xs_std = xs_std.clip(lower=1e-6)  # Small floor to avoid division by zero
                
                carry_xs_raw = (roll_yield_df[root] - xs_mean) / xs_std
                carry_xs_z = carry_xs_raw.clip(-self.clip, self.clip)
                
                feature_name_xs = f"carry_xs_z_{root}"
                features_dict[feature_name_xs] = carry_xs_z
        else:
            logger.warning(
                f"[FxCommodCarry] Insufficient roots ({len(available_roots)}) for cross-sectional features. "
                f"Need at least 2 roots."
            )
        
        # Compute momentum features (63-day change in roll yield, then z-score)
        for root in self.roots:
            if root not in roll_yield_df.columns:
                continue
            
            roll_yield_series = roll_yield_df[root]
            
            # 63-day momentum: change in roll yield over 63 trading days
            carry_mom_63_raw = roll_yield_series - roll_yield_series.shift(63)
            
            # Standardize momentum with rolling z-score
            carry_mom_63_z = _zscore_rolling(
                carry_mom_63_raw,
                window=self.window,
                clip=self.clip,
                min_periods=max(self.window // 2, 126)
            )
            
            feature_name_mom = f"carry_mom_63_z_{root}"
            features_dict[feature_name_mom] = carry_mom_63_z
        
        # Combine into DataFrame
        features = pd.DataFrame(features_dict, index=all_dates)
        features = features.sort_index()
        
        logger.info(
            f"[FxCommodCarry] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features

