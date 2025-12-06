"""
SR3 Curve Features: Carry and Curve Shape Features for SOFR Futures.

Computes Carver-style carry, curve shape, and STIR pack features from
12 SR3 contract ranks (0-11). Features are standardized using rolling z-scores.
"""

import logging
from typing import Optional, Union
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
        min_periods = max(window // 2, 63)  # At least 63 days (1 quarter)
    
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sigma
    return z.clip(-clip, clip)


class Sr3CurveFeatures:
    """
    Builds SOFR (SR3) carry & curve features from 12 ranks (0-11).
    
    Only SR3 rank 0 is tradeable; others are feature-only.
    
    Features computed:
    - sr3_carry_01_z: Carver carry (r1 - r0) standardized
    - sr3_curve_02_z: Curve shape (r2 - r0) standardized
    - sr3_pack_slope_fb_z: Pack slope (front vs back) standardized
    - sr3_front_pack_level_z: Front-pack level (policy expectation) standardized
    - sr3_curvature_belly_z: Belly curvature (hump vs straight) standardized
    """
    
    def __init__(
        self,
        root: str = "SR3",
        ranks: Optional[list] = None,
        window: int = 252
    ):
        """
        Initialize SR3 curve features calculator.
        
        Args:
            root: Root symbol (default: "SR3")
            ranks: List of ranks to use (default: 0-11)
            window: Rolling window for z-score standardization (default: 252 days)
        """
        self.root = root
        self.ranks = ranks if ranks is not None else list(range(12))
        self.window = window
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute SR3 features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - sr3_carry_01_z: Carver carry (r1 - r0) standardized
            - sr3_curve_02_z: Curve shape (r2 - r0) standardized
            - sr3_pack_slope_fb_z: Pack slope (front vs back) standardized
            - sr3_front_pack_level_z: Front-pack level (policy expectation) standardized
            - sr3_curvature_belly_z: Belly curvature (hump vs straight) standardized
        """
        # 1) Load close prices for all SR3 ranks
        # Query ALL available data (don't limit start_date) to build full rolling window history
        # The rolling window will handle the lookback requirement
        close = market.get_contracts_by_root(
            root=self.root,
            ranks=self.ranks,
            fields=("close",),
            start=None,  # Get all available data from the beginning
            end=end_date
        )
        
        if close.empty:
            logger.warning(f"[SR3] No data found for root {self.root}")
            return pd.DataFrame()
        
        # Ensure we have ranks 0, 1, 2 at minimum
        required_ranks = [0, 1, 2]
        available_ranks = [c for c in close.columns if c in required_ranks]
        if len(available_ranks) < len(required_ranks):
            missing = set(required_ranks) - set(available_ranks)
            logger.warning(f"[SR3] Missing required ranks: {missing}. Available: {list(close.columns)}")
            return pd.DataFrame(index=close.index)
        
        # 2) Handle missing data: forward-fill and then backward-fill
        # This is reasonable for futures contracts where missing data is usually due to
        # contract roll timing or data gaps, not fundamental price changes
        # Use ffill() and bfill() instead of deprecated fillna(method=...)
        close_filled = close.ffill().bfill()
        
        # Check if we have enough complete data after filling
        # We need at least window days where ranks 0, 1, 2 all have data
        complete_mask = close_filled[required_ranks].notna().all(axis=1)
        complete_dates = complete_mask.sum()
        
        if complete_dates < self.window:
            logger.warning(
                f"[SR3] Insufficient complete data: {complete_dates} days with all required ranks, "
                f"need {self.window}. Using forward-fill to handle gaps."
            )
            # Continue anyway - we'll compute what we can
        
        # 3) Convert prices to rates: r_k = 100 - P_k
        rates = 100.0 - close_filled
        
        # 4) Extract F0, F1, F2 (as rates: r0, r1, r2)
        r0 = rates[0]
        r1 = rates[1]
        r2 = rates[2]
        
        # 5) Carver Carry and Curve (in rate space)
        # carry_01 = r1 - r0: "how much higher is the next 3M vs the front 3M"
        carry_01_raw = r1 - r0
        
        # curve_02 = r2 - r0: "how much higher is the 3rd 3M vs front 3M"
        curve_02_raw = r2 - r0
        
        # 6) STIR Packs
        # Pack front: ranks 0-3
        pack_front_ranks = [r for r in range(4) if r in rates.columns]
        if len(pack_front_ranks) < 2:
            logger.warning(f"[SR3] Insufficient ranks for pack_front: {pack_front_ranks}")
            pack_front = pd.Series(index=rates.index, dtype=float)
        else:
            pack_front = rates[pack_front_ranks].mean(axis=1)
        
        # Pack belly: ranks 4-7
        pack_belly_ranks = [r for r in range(4, 8) if r in rates.columns]
        if len(pack_belly_ranks) < 2:
            pack_belly = pd.Series(index=rates.index, dtype=float)
        else:
            pack_belly = rates[pack_belly_ranks].mean(axis=1)
        
        # Pack back: ranks 8-11
        pack_back_ranks = [r for r in range(8, 12) if r in rates.columns]
        if len(pack_back_ranks) < 2:
            pack_back = pd.Series(index=rates.index, dtype=float)
        else:
            pack_back = rates[pack_back_ranks].mean(axis=1)
        
        # Pack slope (front vs back): simple difference in rate space
        # pack_slope_fb_raw = pack_back - pack_front
        # Positive = back higher than front (upward sloping curve)
        pack_slope_fb_raw = pack_back - pack_front
        
        # 7) Additional features
        # Front-pack level: absolute level of expected policy over ~1 year
        # High front-pack level → policy very tight → less room for hikes, more room for cuts
        # Low front-pack level → policy very easy → more room for hikes
        front_pack_level_raw = pack_front
        
        # Belly curvature: hump vs straight term structure
        # front_back_avg = (pack_front + pack_back) / 2
        # curvature_belly = belly_pack - front_back_avg
        # Positive: belly rates higher than straight line between front and back → "hump-y" curve
        # Negative: belly lower → "U-shaped" or very concave curve
        front_back_avg = (pack_front + pack_back) / 2.0
        curvature_belly_raw = pack_belly - front_back_avg
        
        # 8) Rolling z-scores
        sr3_carry_01_z = _zscore_rolling(carry_01_raw, window=self.window)
        sr3_curve_02_z = _zscore_rolling(curve_02_raw, window=self.window)
        sr3_pack_slope_fb_z = _zscore_rolling(pack_slope_fb_raw, window=self.window)
        sr3_front_pack_level_z = _zscore_rolling(front_pack_level_raw, window=self.window)
        sr3_curvature_belly_z = _zscore_rolling(curvature_belly_raw, window=self.window)
        
        # 9) Combine into feature DataFrame
        features = pd.DataFrame(
            {
                "sr3_carry_01_z": sr3_carry_01_z,
                "sr3_curve_02_z": sr3_curve_02_z,
                "sr3_pack_slope_fb_z": sr3_pack_slope_fb_z,
                "sr3_front_pack_level_z": sr3_front_pack_level_z,
                "sr3_curvature_belly_z": sr3_curvature_belly_z,
            },
            index=close.index,
        )
        
        logger.info(
            f"[SR3] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features

