"""
SR3 Carry/Curve Strategy: Single-asset strategy based on SOFR carry & curve features.

Uses standardized carry, curve, pack slope, front-pack level, and belly curvature
features to generate signals for SR3 (SOFR futures). This is a feature-only strategy
sleeve that can be combined with other strategies in a portfolio allocator.

Features (5 total):
- sr3_carry_01_z: Carver carry (r1 - r0)
- sr3_curve_02_z: Curve shape (r2 - r0)
- sr3_pack_slope_fb_z: Pack slope (front vs back)
- sr3_front_pack_level_z: Front-pack level (policy expectation)
- sr3_curvature_belly_z: Belly curvature (hump vs straight)
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd

from .feature_sr3_curve import Sr3CurveFeatures

logger = logging.getLogger(__name__)


class Sr3CarryCurveStrategy:
    """
    SOFR (SR3) single-asset strategy based on carry & curve features.
    
    Returns a scalar signal for SR3 on each date based on weighted combination
    of standardized carry, curve, pack slope, front-pack level, and belly curvature features.
    """
    
    def __init__(
        self,
        root: str = "SR3",
        w_carry: float = 0.30,
        w_curve: float = 0.25,
        w_pack_slope: float = 0.20,
        w_front_lvl: float = 0.10,
        w_curv_belly: float = 0.15,
        cap: float = 3.0,
        window: int = 252
    ):
        """
        Initialize SR3 carry/curve strategy.
        
        Args:
            root: Root symbol (default: "SR3")
            w_carry: Weight for carry feature (default: 0.30)
            w_curve: Weight for curve feature (default: 0.25)
            w_pack_slope: Weight for pack slope feature (default: 0.20)
            w_front_lvl: Weight for front-pack level feature (default: 0.10)
            w_curv_belly: Weight for belly curvature feature (default: 0.15)
            cap: Signal cap in standard deviations (default: 3.0)
            window: Rolling window for feature standardization (default: 252)
        """
        self.root = root
        self.w_carry = w_carry
        self.w_curve = w_curve
        self.w_pack_slope = w_pack_slope
        self.w_front_lvl = w_front_lvl
        self.w_curv_belly = w_curv_belly
        self.cap = cap
        
        # Normalize weights to sum to 1.0
        total_weight = w_carry + w_curve + w_pack_slope + w_front_lvl + w_curv_belly
        if total_weight > 0:
            self.w_carry /= total_weight
            self.w_curve /= total_weight
            self.w_pack_slope /= total_weight
            self.w_front_lvl /= total_weight
            self.w_curv_belly /= total_weight
        
        # Initialize feature calculator
        self.feature_calc = Sr3CurveFeatures(root=root, window=window)
        
        # Cache for computed features
        self._features_cache = None
        self._cache_end_date = None
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate signals for SR3 based on carry & curve features.
        
        Args:
            market: MarketData instance
            date: Current rebalance date
            features: Optional pre-computed features DataFrame (if None, computes on demand)
            
        Returns:
            Series with signal for SR3_FRONT (or equivalent symbol)
        """
        date_dt = pd.to_datetime(date)
        
        # Get or compute features
        if features is None:
            # Check cache
            if self._features_cache is None or self._cache_end_date is None or date_dt > self._cache_end_date:
                # Compute features up to current date
                self._features_cache = self.feature_calc.compute(market, end_date=date_dt)
                self._cache_end_date = date_dt
            features = self._features_cache
        
        if features.empty:
            logger.warning(f"[SR3] No features available for date {date_dt}")
            # Return zero signal
            # SR3 uses FRONT_CALENDAR (no _2D suffix) in database
            return pd.Series({f"{self.root}_FRONT_CALENDAR": 0.0})
        
        # Find the closest available date (forward-fill from previous available date)
        if date_dt not in features.index:
            # Find the last available date <= current date
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(f"[SR3] No features available for date {date_dt} (no prior data)")
                # SR3 uses FRONT_CALENDAR (no _2D suffix) in database
                return pd.Series({f"{self.root}_FRONT_CALENDAR": 0.0})
            # Use the most recent available date
            use_date = available_dates[-1]
            logger.debug(f"[SR3] Using features from {use_date} for date {date_dt}")
        else:
            use_date = date_dt
        
        # Get feature values for this date
        f = features.loc[use_date]
        
        # Extract feature values (handle NaNs)
        s_carry = f.get("sr3_carry_01_z", 0.0) if pd.notna(f.get("sr3_carry_01_z")) else 0.0
        s_curve = f.get("sr3_curve_02_z", 0.0) if pd.notna(f.get("sr3_curve_02_z")) else 0.0
        s_pack_slope = f.get("sr3_pack_slope_fb_z", 0.0) if pd.notna(f.get("sr3_pack_slope_fb_z")) else 0.0
        s_front_lvl = f.get("sr3_front_pack_level_z", 0.0) if pd.notna(f.get("sr3_front_pack_level_z")) else 0.0
        s_curv_belly = f.get("sr3_curvature_belly_z", 0.0) if pd.notna(f.get("sr3_curvature_belly_z")) else 0.0
        
        # Weighted combination of all 5 features
        raw = (
            self.w_carry * s_carry +
            self.w_curve * s_curve +
            self.w_pack_slope * s_pack_slope +
            self.w_front_lvl * s_front_lvl +
            self.w_curv_belly * s_curv_belly
        )
        
        # Cap signal
        signal = max(min(raw, self.cap), -self.cap)
        
        # Return as Series with appropriate symbol name
        # SR3 uses FRONT_CALENDAR (no _2D suffix) in database
        symbol_name = f"{self.root}_FRONT_CALENDAR"
        
        return pd.Series({symbol_name: signal})
    
    def clear_cache(self):
        """Clear cached features."""
        self._features_cache = None
        self._cache_end_date = None

