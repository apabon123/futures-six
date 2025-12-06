"""
Rates Curve Strategy: 2s10s and 5s30s Curve Trading Strategy with Curvature.

Generates signals based on curve shape features (slopes and curvatures) computed
from FRED-anchored implied yields. Combines slope and curvature features with
configurable weights. Positive combined signal => flattener, negative => steepener.

Features (4 total):
- curve_2s10s_z: 2s10s slope
- curve_5s30s_z: 5s30s slope
- curve_2s5s10s_curv_z: 2s-5s-10s belly curvature
- curve_5s10s30s_curv_z: 5s-10s-30s belly curvature
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RatesCurveStrategy:
    """
    2s10s and 5s30s curve strategy using anchored implied yields.
    
    Positive curve_z => flattener: long front, short back
    Negative curve_z => steepener: short front, long back
    
    For 2s10s:
    - Positive => long ZT (2y), short ZN (10y)
    - Negative => short ZT (2y), long ZN (10y)
    
    For 5s30s:
    - Positive => long ZF (5y), short UB (30y)
    - Negative => short ZF (5y), long UB (30y)
    """
    
    def __init__(
        self,
        z_cap: float = 3.0,
        w_slope_2s10s: float = 0.35,
        w_slope_5s30s: float = 0.35,
        w_curv_2s5s10s: float = 0.15,
        w_curv_5s10s30s: float = 0.15,
        sym_2y: str = "ZT_FRONT_VOLUME",
        sym_5y: str = "ZF_FRONT_VOLUME",
        sym_10y: str = "ZN_FRONT_VOLUME",
        sym_30y: str = "UB_FRONT_VOLUME",
    ):
        """
        Initialize RatesCurveStrategy.
        
        Args:
            z_cap: Cap on z-score signals (default: 3.0)
            w_slope_2s10s: Weight for 2s10s slope feature (default: 0.35)
            w_slope_5s30s: Weight for 5s30s slope feature (default: 0.35)
            w_curv_2s5s10s: Weight for 2s-5s-10s curvature feature (default: 0.15)
            w_curv_5s10s30s: Weight for 5s-10s-30s curvature feature (default: 0.15)
            sym_2y: Symbol for 2-year futures
            sym_5y: Symbol for 5-year futures
            sym_10y: Symbol for 10-year futures
            sym_30y: Symbol for 30-year futures
        """
        self.z_cap = z_cap
        self.w_slope_2s10s = w_slope_2s10s
        self.w_slope_5s30s = w_slope_5s30s
        self.w_curv_2s5s10s = w_curv_2s5s10s
        self.w_curv_5s10s30s = w_curv_5s10s30s
        self.sym_2y = sym_2y
        self.sym_5y = sym_5y
        self.sym_10y = sym_10y
        self.sym_30y = sym_30y
        
        # Normalize weights within each curve segment
        # For 2s10s: normalize slope + curvature weights
        total_2s10s = w_slope_2s10s + w_curv_2s5s10s
        if total_2s10s > 0:
            self.w_slope_2s10s /= total_2s10s
            self.w_curv_2s5s10s /= total_2s10s
        
        # For 5s30s: normalize slope + curvature weights
        total_5s30s = w_slope_5s30s + w_curv_5s10s30s
        if total_5s30s > 0:
            self.w_slope_5s30s /= total_5s30s
            self.w_curv_5s10s30s /= total_5s30s
    
    def signals(
        self,
        rates_features: pd.DataFrame,
        date: Union[str, datetime]
    ) -> pd.Series:
        """
        Generate signals from rates curve features.
        
        Args:
            rates_features: DataFrame with curve_2s10s_z and curve_5s30s_z columns
            date: Current rebalance date
            
        Returns:
            Series with signals for ZT, ZF, ZN, UB
        """
        date_dt = pd.to_datetime(date)
        
        # Get features for this date (forward-fill if needed)
        if rates_features.empty:
            logger.warning(f"[RatesCurve] No features available (empty DataFrame)")
            return pd.Series(dtype=float)
        
        # Ensure index is DatetimeIndex for comparison
        if not isinstance(rates_features.index, pd.DatetimeIndex):
            rates_features.index = pd.to_datetime(rates_features.index)
        
        if date_dt not in rates_features.index:
            # Find available dates before or equal to date_dt
            mask = rates_features.index <= date_dt
            available_before = rates_features.index[mask]
            if len(available_before) == 0:
                logger.warning(f"[RatesCurve] No features available for date {date_dt}")
                return pd.Series(dtype=float)
            # Use most recent available
            date_dt = available_before.max()
        
        row = rates_features.loc[date_dt]
        
        # Extract all feature z-scores
        z_slope_2s10s = row.get("curve_2s10s_z", np.nan)
        z_slope_5s30s = row.get("curve_5s30s_z", np.nan)
        z_curv_2s5s10s = row.get("curve_2s5s10s_curv_z", np.nan)
        z_curv_5s10s30s = row.get("curve_5s10s30s_curv_z", np.nan)
        
        if (pd.isna(z_slope_2s10s) and pd.isna(z_curv_2s5s10s) and 
            pd.isna(z_slope_5s30s) and pd.isna(z_curv_5s10s30s)):
            logger.warning(f"[RatesCurve] No curve features available for date {date_dt}")
            return pd.Series(dtype=float)
        
        # Initialize signals
        signals = {}
        
        # 2s10s curve: combine slope and curvature
        # signal_2s10s = w_slope * slope_z + w_curv * curv_z
        signal_2s10s = 0.0
        has_2s10s = False
        
        if not pd.isna(z_slope_2s10s):
            signal_2s10s += self.w_slope_2s10s * z_slope_2s10s
            has_2s10s = True
        
        if not pd.isna(z_curv_2s5s10s):
            signal_2s10s += self.w_curv_2s5s10s * z_curv_2s5s10s
            has_2s10s = True
        
        if has_2s10s:
            # Cap combined signal
            signal_2s10s = max(min(signal_2s10s, self.z_cap), -self.z_cap)
            # Positive signal => flattener: long 2y, short 10y
            # Negative signal => steepener: short 2y, long 10y
            s_2y = +signal_2s10s
            s_10y = -signal_2s10s
            signals[self.sym_2y] = s_2y
            signals[self.sym_10y] = s_10y
        
        # 5s30s curve: combine slope and curvature
        # signal_5s30s = w_slope * slope_z + w_curv * curv_z
        signal_5s30s = 0.0
        has_5s30s = False
        
        if not pd.isna(z_slope_5s30s):
            signal_5s30s += self.w_slope_5s30s * z_slope_5s30s
            has_5s30s = True
        
        if not pd.isna(z_curv_5s10s30s):
            signal_5s30s += self.w_curv_5s10s30s * z_curv_5s10s30s
            has_5s30s = True
        
        if has_5s30s:
            # Cap combined signal
            signal_5s30s = max(min(signal_5s30s, self.z_cap), -self.z_cap)
            # Positive signal => flattener: long 5y, short 30y
            # Negative signal => steepener: short 5y, long 30y
            s_5y = +signal_5s30s
            s_30y = -signal_5s30s
            signals[self.sym_5y] = s_5y
            signals[self.sym_30y] = s_30y
        
        # If both curves active, combine signals (additive)
        # This means if both curves suggest same direction, signals add up
        result = pd.Series(signals)
        
        return result

