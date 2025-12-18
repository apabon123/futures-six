"""
SR3 Curve RV Rank Fly Momentum Atomic Sleeve

Atomic sleeve for Rank Fly (2,6,10) momentum strategy.
Integrates with CombinedStrategy architecture.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
from datetime import datetime
import logging

from src.strategies.rates_curve_rv.sr3_curve_rv_momentum import compute_rank_fly_momentum_phase1

logger = logging.getLogger(__name__)


class SR3CurveRVRankFlyAtomic:
    """
    Atomic sleeve for SR3 Curve RV Rank Fly Momentum.
    
    Computes Rank Fly (2,6,10) momentum signals and returns portfolio returns
    that can be combined with other sleeves.
    """
    
    def __init__(
        self,
        zscore_window: int = 252,
        clip: float = 3.0,
        target_vol: float = 0.10,
        vol_lookback: int = 63,
        min_vol_floor: float = 0.01,
        max_leverage: float = 10.0,
        lag: int = 1
    ):
        """
        Initialize Rank Fly atomic sleeve.
        
        Args:
            zscore_window: Rolling window for z-score normalization (default: 252 days)
            clip: Symmetric clipping bounds for normalized signal (default: ±3.0)
            target_vol: Target annualized volatility (default: 0.10 = 10%)
            vol_lookback: Rolling window for realized vol calculation (default: 63 days)
            min_vol_floor: Minimum annualized vol floor (default: 0.01 = 1%)
            max_leverage: Maximum leverage cap (default: 10.0)
            lag: Execution lag in days (default: 1)
        """
        self.zscore_window = zscore_window
        self.clip = clip
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.min_vol_floor = min_vol_floor
        self.max_leverage = max_leverage
        self.lag = lag
        
        # Cache for computed returns
        self._returns_cache = None
        self._cache_start = None
        self._cache_end = None
        
        logger.info(
            f"[SR3CurveRVRankFlyAtomic] Initialized with "
            f"zscore_window={zscore_window}, clip={clip}, target_vol={target_vol}"
        )
    
    def compute_returns(
        self,
        market,
        start: str,
        end: str
    ) -> pd.Series:
        """
        Compute portfolio returns for Rank Fly momentum.
        
        Args:
            market: MarketData instance
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            Series of portfolio returns indexed by date
        """
        # Check cache
        if (self._returns_cache is not None and 
            self._cache_start == start and 
            self._cache_end == end):
            return self._returns_cache
        
        logger.info(f"[SR3CurveRVRankFlyAtomic] Computing returns from {start} to {end}")
        
        # Use Phase-1 computation
        result = compute_rank_fly_momentum_phase1(
            market=market,
            start_date=start,
            end_date=end,
            zscore_window=self.zscore_window,
            clip=self.clip,
            target_vol=self.target_vol,
            vol_lookback=self.vol_lookback,
            min_vol_floor=self.min_vol_floor,
            max_leverage=self.max_leverage,
            lag=self.lag
        )
        
        returns = result['portfolio_returns']
        
        # Cache results
        self._returns_cache = returns
        self._cache_start = start
        self._cache_end = end
        
        return returns
    
    def warmup_periods(self) -> int:
        """
        Return number of trading days required for warmup.
        
        Rank Fly requires:
        - zscore_window days for z-score standardization (252)
        - vol_lookback days for vol targeting (63)
        - Total: max(252, 63) = 252 days
        """
        return max(self.zscore_window, self.vol_lookback)

