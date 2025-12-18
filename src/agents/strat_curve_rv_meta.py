"""
SR3 Curve RV Meta-Sleeve

Meta-sleeve that combines Curve RV atomic sleeves (Rank Fly, Pack Slope, etc.)
and integrates with CombinedStrategy architecture.

Since Curve RV trades spreads/flies (not individual assets), this meta-sleeve
returns a synthetic asset "CURVE_RV" that represents the combined Curve RV position.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Sequence
from datetime import datetime
import logging

from src.agents.strat_curve_rv_rank_fly import SR3CurveRVRankFlyAtomic
from src.agents.strat_curve_rv_pack_slope import SR3CurveRVPackSlopeAtomic

logger = logging.getLogger(__name__)

# Synthetic asset symbol for Curve RV
CURVE_RV_SYMBOL = "CURVE_RV"


class SR3CurveRVMeta:
    """
    SR3 Curve RV Meta-Sleeve wrapper for CombinedStrategy integration.
    
    Combines multiple Curve RV atomic sleeves (Rank Fly, Pack Slope, etc.)
    and returns signals for a synthetic asset "CURVE_RV" that represents
    the combined Curve RV position.
    
    The meta-sleeve computes returns internally from spread/fly positions
    and returns these as a synthetic asset signal.
    """
    
    def __init__(
        self,
        enabled_atomics: Sequence[str] = ("rank_fly", "pack_slope"),
        atomic_weights: Optional[Dict[str, float]] = None,
        zscore_window: int = 252,
        clip: float = 3.0,
        target_vol: float = 0.10,
        vol_lookback: int = 63,
        min_vol_floor: float = 0.01,
        max_leverage: float = 10.0,
        lag: int = 1
    ):
        """
        Initialize SR3 Curve RV Meta-Sleeve.
        
        Args:
            enabled_atomics: List of enabled atomic sleeves ("rank_fly", "pack_slope", "pack_curvature")
            atomic_weights: Optional dict of weights for each atomic (default: equal weight or 100% if single)
            zscore_window: Rolling window for z-score normalization (default: 252 days)
            clip: Symmetric clipping bounds for normalized signal (default: ±3.0)
            target_vol: Target annualized volatility (default: 0.10 = 10%)
            vol_lookback: Rolling window for realized vol calculation (default: 63 days)
            min_vol_floor: Minimum annualized vol floor (default: 0.01 = 1%)
            max_leverage: Maximum leverage cap (default: 10.0)
            lag: Execution lag in days (default: 1)
        """
        self.enabled_atomics = list(enabled_atomics)
        self.zscore_window = zscore_window
        self.clip = clip
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.min_vol_floor = min_vol_floor
        self.max_leverage = max_leverage
        self.lag = lag
        
        # Initialize atomic sleeves
        self.atomics = {}
        
        if "rank_fly" in self.enabled_atomics:
            self.atomics["rank_fly"] = SR3CurveRVRankFlyAtomic(
                zscore_window=zscore_window,
                clip=clip,
                target_vol=target_vol,
                vol_lookback=vol_lookback,
                min_vol_floor=min_vol_floor,
                max_leverage=max_leverage,
                lag=lag
            )
        
        if "pack_slope" in self.enabled_atomics:
            self.atomics["pack_slope"] = SR3CurveRVPackSlopeAtomic(
                zscore_window=zscore_window,
                clip=clip,
                target_vol=target_vol,
                vol_lookback=vol_lookback,
                min_vol_floor=min_vol_floor,
                max_leverage=max_leverage,
                lag=lag
            )
        
        # Set atomic weights
        if atomic_weights is None:
            # Default: Core v9 canonical weights (Rank Fly 0.625, Pack Slope 0.375)
            if "rank_fly" in self.enabled_atomics and "pack_slope" in self.enabled_atomics:
                # Both enabled: canonical Core v9 split
                self.atomic_weights = {"rank_fly": 0.625, "pack_slope": 0.375}
            elif len(self.enabled_atomics) == 1:
                self.atomic_weights = {self.enabled_atomics[0]: 1.0}
            else:
                # Equal weight for other combinations
                self.atomic_weights = {atomic: 1.0 / len(self.enabled_atomics) 
                                     for atomic in self.enabled_atomics}
        else:
            self.atomic_weights = atomic_weights.copy()
            # Normalize weights
            total = sum(self.atomic_weights.values())
            if total > 0:
                self.atomic_weights = {k: v / total for k, v in self.atomic_weights.items()}
        
        # Cache for computed returns
        self._returns_cache = None
        self._cache_start = None
        self._cache_end = None
        
        logger.info(
            f"[SR3CurveRVMeta] Initialized with enabled_atomics={self.enabled_atomics}, "
            f"atomic_weights={self.atomic_weights}"
        )
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        universe: Optional[Sequence[str]] = None
    ) -> pd.Series:
        """
        Get Curve RV signals for a specific date (compatible with CombinedStrategy).
        
        **Note**: Curve RV trades spreads/flies, not individual assets.
        Returns a synthetic asset "CURVE_RV" that represents the combined Curve RV position.
        
        Args:
            market: MarketData instance
            date: Date to get signals for
            universe: Ignored (Curve RV trades spreads, not universe assets)
            
        Returns:
            Series with single entry: CURVE_RV signal at specified date
        """
        # For now, return a placeholder signal
        # The actual returns are computed in compute_returns() and handled specially
        # by ExecSim or the portfolio construction logic
        
        # Return a zero signal for the synthetic asset
        # The actual Curve RV returns will be added to the portfolio separately
        return pd.Series({CURVE_RV_SYMBOL: 0.0})
    
    def compute_returns(
        self,
        market,
        start: str,
        end: str
    ) -> pd.Series:
        """
        Compute combined Curve RV portfolio returns.
        
        Args:
            market: MarketData instance
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            Series of combined portfolio returns indexed by date
        """
        # Check cache
        if (self._returns_cache is not None and 
            self._cache_start == start and 
            self._cache_end == end):
            return self._returns_cache
        
        logger.info(f"[SR3CurveRVMeta] Computing combined returns from {start} to {end}")
        
        # Compute returns for each enabled atomic
        atomic_returns = {}
        for atomic_name in self.enabled_atomics:
            if atomic_name in self.atomics:
                atomic = self.atomics[atomic_name]
                atomic_returns[atomic_name] = atomic.compute_returns(market, start, end)
        
        if not atomic_returns:
            logger.warning("[SR3CurveRVMeta] No atomic returns computed")
            return pd.Series(dtype=float, name='portfolio_return')
        
        # Combine atomic returns with weights
        if len(atomic_returns) == 1:
            # Single atomic: return its returns directly
            combined_returns = list(atomic_returns.values())[0]
        else:
            # Multiple atomics: combine with weights
            # Align all returns to common dates
            all_dates = None
            for rets in atomic_returns.values():
                if all_dates is None:
                    all_dates = rets.index
                else:
                    all_dates = all_dates.intersection(rets.index)
            
            # Combine with weights
            combined_returns = pd.Series(0.0, index=all_dates)
            for atomic_name, rets in atomic_returns.items():
                weight = self.atomic_weights.get(atomic_name, 0.0)
                aligned_rets = rets.loc[all_dates]
                combined_returns += weight * aligned_rets
        
        combined_returns.name = 'portfolio_return'
        
        # Cache results
        self._returns_cache = combined_returns
        self._cache_start = start
        self._cache_end = end
        
        logger.info(
            f"[SR3CurveRVMeta] Combined returns: n={len(combined_returns)}, "
            f"mean={combined_returns.mean():.6f}, std={combined_returns.std():.6f}"
        )
        
        return combined_returns
    
    def warmup_periods(self) -> int:
        """
        Return number of trading days required for warmup.
        
        Curve RV requires the maximum warmup across all enabled atomics.
        """
        max_warmup = 0
        for atomic in self.atomics.values():
            max_warmup = max(max_warmup, atomic.warmup_periods())
        return max_warmup

