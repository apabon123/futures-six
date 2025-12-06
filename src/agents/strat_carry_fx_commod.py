"""
FX/Commodity Carry Strategy: Multi-asset strategy based on roll yield carry features.

Uses standardized carry features (time-series, cross-sectional, and momentum)
to generate signals for FX (6E, 6B, 6J) and commodities (CL, GC).

Features (3 per root):
- carry_ts_z_<root>: Time-series roll yield standardized
- carry_xs_z_<root>: Cross-sectional carry strength (relative to other assets)
- carry_mom_63_z_<root>: 63-day carry momentum (change in roll yield)

Positive combined signal => long (backwardation), negative => short (contango).
"""

import logging
from typing import Optional, Union, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np

from .feature_carry_fx_commod import FxCommodCarryFeatures

logger = logging.getLogger(__name__)


class CarryFxCommodStrategy:
    """
    FX/Commodity carry strategy based on roll yield features.
    
    Returns signals for CL, GC, 6E, 6B, 6J based on standardized carry features.
    Positive carry_z => long (backwardation), negative => short (contango).
    """
    
    def __init__(
        self,
        roots: Optional[List[str]] = None,
        w_ts: float = 0.6,
        w_xs: float = 0.25,
        w_mom: float = 0.15,
        clip: float = 3.0,
        window: int = 252,
        symbol_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize FX/Commodity carry strategy.
        
        Args:
            roots: List of root symbols (default: ["CL", "GC", "6E", "6B", "6J"])
            w_ts: Weight for time-series carry feature (default: 0.6)
            w_xs: Weight for cross-sectional carry feature (default: 0.25)
            w_mom: Weight for carry momentum feature (default: 0.15)
            clip: Signal cap in standard deviations (default: 3.0)
            window: Rolling window for feature standardization (default: 252)
            symbol_map: Optional mapping from root to database symbol
                       (default: auto-generate based on roll type)
        """
        self.roots = roots if roots is not None else ["CL", "GC", "6E", "6B", "6J"]
        self.w_ts = w_ts
        self.w_xs = w_xs
        self.w_mom = w_mom
        self.clip = clip
        self.window = window
        
        # Normalize weights to sum to 1.0
        total_weight = w_ts + w_xs + w_mom
        if total_weight > 0:
            self.w_ts /= total_weight
            self.w_xs /= total_weight
            self.w_mom /= total_weight
        
        # Default symbol mapping: assume volume roll for commodities, calendar for FX
        if symbol_map is None:
            self.symbol_map = {}
            for root in self.roots:
                if root in ["CL", "GC"]:
                    # Commodities use volume roll
                    self.symbol_map[root] = f"{root}_FRONT_VOLUME"
                elif root in ["6E", "6B", "6J"]:
                    # FX uses calendar roll
                    # FX symbols use FRONT_CALENDAR (no _2D suffix) in database
                    self.symbol_map[root] = f"{root}_FRONT_CALENDAR"
                else:
                    # Default to calendar roll
                    # FX symbols use FRONT_CALENDAR (no _2D suffix) in database
                    self.symbol_map[root] = f"{root}_FRONT_CALENDAR"
        else:
            self.symbol_map = symbol_map
        
        # Initialize feature calculator
        self.feature_calc = FxCommodCarryFeatures(
            roots=self.roots,
            window=self.window,
            clip=self.clip
        )
        
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
        Generate signals for FX/Commodity carry based on roll yield features.
        
        Args:
            market: MarketData instance
            date: Current rebalance date
            features: Optional pre-computed features DataFrame (if None, computes on demand)
            
        Returns:
            Series with signals for CL, GC, 6E, 6B, 6J
        """
        date_dt = pd.to_datetime(date)
        
        # Get or compute features
        if features is None:
            # Check cache
            if (self._features_cache is None or 
                self._cache_end_date is None or 
                date_dt > self._cache_end_date):
                # Compute features up to current date
                self._features_cache = self.feature_calc.compute(market, end_date=date_dt)
                self._cache_end_date = date_dt
            features = self._features_cache
        
        if features.empty:
            logger.warning(f"[CarryFxCommod] No features available for date {date_dt}")
            # Return zero signals for all roots
            return pd.Series({self.symbol_map[root]: 0.0 for root in self.roots})
        
        # Find the closest available date (forward-fill from previous available date)
        if date_dt not in features.index:
            # Find the last available date <= current date
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(
                    f"[CarryFxCommod] No features available for date {date_dt} "
                    f"(no prior data)"
                )
                return pd.Series({self.symbol_map[root]: 0.0 for root in self.roots})
            # Use the most recent available date
            use_date = available_dates[-1]
            logger.debug(
                f"[CarryFxCommod] Using features from {use_date} for date {date_dt}"
            )
        else:
            use_date = date_dt
        
        # Get feature values for this date
        f = features.loc[use_date]
        
        # Initialize signals
        signals = {}
        
        # Generate signals for each root
        for root in self.roots:
            symbol_name = self.symbol_map[root]
            
            # Get all three feature values
            feature_ts = f"carry_ts_z_{root}"
            feature_xs = f"carry_xs_z_{root}"
            feature_mom = f"carry_mom_63_z_{root}"
            
            carry_ts_z = f.get(feature_ts, np.nan)
            carry_xs_z = f.get(feature_xs, np.nan)
            carry_mom_z = f.get(feature_mom, np.nan)
            
            # Combine features with weights
            # signal = w_ts * carry_ts_z + w_xs * carry_xs_z + w_mom * carry_mom_z
            signal = 0.0
            has_features = False
            
            if not pd.isna(carry_ts_z):
                signal += self.w_ts * carry_ts_z
                has_features = True
            
            if not pd.isna(carry_xs_z):
                signal += self.w_xs * carry_xs_z
                has_features = True
            
            if not pd.isna(carry_mom_z):
                signal += self.w_mom * carry_mom_z
                has_features = True
            
            if not has_features:
                # No features available for this root
                signals[symbol_name] = 0.0
                logger.debug(
                    f"[CarryFxCommod] No features available for {root} on date {use_date}"
                )
            else:
                # Clip combined signal to bounds
                signal = max(min(signal, self.clip), -self.clip)
                
                # Positive signal => long (backwardation)
                # Negative signal => short (contango)
                signals[symbol_name] = signal
        
        result = pd.Series(signals)
        
        logger.debug(
            f"[CarryFxCommod] Generated signals at {date_dt}: "
            f"mean={result.mean():.3f}, std={result.std():.3f}, "
            f"non-zero={(result != 0).sum()}/{len(result)}"
        )
        
        return result
    
    def clear_cache(self):
        """Clear cached features."""
        self._features_cache = None
        self._cache_end_date = None

