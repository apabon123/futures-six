"""
Momentum Persistence Strategy: Multi-feature persistence (momentum-of-momentum) strategy.

Uses three persistence features:
- persistence_slope_accel_z: Slope acceleration (EMA20-EMA84 acceleration)
- persistence_breakout_accel_z: Breakout acceleration (breakout_126 acceleration)
- persistence_return_accel_z: Return acceleration (ret_84 acceleration)

Combines features with configurable weights and returns signals per symbol.
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MomentumPersistence:
    """
    Momentum persistence strategy using multiple acceleration features.
    
    Combines three persistence features with configurable weights:
    - Slope acceleration (primary)
    - Breakout acceleration (optional)
    - Return acceleration (optional)
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        weights: Optional[dict] = None,
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI"
    ):
        """
        Initialize Momentum Persistence Strategy.
        
        Args:
            symbols: List of symbols to trade (default: None, uses market.universe)
            weights: Dictionary of feature weights:
                - slope_accel: Weight for slope acceleration (default: 0.80)
                - breakout_accel: Weight for breakout acceleration (default: 0.10)
                - return_accel: Weight for return acceleration (default: 0.10)
            signal_cap: Maximum absolute signal value (default: 3.0)
            rebalance: Rebalance frequency (default: "W-FRI")
        """
        self.symbols = symbols
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        
        # Default weights (slope acceleration is primary)
        default_weights = {
            "slope_accel": 0.80,
            "breakout_accel": 0.10,
            "return_accel": 0.10
        }
        
        if weights is None:
            weights = default_weights
        else:
            # Merge with defaults
            for key in default_weights:
                if key not in weights:
                    weights[key] = default_weights[key]
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            logger.warning("[MomentumPersistence] All weights are zero, using defaults")
            self.weights = default_weights
        
        logger.info(
            f"[MomentumPersistence] Initialized with weights={self.weights}, "
            f"cap={signal_cap}, rebalance={rebalance}"
        )
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate persistence signals for a given date.
        
        Args:
            market: MarketData instance
            date: Date for signal generation
            features: Optional pre-computed features DataFrame (PERSISTENCE)
            
        Returns:
            Series of signals indexed by symbol (cross-sectionally z-scored and clipped)
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[MomentumPersistence] No symbols available for date {date_dt}")
            return pd.Series(index=symbols, dtype=float)
        
        # Get features if not provided
        if features is None or features.empty:
            logger.warning(f"[MomentumPersistence] No features available for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        # Find the appropriate date in features
        if date_dt not in features.index:
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(f"[MomentumPersistence] No features available for date {date_dt}")
                return pd.Series(0.0, index=symbols)
            use_date = available_dates[-1]
        else:
            use_date = date_dt
        
        # Get feature row
        feature_row = features.loc[use_date]
        
        # Compute signals per symbol
        signals = {}
        for symbol in symbols:
            # Get features for this symbol
            slope_feature = f"persistence_slope_accel_z_{symbol}"
            breakout_feature = f"persistence_breakout_accel_z_{symbol}"
            return_feature = f"persistence_return_accel_z_{symbol}"
            
            # Extract feature values
            slope_val = feature_row.get(slope_feature, np.nan) if slope_feature in feature_row.index else np.nan
            breakout_val = feature_row.get(breakout_feature, np.nan) if breakout_feature in feature_row.index else np.nan
            return_val = feature_row.get(return_feature, np.nan) if return_feature in feature_row.index else np.nan
            
            # Combine features with weights
            signal_raw = (
                self.weights["slope_accel"] * (slope_val if pd.notna(slope_val) else 0.0) +
                self.weights["breakout_accel"] * (breakout_val if pd.notna(breakout_val) else 0.0) +
                self.weights["return_accel"] * (return_val if pd.notna(return_val) else 0.0)
            )
            
            signals[symbol] = signal_raw
        
        signal_series = pd.Series(signals)
        
        # Cross-sectional z-score
        valid_signals = signal_series.dropna()
        if len(valid_signals) > 1:
            mean = valid_signals.mean()
            std = valid_signals.std()
            if std > 0:
                signal_series = (signal_series - mean) / std
            else:
                signal_series = signal_series * 0
        
        # Clip to signal cap
        signal_series = signal_series.clip(lower=-self.signal_cap, upper=self.signal_cap)
        
        # Fill NaN with 0
        signal_series = signal_series.fillna(0.0)
        
        logger.debug(
            f"[MomentumPersistence] Generated signals at {date_dt}: "
            f"mean={signal_series.mean():.3f}, std={signal_series.std():.3f}, "
            f"min={signal_series.min():.3f}, max={signal_series.max():.3f}"
        )
        
        return signal_series

