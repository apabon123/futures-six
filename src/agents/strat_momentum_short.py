"""
Short-Term Momentum Strategy: Multi-feature momentum strategy.

Uses four short-term momentum features:
- mom_short_ret_21_z: 21-day return momentum
- mom_short_breakout_21_z: 21-day breakout strength
- mom_short_slope_fast_z: Fast trend slope
- mom_short_reversal_filter_z: Reversal filter (RSI-like)

Combines features with configurable weights and returns signals per symbol.
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ShortTermMomentumStrategy:
    """
    Short-term momentum strategy using multiple features.
    
    Combines three short-term momentum features with configurable weights:
    - Return momentum (21-day)
    - Breakout strength (21-day)
    - Fast trend slope (EMA-based)
    - Reversal filter (optional, not used in signal combination by default)
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        weights: Optional[dict] = None,
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI",
        variant: str = "canonical"
    ):
        """
        Initialize Short-Term Momentum Strategy.
        
        Args:
            symbols: List of symbols to trade (default: None, uses market.universe)
            weights: Dictionary of feature weights:
                - ret_21: Weight for return momentum
                - breakout_21: Weight for breakout strength
                - slope_fast: Weight for fast trend slope
                - reversal_filter: Weight for reversal filter (default: 0.0, not used)
            signal_cap: Maximum absolute signal value (default: 3.0)
            rebalance: Rebalance frequency (default: "W-FRI")
            variant: Strategy variant, either "canonical" (default) or "legacy"
                - canonical: Equal-weight 1/3, 1/3, 1/3 composite (ret, breakout, slope)
                - legacy: 0.5, 0.3, 0.2 weighting (for comparison)
        """
        self.symbols = symbols
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        self.variant = variant
        
        # Define variant presets
        if variant == "canonical":
            variant_weights = {
                "ret_21": 1.0 / 3.0,
                "breakout_21": 1.0 / 3.0,
                "slope_fast": 1.0 / 3.0,
                "reversal_filter": 0.0  # Not used in canonical
            }
        elif variant == "legacy":
            variant_weights = {
                "ret_21": 0.5,
                "breakout_21": 0.3,
                "slope_fast": 0.2,
                "reversal_filter": 0.0  # Not used by default
            }
        else:
            raise ValueError(f"Unknown variant: {variant}. Must be 'canonical' or 'legacy'")
        
        # Override with explicit weights if provided
        if weights is not None:
            for key in weights:
                if key in variant_weights:
                    variant_weights[key] = weights[key]
        
        # Normalize weights to sum to 1.0 (excluding reversal_filter if weight is 0)
        active_weights = {k: v for k, v in variant_weights.items() if v > 0 and k != "reversal_filter"}
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in active_weights.items()}
            # Add reversal_filter with its original weight (even if 0)
            self.weights["reversal_filter"] = variant_weights.get("reversal_filter", 0.0)
        else:
            logger.warning("[ShortTermMomentum] All weights are zero, using variant defaults")
            self.weights = variant_weights
        
        logger.info(
            f"[ShortTermMomentum] Initialized with variant={variant}, weights={self.weights}, "
            f"cap={signal_cap}, rebalance={rebalance}"
        )
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate short-term momentum signals for a given date.
        
        Args:
            market: MarketData instance
            date: Date for signal generation
            features: Optional pre-computed features DataFrame (SHORT_MOMENTUM)
            
        Returns:
            Series of signals indexed by symbol
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[ShortTermMomentum] No symbols available for date {date_dt}")
            return pd.Series(index=symbols, dtype=float)
        
        # Get features if not provided
        if features is None or features.empty:
            logger.warning(f"[ShortTermMomentum] No features available for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        # Find the appropriate date in features
        if date_dt not in features.index:
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(f"[ShortTermMomentum] No features available for date {date_dt}")
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
            ret_feature = f"mom_short_ret_21_z_{symbol}"
            breakout_feature = f"mom_short_breakout_21_z_{symbol}"
            slope_feature = f"mom_short_slope_fast_z_{symbol}"
            reversal_feature = f"mom_short_reversal_filter_z_{symbol}"
            
            # Extract feature values
            ret_val = feature_row.get(ret_feature, np.nan) if ret_feature in feature_row.index else np.nan
            breakout_val = feature_row.get(breakout_feature, np.nan) if breakout_feature in feature_row.index else np.nan
            slope_val = feature_row.get(slope_feature, np.nan) if slope_feature in feature_row.index else np.nan
            reversal_val = feature_row.get(reversal_feature, np.nan) if reversal_feature in feature_row.index else np.nan
            
            # Combine features with weights (excluding reversal_filter if weight is 0)
            signal_raw = (
                self.weights.get("ret_21", 0.0) * (ret_val if pd.notna(ret_val) else 0.0) +
                self.weights.get("breakout_21", 0.0) * (breakout_val if pd.notna(breakout_val) else 0.0) +
                self.weights.get("slope_fast", 0.0) * (slope_val if pd.notna(slope_val) else 0.0)
            )
            
            # Optionally apply reversal filter as a modifier (if weight > 0)
            if self.weights.get("reversal_filter", 0.0) > 0 and pd.notna(reversal_val):
                # Use reversal as a dampening factor when overbought/oversold
                signal_raw = signal_raw * (1.0 - abs(reversal_val) * self.weights["reversal_filter"])
            
            signals[symbol] = signal_raw
        
        signal_series = pd.Series(signals)
        
        # Re-standardize across assets (cross-sectional z-score)
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
            f"[ShortTermMomentum] Generated signals at {date_dt}: "
            f"mean={signal_series.mean():.3f}, std={signal_series.std():.3f}"
        )
        
        return signal_series
    
    def describe(self) -> dict:
        """Describe strategy parameters."""
        return {
            'strategy': 'ShortTermMomentum',
            'variant': self.variant,
            'weights': self.weights,
            'signal_cap': self.signal_cap,
            'rebalance': self.rebalance
        }
    
    def reset_state(self):
        """Reset internal state (useful for testing)."""
        logger.debug("[ShortTermMomentum] State reset")

