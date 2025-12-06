"""
Medium-Term Momentum Strategy: Multi-feature momentum strategy.

Uses four medium-term momentum features:
- mom_med_ret_84_z: 84-day return momentum
- mom_med_breakout_126_z: 126-day breakout strength
- mom_med_slope_med_z: Medium trend slope
- mom_med_persistence_z: Trend persistence score

Combines features with configurable weights and returns signals per symbol.
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MediumTermMomentumStrategy:
    """
    Medium-term momentum strategy using multiple features.
    
    Combines four medium-term momentum features with configurable weights:
    - Return momentum (84-day)
    - Breakout strength (126-day)
    - Medium trend slope (EMA-based)
    - Trend persistence
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        weights: Optional[dict] = None,
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI"
    ):
        """
        Initialize Medium-Term Momentum Strategy.
        
        Args:
            symbols: List of symbols to trade (default: None, uses market.universe)
            weights: Dictionary of feature weights:
                - ret_84: Weight for return momentum (default: 0.4)
                - breakout_126: Weight for breakout strength (default: 0.3)
                - slope_med: Weight for medium trend slope (default: 0.2)
                - persistence: Weight for trend persistence (default: 0.1)
            signal_cap: Maximum absolute signal value (default: 3.0)
            rebalance: Rebalance frequency (default: "W-FRI")
        """
        self.symbols = symbols
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        
        # Default weights
        default_weights = {
            "ret_84": 0.4,
            "breakout_126": 0.3,
            "slope_med": 0.2,
            "persistence": 0.1
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
            logger.warning("[MediumTermMomentum] All weights are zero, using defaults")
            self.weights = default_weights
        
        logger.info(
            f"[MediumTermMomentum] Initialized with weights={self.weights}, "
            f"cap={signal_cap}, rebalance={rebalance}"
        )
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate medium-term momentum signals for a given date.
        
        Args:
            market: MarketData instance
            date: Date for signal generation
            features: Optional pre-computed features DataFrame (MEDIUM_MOMENTUM)
            
        Returns:
            Series of signals indexed by symbol
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[MediumTermMomentum] No symbols available for date {date_dt}")
            return pd.Series(index=symbols, dtype=float)
        
        # Get features if not provided
        if features is None or features.empty:
            logger.warning(f"[MediumTermMomentum] No features available for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        # Find the appropriate date in features
        if date_dt not in features.index:
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(f"[MediumTermMomentum] No features available for date {date_dt}")
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
            ret_feature = f"mom_med_ret_84_z_{symbol}"
            breakout_feature = f"mom_med_breakout_126_z_{symbol}"
            slope_feature = f"mom_med_slope_med_z_{symbol}"
            persist_feature = f"mom_med_persistence_z_{symbol}"
            
            # Extract feature values
            ret_val = feature_row.get(ret_feature, np.nan) if ret_feature in feature_row.index else np.nan
            breakout_val = feature_row.get(breakout_feature, np.nan) if breakout_feature in feature_row.index else np.nan
            slope_val = feature_row.get(slope_feature, np.nan) if slope_feature in feature_row.index else np.nan
            persist_val = feature_row.get(persist_feature, np.nan) if persist_feature in feature_row.index else np.nan
            
            # Combine features with weights
            signal_raw = (
                self.weights["ret_84"] * (ret_val if pd.notna(ret_val) else 0.0) +
                self.weights["breakout_126"] * (breakout_val if pd.notna(breakout_val) else 0.0) +
                self.weights["slope_med"] * (slope_val if pd.notna(slope_val) else 0.0) +
                self.weights["persistence"] * (persist_val if pd.notna(persist_val) else 0.0)
            )
            
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
            f"[MediumTermMomentum] Generated signals at {date_dt}: "
            f"mean={signal_series.mean():.3f}, std={signal_series.std():.3f}"
        )
        
        return signal_series
    
    def describe(self) -> dict:
        """Describe strategy parameters."""
        return {
            'strategy': 'MediumTermMomentum',
            'weights': self.weights,
            'signal_cap': self.signal_cap,
            'rebalance': self.rebalance
        }
    
    def reset_state(self):
        """Reset internal state (useful for testing)."""
        logger.debug("[MediumTermMomentum] State reset")


class CanonicalMediumTermMomentumStrategy:
    """
    Canonical medium-term momentum strategy using equal-weight 3-feature composite.
    
    This is the academically-grounded medium-term sleeve (84d canonical horizon)
    with equal-weight composite (1/3, 1/3, 1/3):
    
    Features:
    - 84-day return momentum (skip 10d, vol-scaled, z-scored)
    - 84-day breakout strength (z-scored)
    - EMA21 vs EMA84 slope (vol-scaled, z-scored)
    
    Canonical Parameters:
    - Horizon: 84 trading days (~4 months, canonical medium-term)
    - Skip: 10 trading days (~2 weeks, avoid short-term noise)
    - Vol window: 21 days (standard short-term vol for scaling)
    - Standardization: 252d rolling z-score, clipped at ±3
    - Composite: equal-weight (1/3, 1/3, 1/3)
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI"
    ):
        """
        Initialize Canonical Medium-Term Momentum Strategy.
        
        Args:
            symbols: List of symbols to trade (default: None, uses market.universe)
            signal_cap: Maximum absolute signal value (default: 3.0)
            rebalance: Rebalance frequency (default: "W-FRI")
        """
        self.symbols = symbols
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        
        logger.info(
            f"[CanonicalMediumTermMomentum] Initialized with cap={signal_cap}, rebalance={rebalance}"
        )
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate canonical medium-term momentum signals for a given date.
        
        Args:
            market: MarketData instance
            date: Date for signal generation
            features: Optional pre-computed features DataFrame (CANONICAL_MEDIUM_MOMENTUM)
            
        Returns:
            Series of signals indexed by symbol
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[CanonicalMediumTermMomentum] No symbols available for date {date_dt}")
            return pd.Series(index=symbols, dtype=float)
        
        # Get features if not provided
        if features is None or features.empty:
            logger.warning(f"[CanonicalMediumTermMomentum] No features available for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        # Find the appropriate date in features
        if date_dt not in features.index:
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(f"[CanonicalMediumTermMomentum] No features available for date {date_dt}")
                return pd.Series(0.0, index=symbols)
            use_date = available_dates[-1]
        else:
            use_date = date_dt
        
        # Get feature row
        feature_row = features.loc[use_date]
        
        # Compute signals per symbol
        # Use the pre-computed composite feature (equal-weight combination)
        signals = {}
        for symbol in symbols:
            # Get the canonical composite feature
            composite_feature = f"mom_medcanon_composite_z_{symbol}"
            
            # Extract composite value (already equal-weighted 1/3, 1/3, 1/3)
            if composite_feature in feature_row.index:
                composite_val = feature_row.get(composite_feature, np.nan)
                signal_raw = composite_val if pd.notna(composite_val) else 0.0
            else:
                # Fallback: manually compute from individual features if composite not available
                ret_feature = f"mom_medcanon_ret_84_z_{symbol}"
                breakout_feature = f"mom_medcanon_breakout_84_z_{symbol}"
                slope_feature = f"mom_medcanon_slope_21_84_z_{symbol}"
                
                ret_val = feature_row.get(ret_feature, np.nan) if ret_feature in feature_row.index else np.nan
                breakout_val = feature_row.get(breakout_feature, np.nan) if breakout_feature in feature_row.index else np.nan
                slope_val = feature_row.get(slope_feature, np.nan) if slope_feature in feature_row.index else np.nan
                
                # Equal-weight composite (1/3, 1/3, 1/3)
                signal_raw = (
                    (ret_val if pd.notna(ret_val) else 0.0) +
                    (breakout_val if pd.notna(breakout_val) else 0.0) +
                    (slope_val if pd.notna(slope_val) else 0.0)
                ) / 3.0
            
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
            f"[CanonicalMediumTermMomentum] Generated signals at {date_dt}: "
            f"mean={signal_series.mean():.3f}, std={signal_series.std():.3f}"
        )
        
        return signal_series
    
    def describe(self) -> dict:
        """Describe strategy parameters."""
        return {
            'strategy': 'CanonicalMediumTermMomentum',
            'composite': 'equal_weight_1_3_1_3_1_3',
            'features': ['ret_84_skip10', 'breakout_84_skip10', 'slope_ema21_84'],
            'signal_cap': self.signal_cap,
            'rebalance': self.rebalance
        }
    
    def reset_state(self):
        """Reset internal state (useful for testing)."""
        logger.debug("[CanonicalMediumTermMomentum] State reset")
