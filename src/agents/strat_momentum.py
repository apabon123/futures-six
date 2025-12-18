"""
TSMOM: Time-Series Momentum Strategy Agent

Uses three long-term momentum features:
- mom_long_ret_252_z: 252-day return momentum (vol-standardized, z-scored)
- mom_long_breakout_252_z: 252-day breakout strength
- mom_long_slope_slow_z: Slow trend slope (EMA-based)

Combines features with configurable weights and returns signals per symbol.
Strictly no look-ahead bias - only uses data ≤ asof.
"""

import logging
from typing import Optional, Union, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class TSMOM:
    """
    Time-Series Momentum (TSMOM) strategy agent.
    
    Uses three long-term momentum features with configurable weights:
    - Return momentum (252-day)
    - Breakout strength (252-day)
    - Slow trend slope (EMA-based)
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        weights: Optional[dict] = None,
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI",
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize TSMOM agent.
        
        Args:
            symbols: List of symbols to trade (default: None, uses market.universe)
            weights: Dictionary of feature weights:
                - ret_252: Weight for return momentum (default: 0.5)
                - breakout_252: Weight for breakout strength (default: 0.3)
                - slope_slow: Weight for slow trend slope (default: 0.2)
            signal_cap: Maximum absolute signal value (default: 3.0)
            rebalance: Rebalance frequency ("W-FRI" for weekly Friday, "M" for month-end)
            config_path: Path to strategy configuration file
        """
        self.symbols = symbols
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        
        # Load config if weights not provided
        if weights is None:
            config = self._load_config(config_path)
            tsmom_config = config.get('tsmom', {})
            params = tsmom_config.get('params', {})
            weights = params.get('weights', {})
        
        # Default weights
        default_weights = {
            "ret_252": 0.5,
            "breakout_252": 0.3,
            "slope_slow": 0.2
        }
        
        if not weights:
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
            logger.warning("[TSMOM] All weights are zero, using defaults")
            self.weights = default_weights
        
        # State tracking
        self._last_rebalance = None
        self._last_signals = None
        self._rebalance_dates = None
        
        logger.info(
            f"[TSMOM] Initialized with weights={self.weights}, "
            f"cap={signal_cap}, rebalance={rebalance}"
        )
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[TSMOM] Config not found: {config_path}, using defaults")
            return {}
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _compute_rebalance_dates(
        self,
        date_index: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """
        Compute rebalance dates based on schedule.
        
        Args:
            date_index: Full date range to consider
            
        Returns:
            DatetimeIndex of rebalance dates
        """
        if date_index.empty:
            return pd.DatetimeIndex([])
        
        start = date_index.min()
        end = date_index.max()
        
        if self.rebalance == "W-FRI":
            schedule = pd.date_range(start=start, end=end, freq='W-FRI')
        elif self.rebalance == "M":
            try:
                schedule = pd.date_range(start=start, end=end, freq='ME')
            except ValueError:
                schedule = pd.date_range(start=start, end=end, freq='M')
        elif self.rebalance == "D":
            schedule = pd.date_range(start=start, end=end, freq='D')
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance}")
        
        rebalance_dates = schedule.intersection(date_index)
        logger.debug(f"[TSMOM] Computed {len(rebalance_dates)} rebalance dates")
        return rebalance_dates
    
    def fit_in_sample(
        self,
        market,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ):
        """
        Fit strategy in-sample (optional, mostly no-op for TSMOM).
        
        TSMOM is a rule-based strategy with no parameters to fit.
        This method is provided for API consistency with other agents.
        Pre-computes rebalance dates if data is available.
        
        Args:
            market: MarketData instance
            start: Start date for fitting period
            end: End date for fitting period
        """
        logger.info(f"[TSMOM] fit_in_sample called (pre-computing rebalance dates)")
        
        # Pre-compute rebalance dates if we have data
        symbols = self.symbols if self.symbols is not None else market.universe
        if symbols:
            # Get any price data to determine date range
            prices = market.get_price_panel(symbols=symbols[:1], start=start, end=end, fields=("close",))
            if not prices.empty:
                self._rebalance_dates = self._compute_rebalance_dates(prices.index)
                logger.info(f"[TSMOM] Pre-computed {len(self._rebalance_dates)} rebalance dates")
    
    def signals(
        self,
        market,
        date: Union[str, datetime, pd.Timestamp],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate momentum signals for a given date.
        
        Signals are only recomputed on rebalance dates; otherwise the last
        computed signals are returned (held constant).
        
        Args:
            market: MarketData instance (read-only)
            date: Date for signal generation
            features: Optional pre-computed features DataFrame (LONG_MOMENTUM features)
            
        Returns:
            Series of signals indexed by symbol (roughly mean 0, unit variance)
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[TSMOM] No symbols available for date {date_dt}")
            return pd.Series(index=symbols, dtype=float)
        
        # Get features if not provided
        if features is None or features.empty:
            logger.warning(f"[TSMOM] No features available for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        # Find the appropriate date in features (forward-fill if needed)
        if date_dt not in features.index:
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(f"[TSMOM] No features available for date {date_dt} (no prior data)")
                return pd.Series(0.0, index=symbols)
            use_date = available_dates[-1]
            logger.debug(f"[TSMOM] Using features from {use_date} for date {date_dt}")
        else:
            use_date = date_dt
        
        # Get feature row
        feature_row = features.loc[use_date]
        
        # Compute signals per symbol
        signals = {}
        for symbol in symbols:
            # Get features for this symbol
            ret_feature = f"mom_long_ret_252_z_{symbol}"
            breakout_feature = f"mom_long_breakout_252_z_{symbol}"
            slope_feature = f"mom_long_slope_slow_z_{symbol}"
            
            # Extract feature values
            ret_val = feature_row.get(ret_feature, np.nan) if ret_feature in feature_row.index else np.nan
            breakout_val = feature_row.get(breakout_feature, np.nan) if breakout_feature in feature_row.index else np.nan
            slope_val = feature_row.get(slope_feature, np.nan) if slope_feature in feature_row.index else np.nan
            
            # Combine features with weights
            signal_raw = (
                self.weights["ret_252"] * (ret_val if pd.notna(ret_val) else 0.0) +
                self.weights["breakout_252"] * (breakout_val if pd.notna(breakout_val) else 0.0) +
                self.weights["slope_slow"] * (slope_val if pd.notna(slope_val) else 0.0)
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
            f"[TSMOM] Generated signals at {date_dt}: "
            f"mean={signal_series.mean():.3f}, std={signal_series.std():.3f}, "
            f"min={signal_series.min():.3f}, max={signal_series.max():.3f}"
        )
        
        return signal_series
    
    def describe(self) -> dict:
        """
        Describe strategy parameters and state.
        
        Returns:
            Dictionary with strategy configuration and last update info
        """
        return {
            'strategy': 'TSMOM',
            'weights': self.weights,
            'signal_cap': self.signal_cap,
            'rebalance': self.rebalance,
            'last_rebalance': str(self._last_rebalance) if self._last_rebalance else None,
            'n_rebalance_dates': len(self._rebalance_dates) if self._rebalance_dates is not None else None
        }
    
    def reset_state(self):
        """
        Reset internal state (useful for testing).
        
        Clears cached signals and rebalance tracking.
        """
        self._last_rebalance = None
        self._last_signals = None
        self._rebalance_dates = None
        logger.debug("[TSMOM] State reset")
