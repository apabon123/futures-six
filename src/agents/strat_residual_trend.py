"""
ResidualTrendStrategy: Phase-1 atomic sleeve for Trend Meta-Sleeve.

Uses residual trend feature (long-horizon trend minus short-term movement)
to generate signals. Applies clipped z-scores to preserve magnitude information.
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResidualTrendStrategy:
    """
    Residual Trend strategy agent.
    
    Uses z-scored residual trend feature (trend_resid_ret_252_21_z) to generate signals.
    Applies clipped z-scores (clipped to [-3, 3]) to preserve magnitude information.
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        long_lookback: int = 252,
        short_lookback: int = 21,
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI"
    ):
        """
        Initialize ResidualTrendStrategy agent.
        
        Args:
            symbols: List of symbols to trade (default: None, uses market.universe)
            long_lookback: Long-horizon lookback period (default: 252 days)
            short_lookback: Short-horizon lookback period (default: 21 days)
            signal_cap: Maximum absolute signal value (default: 3.0)
            rebalance: Rebalance frequency ("W-FRI" for weekly Friday, "M" for month-end)
        """
        self.symbols = symbols
        self.long_lookback = long_lookback
        self.short_lookback = short_lookback
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        
        # State tracking
        self._last_rebalance = None
        self._last_signals = None
        self._rebalance_dates = None
        
        logger.info(
            f"[ResidualTrend] Initialized with long_lookback={long_lookback}, "
            f"short_lookback={short_lookback}, cap={signal_cap}, rebalance={rebalance}"
        )
    
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
        logger.debug(f"[ResidualTrend] Computed {len(rebalance_dates)} rebalance dates")
        return rebalance_dates
    
    def fit_in_sample(
        self,
        market,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ):
        """
        Fit strategy in-sample (optional, mostly no-op for ResidualTrend).
        
        ResidualTrend is a rule-based strategy with no parameters to fit.
        This method is provided for API consistency with other agents.
        Pre-computes rebalance dates if data is available.
        
        Args:
            market: MarketData instance
            start: Start date for fitting period
            end: End date for fitting period
        """
        logger.info(f"[ResidualTrend] fit_in_sample called (pre-computing rebalance dates)")
        
        # Pre-compute rebalance dates if we have data
        symbols = self.symbols if self.symbols is not None else market.universe
        if symbols:
            # Get any price data to determine date range
            prices = market.get_price_panel(symbols=symbols[:1], start=start, end=end, fields=("close",))
            if not prices.empty:
                self._rebalance_dates = self._compute_rebalance_dates(prices.index)
                logger.info(f"[ResidualTrend] Pre-computed {len(self._rebalance_dates)} rebalance dates")
    
    def signals(
        self,
        market,
        date: Union[str, datetime, pd.Timestamp],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate residual trend signals for a given date.
        
        Signals are only recomputed on rebalance dates; otherwise the last
        computed signals are returned (held constant).
        
        Args:
            market: MarketData instance (read-only)
            date: Date for signal generation
            features: Optional pre-computed features DataFrame (RESIDUAL_TREND features)
            
        Returns:
            Series of signals indexed by symbol (clipped z-scores, range [-signal_cap, signal_cap])
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[ResidualTrend] No symbols available for date {date_dt}")
            return pd.Series(dtype=float)
        
        # Check if we should rebalance
        if self._rebalance_dates is not None:
            # Find the most recent rebalance date <= current date
            valid_rebalance_dates = self._rebalance_dates[self._rebalance_dates <= date_dt]
            if len(valid_rebalance_dates) == 0:
                # No rebalance date yet, return zero signals
                return pd.Series(0.0, index=symbols)
            
            last_rebalance = valid_rebalance_dates.max()
            
            # If we already computed signals for this rebalance date, return cached
            if self._last_rebalance == last_rebalance and self._last_signals is not None:
                return self._last_signals
        else:
            # No pre-computed rebalance dates, recompute every time
            last_rebalance = date_dt
        
        # Get features if not provided
        if features is None:
            logger.warning(f"[ResidualTrend] No features provided for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        if features.empty:
            logger.warning(f"[ResidualTrend] Empty features DataFrame for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        # Find the appropriate date in features (forward-fill if needed)
        # Use the most recent available date <= date_dt
        available_dates = features.index[features.index <= date_dt]
        if len(available_dates) == 0:
            logger.warning(f"[ResidualTrend] No features available for date {date_dt} or earlier")
            return pd.Series(0.0, index=symbols)
        
        feature_date = available_dates.max()
        
        # Extract z-scored residual trend feature for each symbol
        # Feature name format: trend_resid_ret_252_21_z_{symbol}
        feature_name_template = f"trend_resid_ret_{self.long_lookback}_{self.short_lookback}_z"
        
        signals_dict = {}
        for symbol in symbols:
            feature_col = f"{feature_name_template}_{symbol}"
            
            if feature_col not in features.columns:
                logger.debug(f"[ResidualTrend] Feature {feature_col} not found for {symbol}")
                signals_dict[symbol] = 0.0
                continue
            
            # Get feature value for this date
            feature_value = features.loc[feature_date, feature_col]
            
            if pd.isna(feature_value):
                signals_dict[symbol] = 0.0
            else:
                # Feature is already z-scored and clipped in ResidualTrendFeatures
                # Apply additional clipping to signal_cap (usually same as clip, but allow override)
                signal = np.clip(feature_value, -self.signal_cap, self.signal_cap)
                signals_dict[symbol] = signal
        
        signals = pd.Series(signals_dict, index=symbols)
        
        # Cache signals for this rebalance date
        self._last_rebalance = last_rebalance
        self._last_signals = signals
        
        logger.debug(
            f"[ResidualTrend] Generated signals for {date_dt} (feature_date={feature_date}): "
            f"mean={signals.mean():.3f}, std={signals.std():.3f}, "
            f"non-zero={(signals.abs() > 1e-6).sum()}/{len(signals)}"
        )
        
        return signals

