"""
TSMOM: Time-Series Momentum Strategy Agent

Produces per-asset momentum signals using 12-1 month (default) or 3/6/12 month blend.
Strictly no look-ahead bias - only uses data ≤ asof.

Signal construction:
1. Calculate cumulative returns over lookback period(s), excluding recent skip_recent days
2. Standardize signals (z-score or volatility-scaled)
3. Cap to ±signal_cap
4. Rebalance only on scheduled dates
"""

import logging
from typing import Optional, List, Union
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class TSMOM:
    """
    Time-Series Momentum (TSMOM) strategy agent.
    
    Generates momentum signals based on past returns, with configurable
    lookback periods, standardization methods, and rebalancing schedules.
    """
    
    def __init__(
        self,
        lookbacks: List[int] = None,
        skip_recent: int = 21,
        standardize: str = "vol",
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI",
        return_method: str = "log",
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize TSMOM agent.
        
        Args:
            lookbacks: List of lookback periods in days (e.g., [252] for 12-1 month, 
                      or [63, 126, 252] for 3/6/12 month blend)
            skip_recent: Days to skip at the end (the "-1 month" gap)
            standardize: Method to standardize signals ("zscore" or "vol")
            signal_cap: Maximum absolute signal value (z-score cap)
            rebalance: Rebalance frequency ("W-FRI" for weekly Friday, "M" for month-end)
            return_method: Return calculation method ("log" or "simple")
            config_path: Path to strategy configuration file
        """
        # Load config if not all params provided
        if lookbacks is None:
            config = self._load_config(config_path)
            tsmom_config = config.get('tsmom', {})
            lookbacks = tsmom_config.get('lookbacks', [252])
            skip_recent = tsmom_config.get('skip_recent', 21)
            standardize = tsmom_config.get('standardize', 'vol')
            signal_cap = tsmom_config.get('signal_cap', 3.0)
            rebalance = tsmom_config.get('rebalance', 'W-FRI')
        
        self.lookbacks = lookbacks
        self.skip_recent = skip_recent
        self.standardize = standardize
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        self.return_method = return_method
        
        # State tracking
        self._last_rebalance = None
        self._last_signals = None
        self._rebalance_dates = None
        
        logger.info(
            f"[TSMOM] Initialized with lookbacks={lookbacks}, skip_recent={skip_recent}, "
            f"standardize={standardize}, cap={signal_cap}, rebalance={rebalance}"
        )
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[TSMOM] Config not found: {config_path}, using defaults")
            return {}
        
        with open(path, 'r') as f:
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
        
        # Create a date range covering the full period
        start = date_index.min()
        end = date_index.max()
        
        # Generate schedule
        if self.rebalance == "W-FRI":
            schedule = pd.date_range(start=start, end=end, freq='W-FRI')
        elif self.rebalance == "M":
            # Month-end (use 'ME' for pandas >= 2.2)
            try:
                schedule = pd.date_range(start=start, end=end, freq='ME')
            except ValueError:
                # Fallback for older pandas
                schedule = pd.date_range(start=start, end=end, freq='M')
        elif self.rebalance == "D":
            # Daily (for testing)
            schedule = pd.date_range(start=start, end=end, freq='D')
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance}")
        
        # Only keep dates that exist in the actual trading calendar
        rebalance_dates = schedule.intersection(date_index)
        
        logger.debug(f"[TSMOM] Computed {len(rebalance_dates)} rebalance dates")
        return rebalance_dates
    
    def _calculate_cumulative_returns(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        lookback: int
    ) -> pd.Series:
        """
        Calculate cumulative returns over lookback period, excluding skip_recent days.
        
        Args:
            returns: Wide DataFrame of returns (date x symbols)
            date: Current evaluation date
            lookback: Lookback period in days
            
        Returns:
            Series of cumulative returns per symbol
        """
        # Get data up to current date
        returns_upto = returns.loc[:date]
        
        if len(returns_upto) < self.skip_recent:
            # Not enough data
            return pd.Series(index=returns.columns, dtype=float)
        
        # Exclude the most recent skip_recent days
        returns_window = returns_upto.iloc[:-self.skip_recent] if self.skip_recent > 0 else returns_upto
        
        # Take last lookback days of remaining data
        if len(returns_window) < lookback:
            # Not enough history
            return pd.Series(index=returns.columns, dtype=float)
        
        returns_lookback = returns_window.iloc[-lookback:]
        
        # Calculate cumulative return
        if self.return_method == "log":
            # For log returns, sum them
            cum_ret = returns_lookback.sum()
        else:
            # For simple returns, compound them
            cum_ret = (1 + returns_lookback).prod() - 1
        
        return cum_ret
    
    def _calculate_raw_signals(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculate raw momentum signals (before standardization).
        
        Args:
            returns: Wide DataFrame of returns
            date: Current evaluation date
            
        Returns:
            Series of raw signals per symbol
        """
        if len(self.lookbacks) == 1:
            # Single lookback period
            signals = self._calculate_cumulative_returns(returns, date, self.lookbacks[0])
        else:
            # Blend multiple lookback periods (equal weight)
            signal_list = []
            for lookback in self.lookbacks:
                sig = self._calculate_cumulative_returns(returns, date, lookback)
                signal_list.append(sig)
            
            # Average across lookbacks
            signals = pd.concat(signal_list, axis=1).mean(axis=1)
        
        return signals
    
    def _standardize_signals(
        self,
        raw_signals: pd.Series,
        returns: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Standardize signals using z-score or volatility scaling.
        
        Args:
            raw_signals: Raw momentum signals
            returns: Wide DataFrame of returns (for vol calculation)
            date: Current evaluation date
            
        Returns:
            Standardized signals
        """
        if self.standardize == "zscore":
            # Cross-sectional z-score
            valid_signals = raw_signals.dropna()
            if len(valid_signals) == 0:
                return raw_signals
            
            mean = valid_signals.mean()
            std = valid_signals.std()
            
            if std > 0:
                standardized = (raw_signals - mean) / std
            else:
                standardized = raw_signals * 0  # All zeros if no variation
        
        elif self.standardize == "vol":
            # Divide by trailing volatility
            # Use maximum lookback period for vol estimation
            vol_lookback = max(self.lookbacks) if self.lookbacks else 252
            
            # Get returns up to date
            returns_upto = returns.loc[:date]
            
            if len(returns_upto) < vol_lookback:
                # Not enough data for vol, use simpler window
                vol_lookback = min(63, len(returns_upto))
            
            # Calculate trailing vol (annualized)
            if vol_lookback > 0 and len(returns_upto) >= vol_lookback:
                trailing_vol = returns_upto.iloc[-vol_lookback:].std() * np.sqrt(252)
            else:
                # Fallback: use all available data
                trailing_vol = returns_upto.std() * np.sqrt(252)
            
            # Avoid division by zero - replace zero vol with a small number
            # This handles the edge case of constant prices/returns
            min_vol = 0.01  # 1% minimum annualized vol
            trailing_vol = trailing_vol.clip(lower=min_vol)
            
            # Standardize: signal = raw_signal / vol
            # This gives a rough "return per unit risk" interpretation
            standardized = raw_signals / trailing_vol
        
        else:
            raise ValueError(f"Unknown standardization method: {self.standardize}")
        
        return standardized
    
    def _cap_signals(self, signals: pd.Series) -> pd.Series:
        """
        Cap signals to ±signal_cap.
        
        Args:
            signals: Standardized signals
            
        Returns:
            Capped signals
        """
        return signals.clip(lower=-self.signal_cap, upper=self.signal_cap)
    
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
        
        Args:
            market: MarketData instance
            start: Start date for fitting period
            end: End date for fitting period
        """
        logger.info(f"[TSMOM] fit_in_sample called (no-op for TSMOM)")
        
        # Pre-compute rebalance dates if we have data
        symbols = market.universe
        returns = market.get_returns(symbols=symbols, start=start, end=end, method=self.return_method)
        
        if not returns.empty:
            self._rebalance_dates = self._compute_rebalance_dates(returns.index)
            logger.info(f"[TSMOM] Pre-computed {len(self._rebalance_dates)} rebalance dates")
    
    def signals(
        self,
        market,
        date: Union[str, datetime, pd.Timestamp]
    ) -> pd.Series:
        """
        Generate momentum signals for a given date.
        
        Signals are only recomputed on rebalance dates; otherwise the last
        computed signals are returned (held constant).
        
        Args:
            market: MarketData instance (read-only)
            date: Date for signal generation
            
        Returns:
            Series of signals indexed by symbol (roughly mean 0, unit variance)
        """
        date = pd.to_datetime(date)
        
        # Get returns data up to this date
        symbols = market.universe
        returns = market.get_returns(symbols=symbols, end=date, method=self.return_method)
        
        if returns.empty:
            logger.warning(f"[TSMOM] No returns data available for date {date}")
            return pd.Series(index=symbols, dtype=float)
        
        # Ensure date is in the returns index
        if date not in returns.index:
            # Find the last available date <= requested date
            available_dates = returns.index[returns.index <= date]
            if len(available_dates) == 0:
                logger.warning(f"[TSMOM] No data available on or before {date}")
                return pd.Series(index=symbols, dtype=float)
            date = available_dates[-1]
            logger.debug(f"[TSMOM] Adjusted date to last available: {date}")
        
        # Compute rebalance schedule if not already done
        if self._rebalance_dates is None:
            self._rebalance_dates = self._compute_rebalance_dates(returns.index)
        
        # Check if we need to rebalance
        is_rebalance = date in self._rebalance_dates
        
        if not is_rebalance and self._last_signals is not None:
            # Hold previous signals
            logger.debug(f"[TSMOM] Holding signals from {self._last_rebalance}")
            return self._last_signals
        
        # Rebalance: compute new signals
        logger.debug(f"[TSMOM] Computing signals for {date}")
        
        # Step 1: Calculate raw signals
        raw_signals = self._calculate_raw_signals(returns, date)
        
        # Step 2: Standardize signals
        standardized = self._standardize_signals(raw_signals, returns, date)
        
        # Step 3: Cap signals
        capped = self._cap_signals(standardized)
        
        # Update state
        self._last_signals = capped
        self._last_rebalance = date
        
        logger.debug(
            f"[TSMOM] Generated signals at {date}: "
            f"mean={capped.mean():.3f}, std={capped.std():.3f}, "
            f"min={capped.min():.3f}, max={capped.max():.3f}"
        )
        
        return capped
    
    def describe(self) -> dict:
        """
        Describe strategy parameters and state.
        
        Returns:
            Dictionary with strategy configuration and last update info
        """
        return {
            'strategy': 'TSMOM',
            'lookbacks': self.lookbacks,
            'skip_recent': self.skip_recent,
            'standardize': self.standardize,
            'signal_cap': self.signal_cap,
            'rebalance': self.rebalance,
            'return_method': self.return_method,
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

