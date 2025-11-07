"""
MacroRegimeFilter: Macro Regime-Based Signal Scaling Agent

Applies a continuous scaler k ∈ [k_min, k_max] to strategy signals based on
internal market regime indicators:
- Realized volatility (21-day rolling vol of ES+NQ equal-weighted portfolio)
- Market breadth (fraction of ES/NQ above 200-day SMA)

The scaler adjusts signals to reduce exposure in high-volatility, low-breadth
environments and increase exposure in favorable conditions.

No data writes. No look-ahead. Changes only on rebalance dates.
"""

import logging
from typing import Union, Optional
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MacroRegimeFilter:
    """
    Macro regime filter that scales strategy signals based on volatility and breadth.
    
    Computes a continuous scaler k ∈ [k_min, k_max] that is applied to strategy
    signals. The scaler is based on:
    1. Realized volatility of ES+NQ equal-weighted portfolio (21-day)
    2. Market breadth: fraction of {ES, NQ} above 200-day SMA
    
    The scaler changes only on rebalance dates and is smoothed with EMA.
    """
    
    def __init__(
        self,
        rebalance: str = "W-FRI",
        vol_thresholds: Optional[dict] = None,
        k_bounds: Optional[dict] = None,
        smoothing: float = 0.2,
        vol_lookback: int = 21,
        breadth_lookback: int = 200,
        proxy_symbols: tuple = ("ES", "NQ"),
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize MacroRegimeFilter agent.
        
        Args:
            rebalance: Rebalance frequency (pandas offset alias, e.g., "W-FRI")
            vol_thresholds: Dict with 'low' and 'high' annualized vol thresholds
                           (default: {'low': 0.15, 'high': 0.30})
            k_bounds: Dict with 'min' and 'max' scaler bounds
                     (default: {'min': 0.4, 'max': 1.0})
            smoothing: EMA smoothing parameter α ∈ [0, 1] (default: 0.2)
            vol_lookback: Rolling window for realized vol (default: 21 days)
            breadth_lookback: SMA lookback for breadth calculation (default: 200 days)
            proxy_symbols: Symbols for regime detection (default: ("ES", "NQ"))
            config_path: Path to configuration YAML file
        """
        # Track which parameters were explicitly provided
        defaults = {
            'rebalance': "W-FRI",
            'vol_thresholds': None,
            'k_bounds': None,
            'smoothing': 0.2,
            'vol_lookback': 21,
            'breadth_lookback': 200,
            'proxy_symbols': ("ES", "NQ")
        }
        
        # Try to load from config if exists
        config = self._load_config(config_path)
        
        # Load from config only for parameters that match defaults
        if config and 'macro_regime' in config:
            regime_config = config['macro_regime']
            self.rebalance = regime_config.get('rebalance', rebalance) if rebalance == defaults['rebalance'] else rebalance
            self.smoothing = regime_config.get('smoothing', smoothing) if smoothing == defaults['smoothing'] else smoothing
            self.vol_lookback = regime_config.get('vol_lookback', vol_lookback) if vol_lookback == defaults['vol_lookback'] else vol_lookback
            self.breadth_lookback = regime_config.get('breadth_lookback', breadth_lookback) if breadth_lookback == defaults['breadth_lookback'] else breadth_lookback
            
            # Handle vol_thresholds
            if vol_thresholds is None:
                self.vol_thresholds = regime_config.get('vol_thresholds', {'low': 0.15, 'high': 0.30})
            else:
                self.vol_thresholds = vol_thresholds
            
            # Handle k_bounds
            if k_bounds is None:
                self.k_bounds = regime_config.get('k_bounds', {'min': 0.4, 'max': 1.0})
            else:
                self.k_bounds = k_bounds
            
            # Handle proxy_symbols
            if proxy_symbols == defaults['proxy_symbols']:
                proxy_config = regime_config.get('proxy_symbols', proxy_symbols)
                self.proxy_symbols = tuple(proxy_config) if isinstance(proxy_config, list) else proxy_symbols
            else:
                self.proxy_symbols = proxy_symbols
        else:
            self.rebalance = rebalance
            self.vol_thresholds = vol_thresholds or {'low': 0.15, 'high': 0.30}
            self.k_bounds = k_bounds or {'min': 0.4, 'max': 1.0}
            self.smoothing = smoothing
            self.vol_lookback = vol_lookback
            self.breadth_lookback = breadth_lookback
            self.proxy_symbols = proxy_symbols
        
        # Validate parameters
        if not (0 <= self.smoothing <= 1):
            raise ValueError(f"smoothing must be in [0, 1], got {self.smoothing}")
        
        if self.vol_thresholds['low'] >= self.vol_thresholds['high']:
            raise ValueError(
                f"vol_thresholds['low'] must be < vol_thresholds['high'], "
                f"got {self.vol_thresholds}"
            )
        
        if self.k_bounds['min'] >= self.k_bounds['max']:
            raise ValueError(
                f"k_bounds['min'] must be < k_bounds['max'], got {self.k_bounds}"
            )
        
        if self.vol_lookback < 2:
            raise ValueError(f"vol_lookback must be >= 2, got {self.vol_lookback}")
        
        if self.breadth_lookback < 2:
            raise ValueError(f"breadth_lookback must be >= 2, got {self.breadth_lookback}")
        
        # State for EMA smoothing (persists across rebalances)
        self._last_scaler = None
        self._last_rebalance = None
        
        # Cache for rebalance dates
        self._rebalance_dates = None
        
        logger.info(
            f"[MacroRegimeFilter] Initialized: rebalance={self.rebalance}, "
            f"vol_thresholds={self.vol_thresholds}, k_bounds={self.k_bounds}, "
            f"smoothing={self.smoothing}, vol_lookback={self.vol_lookback}, "
            f"breadth_lookback={self.breadth_lookback}, proxy_symbols={self.proxy_symbols}"
        )
    
    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.debug(f"[MacroRegimeFilter] Config file not found: {config_path}, using defaults")
            return None
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"[MacroRegimeFilter] Failed to load config: {e}, using defaults")
            return None
    
    def _is_rebalance_date(self, date: pd.Timestamp, market) -> bool:
        """
        Check if the given date is a rebalance date.
        
        Args:
            date: Date to check
            market: MarketData instance
            
        Returns:
            True if date is a rebalance date
        """
        # Get all trading days from market
        trading_days = market.trading_days(symbols=self.proxy_symbols)
        
        if len(trading_days) == 0:
            return False
        
        # Generate rebalance schedule
        if self._rebalance_dates is None or date not in trading_days.values:
            # Create date range covering full trading calendar
            start = trading_days.min()
            end = trading_days.max()
            
            # Generate rebalance dates
            rebalance_schedule = pd.date_range(start=start, end=end, freq=self.rebalance)
            
            # Filter to actual trading days (find nearest trading day for each rebalance date)
            actual_rebalance_dates = []
            trading_days_list = list(trading_days)  # Convert to list for easy indexing
            for rb_date in rebalance_schedule:
                # Find nearest trading day on or after rebalance date
                valid_dates = [d for d in trading_days_list if d >= rb_date]
                if len(valid_dates) > 0:
                    actual_rebalance_dates.append(valid_dates[0])
            
            self._rebalance_dates = set(actual_rebalance_dates)
        
        return date in self._rebalance_dates
    
    def _compute_realized_vol(
        self,
        market,
        date: pd.Timestamp
    ) -> float:
        """
        Compute realized volatility of equal-weighted ES+NQ portfolio.
        
        Uses rolling window of daily returns (vol_lookback days) and annualizes.
        
        Args:
            market: MarketData instance
            date: Current date
            
        Returns:
            Annualized realized volatility
        """
        # Get returns for proxy symbols
        returns = market.get_returns(
            symbols=self.proxy_symbols,
            end=date,
            method="log"
        )
        
        if returns.empty:
            logger.warning(f"[MacroRegimeFilter] No returns data for {date}")
            return self.vol_thresholds['low']  # Default to low vol
        
        # Filter to date and lookback window
        returns_at_date = returns[returns.index <= date]
        
        if len(returns_at_date) < self.vol_lookback:
            logger.warning(
                f"[MacroRegimeFilter] Insufficient data for vol calculation: "
                f"got {len(returns_at_date)} days, need {self.vol_lookback}"
            )
            return self.vol_thresholds['low']
        
        # Get last vol_lookback days
        returns_window = returns_at_date.iloc[-self.vol_lookback:]
        
        # Compute equal-weighted portfolio returns
        portfolio_returns = returns_window[list(self.proxy_symbols)].mean(axis=1)
        
        # Compute volatility and annualize
        realized_vol = portfolio_returns.std() * np.sqrt(252)
        
        return realized_vol
    
    def _compute_breadth(
        self,
        market,
        date: pd.Timestamp
    ) -> float:
        """
        Compute market breadth: fraction of proxy symbols above their 200-day SMA.
        
        Args:
            market: MarketData instance
            date: Current date
            
        Returns:
            Breadth value in [0.0, 0.5, 1.0] for 2 symbols
        """
        # Get prices for proxy symbols
        prices = market.get_price_panel(
            symbols=self.proxy_symbols,
            fields=("close",),
            end=date,
            tidy=False
        )
        
        if prices.empty:
            logger.warning(f"[MacroRegimeFilter] No price data for breadth calculation at {date}")
            return 0.5  # Neutral
        
        # Filter to date and lookback window
        prices_at_date = prices[prices.index <= date]
        
        if len(prices_at_date) < self.breadth_lookback:
            logger.warning(
                f"[MacroRegimeFilter] Insufficient data for breadth calculation: "
                f"got {len(prices_at_date)} days, need {self.breadth_lookback}"
            )
            return 0.5  # Neutral
        
        # Compute SMA for each symbol
        sma = prices_at_date.rolling(window=self.breadth_lookback, min_periods=self.breadth_lookback).mean()
        
        # Get latest prices and SMA
        latest_prices = prices_at_date.iloc[-1]
        latest_sma = sma.iloc[-1]
        
        # Count symbols above SMA
        above_sma = (latest_prices > latest_sma).sum()
        total_symbols = len(self.proxy_symbols)
        
        breadth = above_sma / total_symbols
        
        return breadth
    
    def _compute_base_scaler(
        self,
        realized_vol: float,
        breadth: float
    ) -> float:
        """
        Compute base scaler from realized vol and breadth.
        
        Logic:
        1. Map realized vol linearly from [low, high] to [k_max, k_min]
           (higher vol → lower scaler)
        2. Add breadth adjustment: +0.1 if breadth == 1.0, -0.1 if breadth == 0.0
        3. Clamp to [k_min, k_max]
        
        Args:
            realized_vol: Annualized realized volatility
            breadth: Market breadth [0, 1]
            
        Returns:
            Base scaler k
        """
        # 1. Map vol to scaler (linear interpolation)
        vol_low = self.vol_thresholds['low']
        vol_high = self.vol_thresholds['high']
        k_min = self.k_bounds['min']
        k_max = self.k_bounds['max']
        
        # Clamp vol to thresholds
        vol_clamped = np.clip(realized_vol, vol_low, vol_high)
        
        # Linear mapping: low vol → k_max, high vol → k_min
        vol_fraction = (vol_clamped - vol_low) / (vol_high - vol_low)
        base_k = k_max - vol_fraction * (k_max - k_min)
        
        # 2. Breadth adjustment
        if breadth == 1.0:
            breadth_adj = 0.1
        elif breadth == 0.0:
            breadth_adj = -0.1
        else:
            # Linear interpolation for fractional breadth
            breadth_adj = 0.1 * (2 * breadth - 1)  # Maps [0, 1] to [-0.1, 0.1]
        
        k = base_k + breadth_adj
        
        # 3. Clamp to bounds
        k = np.clip(k, k_min, k_max)
        
        return k
    
    def _smooth_scaler(self, new_scaler: float) -> float:
        """
        Apply EMA smoothing to scaler.
        
        EMA: k_smooth = α * k_new + (1 - α) * k_smooth_prev
        
        Args:
            new_scaler: Newly computed scaler
            
        Returns:
            Smoothed scaler
        """
        if self._last_scaler is None:
            # First time: no smoothing
            return new_scaler
        
        # Apply EMA
        smoothed = self.smoothing * new_scaler + (1 - self.smoothing) * self._last_scaler
        
        return smoothed
    
    def scaler(
        self,
        market,
        date: Union[str, datetime]
    ) -> float:
        """
        Compute regime scaler for the given date.
        
        The scaler changes only on rebalance dates. Between rebalances,
        returns the last computed scaler.
        
        Args:
            market: MarketData instance
            date: Current date
            
        Returns:
            Scaler k ∈ [k_min, k_max]
        """
        date = pd.to_datetime(date)
        
        # Check if this is a rebalance date
        is_rebalance = self._is_rebalance_date(date, market)
        
        # If not rebalance and we have a cached scaler, return it
        if not is_rebalance and self._last_scaler is not None:
            logger.debug(
                f"[MacroRegimeFilter] Using cached scaler at {date}: k={self._last_scaler:.3f}"
            )
            return self._last_scaler
        
        # Compute regime indicators
        realized_vol = self._compute_realized_vol(market, date)
        breadth = self._compute_breadth(market, date)
        
        # Compute base scaler
        base_k = self._compute_base_scaler(realized_vol, breadth)
        
        # Apply smoothing
        smoothed_k = self._smooth_scaler(base_k)
        
        # Update state
        self._last_scaler = smoothed_k
        self._last_rebalance = date
        
        logger.info(
            f"[MacroRegimeFilter] Rebalance at {date}: "
            f"vol={realized_vol:.3f}, breadth={breadth:.2f}, "
            f"base_k={base_k:.3f}, smoothed_k={smoothed_k:.3f}"
        )
        
        return smoothed_k
    
    def apply(
        self,
        signals: pd.Series,
        market,
        date: Union[str, datetime]
    ) -> pd.Series:
        """
        Apply regime scaler to strategy signals.
        
        Args:
            signals: Raw strategy signals (pd.Series indexed by symbol)
            market: MarketData instance
            date: Current date
            
        Returns:
            Scaled signals (signals * k)
        """
        date = pd.to_datetime(date)
        
        # Get scaler
        k = self.scaler(market, date)
        
        # Apply to signals
        scaled_signals = signals * k
        
        logger.debug(
            f"[MacroRegimeFilter] Applied scaler k={k:.3f} at {date}: "
            f"gross_leverage {signals.abs().sum():.2f} → {scaled_signals.abs().sum():.2f}"
        )
        
        return scaled_signals
    
    def describe(self) -> dict:
        """
        Return configuration and description of the MacroRegimeFilter agent.
        
        Returns:
            dict with configuration parameters
        """
        return {
            'agent': 'MacroRegimeFilter',
            'role': 'Scale strategy signals based on macro regime (vol + breadth)',
            'rebalance': self.rebalance,
            'vol_thresholds': self.vol_thresholds,
            'k_bounds': self.k_bounds,
            'smoothing': self.smoothing,
            'vol_lookback': self.vol_lookback,
            'breadth_lookback': self.breadth_lookback,
            'proxy_symbols': self.proxy_symbols,
            'outputs': ['scaler(market, date)', 'apply(signals, market, date)']
        }

