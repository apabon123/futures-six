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
        fred_series: Optional[tuple] = None,
        fred_lookback: int = 252,
        fred_weight: float = 0.3,
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
            fred_series: Tuple of FRED series IDs to use (e.g., ("VIXCLS", "DGS10"))
            fred_lookback: Rolling window for FRED indicator normalization (default: 252 days)
            fred_weight: Weight for FRED indicators in scaler calculation (default: 0.3)
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
            'proxy_symbols': ("ES", "NQ"),
            'fred_series': None,
            'fred_lookback': 252,
            'fred_weight': 0.3
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
            
            # Handle FRED indicators
            self.fred_series = regime_config.get('fred_series', fred_series)
            if self.fred_series is not None and isinstance(self.fred_series, list):
                self.fred_series = tuple(self.fred_series)
            self.fred_lookback = regime_config.get('fred_lookback', fred_lookback) if fred_lookback == defaults['fred_lookback'] else fred_lookback
            self.fred_weight = regime_config.get('fred_weight', fred_weight) if fred_weight == defaults['fred_weight'] else fred_weight
        else:
            self.rebalance = rebalance
            self.vol_thresholds = vol_thresholds or {'low': 0.15, 'high': 0.30}
            self.k_bounds = k_bounds or {'min': 0.4, 'max': 1.0}
            self.smoothing = smoothing
            self.vol_lookback = vol_lookback
            self.breadth_lookback = breadth_lookback
            self.proxy_symbols = proxy_symbols
            self.fred_series = fred_series
            self.fred_lookback = fred_lookback
            self.fred_weight = fred_weight
        
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
        
        # State for input smoothing (5-day EMA of breadth and FRED composite)
        self._breadth_history = []
        self._fred_signal_history = []
        self._input_smoothing_window = 5
        
        # Cache for rebalance dates
        self._rebalance_dates = None
        
        logger.info(
            f"[MacroRegimeFilter] Initialized: rebalance={self.rebalance}, "
            f"vol_thresholds={self.vol_thresholds}, k_bounds={self.k_bounds}, "
            f"smoothing={self.smoothing}, vol_lookback={self.vol_lookback}, "
            f"breadth_lookback={self.breadth_lookback}, proxy_symbols={self.proxy_symbols}, "
            f"fred_series={self.fred_series}, fred_lookback={self.fred_lookback}, fred_weight={self.fred_weight}"
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
        # Use continuous returns for realized vol calculation
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.warning(f"[MacroRegimeFilter] No returns data for {date}")
            return self.vol_thresholds['low']  # Default to low vol
        
        # Filter to proxy symbols and end date
        available_symbols = [s for s in self.proxy_symbols if s in returns_cont.columns]
        if not available_symbols:
            logger.warning(f"[MacroRegimeFilter] No matching proxy symbols in continuous returns for {date}")
            return self.vol_thresholds['low']  # Default to low vol
        
        returns = returns_cont[available_symbols].copy()
        date_dt = pd.to_datetime(date)
        returns = returns[returns.index <= date_dt]
        
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
        # Use continuous prices for breadth calculation
        prices_cont = market.prices_cont
        
        if prices_cont.empty:
            logger.warning(f"[MacroRegimeFilter] No price data for breadth calculation at {date}")
            return 0.5  # Neutral
        
        # Filter to proxy symbols and end date
        available_symbols = [s for s in self.proxy_symbols if s in prices_cont.columns]
        if not available_symbols:
            logger.warning(f"[MacroRegimeFilter] No matching proxy symbols in continuous prices for {date}")
            return 0.5  # Neutral
        
        prices = prices_cont[available_symbols].copy()
        date_dt = pd.to_datetime(date)
        prices = prices[prices.index <= date_dt]
        
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
    
    def _compute_fred_signal(
        self,
        market,
        date: pd.Timestamp
    ) -> float:
        """
        Compute combined FRED indicator signal.
        
        Normalizes each FRED indicator using rolling z-score, then combines
        them with equal weights. Returns a signal in [-1, 1] where:
        - Positive = risk-on (favorable conditions)
        - Negative = risk-off (adverse conditions)
        
        Args:
            market: MarketData instance
            date: Current date
            
        Returns:
            Combined FRED signal in [-1, 1]
        """
        if self.fred_series is None or len(self.fred_series) == 0:
            return 0.0  # Neutral if no FRED indicators
        
        try:
            # Get FRED indicators up to date
            fred_data = market.get_fred_indicators(
                series_ids=self.fred_series,
                end=date
            )
            
            if fred_data.empty:
                logger.warning(f"[MacroRegimeFilter] No FRED data available at {date}")
                return 0.0
            
            # Filter to date
            fred_at_date = fred_data[fred_data.index <= date]
            
            if len(fred_at_date) < self.fred_lookback:
                logger.warning(
                    f"[MacroRegimeFilter] Insufficient FRED data: "
                    f"got {len(fred_at_date)} days, need {self.fred_lookback}"
                )
                return 0.0
            
            # Dailyize monthly series before z-scoring
            # Forward-fill to business days for monthly series (CPI, UNRATE)
            monthly_series = ['CPIAUCSL', 'UNRATE']
            
            # Data freshness check: warn if monthly series are stale > 45 days
            for series_id in self.fred_series:
                if series_id in monthly_series and series_id in fred_at_date.columns:
                    series_data = fred_at_date[series_id].dropna()
                    if len(series_data) > 0:
                        last_date = series_data.index[-1]
                        days_since_update = (date - last_date).days
                        if days_since_update > 45:
                            logger.warning(
                                f"[MacroRegimeFilter] FRED series {series_id} is stale: "
                                f"last update {last_date}, {days_since_update} days ago"
                            )
            fred_dailyized = fred_at_date.copy()
            
            for series_id in fred_dailyized.columns:
                if series_id in monthly_series:
                    # Forward-fill monthly data to daily business days
                    series = fred_dailyized[series_id]
                    # Resample to business days with forward-fill
                    fred_dailyized[series_id] = series.asfreq('B', method='pad')
            
            # Get last fred_lookback days (after dailyization)
            fred_window = fred_dailyized.iloc[-self.fred_lookback:]
            
            # Normalize each indicator using z-score
            normalized_signals = []
            for series_id in self.fred_series:
                if series_id not in fred_window.columns:
                    continue
                
                series = fred_window[series_id].dropna()
                if len(series) < 20:  # Need minimum data
                    continue
                
                # Get latest value
                latest_value = series.iloc[-1]
                
                # Compute rolling mean and std for normalization (63-day window after dailyization)
                z_window = min(63, len(series))
                rolling_mean = series.rolling(window=z_window, min_periods=20).mean().iloc[-1]
                rolling_std = series.rolling(window=z_window, min_periods=20).std().iloc[-1]
                
                if rolling_std == 0 or pd.isna(rolling_std):
                    continue
                
                # Z-score normalization
                z_score = (latest_value - rolling_mean) / rolling_std
                
                # Cap z-score to avoid single prints swinging the scaler
                z_score = np.clip(z_score, -5.0, 5.0)
                
                # Map to [-1, 1] using tanh (bounded)
                normalized = np.tanh(z_score / 2.0)  # Divide by 2 to make it less extreme
                normalized_signals.append(normalized)
            
            if len(normalized_signals) == 0:
                return 0.0
            
            # Combine with equal weights
            combined_signal = np.mean(normalized_signals)
            
            return combined_signal
        
        except Exception as e:
            logger.warning(f"[MacroRegimeFilter] Error computing FRED signal: {e}")
            return 0.0
    
    def _compute_base_scaler(
        self,
        realized_vol: float,
        breadth: float,
        fred_signal: float = 0.0
    ) -> float:
        """
        Compute base scaler from realized vol, breadth, and FRED indicators.
        
        Logic:
        1. Map realized vol linearly from [low, high] to [k_max, k_min]
           (higher vol → lower scaler)
        2. Add breadth adjustment: +0.1 if breadth == 1.0, -0.1 if breadth == 0.0
        3. Add FRED signal adjustment: fred_signal * fred_weight
        4. Clamp to [k_min, k_max]
        
        Args:
            realized_vol: Annualized realized volatility
            breadth: Market breadth [0, 1]
            fred_signal: Combined FRED indicator signal [-1, 1]
            
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
        
        # 3. Add FRED signal adjustment
        # FRED signal: positive = risk-on (increase exposure), negative = risk-off (decrease)
        fred_adj = fred_signal * self.fred_weight
        k = k + fred_adj
        
        # 4. Clamp to bounds
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
        breadth_raw = self._compute_breadth(market, date)
        fred_signal_raw = self._compute_fred_signal(market, date)
        
        # Smooth inputs (5-day EMA) to avoid stepwise k-jumps at monthlies
        self._breadth_history.append(breadth_raw)
        self._fred_signal_history.append(fred_signal_raw)
        
        # Keep only last N values for EMA
        if len(self._breadth_history) > self._input_smoothing_window:
            self._breadth_history = self._breadth_history[-self._input_smoothing_window:]
        if len(self._fred_signal_history) > self._input_smoothing_window:
            self._fred_signal_history = self._fred_signal_history[-self._input_smoothing_window:]
        
        # Apply EMA smoothing to inputs
        if len(self._breadth_history) > 1:
            # Simple EMA: new = α * latest + (1-α) * previous_ema
            alpha = 0.2  # 20% weight on new value
            breadth = alpha * breadth_raw + (1 - alpha) * self._breadth_history[-2] if len(self._breadth_history) > 1 else breadth_raw
        else:
            breadth = breadth_raw
        
        if len(self._fred_signal_history) > 1:
            alpha = 0.2
            fred_signal = alpha * fred_signal_raw + (1 - alpha) * self._fred_signal_history[-2] if len(self._fred_signal_history) > 1 else fred_signal_raw
        else:
            fred_signal = fred_signal_raw
        
        # Compute base scaler
        base_k = self._compute_base_scaler(realized_vol, breadth, fred_signal)
        
        # Apply smoothing
        smoothed_k = self._smooth_scaler(base_k)
        
        # Update state
        self._last_scaler = smoothed_k
        self._last_rebalance = date
        
        logger.info(
            f"[MacroRegimeFilter] Rebalance at {date}: "
            f"vol={realized_vol:.3f}, breadth={breadth:.2f}, "
            f"fred_signal={fred_signal:.3f}, base_k={base_k:.3f}, smoothed_k={smoothed_k:.3f}"
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
            'fred_series': self.fred_series,
            'fred_lookback': self.fred_lookback,
            'fred_weight': self.fred_weight,
            'outputs': ['scaler(market, date)', 'apply(signals, market, date)']
        }

