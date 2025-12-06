"""
Momentum Features: Multi-feature momentum signals for different time horizons.

Long-term features:
- mom_long_ret_252_z: 252-day return momentum (vol-standardized, z-scored)
- mom_long_breakout_252_z: 252-day breakout strength (normalized position in range)
- mom_long_slope_slow_z: Slow trend slope (EMA-based trend indicator)

Medium-term features:
- mom_med_ret_84_z: 84-day return momentum
- mom_med_breakout_126_z: 126-day breakout strength
- mom_med_slope_med_z: Medium trend slope (EMA-based)
- mom_med_persistence_z: Trend persistence score

Short-term features:
- mom_short_ret_21_z: 21-day return momentum
- mom_short_breakout_21_z: 21-day breakout strength
- mom_short_slope_fast_z: Fast trend slope (EMA-based)
- mom_short_reversal_filter_z: Reversal filter (RSI-like)
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _zscore_rolling(s: pd.Series, window: int = 252, clip: float = 3.0, min_periods: Optional[int] = None) -> pd.Series:
    """
    Compute rolling z-score with clipping.
    
    Args:
        s: Input series
        window: Rolling window size in days
        clip: Z-score clipping bounds
        min_periods: Minimum periods for rolling calculation (default: window // 2)
        
    Returns:
        Z-scored and clipped series
    """
    if min_periods is None:
        min_periods = max(window // 2, 63)  # At least 63 days (1 quarter)
    
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sigma
    return z.clip(-clip, clip)


class LongMomentumFeatures:
    """
    Computes long-term momentum features for all symbols.
    
    Features:
    - mom_long_ret_252_z: 252-day return momentum (vol-standardized, z-scored)
    - mom_long_breakout_252_z: 252-day breakout strength
    - mom_long_slope_slow_z: Slow trend slope (EMA-based)
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        lookback: int = 252,
        skip_recent: int = 21,
        vol_window: int = 63,
        window: int = 252,
        clip: float = 3.0
    ):
        """
        Initialize long-term momentum features calculator.
        
        Args:
            symbols: List of symbols to compute features for (default: None, uses market.universe)
            lookback: Lookback period for return calculation (default: 252 days)
            skip_recent: Days to skip at the end (default: 21 days)
            vol_window: Window for volatility calculation (default: 63 days)
            window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
        """
        self.symbols = symbols
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.vol_window = vol_window
        self.window = window
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute long-term momentum features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns per symbol:
            - mom_long_ret_252_z_{symbol}: 252-day return momentum
            - mom_long_breakout_252_z_{symbol}: 252-day breakout strength
            - mom_long_slope_slow_z_{symbol}: Slow trend slope
        """
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning("[LongMomentum] No symbols provided")
            return pd.DataFrame()
        
        # Get continuous price data (back-adjusted)
        # Use prices_cont for all "what did the market do?" calculations
        prices_cont = market.prices_cont
        
        if prices_cont.empty:
            logger.warning("[LongMomentum] No continuous price data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        available_symbols = [s for s in symbols if s in prices_cont.columns]
        if not available_symbols:
            logger.warning("[LongMomentum] No matching symbols in continuous prices")
            return pd.DataFrame()
        
        prices = prices_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            prices = prices[prices.index <= end_dt]
        
        if prices.empty:
            logger.warning("[LongMomentum] No price data available after filtering")
            return pd.DataFrame()
        
        # Get continuous returns for volatility calculation
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.warning("[LongMomentum] No continuous returns data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        returns = returns_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            returns = returns[returns.index <= end_dt]
        
        if returns.empty:
            logger.warning("[LongMomentum] No returns data available after filtering")
            return pd.DataFrame()
        
        features_dict = {}
        all_dates = prices.index
        
        for symbol in symbols:
            if symbol not in prices.columns:
                logger.warning(f"[LongMomentum] Symbol {symbol} not in price data")
                continue
            
            price_series = prices[symbol].dropna()
            if price_series.empty:
                logger.warning(f"[LongMomentum] No price data for {symbol}")
                continue
            
            # Get returns for this symbol
            if symbol not in returns.columns:
                logger.warning(f"[LongMomentum] Symbol {symbol} not in returns data")
                continue
            
            ret_series = returns[symbol].dropna()
            
            # Feature 1: 252-day return momentum (vol-standardized, z-scored)
            mom_ret_252 = self._compute_return_momentum(
                price_series, ret_series, all_dates
            )
            
            # Feature 2: 252-day breakout strength
            mom_breakout_252 = self._compute_breakout_strength(
                price_series, all_dates
            )
            
            # Feature 3: Slow trend slope
            mom_slope_slow = self._compute_slope_slow(
                price_series, ret_series, all_dates
            )
            
            # Feature 4: 50-day breakout strength
            mom_breakout_50 = self._compute_breakout_strength_50(
                price_series, all_dates
            )
            
            # Feature 5: 100-day breakout strength
            mom_breakout_100 = self._compute_breakout_strength_100(
                price_series, all_dates
            )
            
            # Store features
            features_dict[f"mom_long_ret_252_z_{symbol}"] = mom_ret_252
            features_dict[f"mom_long_breakout_252_z_{symbol}"] = mom_breakout_252
            features_dict[f"mom_long_slope_slow_z_{symbol}"] = mom_slope_slow
            features_dict[f"mom_breakout_mid_50_z_{symbol}"] = mom_breakout_50
            features_dict[f"mom_breakout_mid_100_z_{symbol}"] = mom_breakout_100
        
        if not features_dict:
            logger.warning("[LongMomentum] No features could be computed")
            return pd.DataFrame()
        
        features = pd.DataFrame(features_dict, index=all_dates).sort_index()
        
        logger.info(
            f"[LongMomentum] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features
    
    def _compute_return_momentum(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute 252-day return momentum (vol-standardized, z-scored).
        
        Formula:
        - r_252 = log(price_t / price_{t-252-21})
        - vol_63 = std(daily_returns_{t-63..t}) * sqrt(252)
        - mom_long_ret_252 = r_252 / vol_63
        - mom_long_ret_252_z = clip(zscore(mom_long_ret_252), -3, 3)
        """
        # Align price and returns to common index
        common_index = price.index.intersection(returns.index)
        price_aligned = price.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        if len(common_index) < self.lookback + self.skip_recent + 1:
            return pd.Series(index=all_dates, dtype=float)
        
        # Compute rolling return: log(price_t / price_{t-252-21})
        # We need to compute: log(price[t-skip_recent] / price[t-skip_recent-lookback])
        # This is equivalent to: log(price[t-skip_recent]) - log(price[t-skip_recent-lookback])
        log_price = np.log(price_aligned)
        
        # Price at t - skip_recent
        price_end_log = log_price.shift(self.skip_recent) if self.skip_recent > 0 else log_price
        
        # Price at t - skip_recent - lookback
        price_start_log = log_price.shift(self.skip_recent + self.lookback)
        
        # Calculate log return
        r_252 = price_end_log - price_start_log
        
        # Compute rolling volatility (63-day)
        vol_63 = returns_aligned.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        
        # Avoid division by zero
        vol_63 = vol_63.clip(lower=1e-6)
        
        # Vol-standardized momentum
        mom_ret_252 = r_252 / vol_63
        
        # Reindex to all_dates
        mom_ret_252 = mom_ret_252.reindex(all_dates)
        
        # Z-score and clip
        mom_ret_252_z = _zscore_rolling(mom_ret_252, window=self.window, clip=self.clip)
        
        return mom_ret_252_z
    
    def _compute_breakout_strength(
        self,
        price: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute 252-day breakout strength.
        
        Formula:
        - rolling_min_252 = rolling_min(price, window=252)
        - rolling_max_252 = rolling_max(price, window=252)
        - raw_breakout_252 = (price_t - rolling_min_252) / (rolling_max_252 - rolling_min_252 + eps)
        - mom_long_breakout_252_z = clip(zscore(raw_breakout_252), -3, 3)
        """
        # Compute rolling min and max
        rolling_min_252 = price.rolling(window=self.lookback, min_periods=self.lookback // 2).min()
        rolling_max_252 = price.rolling(window=self.lookback, min_periods=self.lookback // 2).max()
        
        # Compute raw breakout
        eps = 1e-6
        range_252 = rolling_max_252 - rolling_min_252 + eps
        raw_breakout_252 = (price - rolling_min_252) / range_252
        
        # Map to roughly [-1, +1] by rescaling: (x - 0.5) * 2
        # This centers at 0 and scales to [-1, 1]
        breakout_scaled = (raw_breakout_252 - 0.5) * 2.0
        
        # Z-score and clip
        mom_breakout_252_z = _zscore_rolling(breakout_scaled, window=self.window, clip=self.clip)
        
        return mom_breakout_252_z
    
    def _compute_slope_slow(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute slow trend slope using EMAs.
        
        Formula:
        - EMA_63 and EMA_252 of price (or log price)
        - trend_slow = (EMA_63 - EMA_252) / (vol_63 + eps)
        - mom_long_slope_slow_z = clip(zscore(trend_slow), -3, 3)
        """
        # Use log prices for EMAs (more stable)
        log_price = np.log(price)
        
        # Compute EMAs
        # EMA with span = 2 / (alpha + 1), so alpha = 2 / span - 1
        # For span=63: alpha = 2/63 - 1 ≈ -0.968
        # For span=252: alpha = 2/252 - 1 ≈ -0.992
        ema_63 = log_price.ewm(span=63, adjust=False).mean()
        ema_252 = log_price.ewm(span=252, adjust=False).mean()
        
        # Compute trend slope
        trend_slow = ema_63 - ema_252
        
        # Standardize by volatility
        # Calculate rolling volatility
        vol_63_rolling = returns.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        
        eps = 1e-6
        trend_slow_scaled = trend_slow / (vol_63_rolling + eps)
        
        # Z-score and clip
        mom_slope_slow_z = _zscore_rolling(trend_slow_scaled, window=self.window, clip=self.clip)
        
        return mom_slope_slow_z
    
    def _compute_breakout_strength_50(
        self,
        price: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute 50-day breakout strength for Breakout Mid atomic sleeve.
        
        Formula:
        - rolling_min_50 = rolling_min(price, window=50)
        - rolling_max_50 = rolling_max(price, window=50)
        - raw_breakout_50 = (price_t - rolling_min_50) / (rolling_max_50 - rolling_min_50 + eps)
        - mom_breakout_mid_50_z = clip(zscore(raw_breakout_50), -3, 3)
        """
        lookback_50 = 50
        
        # Compute rolling min and max
        rolling_min_50 = price.rolling(window=lookback_50, min_periods=lookback_50 // 2).min()
        rolling_max_50 = price.rolling(window=lookback_50, min_periods=lookback_50 // 2).max()
        
        # Compute raw breakout
        eps = 1e-6
        range_50 = rolling_max_50 - rolling_min_50 + eps
        raw_breakout_50 = (price - rolling_min_50) / range_50
        
        # Map to roughly [-1, +1] by rescaling: (x - 0.5) * 2
        # This centers at 0 and scales to [-1, 1]
        breakout_scaled = (raw_breakout_50 - 0.5) * 2.0
        
        # Z-score and clip
        mom_breakout_50_z = _zscore_rolling(breakout_scaled, window=self.window, clip=self.clip)
        
        return mom_breakout_50_z
    
    def _compute_breakout_strength_100(
        self,
        price: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute 100-day breakout strength for Breakout Mid atomic sleeve.
        
        Formula:
        - rolling_min_100 = rolling_min(price, window=100)
        - rolling_max_100 = rolling_max(price, window=100)
        - raw_breakout_100 = (price_t - rolling_min_100) / (rolling_max_100 - rolling_min_100 + eps)
        - mom_breakout_mid_100_z = clip(zscore(raw_breakout_100), -3, 3)
        """
        lookback_100 = 100
        
        # Compute rolling min and max
        rolling_min_100 = price.rolling(window=lookback_100, min_periods=lookback_100 // 2).min()
        rolling_max_100 = price.rolling(window=lookback_100, min_periods=lookback_100 // 2).max()
        
        # Compute raw breakout
        eps = 1e-6
        range_100 = rolling_max_100 - rolling_min_100 + eps
        raw_breakout_100 = (price - rolling_min_100) / range_100
        
        # Map to roughly [-1, +1] by rescaling: (x - 0.5) * 2
        # This centers at 0 and scales to [-1, 1]
        breakout_scaled = (raw_breakout_100 - 0.5) * 2.0
        
        # Z-score and clip
        mom_breakout_100_z = _zscore_rolling(breakout_scaled, window=self.window, clip=self.clip)
        
        return mom_breakout_100_z


class MediumMomentumFeatures:
    """
    Computes medium-term momentum features for all symbols.
    
    Features:
    - mom_med_ret_84_z: 84-day return momentum (vol-standardized, z-scored)
    - mom_med_breakout_126_z: 126-day breakout strength
    - mom_med_slope_med_z: Medium trend slope (EMA-based)
    - mom_med_persistence_z: Trend persistence score
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        lookback: int = 84,
        skip_recent: int = 10,
        vol_window: int = 63,
        window: int = 252,
        clip: float = 3.0
    ):
        """
        Initialize medium-term momentum features calculator.
        
        Args:
            symbols: List of symbols to compute features for (default: None, uses market.universe)
            lookback: Lookback period for return calculation (default: 84 days)
            skip_recent: Days to skip at the end (default: 10 days)
            vol_window: Window for volatility calculation (default: 63 days)
            window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
        """
        self.symbols = symbols
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.vol_window = vol_window
        self.window = window
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute medium-term momentum features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns per symbol:
            - mom_med_ret_84_z_{symbol}: 84-day return momentum
            - mom_med_breakout_126_z_{symbol}: 126-day breakout strength
            - mom_med_slope_med_z_{symbol}: Medium trend slope
            - mom_med_persistence_z_{symbol}: Trend persistence
        """
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning("[MediumMomentum] No symbols provided")
            return pd.DataFrame()
        
        # Get continuous price data (back-adjusted)
        # Use prices_cont for all "what did the market do?" calculations
        prices_cont = market.prices_cont
        
        if prices_cont.empty:
            logger.warning("[MediumMomentum] No continuous price data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        available_symbols = [s for s in symbols if s in prices_cont.columns]
        if not available_symbols:
            logger.warning("[MediumMomentum] No matching symbols in continuous prices")
            return pd.DataFrame()
        
        prices = prices_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            prices = prices[prices.index <= end_dt]
        
        if prices.empty:
            logger.warning("[MediumMomentum] No price data available after filtering")
            return pd.DataFrame()
        
        # Get continuous returns for volatility calculation
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.warning("[MediumMomentum] No continuous returns data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        returns = returns_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            returns = returns[returns.index <= end_dt]
        
        if returns.empty:
            logger.warning("[MediumMomentum] No returns data available after filtering")
            return pd.DataFrame()
        
        features_dict = {}
        all_dates = prices.index
        
        for symbol in symbols:
            if symbol not in prices.columns:
                logger.warning(f"[MediumMomentum] Symbol {symbol} not in price data")
                continue
            
            price_series = prices[symbol].dropna()
            if price_series.empty:
                continue
            
            if symbol not in returns.columns:
                logger.warning(f"[MediumMomentum] Symbol {symbol} not in returns data")
                continue
            
            ret_series = returns[symbol].dropna()
            
            # Feature 1: 84-day return momentum
            mom_ret_84 = self._compute_return_momentum(price_series, ret_series, all_dates)
            
            # Feature 2: 126-day breakout strength
            mom_breakout_126 = self._compute_breakout_strength(price_series, all_dates)
            
            # Feature 3: Medium trend slope
            mom_slope_med = self._compute_slope_med(price_series, ret_series, all_dates)
            
            # Feature 4: Persistence score
            mom_persistence = self._compute_persistence(ret_series, all_dates)
            
            # Store features
            features_dict[f"mom_med_ret_84_z_{symbol}"] = mom_ret_84
            features_dict[f"mom_med_breakout_126_z_{symbol}"] = mom_breakout_126
            features_dict[f"mom_med_slope_med_z_{symbol}"] = mom_slope_med
            features_dict[f"mom_med_persistence_z_{symbol}"] = mom_persistence
        
        if not features_dict:
            logger.warning("[MediumMomentum] No features could be computed")
            return pd.DataFrame()
        
        features = pd.DataFrame(features_dict, index=all_dates).sort_index()
        
        logger.info(
            f"[MediumMomentum] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features
    
    def _compute_return_momentum(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute 84-day return momentum (vol-standardized, z-scored)."""
        common_index = price.index.intersection(returns.index)
        price_aligned = price.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        if len(common_index) < self.lookback + self.skip_recent + 1:
            return pd.Series(index=all_dates, dtype=float)
        
        log_price = np.log(price_aligned)
        price_end_log = log_price.shift(self.skip_recent) if self.skip_recent > 0 else log_price
        price_start_log = log_price.shift(self.skip_recent + self.lookback)
        
        r_84 = price_end_log - price_start_log
        
        vol_63 = returns_aligned.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        vol_63 = vol_63.clip(lower=1e-6)
        
        mom_ret_84 = r_84 / vol_63
        mom_ret_84 = mom_ret_84.reindex(all_dates)
        
        mom_ret_84_z = _zscore_rolling(mom_ret_84, window=self.window, clip=self.clip)
        
        return mom_ret_84_z
    
    def _compute_breakout_strength(
        self,
        price: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute 126-day breakout strength."""
        rolling_min_126 = price.rolling(window=126, min_periods=63).min()
        rolling_max_126 = price.rolling(window=126, min_periods=63).max()
        
        eps = 1e-6
        range_126 = rolling_max_126 - rolling_min_126 + eps
        raw_breakout_126 = (price - rolling_min_126) / range_126
        
        breakout_scaled = (raw_breakout_126 - 0.5) * 2.0
        
        mom_breakout_126_z = _zscore_rolling(breakout_scaled, window=self.window, clip=self.clip)
        
        return mom_breakout_126_z
    
    def _compute_slope_med(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute medium trend slope using EMA_20 and EMA_84."""
        log_price = np.log(price)
        
        ema_20 = log_price.ewm(span=20, adjust=False).mean()
        ema_84 = log_price.ewm(span=84, adjust=False).mean()
        
        trend_med = ema_20 - ema_84
        
        vol_63_rolling = returns.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        
        eps = 1e-6
        trend_med_scaled = trend_med / (vol_63_rolling + eps)
        
        mom_slope_med_z = _zscore_rolling(trend_med_scaled, window=self.window, clip=self.clip)
        
        return mom_slope_med_z
    
    def _compute_persistence(
        self,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute trend persistence score."""
        # Get signs of last 20 daily returns
        signs = np.sign(returns)
        
        # Rolling sum of signs over 20 days
        persistence_raw = signs.rolling(window=20, min_periods=10).mean()
        
        # Reindex to all_dates
        persistence_raw = persistence_raw.reindex(all_dates)
        
        # Z-score and clip
        mom_persistence_z = _zscore_rolling(persistence_raw, window=self.window, clip=self.clip)
        
        return mom_persistence_z


class ShortMomentumFeatures:
    """
    Computes short-term momentum features for all symbols.
    
    Features:
    - mom_short_ret_21_z: 21-day return momentum (vol-standardized, z-scored)
    - mom_short_breakout_21_z: 21-day breakout strength
    - mom_short_slope_fast_z: Fast trend slope (EMA-based)
    - mom_short_reversal_filter_z: Reversal filter (RSI-like)
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        lookback: int = 21,
        skip_recent: int = 5,
        vol_window: int = 20,
        window: int = 252,
        clip: float = 3.0
    ):
        """
        Initialize short-term momentum features calculator.
        
        Args:
            symbols: List of symbols to compute features for (default: None, uses market.universe)
            lookback: Lookback period for return calculation (default: 21 days)
            skip_recent: Days to skip at the end (default: 5 days)
            vol_window: Window for volatility calculation (default: 20 days)
            window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
        """
        self.symbols = symbols
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.vol_window = vol_window
        self.window = window
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute short-term momentum features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns per symbol:
            - mom_short_ret_21_z_{symbol}: 21-day return momentum
            - mom_short_breakout_21_z_{symbol}: 21-day breakout strength
            - mom_short_slope_fast_z_{symbol}: Fast trend slope
            - mom_short_reversal_filter_z_{symbol}: Reversal filter
        """
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning("[ShortMomentum] No symbols provided")
            return pd.DataFrame()
        
        # Get continuous price data (back-adjusted)
        # Use prices_cont for all "what did the market do?" calculations
        prices_cont = market.prices_cont
        
        if prices_cont.empty:
            logger.warning("[ShortMomentum] No continuous price data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        available_symbols = [s for s in symbols if s in prices_cont.columns]
        if not available_symbols:
            logger.warning("[ShortMomentum] No matching symbols in continuous prices")
            return pd.DataFrame()
        
        prices = prices_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            prices = prices[prices.index <= end_dt]
        
        if prices.empty:
            logger.warning("[ShortMomentum] No price data available after filtering")
            return pd.DataFrame()
        
        # Get continuous returns for volatility calculation
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.warning("[ShortMomentum] No continuous returns data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        returns = returns_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            returns = returns[returns.index <= end_dt]
        
        if returns.empty:
            logger.warning("[ShortMomentum] No returns data available after filtering")
            return pd.DataFrame()
        
        features_dict = {}
        all_dates = prices.index
        
        for symbol in symbols:
            if symbol not in prices.columns:
                logger.warning(f"[ShortMomentum] Symbol {symbol} not in price data")
                continue
            
            price_series = prices[symbol].dropna()
            if price_series.empty:
                continue
            
            if symbol not in returns.columns:
                logger.warning(f"[ShortMomentum] Symbol {symbol} not in returns data")
                continue
            
            ret_series = returns[symbol].dropna()
            
            # Feature 1: 21-day return momentum
            mom_ret_21 = self._compute_return_momentum(price_series, ret_series, all_dates)
            
            # Feature 2: 21-day breakout strength
            mom_breakout_21 = self._compute_breakout_strength(price_series, all_dates)
            
            # Feature 3: Fast trend slope
            mom_slope_fast = self._compute_slope_fast(price_series, ret_series, all_dates)
            
            # Feature 4: Reversal filter
            mom_reversal = self._compute_reversal_filter(price_series, all_dates)
            
            # Store features
            features_dict[f"mom_short_ret_21_z_{symbol}"] = mom_ret_21
            features_dict[f"mom_short_breakout_21_z_{symbol}"] = mom_breakout_21
            features_dict[f"mom_short_slope_fast_z_{symbol}"] = mom_slope_fast
            features_dict[f"mom_short_reversal_filter_z_{symbol}"] = mom_reversal
        
        if not features_dict:
            logger.warning("[ShortMomentum] No features could be computed")
            return pd.DataFrame()
        
        features = pd.DataFrame(features_dict, index=all_dates).sort_index()
        
        logger.info(
            f"[ShortMomentum] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features
    
    def _compute_return_momentum(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute 21-day return momentum (vol-standardized, z-scored)."""
        common_index = price.index.intersection(returns.index)
        price_aligned = price.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        if len(common_index) < self.lookback + self.skip_recent + 1:
            return pd.Series(index=all_dates, dtype=float)
        
        log_price = np.log(price_aligned)
        price_end_log = log_price.shift(self.skip_recent) if self.skip_recent > 0 else log_price
        price_start_log = log_price.shift(self.skip_recent + self.lookback)
        
        r_21 = price_end_log - price_start_log
        
        vol_20 = returns_aligned.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        vol_20 = vol_20.clip(lower=1e-6)
        
        mom_ret_21 = r_21 / vol_20
        mom_ret_21 = mom_ret_21.reindex(all_dates)
        
        mom_ret_21_z = _zscore_rolling(mom_ret_21, window=self.window, clip=self.clip)
        
        return mom_ret_21_z
    
    def _compute_breakout_strength(
        self,
        price: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute 21-day breakout strength."""
        rolling_min_21 = price.rolling(window=21, min_periods=10).min()
        rolling_max_21 = price.rolling(window=21, min_periods=10).max()
        
        eps = 1e-6
        range_21 = rolling_max_21 - rolling_min_21 + eps
        raw_breakout_21 = (price - rolling_min_21) / range_21
        
        breakout_scaled = (raw_breakout_21 - 0.5) * 2.0
        
        mom_breakout_21_z = _zscore_rolling(breakout_scaled, window=self.window, clip=self.clip)
        
        return mom_breakout_21_z
    
    def _compute_slope_fast(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute fast trend slope using EMA_10 and EMA_40."""
        log_price = np.log(price)
        
        ema_10 = log_price.ewm(span=10, adjust=False).mean()
        ema_40 = log_price.ewm(span=40, adjust=False).mean()
        
        trend_fast = ema_10 - ema_40
        
        vol_20_rolling = returns.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        
        eps = 1e-6
        trend_fast_scaled = trend_fast / (vol_20_rolling + eps)
        
        mom_slope_fast_z = _zscore_rolling(trend_fast_scaled, window=self.window, clip=self.clip)
        
        return mom_slope_fast_z
    
    def _compute_reversal_filter(
        self,
        price: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute reversal filter (RSI-like)."""
        # Compute price changes
        price_change = price.diff()
        
        # Separate gains and losses
        gains = price_change.clip(lower=0)
        losses = -price_change.clip(upper=0)
        
        # Compute average gains and losses over 14 days (RSI-like)
        avg_gain = gains.rolling(window=14, min_periods=7).mean()
        avg_loss = losses.rolling(window=14, min_periods=7).mean()
        
        # Avoid division by zero
        eps = 1e-6
        rs = avg_gain / (avg_loss + eps)
        
        # RSI = 100 - (100 / (1 + RS))
        # Convert to [-1, 1] range: (RSI - 50) / 50
        rsi = 100 - (100 / (1 + rs))
        rsi_normalized = (rsi - 50) / 50
        
        # Reindex to all_dates
        rsi_normalized = rsi_normalized.reindex(all_dates)
        
        # Z-score and clip
        mom_reversal_z = _zscore_rolling(rsi_normalized, window=self.window, clip=self.clip)
        
        return mom_reversal_z


class CanonicalMediumMomentumFeatures:
    """
    Computes canonical medium-term momentum features for all symbols.
    
    This is the academically-grounded medium-term sleeve (84d canonical horizon)
    with equal-weight composite (1/3, 1/3, 1/3):
    
    Features:
    - mom_medcanon_ret_84_z: 84-day return momentum (skip 10d, vol-scaled, z-scored)
    - mom_medcanon_breakout_84_z: 84-day breakout strength (z-scored)
    - mom_medcanon_slope_21_84_z: EMA21 vs EMA84 slope (vol-scaled, z-scored)
    - mom_medcanon_composite_z: Equal-weight composite (1/3 each)
    
    Canonical Parameters:
    - Horizon: 84 trading days (~4 months, canonical medium-term)
    - Skip: 10 trading days (~2 weeks, avoid short-term noise)
    - Vol window: 21 days (standard short-term vol for scaling)
    - Standardization: 252d rolling z-score, clipped at ±3
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        lookback: int = 84,
        skip_recent: int = 10,
        vol_window: int = 21,
        window: int = 252,
        clip: float = 3.0
    ):
        """
        Initialize canonical medium-term momentum features calculator.
        
        Args:
            symbols: List of symbols to compute features for (default: None, uses market.universe)
            lookback: Lookback period for return calculation (default: 84 days, canonical)
            skip_recent: Days to skip at the end (default: 10 days, canonical)
            vol_window: Window for volatility calculation (default: 21 days, canonical)
            window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
        """
        self.symbols = symbols
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.vol_window = vol_window
        self.window = window
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute canonical medium-term momentum features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns per symbol:
            - mom_medcanon_ret_84_z_{symbol}: 84-day return momentum
            - mom_medcanon_breakout_84_z_{symbol}: 84-day breakout strength
            - mom_medcanon_slope_21_84_z_{symbol}: EMA21 vs EMA84 slope
            - mom_medcanon_composite_z_{symbol}: Equal-weight composite (1/3 each)
        """
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning("[CanonicalMediumMomentum] No symbols provided")
            return pd.DataFrame()
        
        # Get continuous price data (back-adjusted)
        prices_cont = market.prices_cont
        
        if prices_cont.empty:
            logger.warning("[CanonicalMediumMomentum] No continuous price data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        available_symbols = [s for s in symbols if s in prices_cont.columns]
        if not available_symbols:
            logger.warning("[CanonicalMediumMomentum] No matching symbols in continuous prices")
            return pd.DataFrame()
        
        prices = prices_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            prices = prices[prices.index <= end_dt]
        
        if prices.empty:
            logger.warning("[CanonicalMediumMomentum] No price data available after filtering")
            return pd.DataFrame()
        
        # Get continuous returns for volatility calculation
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.warning("[CanonicalMediumMomentum] No continuous returns data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        returns = returns_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            returns = returns[returns.index <= end_dt]
        
        if returns.empty:
            logger.warning("[CanonicalMediumMomentum] No returns data available after filtering")
            return pd.DataFrame()
        
        features_dict = {}
        all_dates = prices.index
        
        for symbol in symbols:
            if symbol not in prices.columns:
                logger.warning(f"[CanonicalMediumMomentum] Symbol {symbol} not in price data")
                continue
            
            price_series = prices[symbol].dropna()
            if price_series.empty:
                continue
            
            if symbol not in returns.columns:
                logger.warning(f"[CanonicalMediumMomentum] Symbol {symbol} not in returns data")
                continue
            
            ret_series = returns[symbol].dropna()
            
            # Feature 1: 84-day return momentum (skip 10d, vol-scaled, z-scored)
            mom_ret_84_z = self._compute_return_momentum(price_series, ret_series, all_dates)
            
            # Feature 2: 84-day breakout strength (z-scored)
            mom_breakout_84_z = self._compute_breakout_strength(price_series, all_dates)
            
            # Feature 3: EMA21 vs EMA84 slope (vol-scaled, z-scored)
            mom_slope_21_84_z = self._compute_slope_ema21_84(price_series, ret_series, all_dates)
            
            # Store features
            features_dict[f"mom_medcanon_ret_84_z_{symbol}"] = mom_ret_84_z
            features_dict[f"mom_medcanon_breakout_84_z_{symbol}"] = mom_breakout_84_z
            features_dict[f"mom_medcanon_slope_21_84_z_{symbol}"] = mom_slope_21_84_z
            
            # Canonical composite: equal-weight (1/3, 1/3, 1/3)
            composite = (mom_ret_84_z + mom_breakout_84_z + mom_slope_21_84_z) / 3.0
            features_dict[f"mom_medcanon_composite_z_{symbol}"] = composite
        
        if not features_dict:
            logger.warning("[CanonicalMediumMomentum] No features could be computed")
            return pd.DataFrame()
        
        features = pd.DataFrame(features_dict, index=all_dates).sort_index()
        
        logger.info(
            f"[CanonicalMediumMomentum] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features
    
    def _compute_return_momentum(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute 84-day return momentum (skip 10d, vol-scaled, z-scored).
        
        Formula:
        - r_84 = log(price[t-10]) - log(price[t-10-84])
        - vol_21 = std(daily_returns[t-21..t]) * sqrt(252)
        - mom_ret_84 = r_84 / vol_21
        - mom_ret_84_z = clip(zscore(mom_ret_84), -3, 3)
        """
        common_index = price.index.intersection(returns.index)
        price_aligned = price.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        if len(common_index) < self.lookback + self.skip_recent + 1:
            return pd.Series(index=all_dates, dtype=float)
        
        log_price = np.log(price_aligned)
        
        # Price at t - skip_recent
        price_end_log = log_price.shift(self.skip_recent) if self.skip_recent > 0 else log_price
        
        # Price at t - skip_recent - lookback
        price_start_log = log_price.shift(self.skip_recent + self.lookback)
        
        # Calculate log return
        r_84 = price_end_log - price_start_log
        
        # Compute rolling volatility (21-day, canonical for medium-term)
        vol_21 = returns_aligned.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        vol_21 = vol_21.clip(lower=1e-6)
        
        # Vol-standardized momentum
        mom_ret_84 = r_84 / vol_21
        
        # Reindex to all_dates
        mom_ret_84 = mom_ret_84.reindex(all_dates)
        
        # Z-score and clip
        mom_ret_84_z = _zscore_rolling(mom_ret_84, window=self.window, clip=self.clip)
        
        return mom_ret_84_z
    
    def _compute_breakout_strength(
        self,
        price: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute 84-day breakout strength (z-scored).
        
        Formula:
        - Evaluate at t-10 (skip recent)
        - rolling_min_84 = min(price[t-10-84..t-10])
        - rolling_max_84 = max(price[t-10-84..t-10])
        - raw_breakout_84 = (price[t-10] - rolling_min_84) / (rolling_max_84 - rolling_min_84 + eps)
        - breakout_centered = (raw_breakout_84 - 0.5) * 2  # Map to [-1, +1]
        - mom_breakout_84_z = clip(zscore(breakout_centered), -3, 3)
        """
        # Shift price by skip_recent to evaluate at t-10
        price_shifted = price.shift(self.skip_recent)
        
        # Compute rolling min and max over 84 days at t-10
        rolling_min_84 = price_shifted.rolling(window=self.lookback, min_periods=self.lookback // 2).min()
        rolling_max_84 = price_shifted.rolling(window=self.lookback, min_periods=self.lookback // 2).max()
        
        # Compute raw breakout
        eps = 1e-6
        range_84 = rolling_max_84 - rolling_min_84 + eps
        raw_breakout_84 = (price_shifted - rolling_min_84) / range_84
        
        # Map to roughly [-1, +1] by rescaling: (x - 0.5) * 2
        breakout_centered = (raw_breakout_84 - 0.5) * 2.0
        
        # Reindex to all_dates
        breakout_centered = breakout_centered.reindex(all_dates)
        
        # Z-score and clip
        mom_breakout_84_z = _zscore_rolling(breakout_centered, window=self.window, clip=self.clip)
        
        return mom_breakout_84_z
    
    def _compute_slope_ema21_84(
        self,
        price: pd.Series,
        returns: pd.Series,
        all_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute EMA21 vs EMA84 slope (vol-scaled, z-scored).
        
        Formula:
        - EMA_21 and EMA_84 of log price
        - slope_raw = (EMA_21 - EMA_84) / EMA_84
        - vol_21 = std(daily_returns[t-21..t]) * sqrt(252)
        - slope_scaled = slope_raw / vol_21
        - mom_slope_21_84_z = clip(zscore(slope_scaled), -3, 3)
        """
        # Use log prices for EMAs (more stable)
        log_price = np.log(price)
        
        # Compute EMAs
        ema_21 = log_price.ewm(span=21, adjust=False).mean()
        ema_84 = log_price.ewm(span=84, adjust=False).mean()
        
        # Compute slope
        slope_raw = (ema_21 - ema_84) / (ema_84.abs() + 1e-6)
        
        # Standardize by volatility
        vol_21 = returns.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std() * np.sqrt(252)
        vol_21 = vol_21.clip(lower=1e-6)
        
        slope_scaled = slope_raw / vol_21
        
        # Reindex to all_dates
        slope_scaled = slope_scaled.reindex(all_dates)
        
        # Z-score and clip
        mom_slope_21_84_z = _zscore_rolling(slope_scaled, window=self.window, clip=self.clip)
        
        return mom_slope_21_84_z


class PersistenceFeatures:
    """
    Computes persistence (momentum-of-momentum) features for all symbols.
    
    Persistence captures the rate of change of trend itself.
    
    Features:
    - persistence_slope_accel_z_{symbol}: Slope acceleration (EMA20-EMA84 acceleration)
    - persistence_breakout_accel_z_{symbol}: Breakout acceleration (breakout_126 acceleration)
    - persistence_return_accel_z_{symbol}: Return acceleration (ret_84 acceleration)
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        acceleration_window: int = 21,
        clip: float = 3.0
    ):
        """
        Initialize persistence features calculator.
        
        Args:
            symbols: List of symbols to compute features for (default: None, uses market.universe)
            acceleration_window: Window for acceleration calculation (default: 21 days)
            clip: Z-score clipping bounds for cross-sectional z-scoring (default: 3.0)
        """
        self.symbols = symbols
        self.acceleration_window = acceleration_window
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute persistence features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns per symbol:
            - persistence_slope_accel_z_{symbol}: Cross-sectional z-scored slope acceleration
            - persistence_breakout_accel_z_{symbol}: Cross-sectional z-scored breakout acceleration
            - persistence_return_accel_z_{symbol}: Cross-sectional z-scored return acceleration
        """
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning("[Persistence] No symbols provided")
            return pd.DataFrame()
        
        # Get continuous price data (back-adjusted)
        prices_cont = market.prices_cont
        
        if prices_cont.empty:
            logger.warning("[Persistence] No continuous price data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        available_symbols = [s for s in symbols if s in prices_cont.columns]
        if not available_symbols:
            logger.warning("[Persistence] No matching symbols in continuous prices")
            return pd.DataFrame()
        
        prices = prices_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            prices = prices[prices.index <= end_dt]
        
        if prices.empty:
            logger.warning("[Persistence] No price data available after filtering")
            return pd.DataFrame()
        
        # Get all dates from prices (source of truth for trading dates)
        all_dates = prices.index
        
        # Get daily returns for vol normalization
        returns = prices.pct_change(fill_method=None)
        
        features_dict = {}
        
        # Compute features for each symbol
        for symbol in available_symbols:
            price = prices[symbol]
            ret = returns[symbol]
            
            # 1. Slope acceleration: (EMA20 - EMA84)[t] - (EMA20 - EMA84)[t-21]
            log_price = np.log(price)
            ema20 = log_price.ewm(span=20, adjust=False).mean()
            ema84 = log_price.ewm(span=84, adjust=False).mean()
            
            slope_now = ema20 - ema84
            slope_prev = slope_now.shift(self.acceleration_window)
            slope_accel_raw = slope_now - slope_prev
            
            # Vol-normalize slope acceleration
            vol_63 = ret.rolling(window=63, min_periods=31).std() * np.sqrt(252)
            vol_63 = vol_63.clip(lower=1e-6)
            slope_accel_scaled = slope_accel_raw / vol_63
            slope_accel_scaled = slope_accel_scaled.reindex(all_dates)
            
            features_dict[f"persistence_slope_accel_{symbol}"] = slope_accel_scaled
            
            # 2. Breakout acceleration: breakout_126[t] - breakout_126[t-21]
            rolling_min_126 = price.rolling(window=126, min_periods=63).min()
            rolling_max_126 = price.rolling(window=126, min_periods=63).max()
            
            eps = 1e-6
            range_126 = rolling_max_126 - rolling_min_126 + eps
            raw_breakout_126 = (price - rolling_min_126) / range_126
            breakout_scaled = (raw_breakout_126 - 0.5) * 2.0  # Map to [-1, +1]
            
            breakout_now = breakout_scaled
            breakout_prev = breakout_now.shift(self.acceleration_window)
            breakout_accel_raw = breakout_now - breakout_prev
            breakout_accel_raw = breakout_accel_raw.reindex(all_dates)
            
            features_dict[f"persistence_breakout_accel_{symbol}"] = breakout_accel_raw
            
            # 3. Return acceleration: ret_84[t] - ret_84[t-21]
            # Reuse log_price from slope acceleration calculation
            ret_84_now = log_price - log_price.shift(84)
            ret_84_prev = ret_84_now.shift(self.acceleration_window)
            return_accel_raw = ret_84_now - ret_84_prev
            
            # Vol-normalize return acceleration
            vol_63 = ret.rolling(window=63, min_periods=31).std() * np.sqrt(252)
            vol_63 = vol_63.clip(lower=1e-6)
            return_accel_scaled = return_accel_raw / vol_63
            return_accel_scaled = return_accel_scaled.reindex(all_dates)
            
            features_dict[f"persistence_return_accel_{symbol}"] = return_accel_scaled
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_dict, index=all_dates)
        
        # Cross-sectional z-scoring for each feature type
        # Group features by type (slope, breakout, return)
        feature_types = {
            'slope_accel': [c for c in features_df.columns if 'slope_accel' in c],
            'breakout_accel': [c for c in features_df.columns if 'breakout_accel' in c],
            'return_accel': [c for c in features_df.columns if 'return_accel' in c]
        }
        
        # Create output features dict
        features_dict_output = {}
        
        for feat_type, cols in feature_types.items():
            if not cols:
                continue
            
            # Get raw features for this type
            raw_features = features_df[cols]
            
            # Cross-sectional z-score at each date (vectorized approach)
            z_scored_features = pd.DataFrame(index=all_dates, columns=cols, dtype=float)
            
            for date in all_dates:
                if date not in raw_features.index:
                    continue
                
                row = raw_features.loc[date]
                valid_values = row.dropna()
                
                if len(valid_values) < 2:
                    # Not enough assets for z-scoring - set to NaN
                    z_scored_features.loc[date] = np.nan
                else:
                    mean_val = valid_values.mean()
                    std_val = valid_values.std()
                    
                    if std_val > 0:
                        z_scores = (row - mean_val) / std_val
                        # Clip to ±clip
                        z_scores = z_scores.clip(lower=-self.clip, upper=self.clip)
                    else:
                        z_scores = pd.Series(0.0, index=row.index)
                    
                    z_scored_features.loc[date] = z_scores
            
            # Rename columns to include _z suffix and extract symbol
            # Column format: persistence_{feat_type}_{symbol}
            # Need to extract full symbol name (e.g., "ES_FRONT_CALENDAR_2D")
            for col in cols:
                # Remove prefix "persistence_{feat_type}_" to get symbol
                prefix = f"persistence_{feat_type}_"
                if col.startswith(prefix):
                    symbol = col[len(prefix):]
                else:
                    # Fallback: try to extract symbol from end
                    symbol = col.split('_')[-1]
                feat_name = f"persistence_{feat_type}_z_{symbol}"
                features_dict_output[feat_name] = z_scored_features[col]
        
        # Convert to DataFrame
        if features_dict_output:
            output_df = pd.DataFrame(features_dict_output, index=all_dates).sort_index()
        else:
            output_df = pd.DataFrame(index=all_dates)
        
        logger.info(
            f"[Persistence] Computed features for {len(available_symbols)} symbols, {len(all_dates)} dates "
            f"(non-null: {output_df.notna().sum().to_dict()})"
        )
        
        return output_df


class ResidualTrendFeatures:
    """
    Computes residual trend features for all symbols.
    
    Residual trend = long-horizon trend minus short-term movement.
    
    Features:
    - trend_resid_ret_252_21: Raw residual return (long_ret - short_ret)
    - trend_resid_ret_252_21_z: Cross-sectional z-score of residual return
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        long_lookback: int = 252,
        short_lookback: int = 21,
        clip: float = 3.0
    ):
        """
        Initialize residual trend features calculator.
        
        Args:
            symbols: List of symbols to compute features for (default: None, uses market.universe)
            long_lookback: Long-horizon lookback period (default: 252 days)
            short_lookback: Short-horizon lookback period (default: 21 days)
            clip: Z-score clipping bounds for cross-sectional z-scoring (default: 3.0)
        """
        self.symbols = symbols
        self.long_lookback = long_lookback
        self.short_lookback = short_lookback
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute residual trend features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns per symbol:
            - trend_resid_ret_252_21_{symbol}: Raw residual return
            - trend_resid_ret_252_21_z_{symbol}: Cross-sectional z-scored residual return
        """
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning("[ResidualTrend] No symbols provided")
            return pd.DataFrame()
        
        # Get continuous price data (back-adjusted)
        prices_cont = market.prices_cont
        
        if prices_cont.empty:
            logger.warning("[ResidualTrend] No continuous price data available")
            return pd.DataFrame()
        
        # Filter to requested symbols and date range
        available_symbols = [s for s in symbols if s in prices_cont.columns]
        if not available_symbols:
            logger.warning("[ResidualTrend] No matching symbols in continuous prices")
            return pd.DataFrame()
        
        prices = prices_cont[available_symbols].copy()
        
        # Filter by end_date if provided
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            prices = prices[prices.index <= end_dt]
        
        if prices.empty:
            logger.warning("[ResidualTrend] No price data available after filtering")
            return pd.DataFrame()
        
        # Convert to log prices for log return calculation
        log_prices = np.log(prices)
        
        # Compute long-horizon log returns
        # long_ret_t = log(price_t / price_{t-L_long}) = log(price_t) - log(price_{t-L_long})
        long_log_returns = log_prices - log_prices.shift(self.long_lookback)
        
        # Compute short-horizon log returns
        # short_ret_t = log(price_t / price_{t-L_short}) = log(price_t) - log(price_{t-L_short})
        short_log_returns = log_prices - log_prices.shift(self.short_lookback)
        
        # Compute residual returns: resid_ret = long_ret - short_ret
        residual_returns = long_log_returns - short_log_returns
        
        # Get all dates from prices (source of truth for trading dates)
        all_dates = prices.index
        
        features_dict = {}
        
        # Compute raw residual returns per symbol
        for symbol in available_symbols:
            if symbol not in residual_returns.columns:
                continue
            
            resid_ret_series = residual_returns[symbol].reindex(all_dates)
            features_dict[f"trend_resid_ret_{self.long_lookback}_{self.short_lookback}_{symbol}"] = resid_ret_series
        
        # Compute cross-sectional z-scores for each date
        # For each date, z-score residual returns across all assets
        for symbol in available_symbols:
            if symbol not in residual_returns.columns:
                continue
            
            # Get residual returns for this symbol
            resid_ret_series = residual_returns[symbol].reindex(all_dates)
            
            # Compute cross-sectional z-score for each date
            resid_ret_z_series = pd.Series(index=all_dates, dtype=float)
            
            for date in all_dates:
                # Get residual returns for all assets on this date
                resid_ret_on_date = residual_returns.loc[date] if date in residual_returns.index else pd.Series(dtype=float)
                
                # Filter out NaN values
                valid_returns = resid_ret_on_date.dropna()
                
                if len(valid_returns) < 2:
                    # Need at least 2 assets for z-score
                    resid_ret_z_series.loc[date] = 0.0
                    continue
                
                # Compute z-score
                mean_ret = valid_returns.mean()
                std_ret = valid_returns.std()
                
                if std_ret == 0 or np.isnan(std_ret):
                    resid_ret_z_series.loc[date] = 0.0
                    continue
                
                # Z-score for this symbol on this date
                resid_ret_value = resid_ret_series.loc[date]
                if pd.isna(resid_ret_value):
                    resid_ret_z_series.loc[date] = 0.0
                else:
                    z_score = (resid_ret_value - mean_ret) / std_ret
                    # Clip to bounds
                    z_score = np.clip(z_score, -self.clip, self.clip)
                    resid_ret_z_series.loc[date] = z_score
            
            features_dict[f"trend_resid_ret_{self.long_lookback}_{self.short_lookback}_z_{symbol}"] = resid_ret_z_series
        
        if not features_dict:
            logger.warning("[ResidualTrend] No features could be computed")
            return pd.DataFrame()
        
        features = pd.DataFrame(features_dict, index=all_dates).sort_index()
        
        logger.info(
            f"[ResidualTrend] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features