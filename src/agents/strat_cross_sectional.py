"""
Cross-Sectional Momentum Strategy Agent

Ranks assets by past returns (excluding recent period), produces standardized
long/short signals with market neutrality. Assets are bucketed into top/bottom
performers with equal-weighted positions within each bucket.

Signal construction:
1. Calculate simple returns over lookback period, excluding skip_recent days
2. Rank assets cross-sectionally
3. Long top_frac, short bottom_frac
4. Neutralize to sum ~0
5. Standardize signals (z-score or volatility-scaled)
6. Cap to ±signal_cap
7. Rebalance only on scheduled dates
"""

import logging
from typing import Optional, Union, List, Sequence, Mapping
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CSMOMConfig:
    """Configuration for Phase-1 Cross-Sectional Momentum."""
    lookbacks: Sequence[int]        # e.g. (63, 126, 252)
    weights: Sequence[float]        # e.g. (0.4, 0.35, 0.25)
    vol_lookback: int               # e.g. 63
    rebalance_freq: str             # "D" for daily, "W" for weekly
    neutralize_cross_section: bool  # True → z-score per date
    clip_score: float               # e.g. 3.0 for z-score clipping
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        if len(self.lookbacks) != len(self.weights):
            raise ValueError(f"lookbacks and weights must have same length, got {len(self.lookbacks)} and {len(self.weights)}")
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = tuple(w / total_weight for w in self.weights)
        else:
            raise ValueError("weights must sum to a positive value")


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum strategy agent.
    
    Generates momentum signals by ranking assets against each other based on
    past performance. Long top performers, short bottom performers, with
    market-neutral signal construction.
    """
    
    def __init__(
        self,
        symbols: List[str],
        lookback: int = 126,
        skip_recent: int = 21,
        top_frac: float = 0.33,
        bottom_frac: float = 0.33,
        standardize: str = "vol",
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI"
    ):
        """
        Initialize Cross-Sectional Momentum agent.
        
        Args:
            symbols: List of symbols in the universe
            lookback: Lookback period in days (e.g., 126 for 6-month)
            skip_recent: Days to skip at the end (e.g., 21 for 1-month gap)
            top_frac: Fraction of assets to go long (e.g., 0.33 for top third)
            bottom_frac: Fraction of assets to go short (e.g., 0.33 for bottom third)
            standardize: Method to standardize signals ("zscore" or "vol")
            signal_cap: Maximum absolute signal value (cap after standardization)
            rebalance: Rebalance frequency ("W-FRI" for weekly Friday, "M" for month-end)
        """
        self.symbols = symbols
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.standardize = standardize
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        
        # Validation
        if not 0 < top_frac <= 1.0:
            raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
        if not 0 < bottom_frac <= 1.0:
            raise ValueError(f"bottom_frac must be in (0, 1], got {bottom_frac}")
        if standardize not in ["zscore", "vol"]:
            raise ValueError(f"standardize must be 'zscore' or 'vol', got {standardize}")
        
        # State tracking
        self._last_rebalance = None
        self._last_signals = None
        self._rebalance_dates = None
        
        logger.info(
            f"[CrossSectionalMomentum] Initialized with lookback={lookback}, "
            f"skip_recent={skip_recent}, top_frac={top_frac}, bottom_frac={bottom_frac}, "
            f"standardize={standardize}, cap={signal_cap}, rebalance={rebalance}"
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
        
        logger.debug(f"[CrossSectionalMomentum] Computed {len(rebalance_dates)} rebalance dates")
        return rebalance_dates
    
    def _calculate_past_returns(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculate cumulative returns over lookback period, excluding skip_recent days.
        Uses simple returns compounded.
        
        Args:
            returns: Wide DataFrame of returns (date x symbols)
            date: Current evaluation date
            
        Returns:
            Series of cumulative returns per symbol
        """
        # Get data up to current date
        returns_upto = returns.loc[:date]
        
        if len(returns_upto) < self.skip_recent:
            # Not enough data
            return pd.Series(index=returns.columns, dtype=float)
        
        # Exclude the most recent skip_recent days
        if self.skip_recent > 0:
            returns_window = returns_upto.iloc[:-self.skip_recent]
        else:
            returns_window = returns_upto
        
        # Take last lookback days of remaining data
        if len(returns_window) < self.lookback:
            # Not enough history
            return pd.Series(index=returns.columns, dtype=float)
        
        returns_lookback = returns_window.iloc[-self.lookback:]
        
        # Calculate cumulative simple return (compound them)
        # Use skipna=False to propagate NaN, then handle per-symbol
        cum_ret = pd.Series(index=returns.columns, dtype=float)
        
        for symbol in returns.columns:
            symbol_returns = returns_lookback[symbol].dropna()
            if len(symbol_returns) >= self.lookback * 0.8:  # Require at least 80% of data
                cum_ret[symbol] = (1 + symbol_returns).prod() - 1
            else:
                cum_ret[symbol] = np.nan
        
        return cum_ret
    
    def _rank_and_bucket(
        self,
        past_returns: pd.Series
    ) -> pd.Series:
        """
        Rank assets and assign bucket signals (long/short/neutral).
        
        Args:
            past_returns: Series of past returns per symbol
            
        Returns:
            Series of raw bucket signals (1 for long, -1 for short, 0 for neutral)
        """
        # Drop NaN values
        valid_returns = past_returns.dropna()
        
        if len(valid_returns) == 0:
            return pd.Series(0.0, index=past_returns.index)
        
        # Rank in ascending order (lowest to highest return)
        ranks = valid_returns.rank(method='average', ascending=True)
        n_valid = len(valid_returns)
        
        # Calculate cutoff positions
        n_long = max(1, int(np.ceil(n_valid * self.top_frac)))
        n_short = max(1, int(np.ceil(n_valid * self.bottom_frac)))
        
        # Assign bucket signals
        signals = pd.Series(0.0, index=valid_returns.index)
        
        # Long: top performers (highest ranks)
        long_cutoff = n_valid - n_long + 1
        signals[ranks >= long_cutoff] = 1.0
        
        # Short: bottom performers (lowest ranks)
        short_cutoff = n_short
        signals[ranks <= short_cutoff] = -1.0
        
        # Neutralize: scale so that sum ~0
        # Long and short buckets get equal-weighted positions
        n_long_actual = (signals > 0).sum()
        n_short_actual = (signals < 0).sum()
        
        if n_long_actual > 0 and n_short_actual > 0:
            # Equal dollar-weighted long and short
            signals[signals > 0] = 1.0 / n_long_actual
            signals[signals < 0] = -1.0 / n_short_actual
        
        # Add back NaN symbols with 0 signal
        full_signals = pd.Series(0.0, index=past_returns.index)
        full_signals[signals.index] = signals
        
        return full_signals
    
    def _standardize_signals(
        self,
        raw_signals: pd.Series,
        returns: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Standardize signals using z-score or volatility scaling.
        
        Args:
            raw_signals: Raw bucket signals
            returns: Wide DataFrame of returns (for vol calculation)
            date: Current evaluation date
            
        Returns:
            Standardized signals
        """
        if self.standardize == "zscore":
            # Cross-sectional z-score
            valid_signals = raw_signals[raw_signals != 0]  # Exclude zeros
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
            vol_lookback = max(self.lookback, 63)
            
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
            min_vol = 0.01  # 1% minimum annualized vol
            trailing_vol = trailing_vol.clip(lower=min_vol)
            
            # Standardize: signal = raw_signal / vol
            # This gives a rough "signal per unit risk" interpretation
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
    
    def signals(
        self,
        market,
        date: Union[str, datetime, pd.Timestamp]
    ) -> pd.Series:
        """
        Generate cross-sectional momentum signals for a given date.
        
        Signals are only recomputed on rebalance dates; otherwise the last
        computed signals are returned (held constant).
        
        Args:
            market: MarketData instance (read-only)
            date: Date for signal generation
            
        Returns:
            Series of signals indexed by symbol (neutralized, sum ~0)
        """
        date = pd.to_datetime(date)
        
        # Get returns data up to this date (use simple returns for ranking)
        returns = market.get_returns(symbols=tuple(self.symbols), end=date, method="simple")
        
        if returns.empty:
            logger.warning(f"[CrossSectionalMomentum] No returns data available for date {date}")
            return pd.Series(0.0, index=self.symbols)
        
        # Ensure date is in the returns index
        if date not in returns.index:
            # Find the last available date <= requested date
            available_dates = returns.index[returns.index <= date]
            if len(available_dates) == 0:
                logger.warning(f"[CrossSectionalMomentum] No data available on or before {date}")
                return pd.Series(0.0, index=self.symbols)
            date = available_dates[-1]
            logger.debug(f"[CrossSectionalMomentum] Adjusted date to last available: {date}")
        
        # Compute rebalance schedule if not already done
        if self._rebalance_dates is None:
            self._rebalance_dates = self._compute_rebalance_dates(returns.index)
        
        # Check if we need to rebalance
        is_rebalance = date in self._rebalance_dates
        
        if not is_rebalance and self._last_signals is not None:
            # Hold previous signals
            logger.debug(f"[CrossSectionalMomentum] Holding signals from {self._last_rebalance}")
            return self._last_signals
        
        # Rebalance: compute new signals
        logger.debug(f"[CrossSectionalMomentum] Computing signals for {date}")
        
        # Step 1: Calculate past returns for ranking
        past_returns = self._calculate_past_returns(returns, date)
        
        # Step 2: Rank and bucket assets
        raw_signals = self._rank_and_bucket(past_returns)
        
        # Step 3: Standardize signals
        standardized = self._standardize_signals(raw_signals, returns, date)
        
        # Step 4: Cap signals
        capped = self._cap_signals(standardized)
        
        # Update state
        self._last_signals = capped
        self._last_rebalance = date
        
        logger.debug(
            f"[CrossSectionalMomentum] Generated signals at {date}: "
            f"sum={capped.sum():.6f}, mean={capped.mean():.3f}, std={capped.std():.3f}, "
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
            'strategy': 'CrossSectionalMomentum',
            'symbols': self.symbols,
            'lookback': self.lookback,
            'skip_recent': self.skip_recent,
            'top_frac': self.top_frac,
            'bottom_frac': self.bottom_frac,
            'standardize': self.standardize,
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
        logger.debug("[CrossSectionalMomentum] State reset")


class CSMOMPhase1:
    """
    Phase-1 Cross-Sectional Momentum strategy.
    
    Multi-horizon z-scored momentum with volatility-aware cross-sectional ranking.
    Returns signals in [-1, 1] per asset/date.
    """
    
    def __init__(self, config: CSMOMConfig):
        """
        Initialize Phase-1 CSMOM strategy.
        
        Args:
            config: CSMOMConfig with lookbacks, weights, vol_lookback, etc.
        """
        self.config = config
        logger.info(
            f"[CSMOMPhase1] Initialized with lookbacks={config.lookbacks}, "
            f"weights={config.weights}, vol_lookback={config.vol_lookback}, "
            f"rebalance={config.rebalance_freq}, neutralize={config.neutralize_cross_section}, "
            f"clip={config.clip_score}"
        )
    
    def compute_signals(
        self,
        md,
        start: str,
        end: str,
        universe: Sequence[str]
    ) -> pd.DataFrame:
        """
        Compute Phase-1 cross-sectional momentum signals.
        
        Returns DataFrame with index=date, columns=universe, values in [-1, 1]
        representing cross-sectional momentum scores.
        
        Args:
            md: MarketData instance
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            universe: List of symbols to trade
            
        Returns:
            DataFrame of signals [date x symbols] with values in [-1, 1]
        """
        logger.info(f"[CSMOMPhase1] Computing signals for {len(universe)} assets from {start} to {end}")
        
        # 1) Get daily returns panel (log returns for k-day calculation)
        rets = md.get_returns(universe, start=start, end=end, method="log")
        
        if rets.empty:
            logger.warning("[CSMOMPhase1] No returns data available")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=universe)
        
        logger.info(f"[CSMOMPhase1] Returns data shape: {rets.shape}")
        
        # 2) Compute k-day returns for each horizon
        kret_dict = {}
        for k in self.config.lookbacks:
            # For log returns: k-day log return = rolling sum
            kret = rets.rolling(window=k, min_periods=1).sum()
            kret_dict[k] = kret
            logger.debug(f"[CSMOMPhase1] Computed {k}-day returns")
        
        # 3) Per-horizon cross-sectional z-score on each date
        zscores = []
        for k, w in zip(self.config.lookbacks, self.config.weights):
            kret = kret_dict[k]
            
            # Cross-sectional z-score per date
            # mean_cs = kret.mean(axis=1)  # mean across assets for each date
            # std_cs = kret.std(axis=1)    # std across assets for each date
            # z = (kret.sub(mean_cs, axis=0)).div(std_cs.replace(0.0, np.nan), axis=0)
            
            # More robust: subtract mean, divide by std, handling NaN
            mean_cs = kret.mean(axis=1, skipna=True)
            std_cs = kret.std(axis=1, skipna=True)
            
            # Avoid division by zero
            std_cs = std_cs.replace(0.0, np.nan)
            
            # Z-score: (x - mean) / std
            z = kret.sub(mean_cs, axis=0).div(std_cs, axis=0)
            
            # Fill NaN with 0 (no signal if no variation)
            z = z.fillna(0.0)
            
            zscores.append(w * z)
        
        # 4) Composite score (weighted sum of z-scores)
        composite = sum(zscores)  # (T, N)
        
        # 5) Optional volatility tempering (per asset)
        if self.config.vol_lookback > 0:
            # Get simple returns for volatility calculation
            rets_simple = md.get_returns(universe, start=start, end=end, method="simple")
            
            # Calculate rolling volatility (annualized)
            vol = rets_simple.rolling(window=self.config.vol_lookback, min_periods=1).std() * np.sqrt(252.0)
            
            # Avoid division by zero
            vol = vol.replace(0.0, np.nan)
            
            # Divide composite by vol (vol-adjusted score)
            vol_adjusted = composite.div(vol, axis=0)
            
            # Re-z-score cross-sectionally after vol adjustment
            mean_cs = vol_adjusted.mean(axis=1, skipna=True)
            std_cs = vol_adjusted.std(axis=1, skipna=True)
            std_cs = std_cs.replace(0.0, np.nan)
            
            composite = vol_adjusted.sub(mean_cs, axis=0).div(std_cs, axis=0).fillna(0.0)
            
            logger.debug("[CSMOMPhase1] Applied volatility tempering")
        
        # 6) Cross-sectional neutralization + clipping
        if self.config.neutralize_cross_section:
            # Already z-scored, but ensure mean=0, std=1 per date
            mean_cs = composite.mean(axis=1, skipna=True)
            std_cs = composite.std(axis=1, skipna=True)
            std_cs = std_cs.replace(0.0, np.nan)
            
            composite = composite.sub(mean_cs, axis=0).div(std_cs, axis=0).fillna(0.0)
        
        # 7) Clip and scale to [-1, 1]
        c = float(self.config.clip_score)
        composite = composite.clip(lower=-c, upper=c) / c
        
        # 8) Apply rebalance frequency (forward-fill if not daily)
        if self.config.rebalance_freq == "D":
            # Daily: no forward-fill needed
            signals = composite
        elif self.config.rebalance_freq == "W":
            # Weekly: forward-fill to next rebalance
            # Find weekly rebalance dates (e.g., Fridays)
            rebalance_dates = pd.date_range(start=composite.index.min(), end=composite.index.max(), freq='W-FRI')
            rebalance_dates = rebalance_dates.intersection(composite.index)
            
            # Create signals DataFrame, forward-filling from rebalance dates
            signals = pd.DataFrame(index=composite.index, columns=composite.columns, dtype=float)
            
            for i, date in enumerate(rebalance_dates):
                if date in composite.index:
                    # Get signal values for this rebalance date
                    signal_values = composite.loc[date]
                    
                    # Determine end date for forward-fill
                    if i < len(rebalance_dates) - 1:
                        end_date = rebalance_dates[i + 1]
                        mask = (signals.index >= date) & (signals.index < end_date)
                    else:
                        mask = signals.index >= date
                    
                    # Forward-fill these values
                    for col in composite.columns:
                        signals.loc[mask, col] = signal_values[col]
        else:
            # Default: use as-is
            signals = composite
        
        logger.info(f"[CSMOMPhase1] Generated signals: shape={signals.shape}, range=[{signals.min().min():.3f}, {signals.max().max():.3f}]")
        
        return signals


class CSMOMMeta:
    """
    CSMOM Meta-Sleeve wrapper for CombinedStrategy integration.
    
    Wraps CSMOMPhase1 and provides a date-by-date signals() method
    compatible with CombinedStrategy.
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        lookbacks: Sequence[int] = (63, 126, 252),
        weights: Sequence[float] = (0.4, 0.35, 0.25),
        vol_lookback: int = 63,
        rebalance_freq: str = "D",
        neutralize_cross_section: bool = True,
        clip_score: float = 3.0
    ):
        """
        Initialize CSMOM Meta-Sleeve.
        
        Args:
            symbols: List of symbols (None = use market.universe)
            lookbacks: Lookback periods for multi-horizon momentum
            weights: Weights for each horizon (will be normalized)
            vol_lookback: Lookback for volatility calculation
            rebalance_freq: Rebalance frequency ("D" for daily, "W" for weekly)
            neutralize_cross_section: Whether to z-score cross-sectionally
            clip_score: Z-score clipping threshold
        """
        self.symbols = symbols
        self.config = CSMOMConfig(
            lookbacks=lookbacks,
            weights=weights,
            vol_lookback=vol_lookback,
            rebalance_freq=rebalance_freq,
            neutralize_cross_section=neutralize_cross_section,
            clip_score=clip_score
        )
        self._phase1 = CSMOMPhase1(self.config)
        
        # Cache for pre-computed signals
        self._signals_cache: Optional[pd.DataFrame] = None
        self._cache_start: Optional[str] = None
        self._cache_end: Optional[str] = None
        
        logger.info(
            f"[CSMOMMeta] Initialized with lookbacks={lookbacks}, "
            f"weights={weights}, vol_lookback={vol_lookback}, "
            f"rebalance={rebalance_freq}"
        )
    
    def fit_in_sample(
        self,
        market,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ):
        """
        Pre-compute signals for the full date range.
        
        This is called once before backtesting to compute all signals upfront.
        
        Args:
            market: MarketData instance
            start: Start date (optional, for validation)
            end: End date (optional, for validation)
        """
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning("[CSMOMMeta] No symbols available")
            return
        
        # Get full date range from market
        if start is None or end is None:
            # Use market's available date range
            prices = market.get_price_panel(symbols, fields=("close",), tidy=False)
            if prices.empty:
                logger.warning("[CSMOMMeta] No price data available")
                return
            start = str(prices.index.min().date())
            end = str(prices.index.max().date())
        
        logger.info(f"[CSMOMMeta] Pre-computing signals from {start} to {end}")
        
        # Compute signals for full range
        self._signals_cache = self._phase1.compute_signals(
            md=market,
            start=start,
            end=end,
            universe=list(symbols)
        )
        
        self._cache_start = start
        self._cache_end = end
        
        logger.info(f"[CSMOMMeta] Pre-computed {len(self._signals_cache)} days of signals")
    
    def signals(
        self,
        market,
        date: Union[str, datetime, pd.Timestamp]
    ) -> pd.Series:
        """
        Generate CSMOM signals for a given date.
        
        Args:
            market: MarketData instance (read-only)
            date: Date for signal generation
            
        Returns:
            Series of signals indexed by symbol (roughly mean 0, unit variance)
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[CSMOMMeta] No symbols available for date {date_dt}")
            return pd.Series(index=symbols, dtype=float)
        
        # Use cached signals if available
        if self._signals_cache is not None and not self._signals_cache.empty:
            # Find the appropriate date in cached signals
            if date_dt in self._signals_cache.index:
                signals = self._signals_cache.loc[date_dt]
            else:
                # Forward-fill: find last available date <= requested date
                available_dates = self._signals_cache.index[self._signals_cache.index <= date_dt]
                if len(available_dates) == 0:
                    logger.warning(f"[CSMOMMeta] No signals available for date {date_dt} (no prior data)")
                    return pd.Series(0.0, index=symbols)
                use_date = available_dates[-1]
                signals = self._signals_cache.loc[use_date]
                logger.debug(f"[CSMOMMeta] Using signals from {use_date} for date {date_dt}")
        else:
            # No cache: compute on-the-fly (less efficient but works)
            logger.warning(f"[CSMOMMeta] No pre-computed signals, computing on-the-fly for {date_dt}")
            # Get a reasonable date range around the requested date
            prices = market.get_price_panel(symbols, fields=("close",), tidy=False)
            if prices.empty:
                return pd.Series(0.0, index=symbols)
            
            # Use last 2 years of data for context
            end_date = str(date_dt.date())
            start_date = str((date_dt - pd.Timedelta(days=730)).date())
            
            signals_df = self._phase1.compute_signals(
                md=market,
                start=start_date,
                end=end_date,
                universe=list(symbols)
            )
            
            if date_dt in signals_df.index:
                signals = signals_df.loc[date_dt]
            else:
                available_dates = signals_df.index[signals_df.index <= date_dt]
                if len(available_dates) == 0:
                    return pd.Series(0.0, index=symbols)
                signals = signals_df.loc[available_dates[-1]]
        
        # Ensure all symbols are present (fill missing with 0)
        result = pd.Series(0.0, index=symbols)
        for sym in symbols:
            if sym in signals.index:
                result[sym] = signals[sym]
        
        return result
    
    def describe(self) -> dict:
        """Return strategy description."""
        return {
            'strategy': 'CSMOMMeta',
            'symbols': self.symbols,
            'lookbacks': list(self.config.lookbacks),
            'weights': list(self.config.weights),
            'vol_lookback': self.config.vol_lookback,
            'rebalance_freq': self.config.rebalance_freq,
            'neutralize_cross_section': self.config.neutralize_cross_section,
            'clip_score': self.config.clip_score
        }

