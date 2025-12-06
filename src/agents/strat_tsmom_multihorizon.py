"""
TSMOM Multi-Horizon Strategy: Unified multi-horizon momentum strategy.

Combines long-term (252d), medium-term (84/126d), short-term (21d), and residual trend
(252d-21d) features into a single unified trend-following sleeve using configurable
horizon weights.

Features:
- Long-term: 252d return, 252d breakout, slow slope (EMA-based)
- Medium-term: 84d return, 126d breakout, medium slope, persistence
- Short-term: 21d return, 21d breakout, fast slope, reversal filter (optional)
- Residual Trend: Long-horizon trend minus short-term movement, cross-sectionally z-scored

Horizon weights (recommended):
- Long (252d): 0.40-0.50 (default: 0.45)
- Medium (84/126d): 0.25-0.30 (default: 0.28)
- Short (21d): 0.20 (default: 0.20)
- Residual (252d-21d): 0.10-0.20 (default: 0.15)

All features are combined within each horizon, then horizons are blended,
then cross-sectionally z-scored and clipped to ±3.0.
"""

import logging
from typing import Optional, Union, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class TSMOMMultiHorizonStrategy:
    """
    Multi-horizon Time-Series Momentum (TSMOM) strategy.
    
    Combines long, medium, and short-term momentum features into a single
    unified trend-following signal using configurable horizon and feature weights.
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        horizon_weights: Optional[dict] = None,
        feature_weights: Optional[dict] = None,
        signal_cap: float = 3.0,
        rebalance: str = "W-FRI",
        medium_variant: str = "legacy",
        short_variant: str = "legacy",
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize TSMOM Multi-Horizon Strategy.
        
        Args:
            symbols: List of symbols to trade (default: None, uses market.universe)
            horizon_weights: Dictionary of horizon weights:
                - long_252: Weight for long-term horizon (default: 0.45)
                - med_84: Weight for medium-term horizon (default: 0.28)
                - short_21: Weight for short-term horizon (default: 0.20)
                - residual_252_21: Weight for residual trend horizon (default: 0.15)
            feature_weights: Dictionary of feature weights per horizon:
                - long: {ret_252, breakout_252, slope_slow}
                - med: {ret_84, breakout_126, slope_med, persistence} (legacy)
                      OR {ret_84, breakout_84, slope_21_84} (canonical)
                - short: {ret_21, breakout_21, slope_fast, reversal} (legacy)
                      OR {ret_21, breakout_21, slope_fast} with equal weights (canonical)
            signal_cap: Maximum absolute signal value (default: 3.0)
            rebalance: Rebalance frequency ("W-FRI" for weekly Friday)
            medium_variant: Medium-term variant ("legacy" or "canonical", default: "legacy")
            short_variant: Short-term variant ("legacy" or "canonical", default: "legacy")
            config_path: Path to strategy configuration file
        """
        self.symbols = symbols
        self.signal_cap = signal_cap
        self.rebalance = rebalance
        self.medium_variant = medium_variant
        self.short_variant = short_variant
        
        # Load config if weights not provided
        config = self._load_config(config_path)
        tsmom_mh_config = config.get('strategies', {}).get('tsmom_multihorizon', {})
        params = tsmom_mh_config.get('params', {})
        
        # Override medium_variant from config if provided
        if 'medium_variant' in params:
            self.medium_variant = params['medium_variant']
        
        # Override short_variant from config if provided
        if 'short_variant' in params:
            self.short_variant = params['short_variant']
        
        if horizon_weights is None:
            horizon_weights = params.get('horizon_weights', {})
        if feature_weights is None:
            feature_weights = params.get('feature_weights', {})
        
        # Default horizon weights (including residual trend as 4th atomic sleeve)
        default_horizon_weights = {
            "long_252": 0.45,
            "med_84": 0.28,
            "short_21": 0.20,
            "residual_252_21": 0.15  # Residual trend: long-horizon minus short-term
        }
        
        if not horizon_weights:
            horizon_weights = default_horizon_weights
        else:
            # Merge with defaults
            for key in default_horizon_weights:
                if key not in horizon_weights:
                    horizon_weights[key] = default_horizon_weights[key]
        
        # Normalize horizon weights to sum to 1.0
        total_horizon_weight = sum(horizon_weights.values())
        if total_horizon_weight > 0:
            self.horizon_weights = {k: v / total_horizon_weight for k, v in horizon_weights.items()}
        else:
            logger.warning("[TSMOMMultiHorizon] All horizon weights are zero, using defaults")
            self.horizon_weights = default_horizon_weights
        
        # Default feature weights
        default_feature_weights = {
            "long": {
                "ret_252": 0.5,
                "breakout_252": 0.3,
                "slope_slow": 0.2
            },
            "med": {
                "ret_84": 0.4,
                "breakout_126": 0.3,
                "slope_med": 0.2,
                "persistence": 0.1
            },
            "short": {
                "ret_21": 0.5,
                "breakout_21": 0.3,
                "slope_fast": 0.2,
                "reversal": 0.0
            },
            "breakout_mid_50_100": {
                "breakout_50": 0.5,
                "breakout_100": 0.5
            }
        }
        
        if not feature_weights:
            feature_weights = default_feature_weights
        else:
            # Merge with defaults
            for horizon in default_feature_weights:
                if horizon not in feature_weights:
                    feature_weights[horizon] = default_feature_weights[horizon].copy()
                else:
                    for key in default_feature_weights[horizon]:
                        if key not in feature_weights[horizon]:
                            feature_weights[horizon][key] = default_feature_weights[horizon][key]
        
        # Normalize feature weights within each horizon
        self.feature_weights = {}
        for horizon, weights in feature_weights.items():
            # Exclude reversal if weight is 0
            active_weights = {k: v for k, v in weights.items() if v > 0 or k == "reversal"}
            total_feature_weight = sum(v for k, v in active_weights.items() if k != "reversal")
            
            if total_feature_weight > 0:
                normalized = {k: v / total_feature_weight if k != "reversal" else v 
                             for k, v in active_weights.items()}
                self.feature_weights[horizon] = normalized
            else:
                logger.warning(f"[TSMOMMultiHorizon] All feature weights are zero for {horizon}, using defaults")
                self.feature_weights[horizon] = default_feature_weights[horizon]
        
        # Load vol normalization config (reuse config already loaded)
        vol_norm_cfg = params.get('vol_normalization', {})
        
        # Vol normalization parameters
        self.vol_norm_enabled = vol_norm_cfg.get('enabled', True)
        self.vol_halflife = vol_norm_cfg.get('halflife_days', 63)
        self.sigma_floor = vol_norm_cfg.get('sigma_floor_annual', 0.05)
        self.risk_scale = vol_norm_cfg.get('risk_scale', 0.2)
        
        # Cache for EWMA vol estimates (updated on each signal call)
        self._ewma_vol_cache = {}  # {symbol: Series of vol estimates}
        self._returns_cache = None  # Cached returns DataFrame
        
        # State tracking
        self._last_rebalance = None
        self._last_signals = None
        self._rebalance_dates = None
        
        logger.info(
            f"[TSMOMMultiHorizon] Initialized with horizon_weights={self.horizon_weights}, "
            f"feature_weights={self.feature_weights}, cap={signal_cap}, rebalance={rebalance}, "
            f"medium_variant={self.medium_variant}, short_variant={self.short_variant}, "
            f"vol_norm_enabled={self.vol_norm_enabled}, halflife={self.vol_halflife}, "
            f"sigma_floor={self.sigma_floor}, risk_scale={self.risk_scale}"
        )
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[TSMOMMultiHorizon] Config not found: {config_path}, using defaults")
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
        logger.debug(f"[TSMOMMultiHorizon] Computed {len(rebalance_dates)} rebalance dates")
        return rebalance_dates
    
    def _compute_ewma_vol(
        self,
        market,
        date: pd.Timestamp,
        symbols: list
    ) -> pd.Series:
        """
        Compute EWMA annualized volatility for each symbol at given date.
        
        Uses EWMA with half-life H = 63 days (default):
        λ = 0.5^(1/H)
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
        
        Args:
            market: MarketData instance
            date: Current date (point-in-time)
            symbols: List of symbols to compute vol for
            
        Returns:
            Series of annualized volatility estimates indexed by symbol
        """
        if not self.vol_norm_enabled:
            # Return unit vol if normalization disabled
            return pd.Series(1.0, index=symbols)
        
        # Get returns up to current date (point-in-time)
        # Use continuous returns for consistency with P&L
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.warning(f"[TSMOMMultiHorizon] No returns data for EWMA vol at {date}")
            return pd.Series(self.sigma_floor, index=symbols)
        
        # Filter to available symbols and dates <= current date
        available_symbols = [s for s in symbols if s in returns_cont.columns]
        if not available_symbols:
            logger.warning(f"[TSMOMMultiHorizon] No matching symbols in returns for EWMA vol at {date}")
            return pd.Series(self.sigma_floor, index=symbols)
        
        returns = returns_cont[available_symbols].copy()
        returns = returns[returns.index <= date]
        
        if returns.empty:
            logger.warning(f"[TSMOMMultiHorizon] No returns data up to {date} for EWMA vol")
            return pd.Series(self.sigma_floor, index=symbols)
        
        # Compute EWMA variance using pandas ewm
        # Half-life in days
        ewm_var = returns.pow(2).ewm(halflife=self.vol_halflife, adjust=False).mean()
        
        # Get latest variance estimate (last row)
        if len(ewm_var) == 0:
            return pd.Series(self.sigma_floor, index=symbols)
        
        latest_var = ewm_var.iloc[-1]
        
        # Convert to daily vol (square root of variance)
        ewm_vol_daily = latest_var.pow(0.5)
        
        # Annualize: multiply by sqrt(252)
        sigma_annual = ewm_vol_daily * np.sqrt(252)
        
        # Apply floor
        sigma_annual = sigma_annual.clip(lower=self.sigma_floor)
        
        # Fill missing symbols with floor
        result = pd.Series(self.sigma_floor, index=symbols)
        result[available_symbols] = sigma_annual[available_symbols]
        
        return result
    
    def fit_in_sample(
        self,
        market,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ):
        """
        Fit strategy in-sample (optional, mostly no-op for TSMOM).
        
        Pre-computes rebalance dates if data is available.
        
        Args:
            market: MarketData instance
            start: Start date for fitting period
            end: End date for fitting period
        """
        logger.info(f"[TSMOMMultiHorizon] fit_in_sample called (pre-computing rebalance dates)")
        
        # Pre-compute rebalance dates if we have data
        symbols = self.symbols if self.symbols is not None else market.universe
        if symbols:
            # Get any price data to determine date range
            prices = market.get_price_panel(symbols=symbols[:1], start=start, end=end, fields=("close",))
            if not prices.empty:
                self._rebalance_dates = self._compute_rebalance_dates(prices.index)
                logger.info(f"[TSMOMMultiHorizon] Pre-computed {len(self._rebalance_dates)} rebalance dates")
    
    def signals(
        self,
        market,
        date: Union[str, datetime, pd.Timestamp],
        features: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.Series:
        """
        Generate multi-horizon momentum signals for a given date.
        
        Args:
            market: MarketData instance (read-only)
            date: Date for signal generation
            features: Optional dict of pre-computed features:
                - "LONG_MOMENTUM": Long-term momentum features
                - "MEDIUM_MOMENTUM": Medium-term momentum features
                - "SHORT_MOMENTUM": Short-term momentum features
                - "RESIDUAL_TREND": Residual trend features (optional, 4th atomic sleeve)
                
        Returns:
            Series of signals indexed by symbol (roughly mean 0, unit variance)
        """
        date_dt = pd.to_datetime(date)
        
        # Get symbols
        symbols = self.symbols if self.symbols is not None else market.universe
        
        if not symbols:
            logger.warning(f"[TSMOMMultiHorizon] No symbols available for date {date_dt}")
            return pd.Series(index=symbols, dtype=float)
        
        # Get features if not provided
        if features is None:
            features = {}
        
        long_features = features.get("LONG_MOMENTUM", pd.DataFrame())
        
        # Select medium-term features based on variant
        if self.medium_variant == "canonical":
            med_features = features.get("CANONICAL_MEDIUM_MOMENTUM", pd.DataFrame())
        else:  # legacy
            med_features = features.get("MEDIUM_MOMENTUM", pd.DataFrame())
        
        short_features = features.get("SHORT_MOMENTUM", pd.DataFrame())
        residual_features = features.get("RESIDUAL_TREND", pd.DataFrame())
        
        if long_features.empty and med_features.empty and short_features.empty and residual_features.empty:
            logger.warning(f"[TSMOMMultiHorizon] No features available for date {date_dt}")
            return pd.Series(0.0, index=symbols)
        
        # Find the appropriate date in features (forward-fill if needed)
        def get_feature_date(features_df, target_date):
            if features_df.empty:
                return None
            if target_date in features_df.index:
                return target_date
            available_dates = features_df.index[features_df.index <= target_date]
            if len(available_dates) == 0:
                return None
            return available_dates[-1]
        
        use_date_long = get_feature_date(long_features, date_dt)
        use_date_med = get_feature_date(med_features, date_dt)
        use_date_short = get_feature_date(short_features, date_dt)
        use_date_residual = get_feature_date(residual_features, date_dt)
        
        if use_date_long is None and use_date_med is None and use_date_short is None and use_date_residual is None:
            logger.warning(f"[TSMOMMultiHorizon] No features available for date {date_dt} (no prior data)")
            return pd.Series(0.0, index=symbols)
        
        # Get feature rows
        long_row = long_features.loc[use_date_long] if use_date_long is not None else pd.Series()
        med_row = med_features.loc[use_date_med] if use_date_med is not None else pd.Series()
        short_row = short_features.loc[use_date_short] if use_date_short is not None else pd.Series()
        residual_row = residual_features.loc[use_date_residual] if use_date_residual is not None else pd.Series()
        
        # Compute signals per symbol
        signals = {}
        
        for symbol in symbols:
            # Long-term signal
            long_signal = 0.0
            if use_date_long is not None and not long_row.empty:
                long_weights = self.feature_weights.get("long", {})
                ret_feature = f"mom_long_ret_252_z_{symbol}"
                breakout_feature = f"mom_long_breakout_252_z_{symbol}"
                slope_feature = f"mom_long_slope_slow_z_{symbol}"
                
                ret_val = long_row.get(ret_feature, np.nan) if ret_feature in long_row.index else np.nan
                breakout_val = long_row.get(breakout_feature, np.nan) if breakout_feature in long_row.index else np.nan
                slope_val = long_row.get(slope_feature, np.nan) if slope_feature in long_row.index else np.nan
                
                long_signal = (
                    long_weights.get("ret_252", 0.0) * (ret_val if pd.notna(ret_val) else 0.0) +
                    long_weights.get("breakout_252", 0.0) * (breakout_val if pd.notna(breakout_val) else 0.0) +
                    long_weights.get("slope_slow", 0.0) * (slope_val if pd.notna(slope_val) else 0.0)
                )
            
            # Medium-term signal (variant-dependent)
            med_signal = 0.0
            if use_date_med is not None and not med_row.empty:
                med_weights = self.feature_weights.get("med", {})
                
                if self.medium_variant == "canonical":
                    # Canonical medium-term features (84d return, 84d breakout, EMA21-84 slope)
                    ret_feature = f"mom_medcanon_ret_84_z_{symbol}"
                    breakout_feature = f"mom_medcanon_breakout_84_z_{symbol}"
                    slope_feature = f"mom_medcanon_slope_21_84_z_{symbol}"
                    
                    ret_val = med_row.get(ret_feature, np.nan) if ret_feature in med_row.index else np.nan
                    breakout_val = med_row.get(breakout_feature, np.nan) if breakout_feature in med_row.index else np.nan
                    slope_val = med_row.get(slope_feature, np.nan) if slope_feature in med_row.index else np.nan
                    
                    # Canonical uses equal weights (1/3, 1/3, 1/3) or configured weights
                    # Map legacy feature weight keys to canonical features
                    med_signal = (
                        med_weights.get("ret_84", 0.333333) * (ret_val if pd.notna(ret_val) else 0.0) +
                        med_weights.get("breakout_84", med_weights.get("breakout_126", 0.333333)) * (breakout_val if pd.notna(breakout_val) else 0.0) +
                        med_weights.get("slope_21_84", med_weights.get("slope_med", 0.333334)) * (slope_val if pd.notna(slope_val) else 0.0)
                    )
                else:
                    # Legacy medium-term features (84d return, 126d breakout, EMA20-84 slope, persistence)
                    ret_feature = f"mom_med_ret_84_z_{symbol}"
                    breakout_feature = f"mom_med_breakout_126_z_{symbol}"
                    slope_feature = f"mom_med_slope_med_z_{symbol}"
                    persist_feature = f"mom_med_persistence_z_{symbol}"
                    
                    ret_val = med_row.get(ret_feature, np.nan) if ret_feature in med_row.index else np.nan
                    breakout_val = med_row.get(breakout_feature, np.nan) if breakout_feature in med_row.index else np.nan
                    slope_val = med_row.get(slope_feature, np.nan) if slope_feature in med_row.index else np.nan
                    persist_val = med_row.get(persist_feature, np.nan) if persist_feature in med_row.index else np.nan
                    
                    med_signal = (
                        med_weights.get("ret_84", 0.0) * (ret_val if pd.notna(ret_val) else 0.0) +
                        med_weights.get("breakout_126", 0.0) * (breakout_val if pd.notna(breakout_val) else 0.0) +
                        med_weights.get("slope_med", 0.0) * (slope_val if pd.notna(slope_val) else 0.0) +
                        med_weights.get("persistence", 0.0) * (persist_val if pd.notna(persist_val) else 0.0)
                    )
            
            # Short-term signal (variant-dependent)
            short_signal = 0.0
            if use_date_short is not None and not short_row.empty:
                short_weights = self.feature_weights.get("short", {})
                ret_feature = f"mom_short_ret_21_z_{symbol}"
                breakout_feature = f"mom_short_breakout_21_z_{symbol}"
                slope_feature = f"mom_short_slope_fast_z_{symbol}"
                reversal_feature = f"mom_short_reversal_filter_z_{symbol}"
                
                ret_val = short_row.get(ret_feature, np.nan) if ret_feature in short_row.index else np.nan
                breakout_val = short_row.get(breakout_feature, np.nan) if breakout_feature in short_row.index else np.nan
                slope_val = short_row.get(slope_feature, np.nan) if slope_feature in short_row.index else np.nan
                reversal_val = short_row.get(reversal_feature, np.nan) if reversal_feature in short_row.index else np.nan
                
                if self.short_variant == "canonical":
                    # Canonical short-term: equal weights (1/3, 1/3, 1/3)
                    short_signal = (
                        short_weights.get("ret_21", 0.333333) * (ret_val if pd.notna(ret_val) else 0.0) +
                        short_weights.get("breakout_21", 0.333333) * (breakout_val if pd.notna(breakout_val) else 0.0) +
                        short_weights.get("slope_fast", 0.333334) * (slope_val if pd.notna(slope_val) else 0.0)
                    )
                else:
                    # Legacy short-term: configured weights (default 0.5, 0.3, 0.2)
                    short_signal = (
                        short_weights.get("ret_21", 0.0) * (ret_val if pd.notna(ret_val) else 0.0) +
                        short_weights.get("breakout_21", 0.0) * (breakout_val if pd.notna(breakout_val) else 0.0) +
                        short_weights.get("slope_fast", 0.0) * (slope_val if pd.notna(slope_val) else 0.0)
                    )
                
                # Reversal filter is not used in signal combination (weight = 0.0)
                # But we load it in case we want to use it later
            
            # Residual trend signal (4th atomic sleeve)
            residual_signal = 0.0
            if use_date_residual is not None and not residual_row.empty:
                # Residual trend feature is already z-scored and clipped in ResidualTrendFeatures
                # Feature name format: trend_resid_ret_252_21_z_{symbol}
                residual_feature = f"trend_resid_ret_252_21_z_{symbol}"
                residual_val = residual_row.get(residual_feature, np.nan) if residual_feature in residual_row.index else np.nan
                residual_signal = residual_val if pd.notna(residual_val) else 0.0
            
            # Breakout mid (50-100d) signal (5th atomic sleeve)
            breakout_mid_signal = 0.0
            if use_date_long is not None and not long_row.empty:
                # Breakout mid features are computed in LongMomentumFeatures
                # Feature names: mom_breakout_mid_50_z_{symbol}, mom_breakout_mid_100_z_{symbol}
                breakout_weights = self.feature_weights.get("breakout_mid_50_100", {})
                breakout_50_feature = f"mom_breakout_mid_50_z_{symbol}"
                breakout_100_feature = f"mom_breakout_mid_100_z_{symbol}"
                
                breakout_50_val = long_row.get(breakout_50_feature, np.nan) if breakout_50_feature in long_row.index else np.nan
                breakout_100_val = long_row.get(breakout_100_feature, np.nan) if breakout_100_feature in long_row.index else np.nan
                
                breakout_mid_signal = (
                    breakout_weights.get("breakout_50", 0.0) * (breakout_50_val if pd.notna(breakout_50_val) else 0.0) +
                    breakout_weights.get("breakout_100", 0.0) * (breakout_100_val if pd.notna(breakout_100_val) else 0.0)
                )
            
            # Horizon blend (5 atomic sleeves: long, med, short, residual, breakout_mid)
            raw_signal = (
                self.horizon_weights.get("long_252", 0.0) * long_signal +
                self.horizon_weights.get("med_84", 0.0) * med_signal +
                self.horizon_weights.get("short_21", 0.0) * short_signal +
                self.horizon_weights.get("residual_252_21", 0.0) * residual_signal +
                self.horizon_weights.get("breakout_mid_50_100", 0.0) * breakout_mid_signal
            )
            
            signals[symbol] = raw_signal
        
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
        
        # Apply EWMA vol normalization if enabled
        if self.vol_norm_enabled:
            # Compute EWMA vol for all symbols at this date
            sigma_annual = self._compute_ewma_vol(market, date_dt, symbols)
            
            # Risk-normalize: divide by vol (with floor protection)
            # s_risk = z_clipped / max(sigma, sigma_floor)
            denom = np.maximum(sigma_annual, self.sigma_floor)
            s_risk = signal_series / denom
            
            # Apply global scale factor
            signal_series = self.risk_scale * s_risk
            
            logger.debug(
                f"[TSMOMMultiHorizon] Applied EWMA vol normalization at {date_dt}: "
                f"avg_vol={sigma_annual.mean():.3f}, "
                f"signal_mean={signal_series.mean():.3f}, "
                f"signal_std={signal_series.std():.3f}"
            )
        
        logger.debug(
            f"[TSMOMMultiHorizon] Generated signals at {date_dt}: "
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
            'strategy': 'TSMOMMultiHorizon',
            'horizon_weights': self.horizon_weights,
            'feature_weights': self.feature_weights,
            'signal_cap': self.signal_cap,
            'rebalance': self.rebalance,
            'medium_variant': self.medium_variant,
            'short_variant': self.short_variant,
            'vol_norm_enabled': self.vol_norm_enabled,
            'vol_halflife': self.vol_halflife,
            'sigma_floor': self.sigma_floor,
            'risk_scale': self.risk_scale,
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
        logger.debug("[TSMOMMultiHorizon] State reset")

