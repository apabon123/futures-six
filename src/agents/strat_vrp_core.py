"""
VRP Core Strategy: Volatility Risk Premium directional strategy.

Strategy:
- Signal: Z-scored VRP spread (VIX - realized ES vol)
- Instrument: VX1 front month futures (directional)
- Position: Long VX when VRP is low (vol is cheap)
         Short VX when VRP is high (vol is expensive)

Rationale:
- When VRP > 0 (VIX > realized vol): implied vol is expensive → short vol (fade the premium)
- When VRP < 0 (VIX < realized vol): implied vol is cheap → long vol (fade mean reversion)
- Z-scoring provides consistent signal strength across regimes

Phase-1 Implementation:
- Uses VRPCoreFeatures for feature engineering
- Directional signals in [-1, 1] based on z-scored VRP
- Trades VX1 front month futures only
- No cross-sectional ranking (single-asset directional strategy)
"""

import logging
from typing import Optional, Union, Sequence
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.agents.feature_vrp_core import VRPCoreFeatures

logger = logging.getLogger(__name__)


@dataclass
class VRPCoreConfig:
    """Configuration for VRP Core strategy."""
    rv_lookback: int = 21          # Realized vol lookback (days)
    zscore_window: int = 252       # Z-score rolling window (days)
    clip: float = 3.0              # Z-score clipping bounds
    signal_mode: str = "zscore"    # "zscore" or "tanh" (squash via hyperbolic tangent)
    db_path: Optional[str] = None  # Path to canonical DuckDB


class VRPCorePhase1:
    """
    Phase-1 VRP Core strategy.
    
    Directional volatility strategy trading VX1 based on VRP spread.
    Returns signals in [-1, 1] for VX1.
    """
    
    def __init__(self, config: VRPCoreConfig):
        """
        Initialize Phase-1 VRP Core strategy.
        
        Args:
            config: VRPCoreConfig with rv_lookback, zscore_window, etc.
        """
        self.config = config
        logger.info(
            f"[VRPCorePhase1] Initialized with rv_lookback={config.rv_lookback}, "
            f"zscore_window={config.zscore_window}, clip={config.clip}, "
            f"signal_mode={config.signal_mode}"
        )
    
    def compute_signals(
        self,
        market,
        start: str,
        end: str
    ) -> pd.Series:
        """
        Compute Phase-1 VRP core signals.
        
        Returns Series with index=date, values in [-1, 1] for VX1 position.
        
        **Signal Logic:**
        - Positive vrp_z (VRP > average) → Short vol → Negative signal
        - Negative vrp_z (VRP < average) → Long vol → Positive signal
        - This is mean-reversion: fade extremes in VRP
        
        Args:
            market: MarketData instance
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            Series of signals indexed by date with values in [-1, 1]
        """
        logger.info(f"[VRPCorePhase1] Computing signals from {start} to {end}")
        
        # 1) Compute VRP features
        features_calc = VRPCoreFeatures(
            rv_lookback=self.config.rv_lookback,
            zscore_window=self.config.zscore_window,
            clip=self.config.clip,
            db_path=self.config.db_path
        )
        
        features = features_calc.compute(market, start_date=start, end_date=end)
        
        if features.empty:
            logger.warning("[VRPCorePhase1] No VRP features available")
            return pd.Series(dtype=float, name='signal')
        
        logger.info(f"[VRPCorePhase1] Features shape: {features.shape}")
        
        # 2) Extract z-scored VRP signal
        if 'vrp_z' not in features.columns:
            logger.error("[VRPCorePhase1] vrp_z column not found in features")
            return pd.Series(dtype=float, name='signal')
        
        vrp_z = features['vrp_z'].copy()
        
        # 3) Generate directional signal
        # **Mean-reversion logic**: Fade extreme VRP
        # When VRP is high (positive z-score) → vol is expensive → short vol (negative signal)
        # When VRP is low (negative z-score) → vol is cheap → long vol (positive signal)
        signal = -vrp_z  # Invert: fade high VRP, buy low VRP
        
        # 4) Optional signal transformation
        if self.config.signal_mode == "tanh":
            # Squash to [-1, 1] via hyperbolic tangent
            # This creates smoother transitions than hard clipping
            signal = np.tanh(signal)
            logger.debug("[VRPCorePhase1] Applied tanh signal transformation")
        elif self.config.signal_mode == "zscore":
            # Already in [-clip, clip] from feature computation
            # Scale to [-1, 1] by dividing by clip
            signal = signal / self.config.clip
            signal = signal.clip(-1.0, 1.0)
        else:
            logger.warning(f"[VRPCorePhase1] Unknown signal_mode: {self.config.signal_mode}")
            signal = signal.clip(-1.0, 1.0)
        
        # 5) Set signal name
        signal.name = 'signal'
        
        logger.info(
            f"[VRPCorePhase1] Generated signals: n={len(signal)}, "
            f"mean={signal.mean():.3f}, std={signal.std():.3f}, "
            f"range=[{signal.min():.3f}, {signal.max():.3f}]"
        )
        
        # Log signal distribution
        pct_long = (signal > 0.1).sum() / len(signal) * 100
        pct_short = (signal < -0.1).sum() / len(signal) * 100
        pct_neutral = ((signal >= -0.1) & (signal <= 0.1)).sum() / len(signal) * 100
        logger.info(
            f"[VRPCorePhase1] Signal distribution: "
            f"{pct_long:.1f}% long, {pct_short:.1f}% short, {pct_neutral:.1f}% neutral"
        )
        
        return signal


class VRPCoreMeta:
    """
    VRP Core Meta-Sleeve wrapper for CombinedStrategy integration.
    
    Wraps VRPCorePhase1 and provides a date-by-date signals() method
    compatible with CombinedStrategy.
    """
    
    def __init__(
        self,
        rv_lookback: int = 21,
        zscore_window: int = 252,
        clip: float = 3.0,
        signal_mode: str = "zscore",
        db_path: Optional[str] = None
    ):
        """
        Initialize VRP Core Meta-Sleeve.
        
        Args:
            rv_lookback: Realized vol lookback (days)
            zscore_window: Z-score rolling window (days)
            clip: Z-score clipping bounds
            signal_mode: "zscore" or "tanh"
            db_path: Path to canonical DuckDB
        """
        self.config = VRPCoreConfig(
            rv_lookback=rv_lookback,
            zscore_window=zscore_window,
            clip=clip,
            signal_mode=signal_mode,
            db_path=db_path
        )
        
        self.phase1 = VRPCorePhase1(self.config)
        self._signals_cache = None
        
        logger.info("[VRPCoreMeta] Initialized VRP Core Meta-Sleeve")
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        universe: Optional[Sequence[str]] = None
    ) -> pd.Series:
        """
        Get VRP Core signals for a specific date (compatible with CombinedStrategy).
        
        **Note**: VRP Core trades VX1 only, not the main futures universe.
        The returned signal is for VX1 front month futures.
        
        Args:
            market: MarketData instance
            date: Date to get signals for
            universe: Ignored (VRP trades VX1, not universe assets)
            
        Returns:
            Series with single entry: VX1 signal at specified date
        """
        # Lazy-load signals cache
        if self._signals_cache is None:
            # Compute all signals once
            # TODO: Determine appropriate start date based on warmup requirements
            # For now, use a long history to ensure sufficient warmup
            start = "2010-01-01"  # VIX3M starts 2009-09-18, add buffer for RV warmup
            end = pd.Timestamp.today().strftime('%Y-%m-%d')
            
            logger.info(f"[VRPCoreMeta] Computing signals from {start} to {end}")
            self._signals_cache = self.phase1.compute_signals(market, start, end)
        
        # Get signal for requested date
        date_ts = pd.to_datetime(date)
        
        if date_ts not in self._signals_cache.index:
            logger.warning(f"[VRPCoreMeta] No signal available for {date}")
            return pd.Series(dtype=float)
        
        signal_value = self._signals_cache.loc[date_ts]
        
        # Return as Series with VX1 symbol
        # NOTE: VX1 symbol in canonical DB is '@VX=101XN'
        # For integration with portfolio, we use a standard naming convention
        return pd.Series({'VX1': signal_value})
    
    def warmup_periods(self) -> int:
        """
        Return number of trading days required for warmup.
        
        VRP Core requires:
        - rv_lookback days for realized vol (21)
        - zscore_window days for z-score standardization (252)
        - Total: 252 + 21 = 273 days
        """
        return self.config.zscore_window + self.config.rv_lookback

