"""
VRP Convergence Strategy: VIX vs VX1 convergence directional strategy.

Strategy:
- Signal: Z-scored convergence spread (VIX - VX1)
- Instrument: VX1 front month futures (directional)
- Position: Long VX1 when VX1 is too low vs VIX (expect convergence up)
         Short VX1 when VX1 is too high vs VIX (expect convergence down)

Rationale:
- When VX1 > VIX (VX1 too rich): expect convergence lower → short VX1
- When VX1 < VIX (VX1 too cheap): expect convergence higher → long VX1
- Z-scoring provides consistent signal strength across regimes

Phase-1 Implementation:
- Uses VRPConvergenceFeatures for feature engineering
- Directional signals in [-1, 1] based on z-scored convergence spread
- Trades VX1 front month futures only
- Volatility-targeted position sizing
"""

import logging
from typing import Optional, Union, Sequence
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.agents.feature_vrp_convergence import VRPConvergenceFeatures

logger = logging.getLogger(__name__)


@dataclass
class VRPConvergenceConfig:
    """Configuration for VRP Convergence strategy."""
    zscore_window: int = 252       # Z-score rolling window (days)
    clip: float = 3.0              # Z-score clipping bounds
    signal_mode: str = "zscore"    # Signal transformation: "zscore" or "tanh"
    target_vol: float = 0.10       # Target annualized volatility (10%)
    vol_lookback: int = 63         # Volatility lookback for vol targeting (days)
    vol_floor: float = 0.05         # Minimum volatility floor (5% annualized)
    db_path: Optional[str] = None  # Path to canonical DuckDB


class VRPConvergencePhase1:
    """
    Phase-1 VRP Convergence strategy.
    
    Directional volatility strategy trading VX1 based on VIX-VX1 convergence spread.
    Returns signals in [-1, 1] for VX1.
    """
    
    def __init__(self, config: VRPConvergenceConfig):
        """
        Initialize Phase-1 VRP Convergence strategy.
        
        Args:
            config: VRPConvergenceConfig with zscore_window, clip, etc.
        """
        self.config = config
        logger.info(
            f"[VRPConvergencePhase1] Initialized with zscore_window={config.zscore_window}, "
            f"clip={config.clip}, signal_mode={config.signal_mode}, "
            f"target_vol={config.target_vol}"
        )
    
    def compute_signals(
        self,
        market,
        start: str,
        end: str
    ) -> pd.Series:
        """
        Compute Phase-1 VRP convergence signals.
        
        Returns Series with index=date, values in [-1, 1] for VX1 position.
        
        **Signal Logic:**
        - Positive conv_z (spread > average) → VX1 too cheap → Long VX1 (positive signal)
        - Negative conv_z (spread < average) → VX1 too rich → Short VX1 (negative signal)
        - This is mean-reversion on convergence spread
        
        Args:
            market: MarketData instance
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            Series of signals indexed by date with values in [-1, 1]
        """
        logger.info(f"[VRPConvergencePhase1] Computing signals from {start} to {end}")
        
        # 1) Compute VRP convergence features
        features_calc = VRPConvergenceFeatures(
            zscore_window=self.config.zscore_window,
            clip=self.config.clip,
            db_path=self.config.db_path
        )
        
        features = features_calc.compute(market, start_date=start, end_date=end)
        
        if features.empty:
            logger.warning("[VRPConvergencePhase1] No features computed")
            return pd.Series(dtype=float, name='signal')
        
        logger.info(f"[VRPConvergencePhase1] Features shape: {features.shape}")
        
        # 2) Extract z-scored convergence signal
        if 'conv_z' not in features.columns:
            logger.error("[VRPConvergencePhase1] conv_z column not found in features")
            return pd.Series(dtype=float, name='signal')
        
        conv_z = features['conv_z'].copy()
        
        # 3) Generate short-only signal (matching Phase-0 economics)
        # **Convergence logic**: Only trade the economically valid side (negative spreads)
        # spread_conv = VIX - VX1
        # Under contango: VX1 > VIX, so spread_conv < 0 (negative)
        # Phase-0 rule: if spread_conv < -T → short VX1
        # Phase-1: Only use negative conv_z (negative spread_conv) → short VX1
        # Positive conv_z (positive spread_conv, VIX >= VX1) → flat (no long signal)
        conv_z_neg = np.minimum(conv_z, 0.0)  # Clip positive z-scores to 0 (short-only)
        
        # 4) Signal transformation (short-only, bounded in [-1, 0])
        if self.config.signal_mode == "tanh":
            # Short-only, smoothed, bounded in [-1, 0]
            signal_raw = conv_z_neg
            signal = np.tanh(signal_raw / 2.0)  # Divide by 2.0 to get reasonable scaling
            # Ensure signal stays in [-1, 0] (short-only)
            signal = np.minimum(signal, 0.0)
            logger.debug("[VRPConvergencePhase1] Applied tanh signal transformation (short-only)")
        elif self.config.signal_mode == "zscore":
            # Short-only, linearly scaled, bounded in [-1, 0]
            signal_raw = conv_z_neg
            signal = signal_raw / self.config.clip
            signal = np.clip(signal, -1.0, 0.0)  # Ensure short-only
        else:
            raise ValueError(f"[VRPConvergencePhase1] Unsupported signal_mode: {self.config.signal_mode}")
        
        # 5) Set signal name
        signal.name = 'signal'
        
        logger.info(
            f"[VRPConvergencePhase1] Generated signals: n={len(signal)}, "
            f"mean={signal.mean():.3f}, std={signal.std():.3f}, "
            f"range=[{signal.min():.3f}, {signal.max():.3f}]"
        )
        
        # Log signal distribution (short-only: should have no long signals)
        pct_short = (signal < -0.01).sum() / len(signal) * 100  # Short (signal < -0.01)
        pct_flat = ((signal >= -0.01) & (signal <= 0.01)).sum() / len(signal) * 100  # Flat (near zero)
        pct_long = (signal > 0.01).sum() / len(signal) * 100  # Long (should be 0 for short-only)
        
        # Assert short-only constraint
        if pct_long > 0.1:  # Allow tiny numerical errors
            logger.warning(f"[VRPConvergencePhase1] WARNING: {pct_long:.2f}% long signals detected (should be ~0% for short-only)")
        
        logger.info(
            f"[VRPConvergencePhase1] Signal distribution: "
            f"{pct_short:.1f}% short, {pct_flat:.1f}% flat, {pct_long:.1f}% long (should be ~0%)"
        )
        
        return signal
    
    def compute_positions(
        self,
        signals: pd.Series,
        vx1_returns: pd.Series
    ) -> pd.Series:
        """
        Compute volatility-targeted positions from signals.
        
        Uses 63-day rolling volatility (simple std or EWMA) with vol floor.
        Position = signal * (target_vol / realized_vol)
        
        Args:
            signals: Raw signals in [-1, 1]
            vx1_returns: VX1 daily returns for vol calculation
            
        Returns:
            Series of positions (vol-targeted) indexed by date
        """
        # Align signals and returns
        common_idx = signals.index.intersection(vx1_returns.index)
        if len(common_idx) == 0:
            logger.warning("[VRPConvergencePhase1] No overlapping dates for vol targeting")
            return pd.Series(dtype=float, name='position')
        
        signals_aligned = signals.loc[common_idx]
        vx1_rets_aligned = vx1_returns.loc[common_idx]
        
        # Compute rolling volatility of VX1 returns
        # Use simple rolling std (63-day lookback) - matching VRP-Core approach
        vol_annual = vx1_rets_aligned.rolling(
            window=self.config.vol_lookback,
            min_periods=self.config.vol_lookback
        ).std() * np.sqrt(252)  # Annualize
        
        # Apply vol floor
        vol_annual = vol_annual.clip(lower=self.config.vol_floor)
        
        # Volatility targeting: position = signal * (target_vol / realized_vol)
        # Target vol: 10% annualized (0.10)
        positions = signals_aligned * (self.config.target_vol / vol_annual)
        
        # Clip positions to reasonable bounds (e.g., ±2x notional)
        positions = positions.clip(-2.0, 2.0)
        
        positions.name = 'position'
        
        logger.info(
            f"[VRPConvergencePhase1] Computed positions: mean_vol={vol_annual.mean():.4f}, "
            f"mean_position={positions.abs().mean():.4f}"
        )
        
        return positions


class VRPConvergenceMeta:
    """
    VRP Convergence Meta-Sleeve wrapper for CombinedStrategy integration.
    
    Wraps VRPConvergencePhase1 and provides a date-by-date signals() method
    compatible with CombinedStrategy.
    """
    
    def __init__(
        self,
        zscore_window: int = 252,
        clip: float = 3.0,
        signal_mode: str = "zscore",
        target_vol: float = 0.10,
        vol_lookback: int = 63,
        vol_floor: float = 0.05,
        db_path: Optional[str] = None
    ):
        """
        Initialize VRP Convergence Meta-Sleeve.
        
        Args:
            zscore_window: Z-score rolling window (days)
            clip: Z-score clipping bounds
            signal_mode: "zscore" or "tanh"
            target_vol: Target annualized volatility (default: 0.10 = 10%)
            vol_lookback: Volatility lookback for vol targeting (days)
            vol_floor: Minimum volatility floor (default: 0.05 = 5%)
            db_path: Path to canonical DuckDB
        """
        self.config = VRPConvergenceConfig(
            zscore_window=zscore_window,
            clip=clip,
            signal_mode=signal_mode,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            vol_floor=vol_floor,
            db_path=db_path
        )
        
        self.phase1 = VRPConvergencePhase1(self.config)
        self._signals_cache = None
        
        logger.info("[VRPConvergenceMeta] Initialized VRP Convergence Meta-Sleeve")
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        universe: Optional[Sequence[str]] = None
    ) -> pd.Series:
        """
        Get VRP Convergence signals for a specific date (compatible with CombinedStrategy).
        
        **Note**: VRP Convergence trades VX1 only, not the main futures universe.
        The returned signal is for VX1 front month futures.
        
        Args:
            market: MarketData instance
            date: Date to get signals for
            universe: Ignored (VRP Convergence trades VX1, not universe assets)
            
        Returns:
            Series with single entry: VX1 signal at specified date
        """
        # Lazy-load signals cache
        if self._signals_cache is None:
            # Compute all signals once
            # TODO: Determine appropriate start date based on warmup requirements
            # For now, use a long history to ensure sufficient warmup
            start = "2010-01-01"  # VIX3M starts 2009-09-18, add buffer for warmup
            end = pd.Timestamp.today().strftime('%Y-%m-%d')
            
            logger.info(f"[VRPConvergenceMeta] Computing signals from {start} to {end}")
            self._signals_cache = self.phase1.compute_signals(market, start, end)
        
        # Get signal for requested date
        date_ts = pd.to_datetime(date)
        
        if date_ts not in self._signals_cache.index:
            logger.warning(f"[VRPConvergenceMeta] No signal available for {date}")
            return pd.Series(dtype=float)
        
        signal_value = self._signals_cache.loc[date_ts]
        
        # Return as Series with VX1 symbol
        # NOTE: VX1 symbol in canonical DB is '@VX=101XN'
        # For integration with portfolio, we use a standard naming convention
        return pd.Series({'VX1': signal_value})
    
    def warmup_periods(self) -> int:
        """
        Return warmup period required for this strategy.
        
        Returns:
            Number of trading days needed for warmup (zscore_window)
        """
        return self.config.zscore_window

