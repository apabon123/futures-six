"""
VolManagedOverlay: Volatility-Managed Position Scaling Agent

Scales raw strategy signals to achieve a target volatility level.
Uses RiskVol service to compute ex-ante portfolio or per-asset volatility,
then scales positions accordingly while respecting leverage caps and bounds.

No data writes. No look-ahead. Deterministic outputs given MarketData snapshot.
"""

import logging
from typing import Union, Optional
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VolManagedOverlay:
    """
    Volatility-managed overlay for scaling strategy signals.
    
    Takes raw signals from a strategy and scales them to achieve a target
    annualized volatility, using either:
    - "global" mode: portfolio volatility from signals + covariance matrix
    - "per-asset" mode: individual asset volatilities
    
    Enforces leverage caps and position bounds to ensure risk controls.
    """
    
    def __init__(
        self,
        risk_vol,
        target_vol: float = 0.20,
        lookback_vol: Optional[int] = None,
        floor_vol: float = 0.05,
        cap_leverage: float = 7.0,
        leverage_mode: str = "global",
        position_bounds: tuple = (-3.0, 3.0),
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize VolManagedOverlay agent.
        
        Args:
            risk_vol: RiskVol instance for volatility/covariance calculations
            target_vol: Target annualized volatility (default: 0.20 = 20%)
            lookback_vol: Override RiskVol's vol_lookback if provided
            floor_vol: Minimum volatility floor to avoid extreme scaling (default: 0.05 = 5%)
            cap_leverage: Maximum sum of absolute weights (default: 7.0)
            leverage_mode: "global" or "per-asset" (default: "global")
            position_bounds: Min/max position size in risk units (default: [-3.0, 3.0])
            config_path: Path to configuration YAML file
        """
        self.risk_vol = risk_vol
        
        # Track which parameters were explicitly provided (not defaults)
        # by checking if config should be used
        defaults = {
            'target_vol': 0.20,
            'lookback_vol': None,
            'floor_vol': 0.05,
            'cap_leverage': 7.0,
            'leverage_mode': "global",
            'position_bounds': (-3.0, 3.0)
        }
        
        # Try to load from config if exists
        config = self._load_config(config_path)
        
        # Load from config only for parameters that match defaults
        if config and 'vol_overlay' in config:
            vol_config = config['vol_overlay']
            self.target_vol = vol_config.get('target_vol', target_vol) if target_vol == defaults['target_vol'] else target_vol
            self.lookback_vol = vol_config.get('lookback_vol', lookback_vol) if lookback_vol == defaults['lookback_vol'] else lookback_vol
            self.floor_vol = vol_config.get('floor_vol', floor_vol) if floor_vol == defaults['floor_vol'] else floor_vol
            self.cap_leverage = vol_config.get('cap_leverage', cap_leverage) if cap_leverage == defaults['cap_leverage'] else cap_leverage
            self.leverage_mode = vol_config.get('leverage_mode', leverage_mode) if leverage_mode == defaults['leverage_mode'] else leverage_mode
            
            # Handle position_bounds (list in config, tuple in code)
            if position_bounds == defaults['position_bounds']:
                position_bounds_config = vol_config.get('position_bounds', position_bounds)
                self.position_bounds = tuple(position_bounds_config) if isinstance(position_bounds_config, list) else position_bounds
            else:
                self.position_bounds = position_bounds
        else:
            self.target_vol = target_vol
            self.lookback_vol = lookback_vol
            self.floor_vol = floor_vol
            self.cap_leverage = cap_leverage
            self.leverage_mode = leverage_mode
            self.position_bounds = position_bounds
        
        # Validate parameters
        if self.target_vol <= 0:
            raise ValueError(f"target_vol must be > 0, got {self.target_vol}")
        
        if self.floor_vol <= 0:
            raise ValueError(f"floor_vol must be > 0, got {self.floor_vol}")
        
        if self.cap_leverage <= 0:
            raise ValueError(f"cap_leverage must be > 0, got {self.cap_leverage}")
        
        if self.leverage_mode not in ("global", "per-asset"):
            raise ValueError(f"leverage_mode must be 'global' or 'per-asset', got {self.leverage_mode}")
        
        if len(self.position_bounds) != 2 or self.position_bounds[0] > self.position_bounds[1]:
            raise ValueError(f"position_bounds must be [min, max] with min <= max, got {self.position_bounds}")
        
        logger.info(
            f"[VolManagedOverlay] Initialized: target_vol={self.target_vol:.2%}, "
            f"floor_vol={self.floor_vol:.2%}, cap_leverage={self.cap_leverage}, "
            f"leverage_mode={self.leverage_mode}, position_bounds={self.position_bounds}"
        )
    
    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[VolManagedOverlay] Config file not found: {config_path}, using defaults")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"[VolManagedOverlay] Failed to load config: {e}, using defaults")
            return None
    
    def _compute_portfolio_vol(
        self,
        signals: pd.Series,
        cov: pd.DataFrame
    ) -> float:
        """
        Compute portfolio volatility given signals and covariance matrix.
        
        Portfolio vol = sqrt(w' Σ w) where w are the signals (treated as weights).
        
        Args:
            signals: Signal values (treated as portfolio weights)
            cov: Covariance matrix (annualized)
            
        Returns:
            Portfolio volatility (annualized)
        """
        # Align signals with covariance matrix
        common = signals.index.intersection(cov.index)
        
        if len(common) == 0:
            logger.warning("[VolManagedOverlay] No common symbols between signals and covariance")
            return self.floor_vol
        
        w = signals.loc[common].fillna(0).values
        cov_mat = cov.loc[common, common].values
        
        # Portfolio variance: w' Σ w
        port_var = w @ cov_mat @ w
        
        # Ensure non-negative (numerical precision)
        port_var = max(port_var, 0)
        
        # Portfolio vol
        port_vol = np.sqrt(port_var)
        
        return port_vol
    
    def _scale_global(
        self,
        signals: pd.Series,
        market,
        date: Union[str, datetime]
    ) -> pd.Series:
        """
        Scale signals using global portfolio volatility.
        
        Compute portfolio vol from signals + covariance matrix, then scale
        all signals uniformly to achieve target volatility.
        
        Args:
            signals: Raw strategy signals
            market: MarketData instance
            date: Current date
            
        Returns:
            Scaled signals
        """
        # Get covariance matrix
        try:
            cov = self.risk_vol.covariance(market, date, signals=signals)
        except Exception as e:
            logger.warning(f"[VolManagedOverlay] Could not compute covariance: {e}")
            # Fallback: return signals unscaled but bounded
            return signals.clip(lower=self.position_bounds[0], upper=self.position_bounds[1])
        
        # Filter to non-zero signals for portfolio vol calculation
        active_signals = signals[signals.abs() > 1e-10]
        
        if len(active_signals) == 0:
            logger.debug("[VolManagedOverlay] No active signals, returning zeros")
            return signals * 0
        
        # Compute current portfolio vol
        current_vol = self._compute_portfolio_vol(active_signals, cov)
        
        # Apply floor
        current_vol = max(current_vol, self.floor_vol)
        
        # Calculate scale factor: target / current
        scale = self.target_vol / current_vol
        
        logger.debug(
            f"[VolManagedOverlay] Global mode: current_vol={current_vol:.2%}, "
            f"target_vol={self.target_vol:.2%}, scale={scale:.3f}"
        )
        
        # Scale signals
        scaled = signals * scale
        
        return scaled
    
    def _scale_per_asset(
        self,
        signals: pd.Series,
        market,
        date: Union[str, datetime]
    ) -> pd.Series:
        """
        Scale signals using per-asset volatilities.
        
        Each signal is divided by its asset's volatility, then multiplied
        by target_vol to normalize to target risk level.
        
        Args:
            signals: Raw strategy signals
            market: MarketData instance
            date: Current date
            
        Returns:
            Scaled signals
        """
        # Get volatilities
        try:
            vols = self.risk_vol.vols(market, date, signals=signals)
        except Exception as e:
            logger.warning(f"[VolManagedOverlay] Could not compute vols: {e}")
            # Fallback: return signals unscaled but bounded
            return signals.clip(lower=self.position_bounds[0], upper=self.position_bounds[1])
        
        # Align signals with vols
        common = signals.index.intersection(vols.index)
        
        if len(common) == 0:
            logger.warning("[VolManagedOverlay] No common symbols between signals and vols")
            return signals * 0
        
        # Apply floor to vols
        vols_floored = vols.clip(lower=self.floor_vol)
        
        # Scale: signal * (target_vol / asset_vol)
        # This gives each asset equal risk contribution at target_vol
        scale_factors = self.target_vol / vols_floored
        
        scaled = pd.Series(index=signals.index, dtype=float)
        scaled.loc[common] = signals.loc[common] * scale_factors.loc[common]
        scaled = scaled.fillna(0)
        
        logger.debug(
            f"[VolManagedOverlay] Per-asset mode: scale_factors "
            f"mean={scale_factors.mean():.3f}, std={scale_factors.std():.3f}"
        )
        
        return scaled
    
    def _apply_constraints(self, scaled_signals: pd.Series) -> pd.Series:
        """
        Apply position bounds and leverage cap.
        
        1. Clip to position_bounds
        2. If sum(abs(weights)) > cap_leverage, scale down proportionally
        
        Args:
            scaled_signals: Scaled signals before constraints
            
        Returns:
            Constrained signals
        """
        # Step 1: Apply position bounds
        bounded = scaled_signals.clip(
            lower=self.position_bounds[0],
            upper=self.position_bounds[1]
        )
        
        # Step 2: Apply leverage cap
        gross_leverage = bounded.abs().sum()
        
        if gross_leverage > self.cap_leverage:
            # Scale down proportionally
            leverage_scale = self.cap_leverage / gross_leverage
            bounded = bounded * leverage_scale
            
            logger.debug(
                f"[VolManagedOverlay] Leverage cap applied: "
                f"gross={gross_leverage:.2f} > cap={self.cap_leverage}, "
                f"scale={leverage_scale:.3f}"
            )
        
        return bounded
    
    def scale(
        self,
        signals: pd.Series,
        market,
        date: Union[str, datetime]
    ) -> pd.Series:
        """
        Scale raw strategy signals to achieve target volatility.
        
        Args:
            signals: Raw signals from strategy (pd.Series indexed by symbol)
            market: MarketData instance
            date: Current date for vol/cov calculation
            
        Returns:
            Scaled signals (pd.Series with same index as input)
        """
        date = pd.to_datetime(date)
        
        # Handle empty signals
        if signals.empty or signals.abs().sum() == 0:
            logger.debug("[VolManagedOverlay] Empty or zero signals, returning zeros")
            return signals * 0
        
        # Scale based on mode
        if self.leverage_mode == "global":
            scaled = self._scale_global(signals, market, date)
        else:  # per-asset
            scaled = self._scale_per_asset(signals, market, date)
        
        # Apply constraints (bounds + leverage cap)
        constrained = self._apply_constraints(scaled)
        
        # Ensure same index as input
        result = pd.Series(0.0, index=signals.index)
        result.loc[constrained.index] = constrained
        
        logger.debug(
            f"[VolManagedOverlay] Scaled signals at {date}: "
            f"mean={result.mean():.3f}, std={result.std():.3f}, "
            f"gross_leverage={result.abs().sum():.2f}"
        )
        
        return result
    
    def describe(self) -> dict:
        """
        Return configuration and description of the VolManagedOverlay agent.
        
        Returns:
            dict with configuration parameters
        """
        return {
            'agent': 'VolManagedOverlay',
            'role': 'Scale strategy signals to achieve target volatility',
            'target_vol': self.target_vol,
            'lookback_vol': self.lookback_vol if self.lookback_vol else self.risk_vol.vol_lookback,
            'floor_vol': self.floor_vol,
            'cap_leverage': self.cap_leverage,
            'leverage_mode': self.leverage_mode,
            'position_bounds': self.position_bounds,
            'outputs': ['scale(signals, market, date)']
        }

