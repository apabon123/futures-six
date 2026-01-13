"""
Risk Targeting Layer (Layer 5 in Canonical Stack)

Purpose: Define how large the portfolio is by design.

Key Principle: This layer encodes risk appetite, not risk control.

Allowed:
- Target portfolio volatility
- Equivalent leverage choice (e.g., 7×)
- Static or very slow updates

Not Allowed:
- Regime logic
- Stress detection
- Engine selection
- Dynamic brakes

Risk Targeting answers: "How big do I trade in normal conditions?"

This layer is ALWAYS ON and UPSTREAM of the allocator.
The allocator (Layer 6) can scale DOWN from here, but Risk Targeting sets the baseline.
"""

import logging
import json
from typing import Optional, Dict, Union, Any
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# Default risk targeting parameters
DEFAULT_TARGET_VOL = 0.20  # 20% target portfolio volatility
DEFAULT_VOL_LOOKBACK = 63  # Rolling vol lookback in days
DEFAULT_LEVERAGE_CAP = 7.0  # Maximum leverage
DEFAULT_LEVERAGE_FLOOR = 1.0  # Minimum leverage (no deleveraging below 1x)
DEFAULT_VOL_FLOOR = 0.05  # Minimum vol estimate to avoid extreme scaling


class RiskTargetingLayer:
    """
    Risk Targeting Layer: Converts target volatility to portfolio leverage.
    
    This is Layer 5 in the canonical execution stack.
    
    The layer:
    1. Takes raw portfolio weights from Portfolio Construction / Discretionary Overlay
    2. Estimates current portfolio volatility
    3. Scales weights to achieve target volatility
    4. Outputs weights to be fed into the Allocator (Layer 6)
    
    Key Design Principles:
    - ALWAYS ON: This layer never turns itself off
    - UPSTREAM OF ALLOCATOR: Allocator can only scale down from here
    - NO REGIME LOGIC: Does not detect stress or change behavior based on conditions
    - STATIC OR SLOW: Target vol changes slowly (quarterly at most)
    """
    
    VERSION = "v1.0"
    
    def __init__(
        self,
        target_vol: float = DEFAULT_TARGET_VOL,
        vol_lookback: int = DEFAULT_VOL_LOOKBACK,
        leverage_cap: float = DEFAULT_LEVERAGE_CAP,
        leverage_floor: float = DEFAULT_LEVERAGE_FLOOR,
        vol_floor: float = DEFAULT_VOL_FLOOR,
        update_frequency: str = "static",
        config_path: Optional[str] = "configs/strategies.yaml",
        artifact_writer: Optional[Any] = None  # ArtifactWriter instance
    ):
        """
        Initialize Risk Targeting Layer.
        
        Args:
            target_vol: Target annualized portfolio volatility (default: 0.20 = 20%)
            vol_lookback: Rolling window for volatility estimation (default: 63 days)
            leverage_cap: Maximum leverage allowed (default: 7.0)
            leverage_floor: Minimum leverage (default: 1.0, no deleveraging below)
            vol_floor: Minimum volatility estimate to avoid extreme scaling (default: 0.05)
            update_frequency: How often target_vol can change ("static", "quarterly", "monthly")
            config_path: Path to YAML config file (optional)
            artifact_writer: Optional ArtifactWriter instance for writing artifacts
        """
        # Load from config if available
        config = self._load_config(config_path) if config_path else None
        rt_config = config.get("risk_targeting", {}) if config else {}
        
        # Set parameters (config overrides defaults, explicit args override config)
        self.target_vol = rt_config.get("target_vol", target_vol)
        self.vol_lookback = rt_config.get("vol_lookback", vol_lookback)
        self.leverage_cap = rt_config.get("leverage_cap", leverage_cap)
        self.leverage_floor = rt_config.get("leverage_floor", leverage_floor)
        self.vol_floor = rt_config.get("vol_floor", vol_floor)
        self.update_frequency = rt_config.get("update_frequency", update_frequency)
        
        # Artifact writer (optional)
        self.artifact_writer = artifact_writer
        
        # Validate parameters
        self._validate_params()
        
        # Write params.json once if artifact_writer is provided
        if self.artifact_writer is not None:
            self._write_params()
        
        logger.info(
            f"[RiskTargetingLayer] Initialized (version {self.VERSION}): "
            f"target_vol={self.target_vol:.1%}, leverage_cap={self.leverage_cap}x, "
            f"leverage_floor={self.leverage_floor}x, update_frequency={self.update_frequency}"
        )
    
    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"[RiskTargetingLayer] Failed to load config: {e}")
            return None
    
    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.target_vol <= 0:
            raise ValueError(f"target_vol must be > 0, got {self.target_vol}")
        
        if self.vol_lookback < 5:
            raise ValueError(f"vol_lookback must be >= 5, got {self.vol_lookback}")
        
        if self.leverage_cap <= 0:
            raise ValueError(f"leverage_cap must be > 0, got {self.leverage_cap}")
        
        if self.leverage_floor < 0:
            raise ValueError(f"leverage_floor must be >= 0, got {self.leverage_floor}")
        
        if self.leverage_floor > self.leverage_cap:
            raise ValueError(
                f"leverage_floor ({self.leverage_floor}) must be <= "
                f"leverage_cap ({self.leverage_cap})"
            )
        
        if self.vol_floor <= 0:
            raise ValueError(f"vol_floor must be > 0, got {self.vol_floor}")
        
        if self.update_frequency not in ("static", "quarterly", "monthly"):
            raise ValueError(
                f"update_frequency must be 'static', 'quarterly', or 'monthly', "
                f"got {self.update_frequency}"
            )
    
    def compute_portfolio_vol(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        date: Union[str, datetime],
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Compute ex-ante portfolio volatility.
        
        Uses either:
        1. Provided covariance matrix (if available)
        2. Rolling historical covariance (if returns provided)
        
        Args:
            weights: Portfolio weights indexed by asset
            returns: Historical returns DataFrame (dates x assets)
            date: Current date for rolling window calculation
            cov_matrix: Optional pre-computed covariance matrix
        
        Returns:
            Annualized portfolio volatility estimate
        """
        date = pd.to_datetime(date)
        
        # Filter to non-zero weights
        active_weights = weights[weights.abs() > 1e-10]
        
        if len(active_weights) == 0:
            logger.debug("[RiskTargetingLayer] No active weights, returning vol_floor")
            return self.vol_floor
        
        # Use provided covariance or compute from returns
        if cov_matrix is not None:
            cov = cov_matrix
        else:
            # Compute rolling covariance
            cov = self._compute_rolling_cov(returns, date)
        
        if cov is None or cov.empty:
            logger.warning("[RiskTargetingLayer] Could not compute covariance, using vol_floor")
            return self.vol_floor
        
        # Align weights with covariance
        common_assets = active_weights.index.intersection(cov.index)
        
        if len(common_assets) == 0:
            logger.warning("[RiskTargetingLayer] No common assets, using vol_floor")
            return self.vol_floor
        
        w = active_weights.loc[common_assets].values
        
        # Handle single asset case
        if len(common_assets) == 1:
            # For single asset, covariance is a 1x1 DataFrame or scalar
            asset = common_assets[0]
            if isinstance(cov, pd.DataFrame):
                asset_var = cov.loc[asset, asset]
            else:
                asset_var = cov
            # Portfolio variance = weight^2 * asset_variance
            port_var = (w[0] ** 2) * asset_var
        else:
            # Multi-asset case: Portfolio variance = w' Σ w
            cov_mat = cov.loc[common_assets, common_assets].values
            port_var = w @ cov_mat @ w
        
        port_var = max(port_var, 0)  # Numerical stability
        
        # Annualized volatility (assumes daily returns, 252 trading days)
        # Note: cov is already in daily units, so we annualize by sqrt(252)
        port_vol = np.sqrt(port_var) * np.sqrt(252)
        
        # Apply floor
        port_vol = max(port_vol, self.vol_floor)
        
        return port_vol
    
    def _compute_rolling_cov(
        self,
        returns: pd.DataFrame,
        date: Union[str, datetime]
    ) -> Optional[pd.DataFrame]:
        """
        Compute rolling covariance matrix from historical returns.
        
        Args:
            returns: Historical returns DataFrame (dates x assets)
            date: Current date (compute cov up to this date)
        
        Returns:
            Covariance matrix DataFrame (assets x assets)
        """
        date = pd.to_datetime(date)
        
        # Ensure returns has a DatetimeIndex (not RangeIndex or numpy array)
        if not isinstance(returns.index, pd.DatetimeIndex):
            # Try to convert index to DatetimeIndex
            if 'date' in returns.columns:
                returns = returns.set_index('date')
            else:
                # If we can't convert, assume index represents dates and convert
                returns.index = pd.to_datetime(returns.index)
        
        # Filter to dates before current date (both sides now guaranteed to be datetime-like)
        historical = returns.loc[returns.index < date]
        
        if len(historical) < self.vol_lookback:
            logger.debug(
                f"[RiskTargetingLayer] Insufficient history ({len(historical)} < {self.vol_lookback})"
            )
            return None
        
        # Use last vol_lookback days
        window = historical.tail(self.vol_lookback)
        
        # Handle single asset case
        if len(window.columns) == 1:
            # For single asset, return variance as DataFrame
            asset = window.columns[0]
            var = window[asset].var()
            cov = pd.DataFrame({asset: [var]}, index=[asset])
        else:
            # Compute covariance (daily returns)
            cov = window.cov()
        
        return cov
    
    def compute_leverage(
        self,
        current_vol: float,
        gross_exposure: float = 1.0
    ) -> float:
        """
        Compute target leverage given current portfolio volatility.
        
        Leverage = target_vol / current_vol
        
        This is the multiplier to apply to weights to achieve target volatility.
        If current vol is 10% and target is 20%, leverage = 2.0x.
        If current vol is 30% and target is 20%, leverage = 0.67x (but clipped to floor).
        
        Args:
            current_vol: Current annualized portfolio volatility
            gross_exposure: Current gross exposure (sum of abs weights) - NOT USED, kept for API compatibility
        
        Returns:
            Target leverage multiplier (bounded by leverage_floor and leverage_cap)
        """
        # Avoid division by zero
        current_vol = max(current_vol, self.vol_floor)
        
        # Base leverage to achieve target vol
        # This is the raw multiplier: if vol is half target, need 2x leverage
        raw_leverage = self.target_vol / current_vol
        
        # Apply bounds (clip to floor and cap)
        leverage = np.clip(raw_leverage, self.leverage_floor, self.leverage_cap)
        
        logger.debug(
            f"[RiskTargetingLayer] Leverage calc: current_vol={current_vol:.2%}, "
            f"target_vol={self.target_vol:.2%}, raw_leverage={raw_leverage:.2f}, "
            f"bounded_leverage={leverage:.2f}"
        )
        
        return leverage
    
    def scale_weights(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        date: Union[str, datetime],
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Scale portfolio weights to achieve target volatility.
        
        This is the main interface for the Risk Targeting layer.
        
        Args:
            weights: Raw portfolio weights from Portfolio Construction
            returns: Historical returns for vol estimation
            date: Current date
            cov_matrix: Optional pre-computed covariance matrix
        
        Returns:
            Scaled portfolio weights
        """
        date = pd.to_datetime(date)
        
        # Handle empty weights
        if weights.empty or weights.abs().sum() < 1e-10:
            logger.debug("[RiskTargetingLayer] Empty weights, returning zeros")
            return weights * 0
        
        # Compute current portfolio volatility
        current_vol = self.compute_portfolio_vol(weights, returns, date, cov_matrix)
        
        # Compute current gross exposure and normalize to unit gross to avoid compounding leverage
        gross_exposure = weights.abs().sum()
        if gross_exposure <= 1e-12:
            logger.debug("[RiskTargetingLayer] Zero gross exposure, returning zeros")
            return weights * 0.0
        
        unit_weights = weights / gross_exposure
        
        # Compute target leverage
        leverage = self.compute_leverage(current_vol, gross_exposure)
        
        # Scale weights so that post-RT gross ≈ leverage (avoids compounding upstream leverage)
        scaled_weights = unit_weights * leverage
        
        # Safety: if numerical drift causes gross to exceed cap, renormalize
        gross_after = scaled_weights.abs().sum()
        if gross_after > self.leverage_cap + 1e-6:
            scale = self.leverage_cap / gross_after
            scaled_weights *= scale
            gross_after = scaled_weights.abs().sum()
        
        # Log summary
        logger.info(
            f"[RiskTargetingLayer] {date.strftime('%Y-%m-%d')}: "
            f"vol={current_vol:.2%}, leverage={leverage:.2f}x, "
            f"gross_before={gross_exposure:.2f}, gross_after={gross_after:.2f}"
        )
        
        # Write artifacts
        if self.artifact_writer is not None:
            self._write_artifacts(date, current_vol, leverage, weights, scaled_weights)
        
        return scaled_weights
    
    def get_leverage_series(
        self,
        weights_df: pd.DataFrame,
        returns: pd.DataFrame,
        cov_provider=None
    ) -> pd.Series:
        """
        Compute leverage series for a time series of weights.
        
        Useful for backtesting and analysis.
        
        Args:
            weights_df: DataFrame of weights (dates x assets)
            returns: DataFrame of returns (dates x assets)
            cov_provider: Optional object with .covariance(date) method
        
        Returns:
            Series of leverage values indexed by date
        """
        leverage_series = pd.Series(index=weights_df.index, dtype=float)
        
        for date in weights_df.index:
            weights = weights_df.loc[date]
            
            # Get covariance if provider available
            cov = None
            if cov_provider is not None:
                try:
                    cov = cov_provider.covariance(date)
                except Exception:
                    pass
            
            current_vol = self.compute_portfolio_vol(weights, returns, date, cov)
            gross_exposure = weights.abs().sum()
            leverage = self.compute_leverage(current_vol, gross_exposure)
            
            leverage_series.loc[date] = leverage
        
        return leverage_series
    
    def describe(self) -> dict:
        """
        Return description of the Risk Targeting layer.
        
        Returns:
            Dict with version, parameters, and description
        """
        return {
            "layer": "RiskTargetingLayer",
            "version": self.VERSION,
            "layer_number": 5,
            "purpose": "Define how large the portfolio is by design",
            "key_principle": "Encodes risk appetite, not risk control",
            "parameters": {
                "target_vol": self.target_vol,
                "vol_lookback": self.vol_lookback,
                "leverage_cap": self.leverage_cap,
                "leverage_floor": self.leverage_floor,
                "vol_floor": self.vol_floor,
                "update_frequency": self.update_frequency,
            },
            "allowed": [
                "Target portfolio volatility",
                "Equivalent leverage choice",
                "Static or very slow updates",
            ],
            "not_allowed": [
                "Regime logic",
                "Stress detection",
                "Engine selection",
                "Dynamic brakes",
            ],
            "answers": "How big do I trade in normal conditions?",
        }
    
    def _write_params(self) -> None:
        """Write params.json once per run."""
        import hashlib
        
        # Create version hash from parameters
        params_str = json.dumps({
            "target_vol": self.target_vol,
            "leverage_cap": self.leverage_cap,
            "leverage_floor": self.leverage_floor,
            "vol_floor": self.vol_floor,
            "vol_lookback": self.vol_lookback,
            "update_frequency": self.update_frequency,
            "version": self.VERSION,
        }, sort_keys=True)
        version_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        params = {
            "target_vol": self.target_vol,
            "leverage_cap": self.leverage_cap,
            "leverage_floor": self.leverage_floor,
            "vol_floor": self.vol_floor,
            "vol_lookback": self.vol_lookback,
            "update_frequency": self.update_frequency,
            "estimator": "rolling_covariance",
            "version": self.VERSION,
            "version_hash": version_hash,
        }
        
        self.artifact_writer.write_json("risk_targeting/params.json", params, mode="once")
    
    def _write_artifacts(
        self,
        date: pd.Timestamp,
        current_vol: float,
        leverage: float,
        weights_pre: pd.Series,
        weights_post: pd.Series
    ) -> None:
        """Write artifacts for one date."""
        date_str = date.strftime('%Y-%m-%d')
        
        # Debug logging to catch artifact bugs
        logger.info(
            f"[RT Artifacts] {date_str}: Writing artifacts - "
            f"weights_pre: {len(weights_pre)} assets, gross={weights_pre.abs().sum():.2f}; "
            f"weights_post: {len(weights_post)} assets, gross={weights_post.abs().sum():.2f}"
        )
        
        # 1. Leverage series
        leverage_df = pd.DataFrame({
            'date': [date_str],
            'leverage': [leverage],
        })
        self.artifact_writer.write_csv("risk_targeting/leverage_series.csv", leverage_df, mode="append")
        
        # 2. Realized vol series
        vol_df = pd.DataFrame({
            'date': [date_str],
            'realized_vol': [current_vol],
            'vol_window': [self.vol_lookback],
            'estimator': ['rolling_covariance'],
        })
        self.artifact_writer.write_csv("risk_targeting/realized_vol.csv", vol_df, mode="append")
        
        # 3. Weights pre-risk-targeting
        weights_pre_df = pd.DataFrame({
            'date': [date_str] * len(weights_pre),
            'instrument': weights_pre.index.tolist(),
            'weight': weights_pre.values.tolist(),
        })
        # Sort by instrument for deterministic output
        weights_pre_df = weights_pre_df.sort_values('instrument')
        self.artifact_writer.write_csv("risk_targeting/weights_pre_risk_targeting.csv", weights_pre_df, mode="append")
        
        # 4. Weights post-risk-targeting
        weights_post_df = pd.DataFrame({
            'date': [date_str] * len(weights_post),
            'instrument': weights_post.index.tolist(),
            'weight': weights_post.values.tolist(),
        })
        # Sort by instrument for deterministic output
        weights_post_df = weights_post_df.sort_values('instrument')
        self.artifact_writer.write_csv("risk_targeting/weights_post_risk_targeting.csv", weights_post_df, mode="append")


def create_risk_targeting_layer(
    profile: str = "default",
    config_path: Optional[str] = None  # Don't load from config when using profiles
) -> RiskTargetingLayer:
    """
    Factory function to create Risk Targeting layer with preset profiles.
    
    Profiles:
    - "default": 20% target vol, 7x leverage cap
    - "aggressive": 25% target vol, 10x leverage cap
    - "conservative": 15% target vol, 5x leverage cap
    
    Args:
        profile: Profile name ("default", "aggressive", "conservative")
        config_path: Path to config file (default: None, uses profile values)
    
    Returns:
        Configured RiskTargetingLayer instance
    """
    profiles = {
        "default": {
            "target_vol": 0.20,
            "leverage_cap": 7.0,
            "leverage_floor": 1.0,
        },
        "aggressive": {
            "target_vol": 0.25,
            "leverage_cap": 10.0,
            "leverage_floor": 1.0,
        },
        "conservative": {
            "target_vol": 0.15,
            "leverage_cap": 5.0,
            "leverage_floor": 1.0,
        },
    }
    
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(profiles.keys())}")
    
    params = profiles[profile]
    
    logger.info(f"[RiskTargetingLayer] Creating layer with profile: {profile}")
    
    # Don't pass config_path when using profiles - profile values are explicit
    return RiskTargetingLayer(
        target_vol=params["target_vol"],
        leverage_cap=params["leverage_cap"],
        leverage_floor=params["leverage_floor"],
        config_path=None,  # Explicit: don't load from config
    )

