"""
RiskTransformerV1: Risk Scaling from Regime and State

Consumes regime classifications and allocator state to emit risk scalars:
- risk_scalar: Portfolio-level exposure scaling (RISK_MIN to 1.0)

Maps regimes to target scalars with smoothing to avoid jerkiness.

Default (Profile-L / Institutional):
- NORMAL: risk_scalar = 1.0 (no adjustment)
- ELEVATED: risk_scalar = 0.85 (moderate reduction)
- STRESS: risk_scalar = 0.55 (significant reduction)
- CRISIS: risk_scalar = 0.30 (defensive positioning)

See profiles.py for H/M/L profile configurations.
"""

import logging
from typing import Optional, Dict, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default regime -> risk_scalar mapping (Profile-L / Institutional)
DEFAULT_REGIME_SCALARS = {
    'NORMAL': 1.00,
    'ELEVATED': 0.85,
    'STRESS': 0.55,
    'CRISIS': 0.30
}

# Risk scalar bounds
RISK_MIN = 0.25  # Minimum risk scalar (never go below 25% exposure)
RISK_MAX = 1.00  # Maximum risk scalar

# Smoothing parameters (EWMA)
DEFAULT_HALF_LIFE = 5  # days
DEFAULT_ALPHA = 0.25   # Fixed smoothing factor (alternative to half-life)


class RiskTransformerV1:
    """
    Risk transformer for regime-aware exposure scaling.
    
    Maps regime classifications to risk scalars with smoothing.
    Simple, deterministic, and tunable via a small parameter set.
    """
    
    VERSION = "v1.0"
    
    def __init__(
        self,
        regime_scalars: Optional[Dict[str, float]] = None,
        risk_min: float = RISK_MIN,
        risk_max: float = RISK_MAX,
        smoothing_alpha: float = DEFAULT_ALPHA,
        smoothing_half_life: Optional[int] = None
    ):
        """
        Initialize RiskTransformerV1.
        
        Args:
            regime_scalars: Dict mapping regime names to target risk scalars.
                           If None, uses DEFAULT_REGIME_SCALARS.
            risk_min: Minimum risk scalar (lower bound)
            risk_max: Maximum risk scalar (upper bound)
            smoothing_alpha: Fixed smoothing factor for EWMA (0 to 1).
                            Ignored if smoothing_half_life is provided.
            smoothing_half_life: Half-life in days for EWMA smoothing.
                                If provided, overrides smoothing_alpha.
        """
        self.regime_scalars = regime_scalars or DEFAULT_REGIME_SCALARS
        self.risk_min = risk_min
        self.risk_max = risk_max
        
        # Compute alpha from half-life if provided
        if smoothing_half_life is not None:
            self.smoothing_alpha = 1 - np.exp(-np.log(2) / smoothing_half_life)
            self.smoothing_half_life = smoothing_half_life
        else:
            self.smoothing_alpha = smoothing_alpha
            self.smoothing_half_life = None
        
        # Validate parameters
        if not (0 < self.smoothing_alpha <= 1):
            raise ValueError(f"smoothing_alpha must be in (0, 1], got {self.smoothing_alpha}")
        
        if not (0 < self.risk_min <= self.risk_max <= 1):
            raise ValueError(
                f"Must have 0 < risk_min <= risk_max <= 1, "
                f"got risk_min={self.risk_min}, risk_max={self.risk_max}"
            )
        
        logger.info(f"[RiskTransformerV1] Initialized (version {self.VERSION})")
        logger.info(f"[RiskTransformerV1] Regime scalars: {self.regime_scalars}")
        logger.info(f"[RiskTransformerV1] Risk bounds: [{self.risk_min}, {self.risk_max}]")
        logger.info(f"[RiskTransformerV1] Smoothing alpha: {self.smoothing_alpha:.4f}")
        if self.smoothing_half_life:
            logger.info(f"[RiskTransformerV1] Smoothing half-life: {self.smoothing_half_life} days")
    
    def transform(
        self,
        state_df: pd.DataFrame,
        regime: pd.Series,
        override_regime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform regime and state into risk scalars.
        
        Args:
            state_df: Allocator state DataFrame with canonical features
            regime: Regime series indexed by date
            override_regime: Optional manual regime override (NORMAL/ELEVATED/STRESS/CRISIS)
                           When set, bypasses regime detection and uses this regime for all dates.
                           Useful for testing profile behavior.
        
        Returns:
            DataFrame with 'risk_scalar' column indexed by date
        """
        if override_regime is not None:
            # Manual override mode: use specified regime for all dates
            if override_regime not in self.regime_scalars:
                raise ValueError(
                    f"override_regime must be one of {list(self.regime_scalars.keys())}, "
                    f"got {override_regime}"
                )
            
            # Create regime series with override value
            if regime.empty:
                # If no regime provided, create dummy index from state_df
                if state_df.empty:
                    logger.warning("[RiskTransformerV1] Empty state and regime, returning empty DataFrame")
                    return pd.DataFrame(columns=['risk_scalar'])
                regime = pd.Series(override_regime, index=state_df.index)
            else:
                regime = pd.Series(override_regime, index=regime.index)
            
            logger.info(f"[RiskTransformerV1] Using manual regime override: {override_regime}")
        
        if regime.empty:
            logger.warning("[RiskTransformerV1] Empty regime series, returning empty DataFrame")
            return pd.DataFrame(columns=['risk_scalar'])
        
        logger.info(f"[RiskTransformerV1] Transforming regime to risk scalars for {len(regime)} dates")
        
        # Map regimes to target scalars
        target_scalars = regime.map(self.regime_scalars)
        
        # Check for unmapped regimes
        unmapped = target_scalars.isna()
        if unmapped.any():
            unmapped_regimes = regime[unmapped].unique()
            logger.warning(
                f"[RiskTransformerV1] Unmapped regimes: {unmapped_regimes}. "
                f"Defaulting to NORMAL (1.0)"
            )
            target_scalars = target_scalars.fillna(1.0)
        
        # Apply EWMA smoothing
        risk_scalar = self._apply_smoothing(target_scalars)
        
        # Apply bounds
        risk_scalar = risk_scalar.clip(lower=self.risk_min, upper=self.risk_max)
        
        # Create output DataFrame
        result = pd.DataFrame({'risk_scalar': risk_scalar}, index=regime.index)
        
        # Log statistics
        logger.info(
            f"[RiskTransformerV1] Risk scalar stats: "
            f"mean={risk_scalar.mean():.3f}, "
            f"min={risk_scalar.min():.3f}, "
            f"max={risk_scalar.max():.3f}, "
            f"std={risk_scalar.std():.3f}"
        )
        
        return result
    
    def _apply_smoothing(self, target_scalars: pd.Series) -> pd.Series:
        """
        Apply EWMA smoothing to target scalars.
        
        Args:
            target_scalars: Series of target risk scalars
        
        Returns:
            Smoothed risk scalars
        """
        # Use pandas ewm for efficient EWMA computation
        # alpha parameter: weight of new observation
        smoothed = target_scalars.ewm(alpha=self.smoothing_alpha, adjust=False).mean()
        
        return smoothed
    
    def describe(self) -> dict:
        """
        Return description of RiskTransformerV1.
        
        Returns:
            Dict with version and description
        """
        return {
            'agent': 'RiskTransformerV1',
            'version': self.VERSION,
            'role': 'Transform regime and state into risk scalars',
            'regime_scalars': self.regime_scalars,
            'risk_bounds': [self.risk_min, self.risk_max],
            'smoothing_alpha': self.smoothing_alpha,
            'smoothing_half_life': self.smoothing_half_life,
            'inputs': ['allocator_state_v1', 'regime_series'],
            'outputs': ['risk_scalar']
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_risk_transformer_from_profile(
    profile: Union[str, "AllocatorProfile"]
) -> RiskTransformerV1:
    """
    Create a RiskTransformerV1 configured with a specific allocator profile.
    
    Profiles:
    - "H" (High): Rare intervention, tail-only protection, minimal Sharpe drag
    - "M" (Medium): Balanced approach, active risk management without excessive drag
    - "L" (Low): Conservative, institutional-style (≈ Allocator v1 default)
    
    Args:
        profile: Profile name ("H", "M", "L") or AllocatorProfile instance
    
    Returns:
        Configured RiskTransformerV1 instance
    
    Example:
        >>> transformer_h = create_risk_transformer_from_profile("H")
        >>> transformer_l = create_risk_transformer_from_profile("L")
    """
    # Import here to avoid circular dependency
    from .profiles import get_allocator_profile, AllocatorProfile
    
    if isinstance(profile, str):
        profile_config = get_allocator_profile(profile)
    elif isinstance(profile, AllocatorProfile):
        profile_config = profile
    else:
        raise TypeError(f"profile must be str or AllocatorProfile, got {type(profile)}")
    
    logger.info(
        f"[RiskTransformerV1] Creating transformer with profile: {profile_config.name}"
    )
    
    return RiskTransformerV1(
        regime_scalars=profile_config.regime_scalars,
        risk_min=profile_config.risk_min,
        risk_max=RISK_MAX,
        smoothing_alpha=profile_config.smoothing_alpha,
    )


# Convenience aliases for profile-based creation
def create_allocator_h() -> RiskTransformerV1:
    """Create Allocator-H (High risk tolerance, rare intervention)."""
    return create_risk_transformer_from_profile("H")


def create_allocator_m() -> RiskTransformerV1:
    """Create Allocator-M (Medium risk tolerance, balanced approach)."""
    return create_risk_transformer_from_profile("M")


def create_allocator_l() -> RiskTransformerV1:
    """Create Allocator-L (Low risk tolerance, institutional/conservative)."""
    return create_risk_transformer_from_profile("L")

