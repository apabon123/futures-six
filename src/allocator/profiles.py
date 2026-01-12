"""
Allocator Profiles: H / M / L Risk Tolerance Configurations

This module defines the three canonical allocator profiles:

- Allocator-H (High Risk Tolerance): Rare intervention, tail-only protection
- Allocator-M (Medium Risk Tolerance): Balanced approach
- Allocator-L (Low Risk Tolerance): Conservative, institutional-style (≈ Allocator v1)

Each profile differs ONLY in risk tolerance, not architecture.
The allocator subsystem remains identical:
    A. State Estimation
    B. Regime Interpretation
    C. Risk Transformation
    D. Exposure Application

What changes per profile:
    - Regime → scalar mappings (less aggressive scaling for H, more for L)
    - Smoothing parameters (slower for H, faster for L)
    
Key Design Principles:
    - Allocator is a temporary brake, not a steering wheel
    - Allocator does NOT set leverage (that's Risk Targeting layer)
    - Allocator does NOT target volatility
    - Allocator does NOT optimize Sharpe
    - Allocator encodes pain tolerance for PM
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AllocatorProfile:
    """
    Allocator profile configuration.
    
    Defines the risk tolerance of the allocator through:
    - regime_scalars: Mapping from regime to risk scalar
    - smoothing_alpha: EWMA smoothing factor (lower = smoother)
    - risk_min: Minimum risk scalar (floor)
    - description: Human-readable description
    """
    name: str
    regime_scalars: Dict[str, float]
    smoothing_alpha: float
    risk_min: float
    description: str


# =============================================================================
# CANONICAL ALLOCATOR PROFILES
# =============================================================================

ALLOCATOR_PROFILE_H = AllocatorProfile(
    name="H",
    regime_scalars={
        "NORMAL": 1.00,    # No intervention in normal conditions
        "ELEVATED": 0.98,  # Minimal intervention (2% reduction)
        "STRESS": 0.85,    # Moderate intervention (15% reduction)
        "CRISIS": 0.60,    # Significant but not extreme (40% reduction)
    },
    smoothing_alpha=0.15,  # Slower smoothing (~8d half-life)
    risk_min=0.50,         # Higher floor (never below 50% exposure)
    description=(
        "High risk tolerance profile. Rare intervention, tail-only protection. "
        "Designed for PM with high pain tolerance who wants to capture full "
        "engine alpha while maintaining minimal crisis protection."
    ),
)

ALLOCATOR_PROFILE_M = AllocatorProfile(
    name="M",
    regime_scalars={
        "NORMAL": 1.00,    # No intervention in normal conditions
        "ELEVATED": 0.90,  # Light intervention (10% reduction)
        "STRESS": 0.70,    # Moderate intervention (30% reduction)
        "CRISIS": 0.45,    # Significant intervention (55% reduction)
    },
    smoothing_alpha=0.20,  # Moderate smoothing (~6d half-life)
    risk_min=0.35,         # Moderate floor (never below 35% exposure)
    description=(
        "Medium risk tolerance profile. Balanced approach with active risk "
        "management during elevated conditions but not overly conservative. "
        "Suitable for PM who wants risk control without excessive Sharpe drag."
    ),
)

ALLOCATOR_PROFILE_L = AllocatorProfile(
    name="L",
    regime_scalars={
        "NORMAL": 1.00,    # No intervention in normal conditions
        "ELEVATED": 0.85,  # Active intervention (15% reduction)
        "STRESS": 0.55,    # Significant intervention (45% reduction)
        "CRISIS": 0.30,    # Heavy intervention (70% reduction)
    },
    smoothing_alpha=0.25,  # Faster smoothing (~5d half-life)
    risk_min=0.25,         # Lower floor (can go to 25% exposure)
    description=(
        "Low risk tolerance profile. Conservative, institutional-style. "
        "This is approximately equivalent to Allocator v1. Designed for "
        "PM with low pain tolerance who prioritizes capital preservation."
    ),
)


# Profile registry for easy lookup
ALLOCATOR_PROFILES: Dict[str, AllocatorProfile] = {
    "H": ALLOCATOR_PROFILE_H,
    "M": ALLOCATOR_PROFILE_M,
    "L": ALLOCATOR_PROFILE_L,
    # Aliases
    "high": ALLOCATOR_PROFILE_H,
    "medium": ALLOCATOR_PROFILE_M,
    "low": ALLOCATOR_PROFILE_L,
    "institutional": ALLOCATOR_PROFILE_L,
    "conservative": ALLOCATOR_PROFILE_L,
    "aggressive": ALLOCATOR_PROFILE_H,
    "balanced": ALLOCATOR_PROFILE_M,
}


def get_allocator_profile(name: str) -> AllocatorProfile:
    """
    Get allocator profile by name.
    
    Args:
        name: Profile name ("H", "M", "L" or aliases)
    
    Returns:
        AllocatorProfile configuration
    
    Raises:
        ValueError: If profile name not found
    """
    name_lower = name.lower() if len(name) > 1 else name.upper()
    
    if name_lower not in ALLOCATOR_PROFILES and name.upper() not in ALLOCATOR_PROFILES:
        available = ["H", "M", "L", "high", "medium", "low", "institutional", "conservative", "aggressive", "balanced"]
        raise ValueError(f"Unknown allocator profile: {name}. Available: {available}")
    
    profile = ALLOCATOR_PROFILES.get(name.upper()) or ALLOCATOR_PROFILES.get(name_lower)
    
    logger.info(f"[AllocatorProfiles] Selected profile: {profile.name} - {profile.description[:50]}...")
    
    return profile


def list_profiles() -> Dict[str, str]:
    """
    List all available allocator profiles with descriptions.
    
    Returns:
        Dict mapping profile name to description
    """
    return {
        "H": ALLOCATOR_PROFILE_H.description,
        "M": ALLOCATOR_PROFILE_M.description,
        "L": ALLOCATOR_PROFILE_L.description,
    }


def compare_profiles() -> Dict[str, Dict]:
    """
    Compare all profiles side-by-side.
    
    Returns:
        Dict with profile comparison data
    """
    comparison = {}
    
    for name, profile in [("H", ALLOCATOR_PROFILE_H), ("M", ALLOCATOR_PROFILE_M), ("L", ALLOCATOR_PROFILE_L)]:
        comparison[name] = {
            "regime_scalars": profile.regime_scalars,
            "smoothing_alpha": profile.smoothing_alpha,
            "risk_min": profile.risk_min,
            "sharpe_drag_expected": _estimate_sharpe_drag(profile),
        }
    
    return comparison


def _estimate_sharpe_drag(profile: AllocatorProfile) -> str:
    """
    Estimate expected Sharpe drag from profile.
    
    This is a rough estimate based on regime scalar settings.
    """
    # Rough estimate: average deviation from 1.0 across regimes
    avg_scalar = sum(profile.regime_scalars.values()) / len(profile.regime_scalars)
    drag = 1.0 - avg_scalar
    
    if drag < 0.05:
        return "minimal (<5%)"
    elif drag < 0.10:
        return "low (5-10%)"
    elif drag < 0.15:
        return "moderate (10-15%)"
    else:
        return "significant (>15%)"


# =============================================================================
# PROFILE VALIDATION
# =============================================================================

def validate_profile(profile: AllocatorProfile) -> bool:
    """
    Validate that an allocator profile is well-formed.
    
    Args:
        profile: AllocatorProfile to validate
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If validation fails
    """
    # Check required regimes
    required_regimes = {"NORMAL", "ELEVATED", "STRESS", "CRISIS"}
    if set(profile.regime_scalars.keys()) != required_regimes:
        raise ValueError(
            f"Profile {profile.name} missing regimes. "
            f"Required: {required_regimes}, got: {set(profile.regime_scalars.keys())}"
        )
    
    # Check scalar monotonicity (NORMAL >= ELEVATED >= STRESS >= CRISIS)
    if not (
        profile.regime_scalars["NORMAL"] >= profile.regime_scalars["ELEVATED"] >= 
        profile.regime_scalars["STRESS"] >= profile.regime_scalars["CRISIS"]
    ):
        raise ValueError(
            f"Profile {profile.name} scalars must be monotonically decreasing: "
            f"NORMAL >= ELEVATED >= STRESS >= CRISIS"
        )
    
    # Check scalar bounds
    for regime, scalar in profile.regime_scalars.items():
        if not (0 < scalar <= 1.0):
            raise ValueError(
                f"Profile {profile.name} scalar for {regime} must be in (0, 1], got {scalar}"
            )
    
    # Check smoothing alpha
    if not (0 < profile.smoothing_alpha <= 1.0):
        raise ValueError(
            f"Profile {profile.name} smoothing_alpha must be in (0, 1], got {profile.smoothing_alpha}"
        )
    
    # Check risk_min
    if not (0 < profile.risk_min <= 1.0):
        raise ValueError(
            f"Profile {profile.name} risk_min must be in (0, 1], got {profile.risk_min}"
        )
    
    if profile.risk_min > profile.regime_scalars["CRISIS"]:
        raise ValueError(
            f"Profile {profile.name} risk_min ({profile.risk_min}) cannot exceed "
            f"CRISIS scalar ({profile.regime_scalars['CRISIS']})"
        )
    
    logger.info(f"[AllocatorProfiles] Profile {profile.name} validated successfully")
    return True


# Validate all canonical profiles on import
for _profile in [ALLOCATOR_PROFILE_H, ALLOCATOR_PROFILE_M, ALLOCATOR_PROFILE_L]:
    validate_profile(_profile)

