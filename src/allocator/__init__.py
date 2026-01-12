"""
Allocator module: Portfolio state features and regime-aware allocation.

Canonical Architecture (Layer 6):
- State Estimation → AllocatorStateV1
- Regime Interpretation → RegimeClassifierV1
- Risk Transformation → RiskTransformerV1 (with H/M/L profiles)
- Exposure Application → scalar_loader

Profiles:
- Allocator-H: High risk tolerance, rare intervention
- Allocator-M: Medium risk tolerance, balanced approach
- Allocator-L: Low risk tolerance, institutional/conservative (default v1)
"""

from .state_v1 import AllocatorStateV1, REQUIRED_FEATURES, OPTIONAL_FEATURES, ALL_FEATURES, LOOKBACKS
from .state_validate import validate_allocator_state_v1, validate_inputs_aligned
from .regime_v1 import RegimeClassifierV1
from .regime_rules_v1 import get_default_thresholds as get_regime_thresholds
from .risk_v1 import (
    RiskTransformerV1, 
    DEFAULT_REGIME_SCALARS, 
    RISK_MIN, 
    RISK_MAX,
    create_risk_transformer_from_profile,
    create_allocator_h,
    create_allocator_m,
    create_allocator_l,
)
from .scalar_loader import load_precomputed_applied_scalars, validate_scalar_series
from .profiles import (
    AllocatorProfile,
    ALLOCATOR_PROFILE_H,
    ALLOCATOR_PROFILE_M,
    ALLOCATOR_PROFILE_L,
    ALLOCATOR_PROFILES,
    get_allocator_profile,
    list_profiles,
    compare_profiles,
    validate_profile,
)

__all__ = [
    # State Layer (Stages 1-3, 4A)
    'AllocatorStateV1',
    'REQUIRED_FEATURES',
    'OPTIONAL_FEATURES',
    'ALL_FEATURES',
    'LOOKBACKS',
    'validate_allocator_state_v1',
    'validate_inputs_aligned',
    # Regime Layer (Stage 4B)
    'RegimeClassifierV1',
    'get_regime_thresholds',
    # Risk Layer (Stage 4C)
    'RiskTransformerV1',
    'DEFAULT_REGIME_SCALARS',
    'RISK_MIN',
    'RISK_MAX',
    # Profile-based factory functions
    'create_risk_transformer_from_profile',
    'create_allocator_h',
    'create_allocator_m',
    'create_allocator_l',
    # Profile configurations
    'AllocatorProfile',
    'ALLOCATOR_PROFILE_H',
    'ALLOCATOR_PROFILE_M',
    'ALLOCATOR_PROFILE_L',
    'ALLOCATOR_PROFILES',
    'get_allocator_profile',
    'list_profiles',
    'compare_profiles',
    'validate_profile',
    # Scalar Loader (Stage 5.5)
    'load_precomputed_applied_scalars',
    'validate_scalar_series'
]

