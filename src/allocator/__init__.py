"""
Allocator module: Portfolio state features and regime-aware allocation.
"""

from .state_v1 import AllocatorStateV1, REQUIRED_FEATURES, OPTIONAL_FEATURES, ALL_FEATURES, LOOKBACKS
from .state_validate import validate_allocator_state_v1, validate_inputs_aligned
from .regime_v1 import RegimeClassifierV1
from .regime_rules_v1 import get_default_thresholds as get_regime_thresholds
from .risk_v1 import RiskTransformerV1, DEFAULT_REGIME_SCALARS, RISK_MIN, RISK_MAX
from .scalar_loader import load_precomputed_applied_scalars, validate_scalar_series

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
    # Scalar Loader (Stage 5.5)
    'load_precomputed_applied_scalars',
    'validate_scalar_series'
]

