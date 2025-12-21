"""
Regime Classification Rules v1

Canonical thresholds and logic for regime classification.
Designed to be conservative, intuitive, and avoid thrash.
"""

from typing import Dict

# Regime classification thresholds (v1.0)
# These are starting values - directionally correct but will be tuned

# Volatility acceleration thresholds
VOL_ACCEL_ENTER = 1.30  # Enter stress when short-term vol 30% > long-term
VOL_ACCEL_EXIT = 1.15   # Exit when short-term vol only 15% > long-term

# Correlation shock thresholds
CORR_SHOCK_ENTER = 0.10  # Enter stress when correlation jumps 10%
CORR_SHOCK_EXIT = 0.05   # Exit when correlation shock drops below 5%

# Drawdown level thresholds
DD_ENTER = -0.10         # Enter stress at 10% drawdown
DD_EXIT = -0.06          # Exit when drawdown recovers to 6%
DD_STRESS_ENTER = -0.12  # Stress-specific threshold (12% drawdown)
DD_CRISIS_ENTER = -0.20  # Crisis-specific threshold (20% drawdown)

# Drawdown slope thresholds (negative = worsening)
DD_SLOPE_ENTER = -0.06   # Enter stress when DD worsens 6% over 10d
DD_SLOPE_EXIT = -0.03    # Exit when DD slope improves to -3%

# Anti-thrash: minimum days in regime before downgrade
MIN_DAYS_IN_REGIME = 5

# Regime levels (ordered by severity)
REGIMES = ['NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']
REGIME_LEVELS = {regime: i for i, regime in enumerate(REGIMES)}


def get_default_thresholds() -> Dict:
    """
    Get default regime classification thresholds.
    
    Returns:
        Dict with all threshold parameters
    """
    return {
        'vol_accel_enter': VOL_ACCEL_ENTER,
        'vol_accel_exit': VOL_ACCEL_EXIT,
        'corr_shock_enter': CORR_SHOCK_ENTER,
        'corr_shock_exit': CORR_SHOCK_EXIT,
        'dd_enter': DD_ENTER,
        'dd_exit': DD_EXIT,
        'dd_stress_enter': DD_STRESS_ENTER,
        'dd_crisis_enter': DD_CRISIS_ENTER,
        'dd_slope_enter': DD_SLOPE_ENTER,
        'dd_slope_exit': DD_SLOPE_EXIT,
        'min_days_in_regime': MIN_DAYS_IN_REGIME
    }


def validate_thresholds(thresholds: Dict) -> None:
    """
    Validate that thresholds are consistent.
    
    Args:
        thresholds: Dict of threshold parameters
    
    Raises:
        ValueError: If thresholds are inconsistent
    """
    # Exit thresholds should be less strict than enter thresholds
    if thresholds['vol_accel_exit'] >= thresholds['vol_accel_enter']:
        raise ValueError("vol_accel_exit must be < vol_accel_enter (hysteresis)")
    
    if thresholds['corr_shock_exit'] >= thresholds['corr_shock_enter']:
        raise ValueError("corr_shock_exit must be < corr_shock_enter (hysteresis)")
    
    # Drawdown thresholds (note: more negative = worse)
    if thresholds['dd_exit'] <= thresholds['dd_enter']:
        raise ValueError("dd_exit must be > dd_enter (less negative, hysteresis)")
    
    if thresholds['dd_slope_exit'] <= thresholds['dd_slope_enter']:
        raise ValueError("dd_slope_exit must be > dd_slope_enter (less negative, hysteresis)")
    
    # Drawdown crisis threshold should be worse than stress
    if thresholds['dd_crisis_enter'] >= thresholds['dd_stress_enter']:
        raise ValueError("dd_crisis_enter must be < dd_stress_enter (more negative)")
    
    # Min days should be positive
    if thresholds['min_days_in_regime'] < 1:
        raise ValueError("min_days_in_regime must be >= 1")

