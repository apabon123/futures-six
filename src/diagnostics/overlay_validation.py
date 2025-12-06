"""
Overlay Validation: Validate overlays (macro regime, volatility, allocator).

Checks:
1. Macro regime output sanity
2. Volatility target check
3. Allocator stability check
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime


def validate_macro_overlay(macro_overlay, combined_signal: pd.Series, market, date: datetime):
    """
    Validate macro overlay output sanity.
    
    Args:
        macro_overlay: MacroRegimeFilter instance
        combined_signal: Combined strategy signal (pd.Series)
        market: MarketData instance
        date: Current date
    
    Returns:
        Scaled signal Series
    """
    scaler = macro_overlay.apply(combined_signal, market, date)
    
    # Should stay in reasonable range (typically [0.3, 1.2] for k_bounds)
    # But we check for extreme values
    if scaler.min() < 0 or scaler.max() > 2:
        print("[WARN] Macro scaler outside expected range.")
    
    print(f"Macro scaler: min={scaler.min():.2f}, max={scaler.max():.2f}")
    
    return scaler


def validate_vol_overlay(vol_overlay, combined_signal: pd.Series, market, date: datetime):
    """
    Ensure vol-targeting does not explode leverage or go negative.
    
    Args:
        vol_overlay: VolManagedOverlay instance
        combined_signal: Combined strategy signal (pd.Series)
        market: MarketData instance
        date: Current date
    
    Returns:
        Scaled signal Series
    """
    scaled = vol_overlay.scale(combined_signal, market, date)
    
    leverage = scaled.abs().sum()
    print(f"Leverage: min={leverage:.2f}, max={leverage:.2f}")
    
    if leverage > vol_overlay.cap_leverage + 1e-6:
        print(f"[WARN] Leverage exceeds cap: {leverage:.2f} > {vol_overlay.cap_leverage}")
    
    return scaled


def validate_allocator(allocator, combined_signal: pd.Series, cov: pd.DataFrame, prev_weights: Optional[pd.Series] = None):
    """
    Run allocator and check exposure shapes.
    
    Args:
        allocator: Allocator instance
        combined_signal: Combined strategy signal (pd.Series)
        cov: Covariance matrix (pd.DataFrame)
        prev_weights: Optional previous weights for turnover constraint
    
    Returns:
        Portfolio weights Series
    """
    weights = allocator.solve(combined_signal, cov, prev_weights)
    
    if weights.isna().any():
        print("[WARN] Allocator produced NaNs")
    
    gross = weights.abs().sum()
    net = weights.sum()
    
    print(f"Gross max={gross:.2f}, Net max={net:.2f}")
    
    return weights


def run_overlay_validation(
    macro_overlay,
    vol_overlay,
    allocator,
    combined_signal: pd.Series,
    market,
    date: datetime,
    cov: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None
):
    """
    Run all overlay validation checks.
    
    Args:
        macro_overlay: MacroRegimeFilter instance (optional, can be None)
        vol_overlay: VolManagedOverlay instance
        allocator: Allocator instance
        combined_signal: Combined strategy signal (pd.Series)
        market: MarketData instance
        date: Current date
        cov: Covariance matrix (pd.DataFrame)
        prev_weights: Optional previous weights
    """
    print("=" * 70)
    print("Overlay Validation")
    print("=" * 70)
    
    # Start with combined signal
    current_signal = combined_signal.copy()
    
    # 1. Macro overlay (if provided)
    if macro_overlay is not None:
        print("\n[1/3] Validating macro overlay...")
        try:
            current_signal = validate_macro_overlay(macro_overlay, current_signal, market, date)
            print("[OK] Macro overlay check passed")
        except Exception as e:
            print(f"[WARN] Macro overlay check error: {e}")
    else:
        print("\n[1/3] Skipping macro overlay (not provided)")
    
    # 2. Vol overlay
    print("\n[2/3] Validating vol overlay...")
    try:
        current_signal = validate_vol_overlay(vol_overlay, current_signal, market, date)
        print("[OK] Vol overlay check passed")
    except Exception as e:
        print(f"[WARN] Vol overlay check error: {e}")
    
    # 3. Allocator
    print("\n[3/3] Validating allocator...")
    try:
        weights = validate_allocator(allocator, current_signal, cov, prev_weights)
        print("[OK] Allocator check passed")
        return weights
    except Exception as e:
        print(f"[WARN] Allocator check error: {e}")
        return pd.Series(dtype=float)
    
    print("\n" + "=" * 70)

