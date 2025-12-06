"""
Continuous Price Validation: Validate back-adjusted continuous price series.

Checks that the ContinuousContractBuilder is working correctly:
1. Raw and continuous prices have matching shapes
2. On non-roll days, returns match (cont ≈ raw within contracts)
3. On roll days, raw has jumps but continuous doesn't (gaps vanish)
4. No unexpected NaNs in continuous prices

This helps catch bugs if someone modifies ContinuousContractBuilder later.
"""

import pandas as pd
import numpy as np
from typing import Optional


def validate_continuous_prices(market, verbose: bool = True):
    """
    Sanity checks for prices_cont vs prices_raw and contract_ids.
    
    Validates:
    1. Basic shape / NaNs matching
    2. On non-roll days, cont and raw returns should match exactly
    3. On roll days, raw should have jumps, cont should not
    
    Args:
        market: MarketData instance with prices_raw, prices_cont, contract_ids properties
        verbose: If True, print detailed validation messages
        
    Raises:
        AssertionError: If validation fails
    """
    if verbose:
        print("\n" + "=" * 80)
        print("CONTINUOUS PRICE VALIDATION")
        print("=" * 80)
    
    # Load continuous prices (this triggers the builder if not already built)
    try:
        prices_raw = market.prices_raw
        prices_cont = market.prices_cont
        cids = market.contract_ids
    except AttributeError as e:
        raise AttributeError(
            f"MarketData missing required properties: {e}. "
            "Ensure MarketData has prices_raw, prices_cont, and contract_ids properties."
        )
    
    if prices_raw.empty or prices_cont.empty or cids.empty:
        print("[CONT] WARNING: Empty price data - cannot validate")
        return
    
    # 1) Basic shape / NaNs
    if verbose:
        print("\n[1] Shape and NaN checks...")
    
    assert prices_raw.shape == prices_cont.shape, \
        f"Raw/cont shape mismatch: raw={prices_raw.shape}, cont={prices_cont.shape}"
    
    assert prices_raw.shape == cids.shape, \
        f"Prices/contract_ids shape mismatch: prices={prices_raw.shape}, cids={cids.shape}"
    
    if prices_cont.isna().any().any():
        nans_per_symbol = prices_cont.isna().sum()
        symbols_with_nans = nans_per_symbol[nans_per_symbol > 0]
        print(f"[CONT] WARNING: NaNs in continuous prices:")
        print(f"  {symbols_with_nans.to_dict()}")
    else:
        if verbose:
            print("  [OK] No NaNs in continuous prices")
    
    # 2) On non-roll days, cont and raw should move the same
    if verbose:
        print("\n[2] Non-roll day return matching...")
    
    # Calculate returns (use pct_change for simple returns)
    raw_rets = prices_raw.pct_change()
    cont_rets = prices_cont.pct_change()
    
    # Identify non-roll days (contract_id same as previous day)
    same_contract = cids == cids.shift(1)
    # First row is always False (no previous), so set to True to include first day
    same_contract.iloc[0] = True
    
    # On non-roll days, returns should match
    # Skip first row (no previous return) and NaNs
    non_roll_mask = same_contract & raw_rets.notna() & cont_rets.notna()
    non_roll_mask.iloc[0] = False  # Exclude first row (no return yet)
    
    if non_roll_mask.any().any():
        # Compute difference only on non-roll days
        diff = (raw_rets - cont_rets).where(non_roll_mask).abs()
        max_diff = diff.max().max()
        
        if max_diff > 1e-8:
            # Find worst offenders
            max_diff_per_symbol = diff.max()
            worst_symbols = max_diff_per_symbol[max_diff_per_symbol > 1e-8].sort_values(ascending=False)
            print(f"[CONT] WARNING: Return mismatch on non-roll days, max diff={max_diff:.2e}")
            print(f"  Symbols with large differences:")
            for symbol, max_diff_symbol in worst_symbols.head(5).items():
                print(f"    {symbol}: {max_diff_symbol:.2e}")
        else:
            if verbose:
                print(f"  [OK] Non-roll daily returns match as expected (max diff={max_diff:.2e})")
    else:
        if verbose:
            print("  [WARN] No non-roll days found (all days are rolls or first day)")
    
    # 3) On roll days, raw has jumps, cont shouldn't
    if verbose:
        print("\n[3] Roll day return analysis...")
    
    # Identify roll days (contract_id changes)
    roll_days = cids != cids.shift(1)
    roll_days.iloc[0] = False  # First day is not a roll
    
    if roll_days.any().any():
        # Calculate max returns on roll days per symbol
        # Only consider roll days that have valid returns
        valid_roll_days = roll_days & raw_rets.notna() & cont_rets.notna()
        
        raw_roll_rets = raw_rets.where(valid_roll_days)
        cont_roll_rets = cont_rets.where(valid_roll_days)
        
        raw_roll_moves = raw_roll_rets.abs().max()
        cont_roll_moves = cont_roll_rets.abs().max()
        
        # Find symbols with significant roll jumps in raw but not in cont
        significant_raw_rolls = raw_roll_moves > 0.01  # > 1%
        if significant_raw_rolls.any():
            if verbose:
                print("[CONT] Max raw return on roll days (per symbol, in decimal):")
                print(raw_roll_moves[significant_raw_rolls].sort_values(ascending=False).head(10))
            
            # Check that continuous returns are much smaller on roll days
            roll_ratio = cont_roll_moves / (raw_roll_moves + 1e-10)  # Avoid div by zero
            large_cont_rolls = (roll_ratio > 0.5) & significant_raw_rolls  # Cont > 50% of raw
            
            if large_cont_rolls.any():
                print(f"[CONT] WARNING: Continuous returns still large on roll days:")
                print(f"  Symbols where cont_return > 50% of raw_return:")
                for symbol in large_cont_rolls[large_cont_rolls].index:
                    print(f"    {symbol}: raw={raw_roll_moves[symbol]:.4f}, cont={cont_roll_moves[symbol]:.4f}, ratio={roll_ratio[symbol]:.2f}")
            else:
                if verbose:
                    print("[CONT] [OK] Roll jumps successfully removed from continuous series")
                    print("  Max continuous return on roll days (per symbol):")
                    print(cont_roll_moves[significant_raw_rolls].sort_values(ascending=False).head(10))
        else:
            if verbose:
                print("  [OK] No significant roll jumps found (all < 1%)")
    else:
        if verbose:
            print("  [WARN] No roll days found (contract_id never changes)")
    
    # 4) Summary statistics
    if verbose:
        print("\n[4] Summary statistics...")
        n_roll_days = roll_days.sum().sum() if roll_days.any().any() else 0
        n_symbols = len(prices_raw.columns)
        n_days = len(prices_raw)
        
        print(f"  Symbols: {n_symbols}")
        print(f"  Total days: {n_days}")
        if n_roll_days > 0:
            print(f"  Roll days: {n_roll_days} ({100 * n_roll_days / (n_symbols * n_days):.2f}%)")
        else:
            print(f"  Roll days: 0 (contract_id never changes)")
        
        print("\n" + "=" * 80)
        print("CONTINUOUS PRICE VALIDATION COMPLETE")
        print("=" * 80 + "\n")


def run_continuous_validation(market, verbose: bool = True):
    """
    Run continuous price validation and return summary.
    
    This is a convenience wrapper that catches exceptions and provides
    a summary return value.
    
    Args:
        market: MarketData instance
        verbose: If True, print detailed validation messages
        
    Returns:
        dict with validation results:
            - success: bool
            - errors: list of error messages
            - warnings: list of warning messages
    """
    errors = []
    warnings = []
    
    try:
        validate_continuous_prices(market, verbose=verbose)
        return {
            'success': True,
            'errors': errors,
            'warnings': warnings
        }
    except AssertionError as e:
        errors.append(str(e))
        if verbose:
            print(f"\n[CONT] VALIDATION FAILED: {e}")
        return {
            'success': False,
            'errors': errors,
            'warnings': warnings
        }
    except Exception as e:
        errors.append(f"Unexpected error: {e}")
        if verbose:
            print(f"\n[CONT] VALIDATION ERROR: {e}")
        return {
            'success': False,
            'errors': errors,
            'warnings': warnings
        }

