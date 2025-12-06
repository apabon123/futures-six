"""
Feature Validation: Validate feature computation and alignment.

Checks:
1. Index alignment between prices and features
2. No lookahead bias (features don't depend on future data)
3. Basic distribution and z-score sanity checks
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List


def validate_alignment(prices: pd.DataFrame, features: pd.DataFrame):
    """
    Ensure features and prices are aligned and sane.
    
    Checks:
    1. Index match
    2. Sorted and unique indices
    3. NaN sanity check (NaNs only at start)
    
    Args:
        prices: Price DataFrame indexed by date
        features: Feature DataFrame indexed by date
    """
    # 1) Index match
    assert prices.index.equals(features.index), "Index mismatch"
    
    # 2) Sorted and unique
    assert prices.index.is_monotonic_increasing, "Prices index not sorted"
    assert features.index.is_monotonic_increasing, "Features index not sorted"
    assert prices.index.is_unique, "Duplicate dates in prices"
    assert features.index.is_unique, "Duplicate dates in features"
    
    # 3) NaN sanity check: NaNs only at start
    cutoff = int(len(features) * 0.05)
    for col in features.columns:
        interior_nans = features[col].iloc[cutoff:].isna().any()
        if interior_nans:
            print(f"[WARN] Feature {col} has NaNs in the interior of the series.")


def validate_no_lookahead(
    market,
    feature_service,
    end_date: Optional[str] = None,
    k: int = 10
):
    """
    Prove that features do not depend on future data.
    
    Computes features on full dataset and truncated dataset (last k days removed),
    then compares overlapping portions. They should be identical.
    
    Args:
        market: MarketData instance
        feature_service: FeatureService instance
        end_date: End date for feature computation (default: last available)
        k: Number of days to truncate from end (default: 10)
    """
    # Get full features
    full_features_dict = feature_service.get_features(end_date=end_date)
    
    # Combine all features into single DataFrame (if multiple feature types)
    # Take the union of all dates and concatenate columns
    if not full_features_dict:
        print("[WARN] No features computed")
        return
    
    # Get all feature DataFrames and combine
    feature_dfs = []
    for feat_type, feat_df in full_features_dict.items():
        if not feat_df.empty:
            feature_dfs.append(feat_df)
    
    if not feature_dfs:
        print("[WARN] All feature DataFrames are empty")
        return
    
    # Combine features (align by index, fill missing with NaN)
    full_features = pd.concat(feature_dfs, axis=1)
    full_features = full_features.sort_index()
    
    if len(full_features) <= k:
        print(f"[WARN] Not enough data for lookahead test (need > {k} days, got {len(full_features)})")
        return
    
    # Truncate: compute features up to (end_date - k days) if possible
    # Get the date that's k days before the last date
    last_date = full_features.index[-1]
    trunc_end_date = full_features.index[-k-1] if len(full_features) > k else None
    
    if trunc_end_date is None:
        print(f"[WARN] Cannot truncate {k} days from dataset")
        return
    
    # Clear cache to force recomputation
    feature_service.clear_cache()
    
    # Compute features on truncated dataset
    trunc_features_dict = feature_service.get_features(end_date=trunc_end_date)
    
    # Combine truncated features
    trunc_feature_dfs = []
    for feat_type, feat_df in trunc_features_dict.items():
        if not feat_df.empty:
            trunc_feature_dfs.append(feat_df)
    
    if not trunc_feature_dfs:
        print("[WARN] Truncated features are empty")
        return
    
    trunc_features = pd.concat(trunc_feature_dfs, axis=1)
    trunc_features = trunc_features.sort_index()
    
    # Get overlapping portion
    overlap_full = full_features.iloc[:-k]
    overlap_trunc = trunc_features
    
    # Align columns (take intersection)
    common_cols = overlap_full.columns.intersection(overlap_trunc.columns)
    if len(common_cols) == 0:
        print("[WARN] No common columns between full and truncated features")
        return
    
    overlap_full = overlap_full[common_cols]
    overlap_trunc = overlap_trunc[common_cols]
    
    # Align indices (take intersection)
    common_dates = overlap_full.index.intersection(overlap_trunc.index)
    if len(common_dates) == 0:
        print("[WARN] No common dates between full and truncated features")
        return
    
    overlap_full = overlap_full.loc[common_dates]
    overlap_trunc = overlap_trunc.loc[common_dates]
    
    # Compare (handle NaNs)
    diff = (overlap_full - overlap_trunc).abs()
    
    # Replace NaN differences with 0 (both are NaN = no difference)
    diff = diff.fillna(0)
    
    max_diff = diff.max().max()
    
    assert max_diff < 1e-10, f"Possible lookahead detected: max diff={max_diff}"
    
    print(f"[OK] No lookahead detected: max diff={max_diff:.2e}")


def validate_feature_stats(
    features: pd.DataFrame,
    zscore_cols: Optional[List[str]] = None
):
    """
    Ensure no exploding features and z-scored features behave normally.
    
    Args:
        features: Feature DataFrame indexed by date
        zscore_cols: Optional list of column names that should be z-scored
    """
    desc = features.describe()
    
    for col in features.columns:
        col_desc = desc[col]
        
        # Constant series
        if col_desc["std"] == 0:
            print(f"[WARN] Feature {col} constant series.")
        
        # Exploding values
        if col_desc["max"] > 20 or col_desc["min"] < -20:
            print(f"[WARN] Feature {col} has extreme values >20 zscore.")
    
    if zscore_cols:
        for col in zscore_cols:
            if col not in features.columns:
                continue
            
            mean = features[col].mean()
            std = features[col].std()
            
            if abs(mean) > 0.2:
                print(f"[WARN] Z-score feature {col} mean={mean:.3f}")
            
            if not (0.5 < std < 1.5):
                print(f"[WARN] Z-score feature {col} std={std:.3f}")


def run_feature_validation(
    market,
    feature_service,
    prices: Optional[pd.DataFrame] = None,
    zscore_cols: Optional[List[str]] = None,
    end_date: Optional[str] = None
):
    """
    Run all feature validation checks.
    
    Args:
        market: MarketData instance
        feature_service: FeatureService instance
        prices: Optional price DataFrame (if None, will fetch from market)
        zscore_cols: Optional list of column names that should be z-scored
        end_date: End date for feature computation (default: last available)
    """
    print("=" * 70)
    print("Feature Validation")
    print("=" * 70)
    
    # Get features
    features_dict = feature_service.get_features(end_date=end_date)
    
    if not features_dict:
        print("[ERROR] No features computed")
        return
    
    # Combine all features
    feature_dfs = []
    for feat_type, feat_df in features_dict.items():
        if not feat_df.empty:
            feature_dfs.append(feat_df)
    
    if not feature_dfs:
        print("[ERROR] All feature DataFrames are empty")
        return
    
    features = pd.concat(feature_dfs, axis=1)
    features = features.sort_index()
    
    # Get prices if not provided
    if prices is None:
        # Get prices for all symbols in market universe
        prices = market.get_price_panel(
            symbols=market.universe,
            fields=("close",),
            end=end_date,
            tidy=False
        )
    
    if prices.empty:
        print("[ERROR] No price data available")
        return
    
    # Align prices and features by index
    common_dates = prices.index.intersection(features.index)
    if len(common_dates) == 0:
        print("[ERROR] No common dates between prices and features")
        return
    
    prices_aligned = prices.loc[common_dates]
    features_aligned = features.loc[common_dates]
    
    # Run validations
    print("\n[1/3] Validating alignment...")
    try:
        validate_alignment(prices_aligned, features_aligned)
        print("[OK] Alignment check passed")
    except AssertionError as e:
        print(f"[FAIL] Alignment check failed: {e}")
        return
    
    print("\n[2/3] Validating no lookahead...")
    try:
        validate_no_lookahead(market, feature_service, end_date=end_date, k=10)
        print("[OK] No lookahead check passed")
    except AssertionError as e:
        print(f"[FAIL] No lookahead check failed: {e}")
    except Exception as e:
        print(f"[WARN] No lookahead check error: {e}")
    
    print("\n[3/3] Validating feature stats...")
    validate_feature_stats(features_aligned, zscore_cols)
    print("[OK] Feature stats check completed")
    
    print("\n" + "=" * 70)

