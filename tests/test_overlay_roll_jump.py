"""
Tests for RollJumpFilter overlay.

Verifies that:
1. Injected gaps are reduced per mode (drop/winsorize)
2. Shape/index/columns are preserved
3. Filter is idempotent (applying twice doesn't change results further)
4. Only flagged dates are modified
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents.overlay_roll_jump import RollJumpFilter


@pytest.fixture
def sample_returns():
    """
    Create sample returns DataFrame (wide format).
    
    Returns: index=dates, columns=symbols, values=simple returns
    """
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    symbols = ['ES', 'NQ', 'CL', 'GC', 'ZN']
    
    # Generate realistic returns (mean ~0, std ~0.01)
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols
    )
    
    return returns


@pytest.fixture
def sample_jumps():
    """
    Create sample jumps DataFrame (tidy format).
    
    Returns: columns=[date, symbol, return, flagged]
    """
    jumps = pd.DataFrame([
        {'date': pd.Timestamp('2024-01-05'), 'symbol': 'ES', 'return': 0.025, 'flagged': True},
        {'date': pd.Timestamp('2024-01-10'), 'symbol': 'CL', 'return': -0.032, 'flagged': True},
        {'date': pd.Timestamp('2024-01-15'), 'symbol': 'GC', 'return': 0.018, 'flagged': True},
    ])
    
    return jumps


def test_initialization_default():
    """Test default initialization."""
    filter = RollJumpFilter()
    
    assert filter.threshold_bp == 100.0
    assert filter.mode == "winsorize"


def test_initialization_custom():
    """Test custom initialization."""
    filter = RollJumpFilter(threshold_bp=150.0, mode="drop")
    
    assert filter.threshold_bp == 150.0
    assert filter.mode == "drop"


def test_initialization_validation():
    """Test parameter validation."""
    # Invalid threshold
    with pytest.raises(ValueError, match="threshold_bp must be > 0"):
        RollJumpFilter(threshold_bp=-50.0)
    
    with pytest.raises(ValueError, match="threshold_bp must be > 0"):
        RollJumpFilter(threshold_bp=0.0)
    
    # Invalid mode
    with pytest.raises(ValueError, match="mode must be 'drop' or 'winsorize'"):
        RollJumpFilter(mode="invalid")


def test_gap_injection_winsorize():
    """
    Test that injected +5% gap is reduced to threshold in winsorize mode.
    
    This is the core test: verify that a large roll jump gets clipped.
    """
    # Create returns with a large injected gap
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.002, 0.050, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001],  # 5% jump on day 3
        'NQ': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    }, index=dates)
    
    # Flag the 5% jump
    jumps = pd.DataFrame([
        {'date': dates[2], 'symbol': 'ES', 'return': 0.050, 'flagged': True}
    ])
    
    # Apply filter with 1% threshold
    filter = RollJumpFilter(threshold_bp=100.0, mode="winsorize")
    filtered = filter.apply(returns, jumps)
    
    # Check that the 5% return was clipped to 1%
    threshold_decimal = 0.01
    assert filtered.loc[dates[2], 'ES'] == pytest.approx(threshold_decimal, abs=1e-10), \
        f"Expected {threshold_decimal}, got {filtered.loc[dates[2], 'ES']}"
    
    # Check that other returns are unchanged
    assert filtered.loc[dates[0], 'ES'] == 0.001
    assert filtered.loc[dates[1], 'ES'] == 0.002
    assert filtered.loc[dates[3], 'ES'] == 0.001
    
    # NQ should be completely unchanged
    pd.testing.assert_series_equal(filtered['NQ'], returns['NQ'])
    
    print(f"✓ Gap injection (winsorize): 5% jump clipped to {threshold_decimal:.2%}")


def test_gap_injection_drop():
    """
    Test that injected +5% gap is set to 0 in drop mode.
    """
    # Create returns with a large injected gap
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.002, 0.050, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001],  # 5% jump on day 3
        'NQ': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    }, index=dates)
    
    # Flag the 5% jump
    jumps = pd.DataFrame([
        {'date': dates[2], 'symbol': 'ES', 'return': 0.050, 'flagged': True}
    ])
    
    # Apply filter in drop mode
    filter = RollJumpFilter(threshold_bp=100.0, mode="drop")
    filtered = filter.apply(returns, jumps)
    
    # Check that the 5% return was set to 0
    assert filtered.loc[dates[2], 'ES'] == 0.0, \
        f"Expected 0.0, got {filtered.loc[dates[2], 'ES']}"
    
    # Check that other returns are unchanged
    assert filtered.loc[dates[0], 'ES'] == 0.001
    assert filtered.loc[dates[1], 'ES'] == 0.002
    assert filtered.loc[dates[3], 'ES'] == 0.001
    
    # NQ should be completely unchanged
    pd.testing.assert_series_equal(filtered['NQ'], returns['NQ'])
    
    print("✓ Gap injection (drop): 5% jump set to 0")


def test_shape_preserved(sample_returns, sample_jumps):
    """
    Test that output has same shape/index/columns as input.
    """
    filter = RollJumpFilter(threshold_bp=100.0, mode="winsorize")
    filtered = filter.apply(sample_returns, sample_jumps)
    
    # Check shape
    assert filtered.shape == sample_returns.shape, \
        f"Shape changed: {sample_returns.shape} -> {filtered.shape}"
    
    # Check index
    pd.testing.assert_index_equal(filtered.index, sample_returns.index)
    
    # Check columns
    pd.testing.assert_index_equal(filtered.columns, sample_returns.columns)
    
    print(f"✓ Shape preserved: {filtered.shape}")


def test_idempotent(sample_returns, sample_jumps):
    """
    Test that applying filter twice doesn't change results further.
    
    Filter should be idempotent: F(F(x)) = F(x)
    """
    filter = RollJumpFilter(threshold_bp=100.0, mode="winsorize")
    
    # Apply once
    filtered_once = filter.apply(sample_returns, sample_jumps)
    
    # Apply again to the already-filtered data
    # Note: jumps_df should still reference the original jumps
    filtered_twice = filter.apply(filtered_once, sample_jumps)
    
    # Should be identical
    pd.testing.assert_frame_equal(filtered_once, filtered_twice, check_exact=True)
    
    print("✓ Idempotent: applying twice yields same result")


def test_idempotent_drop_mode(sample_returns, sample_jumps):
    """Test idempotency in drop mode."""
    filter = RollJumpFilter(threshold_bp=100.0, mode="drop")
    
    filtered_once = filter.apply(sample_returns, sample_jumps)
    filtered_twice = filter.apply(filtered_once, sample_jumps)
    
    pd.testing.assert_frame_equal(filtered_once, filtered_twice, check_exact=True)
    
    print("✓ Idempotent (drop mode)")


def test_only_flagged_modified():
    """
    Test that only flagged dates/symbols are modified.
    
    Unflagged returns should remain exactly as they were.
    """
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.020, 0.003, 0.004, 0.005],  # Day 2 has 2% return
        'NQ': [0.002, 0.003, 0.004, 0.005, 0.006],
        'CL': [0.001, 0.002, 0.003, 0.004, 0.005],
    }, index=dates)
    
    # Flag only ES on day 2
    jumps = pd.DataFrame([
        {'date': dates[1], 'symbol': 'ES', 'return': 0.020, 'flagged': True}
    ])
    
    filter = RollJumpFilter(threshold_bp=100.0, mode="winsorize")
    filtered = filter.apply(returns, jumps)
    
    # ES day 2 should be clipped to 1%
    assert filtered.loc[dates[1], 'ES'] == 0.01
    
    # All other values should be exactly the same
    # ES other days
    assert filtered.loc[dates[0], 'ES'] == returns.loc[dates[0], 'ES']
    assert filtered.loc[dates[2], 'ES'] == returns.loc[dates[2], 'ES']
    assert filtered.loc[dates[3], 'ES'] == returns.loc[dates[3], 'ES']
    assert filtered.loc[dates[4], 'ES'] == returns.loc[dates[4], 'ES']
    
    # NQ and CL completely unchanged
    pd.testing.assert_series_equal(filtered['NQ'], returns['NQ'])
    pd.testing.assert_series_equal(filtered['CL'], returns['CL'])
    
    print("✓ Only flagged date/symbol modified")


def test_negative_jump_winsorize():
    """Test that negative jumps are also clipped correctly."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, -0.035, 0.003, 0.004, 0.005],  # -3.5% jump on day 2
    }, index=dates)
    
    jumps = pd.DataFrame([
        {'date': dates[1], 'symbol': 'ES', 'return': -0.035, 'flagged': True}
    ])
    
    filter = RollJumpFilter(threshold_bp=100.0, mode="winsorize")
    filtered = filter.apply(returns, jumps)
    
    # Should be clipped to -1%
    assert filtered.loc[dates[1], 'ES'] == pytest.approx(-0.01, abs=1e-10)
    
    print("✓ Negative jump clipped correctly")


def test_multiple_symbols_same_date():
    """Test filtering when multiple symbols have jumps on same date."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.025, 0.003, 0.004, 0.005],  # 2.5% on day 2
        'NQ': [0.002, -0.030, 0.004, 0.005, 0.006],  # -3% on day 2
        'CL': [0.001, 0.002, 0.003, 0.004, 0.005],  # Normal
    }, index=dates)
    
    jumps = pd.DataFrame([
        {'date': dates[1], 'symbol': 'ES', 'return': 0.025, 'flagged': True},
        {'date': dates[1], 'symbol': 'NQ', 'return': -0.030, 'flagged': True},
    ])
    
    filter = RollJumpFilter(threshold_bp=100.0, mode="winsorize")
    filtered = filter.apply(returns, jumps)
    
    # Both should be clipped
    assert filtered.loc[dates[1], 'ES'] == 0.01
    assert filtered.loc[dates[1], 'NQ'] == -0.01
    
    # CL unchanged
    pd.testing.assert_series_equal(filtered['CL'], returns['CL'])
    
    print("✓ Multiple symbols on same date handled correctly")


def test_empty_returns():
    """Test handling of empty returns DataFrame."""
    empty_returns = pd.DataFrame()
    jumps = pd.DataFrame([
        {'date': pd.Timestamp('2024-01-05'), 'symbol': 'ES', 'return': 0.025, 'flagged': True}
    ])
    
    filter = RollJumpFilter()
    result = filter.apply(empty_returns, jumps)
    
    assert result.empty
    print("✓ Empty returns handled correctly")


def test_empty_jumps(sample_returns):
    """Test handling of empty jumps DataFrame."""
    empty_jumps = pd.DataFrame(columns=['date', 'symbol', 'return', 'flagged'])
    
    filter = RollJumpFilter()
    result = filter.apply(sample_returns, empty_jumps)
    
    # Should return unchanged copy
    pd.testing.assert_frame_equal(result, sample_returns, check_exact=True)
    print("✓ Empty jumps handled correctly")


def test_jump_date_not_in_returns():
    """Test handling when jump date is not in returns DataFrame."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.002, 0.003, 0.004, 0.005],
    }, index=dates)
    
    # Jump on a date not in returns
    jumps = pd.DataFrame([
        {'date': pd.Timestamp('2024-01-10'), 'symbol': 'ES', 'return': 0.025, 'flagged': True}
    ])
    
    filter = RollJumpFilter()
    result = filter.apply(returns, jumps)
    
    # Should return unchanged
    pd.testing.assert_frame_equal(result, returns, check_exact=True)
    print("✓ Jump date not in returns handled correctly")


def test_jump_symbol_not_in_returns():
    """Test handling when jump symbol is not in returns DataFrame."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.002, 0.003, 0.004, 0.005],
    }, index=dates)
    
    # Jump for a symbol not in returns
    jumps = pd.DataFrame([
        {'date': dates[1], 'symbol': 'GC', 'return': 0.025, 'flagged': True}
    ])
    
    filter = RollJumpFilter()
    result = filter.apply(returns, jumps)
    
    # Should return unchanged
    pd.testing.assert_frame_equal(result, returns, check_exact=True)
    print("✓ Jump symbol not in returns handled correctly")


def test_nan_handling():
    """Test that NaN values are left as NaN."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, np.nan, 0.003, 0.004, 0.005],
    }, index=dates)
    
    jumps = pd.DataFrame([
        {'date': dates[1], 'symbol': 'ES', 'return': 0.025, 'flagged': True}
    ])
    
    filter = RollJumpFilter()
    result = filter.apply(returns, jumps)
    
    # NaN should remain NaN (not modified)
    assert pd.isna(result.loc[dates[1], 'ES'])
    
    print("✓ NaN values preserved")


def test_threshold_edge_case_winsorize():
    """
    Test behavior when return exactly equals threshold.
    
    A return at exactly the threshold should not be modified.
    """
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.010, 0.003, 0.004, 0.005],  # Exactly 1% on day 2
    }, index=dates)
    
    jumps = pd.DataFrame([
        {'date': dates[1], 'symbol': 'ES', 'return': 0.010, 'flagged': True}
    ])
    
    filter = RollJumpFilter(threshold_bp=100.0, mode="winsorize")
    filtered = filter.apply(returns, jumps)
    
    # Should remain at 1% (not clipped)
    assert filtered.loc[dates[1], 'ES'] == 0.010
    
    print("✓ Threshold edge case handled correctly")


def test_custom_threshold():
    """Test with custom threshold."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.DataFrame({
        'ES': [0.001, 0.025, 0.003, 0.004, 0.005],  # 2.5% jump
    }, index=dates)
    
    jumps = pd.DataFrame([
        {'date': dates[1], 'symbol': 'ES', 'return': 0.025, 'flagged': True}
    ])
    
    # Use 2% threshold
    filter = RollJumpFilter(threshold_bp=200.0, mode="winsorize")
    filtered = filter.apply(returns, jumps)
    
    # Should be clipped to 2%
    assert filtered.loc[dates[1], 'ES'] == pytest.approx(0.02, abs=1e-10)
    
    print("✓ Custom threshold works correctly")


def test_describe():
    """Test describe method returns correct configuration."""
    filter = RollJumpFilter(threshold_bp=150.0, mode="drop")
    
    desc = filter.describe()
    
    assert desc['agent'] == 'RollJumpFilter'
    assert desc['threshold_bp'] == 150.0
    assert desc['mode'] == 'drop'
    assert 'role' in desc
    assert 'modes_available' in desc
    assert 'drop' in desc['modes_available']
    assert 'winsorize' in desc['modes_available']
    assert 'apply' in desc['outputs'][0]
    
    print("✓ Describe method returns correct configuration")


def test_original_not_modified():
    """Test that original DataFrames are not modified."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    returns_original = pd.DataFrame({
        'ES': [0.001, 0.025, 0.003, 0.004, 0.005],
    }, index=dates)
    
    jumps_original = pd.DataFrame([
        {'date': dates[1], 'symbol': 'ES', 'return': 0.025, 'flagged': True}
    ])
    
    # Create copies to compare later
    returns_copy = returns_original.copy()
    jumps_copy = jumps_original.copy()
    
    filter = RollJumpFilter()
    _ = filter.apply(returns_original, jumps_original)
    
    # Original should be unchanged
    pd.testing.assert_frame_equal(returns_original, returns_copy)
    pd.testing.assert_frame_equal(jumps_original, jumps_copy)
    
    print("✓ Original DataFrames not modified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

