"""
Tests for MacroRegimeFilter agent.

Verifies that:
1. Scaler stays within bounds [k_min, k_max]
2. Higher realized vol → lower scaler (monotonicity)
3. Scaler changes only on rebalance dates
4. Breadth adjustments work correctly
5. EMA smoothing is applied
6. Deterministic outputs given same inputs
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.agents.overlay_macro_regime import MacroRegimeFilter


@pytest.fixture
def mock_market():
    """Create a mock MarketData instance with realistic data."""
    market = Mock()
    market.universe = ['ES', 'NQ', 'GC', 'CL', 'TY']
    
    # Create a date range for trading days
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')  # Business days
    market._dates = dates
    
    # Mock trading_days method
    def mock_trading_days(symbols=None):
        return market._dates
    
    market.trading_days = mock_trading_days
    
    # Store mock data for returns and prices
    market._returns_data = None
    market._prices_data = None
    
    return market


def create_returns_data(dates, es_vol=0.18, nq_vol=0.25, seed=42):
    """Create mock returns data with specified volatility."""
    n = len(dates)
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate returns with specified annualized vol
    # Daily vol = annual vol / sqrt(252)
    es_returns = np.random.randn(n) * (es_vol / np.sqrt(252))
    nq_returns = np.random.randn(n) * (nq_vol / np.sqrt(252))
    
    returns_df = pd.DataFrame({
        'ES': es_returns,
        'NQ': nq_returns
    }, index=dates)
    
    return returns_df


def create_prices_data(dates, es_trend=1.0, nq_trend=1.0, seed=43):
    """
    Create mock price data.
    
    trend=1.0 means above SMA, trend=-1.0 means below SMA, trend=0.0 means neutral
    """
    n = len(dates)
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create prices with trend
    # Start at 100, add small random walk + strong trend
    es_prices = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.arange(n) * es_trend * 0.3
    nq_prices = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.arange(n) * nq_trend * 0.3
    
    prices_df = pd.DataFrame({
        'ES': es_prices,
        'NQ': nq_prices
    }, index=dates)
    
    return prices_df


def setup_market_data(market, vol_level='medium', breadth_level='neutral'):
    """
    Setup market data with specific vol and breadth regimes.
    
    Args:
        market: Mock market instance
        vol_level: 'low' (~12%), 'medium' (~22%), 'high' (~35%)
        breadth_level: 'bearish' (0), 'neutral' (0.5), 'bullish' (1.0)
    """
    dates = market._dates
    
    # Set volatility levels (use different seeds for different levels)
    vol_map = {
        'low': ((0.12, 0.12), 100),
        'medium': ((0.22, 0.22), 200),
        'high': ((0.35, 0.35), 300)
    }
    (es_vol, nq_vol), vol_seed = vol_map[vol_level]
    
    # Set breadth levels (trends: positive=bullish, negative=bearish)
    breadth_map = {
        'bearish': ((-1.0, -1.0), 400),   # Both below SMA (downtrend)
        'neutral': ((1.0, -1.0), 500),     # One above, one below
        'bullish': ((1.0, 1.0), 600)       # Both above SMA (uptrend)
    }
    (es_trend, nq_trend), breadth_seed = breadth_map[breadth_level]
    
    # Create data with specific seeds for reproducibility
    returns_data = create_returns_data(dates, es_vol, nq_vol, seed=vol_seed)
    prices_data = create_prices_data(dates, es_trend, nq_trend, seed=breadth_seed)
    
    # Mock get_returns method
    def mock_get_returns(symbols=None, end=None, method='log', **kwargs):
        data = returns_data.copy()
        if end:
            data = data[data.index <= end]
        if symbols:
            data = data[[s for s in symbols if s in data.columns]]
        return data
    
    # Mock get_price_panel method
    def mock_get_price_panel(symbols=None, fields=None, end=None, tidy=False, **kwargs):
        data = prices_data.copy()
        if end:
            data = data[data.index <= end]
        if symbols:
            data = data[[s for s in symbols if s in data.columns]]
        return data
    
    market.get_returns = mock_get_returns
    market.get_price_panel = mock_get_price_panel


def test_initialization_default():
    """Test default initialization."""
    filter = MacroRegimeFilter(config_path="nonexistent.yaml")
    
    assert filter.rebalance == "W-FRI"
    assert filter.vol_thresholds == {'low': 0.15, 'high': 0.30}
    assert filter.k_bounds == {'min': 0.4, 'max': 1.0}
    assert filter.smoothing == 0.2
    assert filter.vol_lookback == 21
    assert filter.breadth_lookback == 200
    assert filter.proxy_symbols == ("ES", "NQ")


def test_initialization_custom():
    """Test custom initialization."""
    filter = MacroRegimeFilter(
        rebalance="M",
        vol_thresholds={'low': 0.10, 'high': 0.40},
        k_bounds={'min': 0.3, 'max': 1.2},
        smoothing=0.3,
        vol_lookback=42,
        breadth_lookback=100,
        proxy_symbols=("ES", "NQ", "GC"),
        config_path="nonexistent.yaml"
    )
    
    assert filter.rebalance == "M"
    assert filter.vol_thresholds == {'low': 0.10, 'high': 0.40}
    assert filter.k_bounds == {'min': 0.3, 'max': 1.2}
    assert filter.smoothing == 0.3
    assert filter.vol_lookback == 42
    assert filter.breadth_lookback == 100
    assert filter.proxy_symbols == ("ES", "NQ", "GC")


def test_initialization_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match="smoothing must be in"):
        MacroRegimeFilter(smoothing=-0.1, config_path="nonexistent.yaml")
    
    with pytest.raises(ValueError, match="smoothing must be in"):
        MacroRegimeFilter(smoothing=1.5, config_path="nonexistent.yaml")
    
    with pytest.raises(ValueError, match="vol_thresholds"):
        MacroRegimeFilter(
            vol_thresholds={'low': 0.30, 'high': 0.15},
            config_path="nonexistent.yaml"
        )
    
    with pytest.raises(ValueError, match="k_bounds"):
        MacroRegimeFilter(
            k_bounds={'min': 1.0, 'max': 0.4},
            config_path="nonexistent.yaml"
        )
    
    with pytest.raises(ValueError, match="vol_lookback"):
        MacroRegimeFilter(vol_lookback=1, config_path="nonexistent.yaml")
    
    with pytest.raises(ValueError, match="breadth_lookback"):
        MacroRegimeFilter(breadth_lookback=0, config_path="nonexistent.yaml")


def test_bounds(mock_market):
    """
    Test that scaler always stays within [k_min, k_max].
    
    Test various vol and breadth combinations.
    """
    k_min = 0.4
    k_max = 1.0
    
    filter = MacroRegimeFilter(
        vol_thresholds={'low': 0.10, 'high': 0.40},  # Wider range to capture test data
        k_bounds={'min': k_min, 'max': k_max},
        config_path="nonexistent.yaml"
    )
    
    # Test different regime combinations
    test_cases = [
        ('low', 'bullish'),      # Best case
        ('low', 'neutral'),
        ('low', 'bearish'),
        ('medium', 'bullish'),
        ('medium', 'neutral'),
        ('medium', 'bearish'),
        ('high', 'bullish'),
        ('high', 'neutral'),
        ('high', 'bearish'),     # Worst case
    ]
    
    for vol_level, breadth_level in test_cases:
        # Setup market data
        setup_market_data(mock_market, vol_level, breadth_level)
        
        # Get scaler at a date with sufficient history
        date = mock_market._dates[250]  # After 200 days for breadth
        k = filter.scaler(mock_market, date)
        
        # Check bounds
        assert k_min <= k <= k_max, \
            f"Scaler {k:.3f} out of bounds [{k_min}, {k_max}] for {vol_level} vol, {breadth_level} breadth"
        
        print(f"[OK] Bounds test ({vol_level}, {breadth_level}): k={k:.3f} in [{k_min}, {k_max}]")


def test_monotone_vol(mock_market):
    """
    Test that higher realized vol → lower scaler (holding breadth fixed).
    
    This tests the monotonic relationship between vol and scaler.
    """
    filter = MacroRegimeFilter(
        vol_thresholds={'low': 0.10, 'high': 0.40},  # Wider range to capture test data
        smoothing=0.0,  # No smoothing for clearer comparison
        config_path="nonexistent.yaml"
    )
    
    # Fix breadth to neutral
    breadth_level = 'neutral'
    
    # Test low → medium → high vol
    vol_levels = ['low', 'medium', 'high']
    scalers = []
    
    for vol_level in vol_levels:
        # Reset filter state for each test
        filter._last_scaler = None
        filter._last_rebalance = None
        filter._rebalance_dates = None
        
        setup_market_data(mock_market, vol_level, breadth_level)
        
        date = mock_market._dates[250]
        k = filter.scaler(mock_market, date)
        scalers.append(k)
        
        print(f"  {vol_level} vol: k={k:.3f}")
    
    # Check monotonicity: low vol → high scaler, high vol → low scaler
    assert scalers[0] > scalers[1], \
        f"Low vol scaler {scalers[0]:.3f} should be > medium vol scaler {scalers[1]:.3f}"
    
    assert scalers[1] > scalers[2], \
        f"Medium vol scaler {scalers[1]:.3f} should be > high vol scaler {scalers[2]:.3f}"
    
    print(f"[OK] Monotone vol test: low={scalers[0]:.3f} > medium={scalers[1]:.3f} > high={scalers[2]:.3f}")


def test_rebalance_only(mock_market):
    """
    Test that scaler changes only on rebalance dates.
    
    Between rebalances, scaler should remain constant.
    """
    filter = MacroRegimeFilter(
        rebalance="W-FRI",  # Weekly on Fridays
        smoothing=0.1,
        config_path="nonexistent.yaml"
    )
    
    setup_market_data(mock_market, 'medium', 'neutral')
    
    # Test dates spanning multiple weeks
    test_dates = mock_market._dates[250:280]  # About 4 weeks
    
    scalers = []
    for date in test_dates:
        k = filter.scaler(mock_market, date)
        scalers.append(k)
    
    # Count unique scalers (should be small, ~4-5 for 4 weeks)
    unique_scalers = len(set(np.round(scalers, 6)))  # Round to avoid float precision issues
    
    # Should have fewer unique scalers than days (since it only changes on rebalances)
    assert unique_scalers < len(test_dates), \
        f"Expected fewer unique scalers ({unique_scalers}) than days ({len(test_dates)})"
    
    # Check that consecutive non-rebalance days have same scaler
    consecutive_same = 0
    for i in range(1, len(scalers)):
        if abs(scalers[i] - scalers[i-1]) < 1e-6:
            consecutive_same += 1
    
    # At least some consecutive days should have same scaler
    assert consecutive_same > len(test_dates) * 0.5, \
        f"Expected >50% consecutive same scalers, got {consecutive_same}/{len(test_dates)-1}"
    
    print(f"[OK] Rebalance only test: {unique_scalers} unique scalers over {len(test_dates)} days")
    print(f"  {consecutive_same}/{len(test_dates)-1} consecutive days with same scaler")


def test_breadth_adjustment(mock_market):
    """
    Test that breadth adjustments work correctly.
    
    Bullish breadth → higher scaler, bearish breadth → lower scaler.
    """
    filter = MacroRegimeFilter(
        vol_thresholds={'low': 0.10, 'high': 0.40},  # Wider range to capture test data
        smoothing=0.0,  # No smoothing for clearer comparison
        config_path="nonexistent.yaml"
    )
    
    # Fix vol to medium
    vol_level = 'medium'
    
    # Test bearish → neutral → bullish breadth
    breadth_levels = ['bearish', 'neutral', 'bullish']
    scalers = []
    
    for breadth_level in breadth_levels:
        # Reset filter state
        filter._last_scaler = None
        filter._last_rebalance = None
        filter._rebalance_dates = None
        
        setup_market_data(mock_market, vol_level, breadth_level)
        
        date = mock_market._dates[250]
        k = filter.scaler(mock_market, date)
        scalers.append(k)
        
        print(f"  {breadth_level} breadth: k={k:.3f}")
    
    # Check ordering: bearish < neutral < bullish
    assert scalers[0] < scalers[1], \
        f"Bearish scaler {scalers[0]:.3f} should be < neutral scaler {scalers[1]:.3f}"
    
    assert scalers[1] < scalers[2], \
        f"Neutral scaler {scalers[1]:.3f} should be < bullish scaler {scalers[2]:.3f}"
    
    print(f"[OK] Breadth adjustment test: bearish={scalers[0]:.3f} < neutral={scalers[1]:.3f} < bullish={scalers[2]:.3f}")


def test_smoothing(mock_market):
    """
    Test that EMA smoothing is applied correctly across multiple rebalances.
    
    With smoothing, the scaler should change gradually, not jump immediately.
    """
    # Create filter with strong smoothing
    filter = MacroRegimeFilter(
        vol_thresholds={'low': 0.10, 'high': 0.40},
        smoothing=0.3,  # 30% of new value, 70% of old
        rebalance="W-MON",
        config_path="nonexistent.yaml"
    )
    
    # Start with high vol regime
    setup_market_data(mock_market, 'high', 'bearish')
    
    # Find first Monday
    date1 = None
    for i in range(250, len(mock_market._dates)):
        if mock_market._dates[i].weekday() == 0:
            date1 = mock_market._dates[i]
            break
    
    k1 = filter.scaler(mock_market, date1)
    
    # Switch to low vol (favorable regime)
    setup_market_data(mock_market, 'low', 'bullish')
    
    # Get scalers at next two Mondays to see smoothing effect
    monday_scalers = [k1]
    current_idx = mock_market._dates.get_loc(date1) + 1
    
    for _ in range(3):  # Get 3 more rebalance dates
        for i in range(current_idx, len(mock_market._dates)):
            if mock_market._dates[i].weekday() == 0:
                k = filter.scaler(mock_market, mock_market._dates[i])
                monday_scalers.append(k)
                current_idx = i + 1
                break
    
    # With smoothing, scalers should gradually increase, not jump
    # Each scaler should be > previous but the change should be moderate
    for i in range(1, len(monday_scalers)):
        assert monday_scalers[i] >= monday_scalers[i-1], \
            f"Scaler should not decrease in favorable regime: {monday_scalers}"
        
        # The increase should be bounded (smoothing dampens changes)
        if i == 1:
            # First change might be larger
            continue
        else:
            # Subsequent changes should be smaller (converging)
            change_prev = monday_scalers[i-1] - monday_scalers[i-2] if i >= 2 else 1.0
            change_current = monday_scalers[i] - monday_scalers[i-1]
            # Current change should be <= previous change (converging)
            assert change_current <= change_prev + 0.01, \
                f"Smoothing should cause convergence: {monday_scalers}"
    
    print(f"[OK] Smoothing test:")
    print(f"  Scalers over time: {[f'{k:.3f}' for k in monday_scalers]}")
    print(f"  (Gradually increasing, not jumping)")


def test_apply(mock_market):
    """Test apply method multiplies signals by scaler."""
    filter = MacroRegimeFilter(config_path="nonexistent.yaml")
    
    setup_market_data(mock_market, 'medium', 'neutral')
    
    # Create sample signals
    signals = pd.Series({
        'ES': 1.5,
        'NQ': -0.8,
        'GC': 0.5,
        'CL': -1.2,
        'TY': 0.3
    })
    
    date = mock_market._dates[250]
    
    # Get scaler
    k = filter.scaler(mock_market, date)
    
    # Apply to signals
    scaled_signals = filter.apply(signals, mock_market, date)
    
    # Check that signals are multiplied by k
    expected = signals * k
    pd.testing.assert_series_equal(scaled_signals, expected)
    
    print(f"[OK] Apply test: signals scaled by k={k:.3f}")


def test_deterministic(mock_market):
    """Test that outputs are deterministic given same inputs."""
    filter = MacroRegimeFilter(
        smoothing=0.2,
        config_path="nonexistent.yaml"
    )
    
    setup_market_data(mock_market, 'medium', 'neutral')
    
    date = mock_market._dates[250]
    
    # Run twice with same inputs
    k1 = filter.scaler(mock_market, date)
    
    # Reset and run again
    filter._last_scaler = None
    filter._last_rebalance = None
    filter._rebalance_dates = None
    
    k2 = filter.scaler(mock_market, date)
    
    # Should be identical
    assert abs(k1 - k2) < 1e-10, f"Non-deterministic: k1={k1:.6f}, k2={k2:.6f}"
    
    print(f"[OK] Deterministic test: k1={k1:.6f}, k2={k2:.6f}")


def test_extreme_regimes(mock_market):
    """
    Test behavior in extreme regimes.
    
    Verify that worst case produces lower scaler than best case.
    """
    filter = MacroRegimeFilter(
        vol_thresholds={'low': 0.10, 'high': 0.40},  # Wider range to capture test data
        smoothing=0.0,
        config_path="nonexistent.yaml"
    )
    
    k_min = filter.k_bounds['min']
    k_max = filter.k_bounds['max']
    
    # Worst case: high vol + bearish
    setup_market_data(mock_market, 'high', 'bearish')
    date = mock_market._dates[250]
    k_worst = filter.scaler(mock_market, date)
    
    # Reset
    filter._last_scaler = None
    filter._last_rebalance = None
    filter._rebalance_dates = None
    
    # Best case: low vol + bullish
    setup_market_data(mock_market, 'low', 'bullish')
    k_best = filter.scaler(mock_market, date)
    
    # Worst should be significantly lower than best
    assert k_worst < k_best - 0.2, \
        f"Worst case scaler {k_worst:.3f} should be much lower than best case {k_best:.3f}"
    
    # Both should be within bounds
    assert k_min <= k_worst <= k_max
    assert k_min <= k_best <= k_max
    
    # Best should be near k_max
    assert k_best >= k_max - 0.1, \
        f"Best case scaler {k_best:.3f} should be near k_max {k_max}"
    
    print(f"[OK] Extreme regimes test:")
    print(f"  Worst (high vol, bearish): k={k_worst:.3f}")
    print(f"  Best (low vol, bullish): k={k_best:.3f}")
    print(f"  Spread: {k_best - k_worst:.3f}")


def test_insufficient_data(mock_market):
    """Test handling when insufficient data is available."""
    filter = MacroRegimeFilter(
        vol_lookback=21,
        breadth_lookback=200,
        config_path="nonexistent.yaml"
    )
    
    setup_market_data(mock_market, 'medium', 'neutral')
    
    # Try to compute scaler with insufficient data (early date)
    date = mock_market._dates[10]  # Only 10 days of history
    
    # Should handle gracefully and return a scaler (defaults)
    k = filter.scaler(mock_market, date)
    
    # Should still be within bounds
    assert filter.k_bounds['min'] <= k <= filter.k_bounds['max']
    
    print(f"[OK] Insufficient data test: k={k:.3f} (with defaults)")


def test_describe():
    """Test describe method returns correct configuration."""
    filter = MacroRegimeFilter(
        rebalance="M",
        vol_thresholds={'low': 0.12, 'high': 0.28},
        k_bounds={'min': 0.5, 'max': 0.9},
        smoothing=0.3,
        vol_lookback=30,
        breadth_lookback=150,
        proxy_symbols=("ES", "NQ", "RTY"),
        config_path="nonexistent.yaml"
    )
    
    desc = filter.describe()
    
    assert desc['agent'] == 'MacroRegimeFilter'
    assert desc['rebalance'] == 'M'
    assert desc['vol_thresholds'] == {'low': 0.12, 'high': 0.28}
    assert desc['k_bounds'] == {'min': 0.5, 'max': 0.9}
    assert desc['smoothing'] == 0.3
    assert desc['vol_lookback'] == 30
    assert desc['breadth_lookback'] == 150
    assert desc['proxy_symbols'] == ("ES", "NQ", "RTY")
    assert 'scaler' in desc['outputs'][0]
    assert 'apply' in desc['outputs'][1]
    
    print("[OK] Describe method returns correct configuration")


def test_multiple_rebalances(mock_market):
    """
    Test behavior across multiple rebalances.
    
    Verify that scaler updates at each rebalance and smoothing works correctly.
    """
    filter = MacroRegimeFilter(
        rebalance="W",
        smoothing=0.3,
        config_path="nonexistent.yaml"
    )
    
    # Start with high vol regime
    setup_market_data(mock_market, 'high', 'bearish')
    
    # Get initial scaler
    date1 = mock_market._dates[250]
    k1 = filter.scaler(mock_market, date1)
    
    # Switch to medium vol
    setup_market_data(mock_market, 'medium', 'neutral')
    date2 = mock_market._dates[260]
    k2 = filter.scaler(mock_market, date2)
    
    # Switch to low vol
    setup_market_data(mock_market, 'low', 'bullish')
    date3 = mock_market._dates[270]
    k3 = filter.scaler(mock_market, date3)
    
    # Scalers should increase over time (improving regime)
    assert k1 < k2 < k3, \
        f"Scalers should increase: k1={k1:.3f}, k2={k2:.3f}, k3={k3:.3f}"
    
    print(f"[OK] Multiple rebalances test:")
    print(f"  Rebalance 1 (high vol, bearish): k={k1:.3f}")
    print(f"  Rebalance 2 (medium vol, neutral): k={k2:.3f}")
    print(f"  Rebalance 3 (low vol, bullish): k={k3:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

