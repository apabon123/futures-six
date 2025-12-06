"""
Tests for TSMOM (Time-Series Momentum) Strategy Agent

Test suite validates:
1. No look-ahead bias
2. Rebalance schedule adherence
3. Monotone relationship between past returns and signals
4. Signal standardization and capping
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.strat_momentum import TSMOM


class MockMarketData:
    """Mock MarketData for testing without database dependency."""
    
    def __init__(self, returns_df: pd.DataFrame, universe: tuple):
        self.returns_df = returns_df
        self.universe = universe
        self.asof = None
    
    def get_returns(
        self,
        symbols=None,
        start=None,
        end=None,
        method="log",
        price="close"
    ) -> pd.DataFrame:
        """Return mock returns filtered by date."""
        df = self.returns_df.copy()
        
        # Handle empty dataframe
        if df.empty:
            return df
        
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        if symbols:
            df = df[[s for s in symbols if s in df.columns]]
        
        return df


def create_synthetic_returns(
    n_days: int = 500,
    n_symbols: int = 6,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic returns for testing.
    
    Args:
        n_days: Number of trading days
        n_symbols: Number of symbols
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame of returns (date x symbols)
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    symbols = [f'SYM{i}' for i in range(n_symbols)]
    
    # Generate returns with some autocorrelation (momentum)
    returns = np.random.randn(n_days, n_symbols) * 0.01
    
    # Add some momentum (positive autocorrelation)
    for i in range(1, n_days):
        returns[i] += 0.3 * returns[i-1]
    
    df = pd.DataFrame(returns, index=dates, columns=symbols)
    return df


def create_trending_returns(
    n_days: int = 500,
    n_symbols: int = 3
) -> pd.DataFrame:
    """
    Create returns with clear trends for monotonicity testing.
    
    Symbol 0: Strong positive trend
    Symbol 1: Weak positive trend
    Symbol 2: Negative trend
    
    Returns:
        DataFrame of returns with controlled trends
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    returns = pd.DataFrame(index=dates)
    returns['STRONG_UP'] = 0.002 + np.random.randn(n_days) * 0.005  # +0.2% avg
    returns['WEAK_UP'] = 0.0005 + np.random.randn(n_days) * 0.005   # +0.05% avg
    returns['DOWN'] = -0.001 + np.random.randn(n_days) * 0.005      # -0.1% avg
    
    return returns


class TestTSMOMInitialization:
    """Test TSMOM initialization and configuration."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        assert strategy.lookbacks == [252]
        assert strategy.skip_recent == 21
        assert strategy.standardize == "vol"
        assert strategy.signal_cap == 3.0
        assert strategy.rebalance == "W-FRI"
    
    def test_multi_lookback_initialization(self):
        """Test initialization with multiple lookback periods."""
        strategy = TSMOM(
            lookbacks=[63, 126, 252],
            skip_recent=21,
            standardize="zscore",
            signal_cap=2.0,
            rebalance="M"
        )
        
        assert strategy.lookbacks == [63, 126, 252]
        assert strategy.standardize == "zscore"
    
    def test_config_file_loading(self):
        """Test loading configuration from YAML file."""
        # This will use the actual config file if it exists
        strategy = TSMOM(config_path="configs/strategies.yaml")
        
        # Should have loaded from config
        assert strategy.lookbacks is not None
        assert strategy.skip_recent is not None
        assert strategy.signal_cap is not None


class TestTSMOMNoLookAhead:
    """Test that signals don't use future information (no look-ahead bias)."""
    
    def test_no_lookahead_basic(self):
        """Verify that signals at time t don't depend on data after t."""
        returns = create_synthetic_returns(n_days=300, n_symbols=3)
        universe = tuple(returns.columns)
        
        # Create two market snapshots: one full, one truncated
        market_full = MockMarketData(returns, universe)
        market_truncated = MockMarketData(returns.iloc[:200], universe)
        
        strategy = TSMOM(
            lookbacks=[63],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"  # Daily for testing
        )
        
        # Evaluate at day 150 (well before truncation point)
        test_date = returns.index[150]
        
        signals_full = strategy.signals(market_full, test_date)
        
        # Reset state for second run
        strategy.reset_state()
        
        signals_truncated = strategy.signals(market_truncated, test_date)
        
        # Signals should be identical - future data (after day 150) shouldn't affect them
        pd.testing.assert_series_equal(signals_full, signals_truncated, check_names=False)
    
    def test_no_lookahead_with_modified_future(self):
        """Test that modifying future returns doesn't change past signals."""
        returns = create_synthetic_returns(n_days=300, n_symbols=3)
        universe = tuple(returns.columns)
        
        # Create original market
        market_original = MockMarketData(returns.copy(), universe)
        
        # Create modified market with different future returns
        returns_modified = returns.copy()
        returns_modified.iloc[200:] = returns_modified.iloc[200:] * -10  # Drastically change future
        market_modified = MockMarketData(returns_modified, universe)
        
        strategy = TSMOM(
            lookbacks=[63],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        # Evaluate at day 150 (before modification)
        test_date = returns.index[150]
        
        signals_original = strategy.signals(market_original, test_date)
        strategy.reset_state()
        signals_modified = strategy.signals(market_modified, test_date)
        
        # Should be identical
        pd.testing.assert_series_equal(signals_original, signals_modified, check_names=False)
    
    def test_skip_recent_excludes_latest_data(self):
        """Verify that skip_recent actually excludes the most recent data."""
        # Create returns with a big jump in the last few days
        n_days = 300
        returns = create_synthetic_returns(n_days=n_days, n_symbols=2)
        
        # Add a huge return in the last 10 days that should be skipped
        returns.iloc[-10:, 0] = 0.05  # 5% daily return
        
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        # Strategy with skip_recent=21 should ignore the last 21 days
        strategy_skip = TSMOM(
            lookbacks=[63],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        # Strategy with skip_recent=0 should include all data
        strategy_no_skip = TSMOM(
            lookbacks=[63],
            skip_recent=0,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-1]
        
        signals_skip = strategy_skip.signals(market, test_date)
        strategy_no_skip.reset_state()
        signals_no_skip = strategy_no_skip.signals(market, test_date)
        
        # Signal for symbol 0 should be higher when we DON'T skip recent data
        # (because recent data has huge positive returns)
        assert signals_no_skip.iloc[0] > signals_skip.iloc[0]


class TestTSMOMRebalanceSchedule:
    """Test that signals only change on rebalance dates."""
    
    def test_signals_constant_between_rebalances(self):
        """Verify signals don't change between rebalance dates."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[63],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"  # Weekly Friday
        )
        
        # Get signals for a week of consecutive days
        test_dates = returns.index[-10:-3]  # Last week
        signals_list = []
        
        for date in test_dates:
            signals = strategy.signals(market, date)
            signals_list.append(signals)
        
        # Find which dates are Fridays (rebalance dates)
        fridays = [d for d in test_dates if d.dayofweek == 4]
        
        if len(fridays) > 0:
            # Between rebalances, signals should be identical
            for i in range(len(signals_list) - 1):
                # Check if we crossed a Friday
                crossed_friday = any(
                    test_dates[i] < friday <= test_dates[i+1] 
                    for friday in fridays
                )
                
                if not crossed_friday:
                    # Should be identical
                    pd.testing.assert_series_equal(
                        signals_list[i], 
                        signals_list[i+1],
                        check_names=False
                    )
    
    def test_signals_change_on_rebalance_dates(self):
        """Verify signals DO change on rebalance dates."""
        # Create returns with a trend that will change signals
        n_days = 150
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
        
        returns = pd.DataFrame(index=dates)
        # Early period: positive returns, later: negative
        returns['SYM1'] = 0.002  # Constant positive
        returns.loc[dates[100]:, 'SYM1'] = -0.002  # Switch to negative
        
        universe = ('SYM1',)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[63],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Find Friday dates in our range
        fridays = [d for d in dates if d.dayofweek == 4]
        
        # Get signals on multiple rebalance dates
        signals_over_time = []
        for friday in fridays[10:20]:  # Sample of Fridays
            sig = strategy.signals(market, friday)
            signals_over_time.append(sig.iloc[0])
        
        # Signals should vary over time (not all identical)
        assert len(set(np.round(signals_over_time, 4))) > 1
    
    def test_monthly_rebalance_schedule(self):
        """Test monthly rebalancing."""
        returns = create_synthetic_returns(n_days=200, n_symbols=2)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[63],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="M"  # Monthly
        )
        
        # Get signals on consecutive days in mid-month
        mid_month_dates = [d for d in returns.index if d.day >= 10 and d.day <= 20][:5]
        
        signals_list = []
        for date in mid_month_dates:
            signals = strategy.signals(market, date)
            signals_list.append(signals)
        
        # All should be identical (no month-end in this range)
        for i in range(len(signals_list) - 1):
            pd.testing.assert_series_equal(
                signals_list[i],
                signals_list[i+1],
                check_names=False
            )


class TestTSMOMMonotoneRelation:
    """Test monotone relationship between past returns and signals."""
    
    def test_positive_return_positive_signal(self):
        """Assets with positive past returns should have positive signals."""
        returns = create_trending_returns(n_days=300, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-1]
        signals = strategy.signals(market, test_date)
        
        # STRONG_UP had positive returns, should have positive signal
        assert signals['STRONG_UP'] > 0
        
        # DOWN had negative returns, should have negative signal
        assert signals['DOWN'] < 0
    
    def test_stronger_return_stronger_signal(self):
        """Stronger past return should lead to stronger signal."""
        returns = create_trending_returns(n_days=300, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-1]
        signals = strategy.signals(market, test_date)
        
        # STRONG_UP should have higher signal than WEAK_UP
        assert signals['STRONG_UP'] > signals['WEAK_UP']
        
        # WEAK_UP should have higher signal than DOWN
        assert signals['WEAK_UP'] > signals['DOWN']
    
    def test_monotone_across_lookbacks(self):
        """Test monotonicity with different lookback periods."""
        # Create asset with consistently positive returns
        n_days = 400
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        returns = pd.DataFrame(index=dates)
        returns['UP'] = 0.001 + np.random.randn(n_days) * 0.002  # Positive drift
        returns['FLAT'] = np.random.randn(n_days) * 0.002  # No drift
        
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[63, 126, 252],  # Blend of lookbacks
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]  # Well before end
        signals = strategy.signals(market, test_date)
        
        # UP should have positive signal (positive returns)
        assert signals['UP'] > 0
        
        # UP should be significantly stronger than FLAT
        assert signals['UP'] > signals['FLAT']


class TestTSMOMStandardization:
    """Test signal standardization and capping."""
    
    def test_signal_capping(self):
        """Test that signals are capped to ±signal_cap."""
        # Create extreme returns that would produce large signals
        # Need enough data: 252 lookback + 21 skip + buffer
        n_days = 350
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        returns = pd.DataFrame(index=dates)
        returns['EXTREME_UP'] = 0.005  # 0.5% daily return (extreme)
        returns['EXTREME_DOWN'] = -0.005
        returns['NORMAL'] = np.random.randn(n_days) * 0.005
        
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        signal_cap = 2.5
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=signal_cap,
            rebalance="D"
        )
        
        # Use a date that's definitely in the index
        test_date = returns.index[-1]
        signals = strategy.signals(market, test_date)
        
        # All signals should be within [-signal_cap, +signal_cap]
        assert (signals.abs() <= signal_cap).all()
        assert signals.max() <= signal_cap
        assert signals.min() >= -signal_cap
    
    def test_zscore_standardization(self):
        """Test z-score standardization produces reasonable distribution."""
        # Need enough data: 252 lookback + 21 skip + buffer
        returns = create_synthetic_returns(n_days=350, n_symbols=6)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="zscore",
            signal_cap=3.0,
            rebalance="D"
        )
        
        # Use last date to ensure enough history
        test_date = returns.index[-1]
        signals = strategy.signals(market, test_date)
        
        # Z-score should have roughly mean 0
        assert abs(signals.mean()) < 0.5
        
        # Should use reasonable range
        assert signals.std() > 0
    
    def test_vol_standardization(self):
        """Test volatility-based standardization."""
        # Need enough data: 252 lookback + 21 skip + buffer
        returns = create_synthetic_returns(n_days=350, n_symbols=6)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        # Use last date to ensure enough history
        test_date = returns.index[-1]
        signals = strategy.signals(market, test_date)
        
        # Signals should be finite
        assert signals.notna().all()
        assert np.isfinite(signals).all()
    
    def test_cap_affects_extreme_signals(self):
        """Test that capping actually affects extreme signals."""
        # Create returns that will produce extreme signals
        # Need enough data: 252 lookback + 21 skip + buffer
        n_days = 350
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        returns = pd.DataFrame(index=dates)
        returns['EXTREME'] = 0.005  # Large daily return
        returns['NORMAL'] = np.random.randn(n_days) * 0.005
        
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        # Strategy with large cap
        strategy_large_cap = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=10.0,  # Large cap
            rebalance="D"
        )
        
        # Strategy with small cap
        strategy_small_cap = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=1.0,  # Small cap
            rebalance="D"
        )
        
        # Use last date to ensure enough history
        test_date = returns.index[-1]
        
        signals_large = strategy_large_cap.signals(market, test_date)
        strategy_small_cap.reset_state()
        signals_small = strategy_small_cap.signals(market, test_date)
        
        # The EXTREME asset signal should be capped in small_cap strategy
        assert abs(signals_small['EXTREME']) <= 1.0
        
        # And it should be different (smaller) than with large cap
        assert abs(signals_small['EXTREME']) < abs(signals_large['EXTREME'])


class TestTSMOMEdgeCases:
    """Test edge cases and error handling."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data for lookback."""
        # Only 30 days of data, but lookback is 252
        returns = create_synthetic_returns(n_days=30, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-1]
        signals = strategy.signals(market, test_date)
        
        # Should return a Series (possibly with NaN values)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(universe)
    
    def test_empty_returns(self):
        """Test handling of empty returns data."""
        returns = pd.DataFrame()  # Empty
        universe = ('SYM1', 'SYM2')
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = pd.Timestamp('2023-01-15')
        signals = strategy.signals(market, test_date)
        
        # Should return a Series with the universe symbols
        assert isinstance(signals, pd.Series)
        assert list(signals.index) == list(universe)
    
    def test_single_symbol(self):
        """Test with a single symbol."""
        returns = create_synthetic_returns(n_days=300, n_symbols=1)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        assert len(signals) == 1
        assert isinstance(signals.iloc[0], (int, float))
    
    def test_reset_state(self):
        """Test that reset_state clears internal caches."""
        returns = create_synthetic_returns(n_days=200, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[63],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Generate signals
        test_date = returns.index[-10]
        signals1 = strategy.signals(market, test_date)
        
        # State should be set
        assert strategy._last_rebalance is not None
        assert strategy._last_signals is not None
        
        # Reset
        strategy.reset_state()
        
        # State should be cleared
        assert strategy._last_rebalance is None
        assert strategy._last_signals is None
        assert strategy._rebalance_dates is None


class TestTSMOMAPI:
    """Test public API methods."""
    
    def test_describe(self):
        """Test describe() method returns correct info."""
        strategy = TSMOM(
            lookbacks=[63, 126, 252],
            skip_recent=21,
            standardize="zscore",
            signal_cap=2.5,
            rebalance="M"
        )
        
        desc = strategy.describe()
        
        assert desc['strategy'] == 'TSMOM'
        assert desc['lookbacks'] == [63, 126, 252]
        assert desc['skip_recent'] == 21
        assert desc['standardize'] == 'zscore'
        assert desc['signal_cap'] == 2.5
        assert desc['rebalance'] == 'M'
        assert 'last_rebalance' in desc
    
    def test_fit_in_sample(self):
        """Test fit_in_sample() method (should be no-op)."""
        returns = create_synthetic_returns(n_days=300, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Should not raise error
        strategy.fit_in_sample(market, start='2020-01-01', end='2021-12-31')
        
        # Should have pre-computed rebalance dates
        assert strategy._rebalance_dates is not None
    
    def test_signals_returns_series(self):
        """Test that signals() returns a pandas Series."""
        returns = create_synthetic_returns(n_days=300, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        assert isinstance(signals, pd.Series)
        assert set(signals.index) == set(universe)
    
    def test_multiple_signals_calls(self):
        """Test calling signals() multiple times."""
        returns = create_synthetic_returns(n_days=300, n_symbols=3)
        universe = tuple(returns.columns)
        market = MockMarketData(returns, universe)
        
        strategy = TSMOM(
            lookbacks=[252],
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Call multiple times with different dates
        dates = returns.index[-20:]
        for date in dates:
            signals = strategy.signals(market, date)
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(universe)


class TestShortTermMomentumVariants:
    """Test Short-Term Momentum Strategy variants (canonical vs legacy)."""
    
    def test_canonical_variant_initialization(self):
        """Test canonical variant initializes with equal weights."""
        from agents.strat_momentum_short import ShortTermMomentumStrategy
        
        strategy = ShortTermMomentumStrategy(variant="canonical")
        
        # Canonical should have equal weights (1/3, 1/3, 1/3)
        assert abs(strategy.weights["ret_21"] - 1.0/3.0) < 0.01
        assert abs(strategy.weights["breakout_21"] - 1.0/3.0) < 0.01
        assert abs(strategy.weights["slope_fast"] - 1.0/3.0) < 0.01
        assert strategy.weights["reversal_filter"] == 0.0
        assert strategy.variant == "canonical"
    
    def test_legacy_variant_initialization(self):
        """Test legacy variant initializes with legacy weights."""
        from agents.strat_momentum_short import ShortTermMomentumStrategy
        
        strategy = ShortTermMomentumStrategy(variant="legacy")
        
        # Legacy should have 0.5, 0.3, 0.2 weights
        assert abs(strategy.weights["ret_21"] - 0.5) < 0.01
        assert abs(strategy.weights["breakout_21"] - 0.3) < 0.01
        assert abs(strategy.weights["slope_fast"] - 0.2) < 0.01
        assert strategy.weights["reversal_filter"] == 0.0
        assert strategy.variant == "legacy"
    
    def test_default_variant_is_canonical(self):
        """Test that default variant is canonical."""
        from agents.strat_momentum_short import ShortTermMomentumStrategy
        
        strategy = ShortTermMomentumStrategy()
        
        assert strategy.variant == "canonical"
        assert abs(strategy.weights["ret_21"] - 1.0/3.0) < 0.01
    
    def test_invalid_variant_raises_error(self):
        """Test that invalid variant raises error."""
        from agents.strat_momentum_short import ShortTermMomentumStrategy
        
        with pytest.raises(ValueError):
            ShortTermMomentumStrategy(variant="invalid")
    
    def test_explicit_weights_override_variant(self):
        """Test that explicit weights override variant defaults."""
        from agents.strat_momentum_short import ShortTermMomentumStrategy
        
        custom_weights = {
            "ret_21": 0.6,
            "breakout_21": 0.2,
            "slope_fast": 0.2
        }
        
        strategy = ShortTermMomentumStrategy(
            variant="canonical",
            weights=custom_weights
        )
        
        # Weights should be normalized but preserve ratios
        total = sum(custom_weights.values())
        assert abs(strategy.weights["ret_21"] - 0.6/total) < 0.01
        assert abs(strategy.weights["breakout_21"] - 0.2/total) < 0.01
        assert abs(strategy.weights["slope_fast"] - 0.2/total) < 0.01
    
    def test_describe_includes_variant(self):
        """Test that describe() includes variant information."""
        from agents.strat_momentum_short import ShortTermMomentumStrategy
        
        strategy_canonical = ShortTermMomentumStrategy(variant="canonical")
        strategy_legacy = ShortTermMomentumStrategy(variant="legacy")
        
        desc_canonical = strategy_canonical.describe()
        desc_legacy = strategy_legacy.describe()
        
        assert 'variant' in desc_canonical
        assert desc_canonical['variant'] == 'canonical'
        
        assert 'variant' in desc_legacy
        assert desc_legacy['variant'] == 'legacy'


class TestTSMOMMultiHorizonShortVariant:
    """Test TSMOMMultiHorizon strategy with short_variant parameter."""
    
    def test_short_variant_initialization(self):
        """Test that short_variant parameter is accepted."""
        from agents.strat_tsmom_multihorizon import TSMOMMultiHorizonStrategy
        
        strategy_canonical = TSMOMMultiHorizonStrategy(short_variant="canonical")
        strategy_legacy = TSMOMMultiHorizonStrategy(short_variant="legacy")
        
        assert strategy_canonical.short_variant == "canonical"
        assert strategy_legacy.short_variant == "legacy"
    
    def test_default_short_variant_is_legacy(self):
        """Test that default short_variant is legacy."""
        from agents.strat_tsmom_multihorizon import TSMOMMultiHorizonStrategy
        
        strategy = TSMOMMultiHorizonStrategy()
        
        assert strategy.short_variant == "legacy"
    
    def test_describe_includes_short_variant(self):
        """Test that describe() includes short_variant."""
        from agents.strat_tsmom_multihorizon import TSMOMMultiHorizonStrategy
        
        strategy_canonical = TSMOMMultiHorizonStrategy(short_variant="canonical")
        strategy_legacy = TSMOMMultiHorizonStrategy(short_variant="legacy")
        
        desc_canonical = strategy_canonical.describe()
        desc_legacy = strategy_legacy.describe()
        
        assert 'short_variant' in desc_canonical
        assert desc_canonical['short_variant'] == 'canonical'
        
        assert 'short_variant' in desc_legacy
        assert desc_legacy['short_variant'] == 'legacy'


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

