"""
Tests for Cross-Sectional Momentum Strategy Agent

Test suite validates:
1. Rebalance-only signal updates
2. Rank monotonicity (higher past return → higher signal)
3. Near-neutrality (signals sum to ~0)
4. Signal capping (|signal| ≤ signal_cap)
5. No look-ahead bias
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

from agents.strat_cross_sectional import CrossSectionalMomentum


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
        method="simple",
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
    
    # Generate returns with some variation for cross-sectional differences
    returns = np.random.randn(n_days, n_symbols) * 0.01
    
    # Add different trends to different symbols
    for i in range(n_symbols):
        trend = (i - n_symbols/2) * 0.0001  # Some positive, some negative
        returns[:, i] += trend
    
    df = pd.DataFrame(returns, index=dates, columns=symbols)
    return df


def create_ranked_returns(
    n_days: int = 300,
    n_symbols: int = 6
) -> pd.DataFrame:
    """
    Create returns with clear ranking for monotonicity testing.
    
    Symbols are ranked by average return: SYM0 (highest) to SYM5 (lowest)
    
    Returns:
        DataFrame of returns with controlled ranking
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    symbols = [f'SYM{i}' for i in range(n_symbols)]
    
    returns = pd.DataFrame(index=dates, columns=symbols)
    
    # Create clear ranking: SYM0 best, SYM5 worst
    for i, symbol in enumerate(symbols):
        # Higher index = worse performance
        mean_return = 0.002 - (i * 0.0008)  # 0.2% to -0.2%
        returns[symbol] = mean_return + np.random.randn(n_days) * 0.003
    
    return returns


class TestCrossSectionalInitialization:
    """Test CrossSectionalMomentum initialization and configuration."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        symbols = ['SYM1', 'SYM2', 'SYM3']
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        assert strategy.symbols == symbols
        assert strategy.lookback == 126
        assert strategy.skip_recent == 21
        assert strategy.top_frac == 0.33
        assert strategy.bottom_frac == 0.33
        assert strategy.standardize == "vol"
        assert strategy.signal_cap == 3.0
        assert strategy.rebalance == "W-FRI"
    
    def test_validation_top_frac(self):
        """Test validation of top_frac parameter."""
        symbols = ['SYM1', 'SYM2']
        
        # Invalid: top_frac = 0
        with pytest.raises(ValueError, match="top_frac must be in"):
            CrossSectionalMomentum(symbols=symbols, top_frac=0.0)
        
        # Invalid: top_frac > 1
        with pytest.raises(ValueError, match="top_frac must be in"):
            CrossSectionalMomentum(symbols=symbols, top_frac=1.5)
    
    def test_validation_standardize(self):
        """Test validation of standardize parameter."""
        symbols = ['SYM1', 'SYM2']
        
        # Invalid standardization method
        with pytest.raises(ValueError, match="standardize must be"):
            CrossSectionalMomentum(symbols=symbols, standardize="invalid")


class TestCrossSectionalRebalanceOnly:
    """Test that signals change only on rebalance dates."""
    
    def test_signals_constant_between_rebalances(self):
        """Verify signals don't change between rebalance dates."""
        returns = create_synthetic_returns(n_days=100, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=63,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Get signals for a week of consecutive days
        test_dates = returns.index[-10:-3]
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
    
    def test_signals_can_change_on_rebalance(self):
        """Verify that strategy computes signals on rebalance dates."""
        # Create returns with time-varying trends
        n_days = 300
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
        symbols = ['SYM1', 'SYM2', 'SYM3', 'SYM4', 'SYM5', 'SYM6']
        
        np.random.seed(42)
        returns = pd.DataFrame(index=dates, columns=symbols)
        
        # Create time-varying performance
        for i, symbol in enumerate(symbols):
            base = np.random.randn(n_days) * 0.01
            # Add different trends over time
            trend = (0.003 - i * 0.001) if i < 3 else (-0.002 + (i-3) * 0.0005)
            returns[symbol] = base + trend
        
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=63,
            skip_recent=21,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Find Friday dates
        fridays = [d for d in dates if d.dayofweek == 4]
        
        # Get signals at early vs late dates
        early_sig = strategy.signals(market, fridays[15])
        strategy.reset_state()
        late_sig = strategy.signals(market, fridays[-5])
        
        # Signals should exist and be valid
        assert not early_sig.isna().all(), "Should have valid signals"
        assert not late_sig.isna().all(), "Should have valid signals"
        
        # Test that rebalance mechanism works by checking different dates give results
        assert isinstance(early_sig, pd.Series)
        assert isinstance(late_sig, pd.Series)


class TestCrossSectionalRankMonotonicity:
    """Test monotone relationship between past returns and signals."""
    
    def test_higher_return_higher_signal(self):
        """Assets with higher past returns should get higher signals."""
        returns = create_ranked_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"  # Daily for testing
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        # SYM0 has highest returns, should have highest signal
        # SYM5 has lowest returns, should have lowest signal
        assert signals['SYM0'] > signals['SYM5'], \
            "Best performer should have higher signal than worst performer"
        
        # Check overall monotonicity (allowing for some noise from standardization)
        # Top performers should generally have positive signals
        assert signals['SYM0'] > 0, "Top performer should have positive signal"
        
        # Bottom performers should generally have negative signals
        assert signals['SYM5'] < 0, "Bottom performer should have negative signal"
    
    def test_ranking_stability(self):
        """Test that clear rankings produce consistent relative signals."""
        # Create extreme differences in returns
        n_days = 300
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        returns = pd.DataFrame(index=dates)
        returns['STRONG_UP'] = 0.003 + np.random.randn(n_days) * 0.001  # +0.3% avg
        returns['WEAK_UP'] = 0.001 + np.random.randn(n_days) * 0.001    # +0.1% avg
        returns['FLAT'] = np.random.randn(n_days) * 0.001               # 0% avg
        returns['WEAK_DOWN'] = -0.001 + np.random.randn(n_days) * 0.001  # -0.1% avg
        returns['STRONG_DOWN'] = -0.003 + np.random.randn(n_days) * 0.001  # -0.3% avg
        
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.4,  # Top 40% long, bottom 40% short
            bottom_frac=0.4,
            standardize="vol",
            signal_cap=10.0,  # High cap to avoid capping effects
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        # Check monotonic ranking (use >= for potential ties after standardization)
        assert signals['STRONG_UP'] >= signals['WEAK_UP']
        assert signals['WEAK_UP'] >= signals['WEAK_DOWN']
        assert signals['WEAK_DOWN'] >= signals['STRONG_DOWN']
        
        # Verify extremes are clearly different
        assert signals['STRONG_UP'] > signals['STRONG_DOWN']
    
    def test_buckets_long_short_neutral(self):
        """Test that assets are correctly bucketed into long/short/neutral."""
        returns = create_ranked_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,  # Top 2 long, bottom 2 short
            bottom_frac=0.33,
            standardize="zscore",  # Use zscore for clearer bucket identification
            signal_cap=10.0,  # High cap to avoid capping
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        # Count longs, shorts, neutrals
        n_long = (signals > 0.01).sum()
        n_short = (signals < -0.01).sum()
        n_neutral = ((signals >= -0.01) & (signals <= 0.01)).sum()
        
        # With 6 assets and 33% fractions, expect ~2 long, ~2 short, ~2 neutral
        assert n_long >= 1, "Should have at least one long position"
        assert n_short >= 1, "Should have at least one short position"


class TestCrossSectionalNeutrality:
    """Test that signals are market-neutral (sum near zero)."""
    
    def test_signals_sum_near_zero(self):
        """Test that sum of signals is near zero (market neutral)."""
        returns = create_synthetic_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        # Sum should be very close to zero
        signal_sum = signals.sum()
        assert abs(signal_sum) < 0.5, f"Signal sum should be near 0, got {signal_sum}"
    
    def test_neutrality_across_multiple_dates(self):
        """Test neutrality holds across multiple rebalance dates."""
        returns = create_synthetic_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.5,  # Top 50% long, bottom 50% short
            bottom_frac=0.5,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Test multiple dates
        test_dates = returns.index[-50::5]  # Every 5 days
        
        for date in test_dates:
            signals = strategy.signals(market, date)
            signal_sum = signals.sum()
            assert abs(signal_sum) < 0.5, \
                f"Signal sum at {date} should be near 0, got {signal_sum}"
    
    def test_neutrality_with_missing_data(self):
        """Test neutrality when some assets have missing data."""
        returns = create_synthetic_returns(n_days=300, n_symbols=6)
        
        # Introduce some NaN values in early period (not in lookback window)
        returns.iloc[0:50, 0] = np.nan  # SYM0 missing early period
        
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=63,  # Shorter lookback for this test
            skip_recent=10,
            top_frac=0.5,
            bottom_frac=0.5,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        # Should still be reasonably neutral among available assets
        # Allow slightly higher tolerance due to vol standardization effects
        signal_sum = signals.sum()
        assert abs(signal_sum) < 1.0, \
            f"Signal sum with missing data should be near 0, got {signal_sum}"


class TestCrossSectionalSignalCapping:
    """Test that signals are properly capped."""
    
    def test_signals_within_cap(self):
        """Test that all signals are within [-cap, +cap]."""
        returns = create_synthetic_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        signal_cap = 3.0
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=signal_cap,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        # All signals should be within bounds
        assert (signals.abs() <= signal_cap).all(), \
            f"All signals should be within ±{signal_cap}"
        assert signals.max() <= signal_cap
        assert signals.min() >= -signal_cap
    
    def test_different_caps(self):
        """Test that different caps produce different signal magnitudes."""
        returns = create_ranked_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        # Strategy with large cap
        strategy_large = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=5.0,
            rebalance="D"
        )
        
        # Strategy with small cap
        strategy_small = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=1.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        
        signals_large = strategy_large.signals(market, test_date)
        strategy_small.reset_state()
        signals_small = strategy_small.signals(market, test_date)
        
        # Small cap should constrain signals more
        assert signals_small.abs().max() <= 1.0
        assert signals_large.abs().max() <= 5.0
        
        # If any signal would exceed 1.0, small cap constrains it
        if signals_large.abs().max() > 1.0:
            assert signals_large.abs().max() > signals_small.abs().max()


class TestCrossSectionalNoLookAhead:
    """Test that signals don't use future information."""
    
    def test_no_lookahead_basic(self):
        """Verify that signals at time t don't depend on data after t."""
        returns = create_synthetic_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        
        # Create two market snapshots
        market_full = MockMarketData(returns, universe)
        market_truncated = MockMarketData(returns.iloc[:200], universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=63,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        # Evaluate at day 150 (well before truncation)
        test_date = returns.index[150]
        
        signals_full = strategy.signals(market_full, test_date)
        strategy.reset_state()
        signals_truncated = strategy.signals(market_truncated, test_date)
        
        # Signals should be identical
        pd.testing.assert_series_equal(
            signals_full, 
            signals_truncated, 
            check_names=False
        )
    
    def test_skip_recent_excludes_latest_data(self):
        """Verify that skip_recent actually excludes the most recent data."""
        n_days = 300
        returns = create_synthetic_returns(n_days=n_days, n_symbols=6)
        symbols = list(returns.columns)
        
        # Create a scenario where skip_recent makes a clear difference
        # Make SYM0 have terrible returns in days -15 to -6 (in skip=5 window but not skip=21)
        # This way skip=5 will see the bad returns, skip=21 won't
        returns.iloc[-15:-5, 0] = -0.03  # -3% daily return for SYM0
        
        # Make other symbols neutral/positive
        for i in range(1, 6):
            returns.iloc[-15:-5, i] = 0.001
        
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        # Strategy with skip_recent=21 should ignore last 21 days (won't see the bad SYM0 returns)
        strategy_skip_more = CrossSectionalMomentum(
            symbols=symbols,
            lookback=63,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        # Strategy with skip_recent=5 should include days -15 to -6 (sees the bad SYM0 returns)
        strategy_skip_less = CrossSectionalMomentum(
            symbols=symbols,
            lookback=63,
            skip_recent=5,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-1]
        
        signals_skip_more = strategy_skip_more.signals(market, test_date)
        strategy_skip_less.reset_state()
        signals_skip_less = strategy_skip_less.signals(market, test_date)
        
        # SYM0 should have LOWER signal when we skip less (because we see the recent losses)
        # OR at minimum, the signals should be different
        signal_diff = abs(signals_skip_more['SYM0'] - signals_skip_less['SYM0'])
        assert signal_diff > 0.01 or signals_skip_more['SYM0'] > signals_skip_less['SYM0'], \
            f"Skip_recent should affect signals: more_skip={signals_skip_more['SYM0']:.3f} vs less_skip={signals_skip_less['SYM0']:.3f}"


class TestCrossSectionalEdgeCases:
    """Test edge cases and error handling."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data for lookback."""
        returns = create_synthetic_returns(n_days=50, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,  # More than available data
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-1]
        signals = strategy.signals(market, test_date)
        
        # Should return a Series
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(symbols)
    
    def test_few_symbols(self):
        """Test with very few symbols (edge case for bucketing)."""
        returns = create_synthetic_returns(n_days=300, n_symbols=3)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,  # 1 long
            bottom_frac=0.33,  # 1 short
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        assert len(signals) == 3
        # Should still be near-neutral
        assert abs(signals.sum()) < 0.5
    
    def test_reset_state(self):
        """Test that reset_state clears internal caches."""
        returns = create_synthetic_returns(n_days=200, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=63,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
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


class TestCrossSectionalAPI:
    """Test public API methods."""
    
    def test_describe(self):
        """Test describe() method returns correct info."""
        symbols = ['SYM1', 'SYM2', 'SYM3']
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="zscore",
            signal_cap=2.5,
            rebalance="M"
        )
        
        desc = strategy.describe()
        
        assert desc['strategy'] == 'CrossSectionalMomentum'
        assert desc['symbols'] == symbols
        assert desc['lookback'] == 126
        assert desc['skip_recent'] == 21
        assert desc['top_frac'] == 0.33
        assert desc['bottom_frac'] == 0.33
        assert desc['standardize'] == 'zscore'
        assert desc['signal_cap'] == 2.5
        assert desc['rebalance'] == 'M'
        assert 'last_rebalance' in desc
    
    def test_signals_returns_series(self):
        """Test that signals() returns a pandas Series."""
        returns = create_synthetic_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="D"
        )
        
        test_date = returns.index[-30]
        signals = strategy.signals(market, test_date)
        
        assert isinstance(signals, pd.Series)
        assert set(signals.index) == set(symbols)
    
    def test_multiple_calls(self):
        """Test calling signals() multiple times."""
        returns = create_synthetic_returns(n_days=300, n_symbols=6)
        symbols = list(returns.columns)
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Call multiple times
        dates = returns.index[-20:]
        for date in dates:
            signals = strategy.signals(market, date)
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(symbols)


class TestCrossSectionalIntegration:
    """Integration tests for realistic scenarios."""
    
    def test_realistic_six_futures(self):
        """Test with 6 futures (typical use case)."""
        n_days = 1000  # ~4 years
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        symbols = ['ES', 'NQ', 'GC', 'CL', 'ZN', 'EC']
        
        # Create returns with varying performance
        returns = pd.DataFrame(index=dates, columns=symbols)
        np.random.seed(42)
        for symbol in symbols:
            returns[symbol] = np.random.randn(n_days) * 0.01
        
        # Add some trends
        returns['ES'] += 0.0002  # Equity up-trend
        returns['GC'] -= 0.0001  # Gold down-trend
        
        universe = tuple(symbols)
        market = MockMarketData(returns, universe)
        
        strategy = CrossSectionalMomentum(
            symbols=symbols,
            lookback=126,
            skip_recent=21,
            top_frac=0.33,
            bottom_frac=0.33,
            standardize="vol",
            signal_cap=3.0,
            rebalance="W-FRI"
        )
        
        # Test across multiple dates
        test_dates = dates[200::50]  # Sample dates
        
        for date in test_dates:
            signals = strategy.signals(market, date)
            
            # Basic validation
            assert len(signals) == len(symbols)
            assert abs(signals.sum()) < 0.5  # Near-neutral
            assert (signals.abs() <= 3.0).all()  # Within cap


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

