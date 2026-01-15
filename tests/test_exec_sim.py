"""
Tests for ExecSim (Backtest Orchestrator & Metrics Agent)

Test suite validates:
1. Rebalance schedule construction
2. No look-ahead bias (PnL uses only returns from t→t+1)
3. Monotone metrics (higher vol target → higher realized vol)
4. Panel shapes and alignment
5. Deterministic outputs
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.exec_sim import ExecSim


class MockMarketData:
    """Mock MarketData for testing without database dependency."""
    
    def __init__(self, returns_df: pd.DataFrame, universe: tuple):
        self.returns_df = returns_df
        self.returns_cont = returns_df  # ExecSim expects continuous returns
        self.universe = universe
        self.asof = None
    
    def get_returns(self, symbols=None, start=None, end=None, method="log", price="close"):
        """Return mock returns filtered by date."""
        df = self.returns_df.copy()
        
        if df.empty:
            return df
        
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        if symbols:
            df = df[[s for s in symbols if s in df.columns]]
        
        return df
    
    def trading_days(self, symbols=None):
        """Return trading days."""
        return self.returns_df.index


class MockStrategy:
    """Mock strategy agent for testing."""
    
    def __init__(self, signal_value: float = 1.0):
        self.signal_value = signal_value
        self.call_count = 0
    
    def signals(self, market, date):
        """Return constant signals."""
        self.call_count += 1
        return pd.Series(self.signal_value, index=market.universe)


class MockOverlay:
    """Mock vol-managed overlay for testing."""
    
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
    
    def scale(self, signals, market, date):
        """Scale signals by constant factor."""
        return signals * self.scale_factor


class MockRiskVol:
    """Mock RiskVol agent for testing."""
    
    def __init__(self, universe: tuple):
        self.universe = universe
    
    def mask(self, market, date, signals=None):
        """Return tradable symbols."""
        return pd.Index(self.universe)
    
    def vols(self, market, date):
        """Return constant volatilities."""
        return pd.Series(0.15, index=self.universe)
    
    def covariance(self, market, date, signals=None):
        """Return identity covariance matrix."""
        n = len(self.universe)
        cov = np.eye(n) * 0.15**2
        return pd.DataFrame(cov, index=self.universe, columns=self.universe)


class MockAllocator:
    """Mock allocator agent for testing."""
    
    def __init__(self):
        pass
    
    def solve(self, signals, cov, weights_prev=None):
        """Return signals as weights (pass-through)."""
        return signals.copy()


def create_synthetic_returns(
    n_days: int = 252,
    n_symbols: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """Create synthetic returns for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    symbols = [f'SYM{i}' for i in range(n_symbols)]
    
    # Small positive drift
    returns = np.random.randn(n_days, n_symbols) * 0.01 + 0.0002
    
    df = pd.DataFrame(returns, index=dates, columns=symbols)
    return df


class TestExecSimInitialization:
    """Test ExecSim initialization and configuration."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        exec_sim = ExecSim(
            rebalance="W-FRI",
            slippage_bps=0.5,
            commission_per_contract=0.0,
            cash_rate=0.0,
            position_notional_scale=1.0
        )
        
        assert exec_sim.rebalance == "W-FRI"
        assert exec_sim.slippage_bps == 0.5
        assert exec_sim.commission_per_contract == 0.0
        assert exec_sim.cash_rate == 0.0
        assert exec_sim.position_notional_scale == 1.0
    
    def test_config_file_loading(self):
        """Test loading configuration from YAML file."""
        exec_sim = ExecSim(config_path="configs/strategies.yaml")
        
        # Should have loaded from config
        assert exec_sim.rebalance is not None
        assert exec_sim.slippage_bps >= 0
    
    def test_invalid_slippage(self):
        """Test that invalid slippage raises error."""
        with pytest.raises(ValueError, match="slippage_bps must be >= 0"):
            ExecSim(slippage_bps=-1.0)
    
    def test_invalid_commission(self):
        """Test that invalid commission raises error."""
        with pytest.raises(ValueError, match="commission_per_contract must be >= 0"):
            ExecSim(commission_per_contract=-1.0)


class TestExecSimRebalanceSchedule:
    """Test rebalance schedule construction."""
    
    def test_build_rebalance_dates(self):
        """Test that rebalance dates are correctly generated."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        risk_vol = MockRiskVol(universe)
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        start = returns.index[20]
        end = returns.index[80]
        
        rebalance_dates = exec_sim._build_rebalance_dates(market, risk_vol, start, end)
        
        # Should have some Friday dates
        assert len(rebalance_dates) > 0
        
        # All dates should be Fridays (dayofweek == 4)
        assert all(d.dayofweek == 4 for d in rebalance_dates)
        
        # All dates should be in range
        assert all(start <= d <= end for d in rebalance_dates)
    
    def test_monthly_rebalance_schedule(self):
        """Test monthly rebalancing schedule."""
        returns = create_synthetic_returns(n_days=252, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        risk_vol = MockRiskVol(universe)
        
        exec_sim = ExecSim(rebalance="M")
        
        start = returns.index[0]
        end = returns.index[-1]
        
        rebalance_dates = exec_sim._build_rebalance_dates(market, risk_vol, start, end)
        
        # Should have ~12 dates for ~1 year of data
        assert 10 <= len(rebalance_dates) <= 14
    
    def test_empty_date_range(self):
        """Test handling of empty date range."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        risk_vol = MockRiskVol(universe)
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        # Start after end
        start = returns.index[80]
        end = returns.index[20]
        
        rebalance_dates = exec_sim._build_rebalance_dates(market, risk_vol, start, end)
        
        # Should be empty
        assert len(rebalance_dates) == 0


class TestExecSimBacktestLoop:
    """Test backtest execution loop."""
    
    def test_run_produces_expected_outputs(self):
        """Test that run() produces all expected output fields."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        start = returns.index[20]
        end = returns.index[80]
        
        results = exec_sim.run(market, start, end, components)
        
        # Check that all expected keys are present
        assert 'equity_curve' in results
        assert 'weights_panel' in results
        assert 'signals_panel' in results
        assert 'report' in results
    
    def test_equity_curve_is_monotone_positive_returns(self):
        """Test that equity curve increases with positive returns."""
        # Create returns that are always positive
        n_days = 100
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        symbols = ['SYM0', 'SYM1']
        
        returns = pd.DataFrame(0.005, index=dates, columns=symbols)  # 0.5% daily return
        universe = tuple(symbols)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        results = exec_sim.run(market, dates[10], dates[80], components)
        
        equity = results['equity_curve']
        
        # Equity should be increasing (all returns positive)
        assert equity.iloc[-1] > equity.iloc[0]
    
    def test_weights_panel_shape(self):
        """Test that weights panel has correct shape."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        start = returns.index[20]
        end = returns.index[80]
        
        results = exec_sim.run(market, start, end, components)
        
        weights = results['weights_panel']
        
        # Weights should have columns = symbols
        assert set(weights.columns) == set(universe)
        
        # Should have multiple rows (rebalance dates)
        assert len(weights) > 0
    
    def test_signals_panel_shape(self):
        """Test that signals panel has correct shape."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        start = returns.index[20]
        end = returns.index[80]
        
        results = exec_sim.run(market, start, end, components)
        
        signals = results['signals_panel']
        
        # Signals should have columns = symbols
        assert set(signals.columns) == set(universe)
        
        # Should have same shape as weights
        assert signals.shape == results['weights_panel'].shape


class TestExecSimNoLookAhead:
    """Test that PnL computation doesn't use future returns (no look-ahead bias)."""
    
    def test_no_future_returns_used(self):
        """Test that PnL for date t only uses returns from t→t+1."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        # Create two markets: one full, one with modified future
        market_full = MockMarketData(returns.copy(), universe)
        
        returns_modified = returns.copy()
        # Modify returns in the second half drastically
        returns_modified.iloc[60:] = returns_modified.iloc[60:] * -10
        market_modified = MockMarketData(returns_modified, universe)
        
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol_full = MockRiskVol(universe)
        risk_vol_modified = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components_full = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol_full,
            'allocator': allocator
        }
        
        components_modified = {
            'strategy': MockStrategy(signal_value=1.0),
            'overlay': MockOverlay(scale_factor=1.0),
            'risk_vol': risk_vol_modified,
            'allocator': MockAllocator()
        }
        
        # Run backtest only on first half (before modification)
        start = returns.index[20]
        end = returns.index[50]  # Before modification at 60
        
        results_full = exec_sim.run(market_full, start, end, components_full)
        results_modified = exec_sim.run(market_modified, start, end, components_modified)
        
        # Equity curves should be identical (future data shouldn't affect past PnL)
        equity_full = results_full['equity_curve']
        equity_modified = results_modified['equity_curve']
        
        # They should be close (allowing for small numerical differences)
        pd.testing.assert_series_equal(equity_full, equity_modified, check_names=False)


class TestExecSimMonotoneMetrics:
    """Test monotone relationship: higher target_vol → higher realized vol."""
    
    def test_higher_target_vol_higher_realized_vol(self):
        """Test that higher target vol leads to higher realized vol."""
        returns = create_synthetic_returns(n_days=200, n_symbols=3, seed=42)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        # Low target vol
        overlay_low = MockOverlay(scale_factor=0.5)
        
        # High target vol
        overlay_high = MockOverlay(scale_factor=2.0)
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        start = returns.index[20]
        end = returns.index[180]
        
        # Run with low vol
        components_low = {
            'strategy': strategy,
            'overlay': overlay_low,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        results_low = exec_sim.run(market, start, end, components_low)
        
        # Run with high vol (need fresh strategy/overlay instances)
        components_high = {
            'strategy': MockStrategy(signal_value=1.0),
            'overlay': overlay_high,
            'risk_vol': MockRiskVol(universe),
            'allocator': MockAllocator()
        }
        
        results_high = exec_sim.run(market, start, end, components_high)
        
        # Higher scale factor should lead to higher realized vol
        vol_low = results_low['report']['vol']
        vol_high = results_high['report']['vol']
        
        assert vol_high > vol_low


class TestExecSimMetrics:
    """Test metrics calculation."""
    
    def test_metrics_in_report(self):
        """Test that all expected metrics are in report."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        start = returns.index[20]
        end = returns.index[80]
        
        results = exec_sim.run(market, start, end, components)
        
        report = results['report']
        
        # Check all expected metrics
        assert 'cagr' in report
        assert 'vol' in report
        assert 'sharpe' in report
        assert 'max_drawdown' in report
        assert 'hit_rate' in report
        assert 'avg_turnover' in report
        assert 'avg_gross' in report
        assert 'avg_net' in report
        assert 'n_periods' in report
    
    def test_positive_returns_positive_cagr(self):
        """Test that positive returns lead to positive CAGR."""
        # All positive returns
        n_days = 100
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        symbols = ['SYM0', 'SYM1']
        
        returns = pd.DataFrame(0.003, index=dates, columns=symbols)  # 0.3% daily
        universe = tuple(symbols)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        results = exec_sim.run(market, dates[10], dates[80], components)
        
        # CAGR should be positive
        assert results['report']['cagr'] > 0
    
    def test_max_drawdown_negative_or_zero(self):
        """Test that max drawdown is <= 0."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        start = returns.index[20]
        end = returns.index[80]
        
        results = exec_sim.run(market, start, end, components)
        
        # Max drawdown should be <= 0
        assert results['report']['max_drawdown'] <= 0


class TestExecSimDeterminism:
    """Test that ExecSim produces deterministic outputs."""
    
    def test_deterministic_backtest(self):
        """Test that repeated runs with same inputs give same outputs."""
        returns = create_synthetic_returns(n_days=100, n_symbols=3, seed=42)
        universe = tuple(returns.columns)
        
        market = MockMarketData(returns.copy(), universe)
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        start = returns.index[20]
        end = returns.index[80]
        
        # Run twice with same components
        for run_num in range(2):
            strategy = MockStrategy(signal_value=1.0)
            overlay = MockOverlay(scale_factor=1.0)
            risk_vol = MockRiskVol(universe)
            allocator = MockAllocator()
            
            components = {
                'strategy': strategy,
                'overlay': overlay,
                'risk_vol': risk_vol,
                'allocator': allocator
            }
            
            if run_num == 0:
                results1 = exec_sim.run(market, start, end, components)
            else:
                results2 = exec_sim.run(market, start, end, components)
        
        # Results should be identical
        pd.testing.assert_series_equal(
            results1['equity_curve'],
            results2['equity_curve'],
            check_names=False
        )
        
        pd.testing.assert_frame_equal(
            results1['weights_panel'],
            results2['weights_panel']
        )
        
        assert results1['report']['cagr'] == results2['report']['cagr']
        assert results1['report']['vol'] == results2['report']['vol']


class TestExecSimEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_returns(self):
        """Test handling of empty returns."""
        returns = pd.DataFrame()  # Empty
        universe = ('SYM1', 'SYM2')
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-12-31')
        
        results = exec_sim.run(market, start, end, components)
        
        # Should return empty results
        assert results['equity_curve'].empty
        assert results['weights_panel'].empty
        assert results['signals_panel'].empty
    
    def test_no_rebalance_dates(self):
        """Test handling when no valid rebalance dates."""
        # Very short date range with no Fridays
        dates = pd.date_range(start='2020-01-06', periods=3, freq='B')  # Mon-Wed
        symbols = ['SYM0']
        returns = pd.DataFrame(0.001, index=dates, columns=symbols)
        universe = tuple(symbols)
        
        market = MockMarketData(returns, universe)
        strategy = MockStrategy(signal_value=1.0)
        overlay = MockOverlay(scale_factor=1.0)
        risk_vol = MockRiskVol(universe)
        allocator = MockAllocator()
        
        exec_sim = ExecSim(rebalance="W-FRI")
        
        components = {
            'strategy': strategy,
            'overlay': overlay,
            'risk_vol': risk_vol,
            'allocator': allocator
        }
        
        results = exec_sim.run(market, dates[0], dates[-1], components)
        
        # Should handle gracefully
        assert isinstance(results, dict)


class TestExecSimAPI:
    """Test public API methods."""
    
    def test_describe(self):
        """Test describe() method returns correct info."""
        exec_sim = ExecSim(
            rebalance="M",
            slippage_bps=1.0,
            commission_per_contract=2.5,
            cash_rate=0.02,
            position_notional_scale=10000
        )
        
        desc = exec_sim.describe()
        
        assert desc['agent'] == 'ExecSim'
        assert desc['rebalance'] == 'M'
        assert desc['slippage_bps'] == 1.0
        assert desc['commission_per_contract'] == 2.5
        assert desc['cash_rate'] == 0.02
        assert desc['position_notional_scale'] == 10000


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

