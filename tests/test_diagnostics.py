"""
Tests for Diagnostics & Attribution Utility

Test suite validates:
1. Metric shapes and structure
2. File output existence and paths
3. Sharpe ratio definition and calculation
4. Time alignment across panels
5. Backward compatibility with ExecSim output format
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.diagnostics import make_report


def create_synthetic_equity_curve(
    n_days: int = 252,
    start_value: float = 1.0,
    daily_return: float = 0.0005,
    volatility: float = 0.01,
    seed: int = 42
) -> pd.Series:
    """Create synthetic equity curve for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    # Generate returns with drift and volatility
    returns = np.random.randn(n_days) * volatility + daily_return
    
    # Compute equity curve
    equity = start_value * np.cumprod(1 + returns)
    
    return pd.Series(equity, index=dates, name='equity')


def create_synthetic_weights(
    dates: pd.DatetimeIndex,
    n_symbols: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """Create synthetic weights panel for testing."""
    np.random.seed(seed)
    
    symbols = [f'SYM{i}' for i in range(n_symbols)]
    
    # Random weights that sum to ~1.0
    weights = np.random.randn(len(dates), n_symbols) * 0.3
    weights = weights / np.abs(weights).sum(axis=1, keepdims=True)
    
    df = pd.DataFrame(weights, index=dates, columns=symbols)
    return df


class TestMetricShapes:
    """Test that outputs have expected columns and index lengths."""
    
    def test_metrics_dict_keys(self, tmp_path):
        """Test that metrics dict contains all expected keys."""
        equity = create_synthetic_equity_curve(n_days=100)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test1"))
        
        # Check top-level keys
        assert 'metrics' in report
        assert 'files' in report
        
        # Check all expected metric keys
        expected_metrics = [
            'cagr', 'vol', 'sharpe', 'max_drawdown', 'calmar',
            'hit_rate', 'avg_drawdown_length', 'avg_gross_exposure',
            'avg_net_exposure', 'avg_turnover', 'cost_drag'
        ]
        
        for metric in expected_metrics:
            assert metric in report['metrics'], f"Missing metric: {metric}"
    
    def test_equity_csv_shape(self, tmp_path):
        """Test that equity CSV has correct shape and columns."""
        equity = create_synthetic_equity_curve(n_days=100)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test2"))
        
        # Load equity CSV
        equity_path = Path(report['files']['equity'])
        equity_df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        
        # Check shape
        assert len(equity_df) == len(equity)
        assert 'equity' in equity_df.columns
        assert equity_df.index.name == 'date'
    
    def test_weights_csv_shape(self, tmp_path):
        """Test that weights CSV has correct shape."""
        equity = create_synthetic_equity_curve(n_days=100)
        weights = create_synthetic_weights(equity.index, n_symbols=3)
        
        results = {
            'equity_curve': equity,
            'weights': {'total': weights}
        }
        
        report = make_report(results, outdir=str(tmp_path / "test3"))
        
        # Load weights CSV
        weights_path = Path(report['files']['weights_total'])
        weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        
        # Check shape
        assert weights_df.shape == weights.shape
        assert weights_df.index.name == 'date'
        assert list(weights_df.columns) == list(weights.columns)
    
    def test_sleeve_pnl_shape(self, tmp_path):
        """Test that sleeve P&L CSV has correct shape."""
        equity = create_synthetic_equity_curve(n_days=100)
        
        # Create per-sleeve P&L
        sleeve_pnl = {
            'momentum': pd.Series(np.random.randn(len(equity)) * 0.01, index=equity.index),
            'carry': pd.Series(np.random.randn(len(equity)) * 0.008, index=equity.index),
            'value': pd.Series(np.random.randn(len(equity)) * 0.005, index=equity.index)
        }
        
        results = {
            'equity_curve': equity,
            'pnl': sleeve_pnl
        }
        
        report = make_report(results, outdir=str(tmp_path / "test4"))
        
        # Load sleeve P&L CSV
        sleeve_path = Path(report['files']['sleeve_pnl'])
        sleeve_df = pd.read_csv(sleeve_path, index_col=0, parse_dates=True)
        
        # Check shape
        assert len(sleeve_df) == len(equity)
        assert set(sleeve_df.columns) == set(sleeve_pnl.keys())
        assert sleeve_df.index.name == 'date'


class TestPathsExist:
    """Test that CSV files exist at returned paths after run."""
    
    def test_equity_file_exists(self, tmp_path):
        """Test that equity.csv exists after report generation."""
        equity = create_synthetic_equity_curve(n_days=50)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test5"))
        
        # Check file exists
        equity_path = Path(report['files']['equity'])
        assert equity_path.exists()
        assert equity_path.name == 'equity.csv'
    
    def test_all_files_exist(self, tmp_path):
        """Test that all expected files exist when full results provided."""
        equity = create_synthetic_equity_curve(n_days=100)
        weights = create_synthetic_weights(equity.index, n_symbols=3)
        
        pnl_dict = {
            'momentum': pd.Series(np.random.randn(len(equity)) * 0.01, index=equity.index),
            'carry': pd.Series(np.random.randn(len(equity)) * 0.008, index=equity.index)
        }
        
        asset_pnl = pd.DataFrame(
            np.random.randn(len(equity), 3) * 0.005,
            index=equity.index,
            columns=['SYM0', 'SYM1', 'SYM2']
        )
        
        turnover = pd.Series(np.random.rand(len(equity)) * 0.1, index=equity.index)
        costs = pd.Series(np.random.rand(len(equity)) * 0.0005, index=equity.index)
        
        results = {
            'equity_curve': equity,
            'weights': {'total': weights},
            'pnl': pnl_dict,
            'asset_pnl': asset_pnl,
            'turnover': turnover,
            'costs': costs
        }
        
        report = make_report(results, outdir=str(tmp_path / "test6"))
        
        # Check all files exist
        expected_files = ['equity', 'sleeve_pnl', 'asset_pnl', 'weights_total', 'turnover_costs']
        
        for file_key in expected_files:
            assert file_key in report['files']
            file_path = Path(report['files'][file_key])
            assert file_path.exists(), f"File not found: {file_key}"
    
    def test_files_in_correct_directory(self, tmp_path):
        """Test that files are saved in specified directory."""
        equity = create_synthetic_equity_curve(n_days=50)
        
        custom_dir = tmp_path / "custom" / "reports"
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(custom_dir))
        
        # Check that equity file is in custom directory
        equity_path = Path(report['files']['equity'])
        assert equity_path.parent == custom_dir
        assert custom_dir.exists()


class TestSharpeDefinition:
    """Test that Sharpe ratio follows correct definition."""
    
    def test_sharpe_calculation_formula(self, tmp_path):
        """Test that Sharpe = mean(daily)/std(daily)*sqrt(252)."""
        # Create equity curve with known properties
        n_days = 252
        equity = create_synthetic_equity_curve(n_days=n_days, daily_return=0.001, volatility=0.015, seed=123)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test7"))
        
        # Compute Sharpe manually
        daily_returns = equity.pct_change().dropna()
        expected_sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        
        # Compare
        actual_sharpe = report['metrics']['sharpe']
        
        assert abs(actual_sharpe - expected_sharpe) < 1e-10, \
            f"Sharpe mismatch: expected {expected_sharpe:.6f}, got {actual_sharpe:.6f}"
    
    def test_sharpe_positive_returns(self, tmp_path):
        """Test Sharpe is positive for consistently positive returns."""
        # Create equity curve that always increases
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        equity = pd.Series(np.linspace(1.0, 1.5, 100), index=dates)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test8"))
        
        # Sharpe should be positive
        assert report['metrics']['sharpe'] > 0
    
    def test_sharpe_zero_volatility(self, tmp_path):
        """Test Sharpe calculation when volatility is zero."""
        # Constant equity curve (no volatility)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        equity = pd.Series(1.0, index=dates)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test9"))
        
        # Sharpe should be 0 when vol is 0
        assert report['metrics']['sharpe'] == 0.0


class TestBackwardCompatibility:
    """Test backward compatibility with ExecSim output format."""
    
    def test_execsim_output_format(self, tmp_path):
        """Test that diagnostics handles ExecSim's output format."""
        # ExecSim returns: equity_curve, weights_panel, signals_panel, report
        equity = create_synthetic_equity_curve(n_days=100)
        weights_panel = create_synthetic_weights(equity.index, n_symbols=3)
        signals_panel = create_synthetic_weights(equity.index, n_symbols=3) * 2.0
        
        results = {
            'equity_curve': equity,
            'weights_panel': weights_panel,
            'signals_panel': signals_panel,
            'report': {'cagr': 0.15, 'sharpe': 1.2}
        }
        
        report = make_report(results, outdir=str(tmp_path / "test10"))
        
        # Should process without errors
        assert 'metrics' in report
        assert 'files' in report
        assert 'equity' in report['files']
        assert 'weights_total' in report['files']
    
    def test_dict_weights_format(self, tmp_path):
        """Test that diagnostics handles dict-based weights format."""
        equity = create_synthetic_equity_curve(n_days=100)
        
        weights_dict = {
            'momentum': create_synthetic_weights(equity.index, n_symbols=3, seed=1),
            'carry': create_synthetic_weights(equity.index, n_symbols=3, seed=2),
            'total': create_synthetic_weights(equity.index, n_symbols=3, seed=3)
        }
        
        results = {
            'equity_curve': equity,
            'weights': weights_dict
        }
        
        report = make_report(results, outdir=str(tmp_path / "test11"))
        
        # Should use 'total' weights
        assert 'weights_total' in report['files']
        
        weights_df = pd.read_csv(report['files']['weights_total'], index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(weights_df, weights_dict['total'], check_names=False, check_freq=False)


class TestMetricCalculations:
    """Test individual metric calculations."""
    
    def test_cagr_calculation(self, tmp_path):
        """Test CAGR calculation."""
        # Create 1-year equity curve with 20% return
        dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
        equity = pd.Series(np.linspace(1.0, 1.2, 252), index=dates)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test12"))
        
        # Expected CAGR ~20%
        cagr = report['metrics']['cagr']
        assert 0.18 < cagr < 0.22, f"CAGR out of expected range: {cagr:.2%}"
    
    def test_max_drawdown_calculation(self, tmp_path):
        """Test max drawdown calculation."""
        # Create equity curve with known drawdown
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        equity_values = np.ones(100)
        equity_values[0:50] = np.linspace(1.0, 1.5, 50)  # Rise to 1.5
        equity_values[50:70] = np.linspace(1.5, 1.2, 20)  # Drop to 1.2 (20% drawdown)
        equity_values[70:] = np.linspace(1.2, 1.6, 30)  # Recover to 1.6
        
        equity = pd.Series(equity_values, index=dates)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test13"))
        
        # Expected max drawdown ~-20%
        max_dd = report['metrics']['max_drawdown']
        assert -0.22 < max_dd < -0.18, f"MaxDD out of expected range: {max_dd:.2%}"
    
    def test_hit_rate_calculation(self, tmp_path):
        """Test hit rate calculation."""
        # Create equity curve where 60% of returns are positive
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        
        returns = np.random.randn(100) * 0.01
        returns[returns > -0.002] = abs(returns[returns > -0.002])  # Make ~60% positive
        
        equity = pd.Series(np.cumprod(1 + returns), index=dates)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test14"))
        
        # Hit rate should be between 0 and 1
        hit_rate = report['metrics']['hit_rate']
        assert 0.0 <= hit_rate <= 1.0
    
    def test_exposure_metrics(self, tmp_path):
        """Test gross and net exposure calculations."""
        equity = create_synthetic_equity_curve(n_days=100)
        
        # Create weights with known gross/net exposure
        dates = equity.index
        weights = pd.DataFrame({
            'SYM0': [0.5] * len(dates),
            'SYM1': [0.3] * len(dates),
            'SYM2': [-0.2] * len(dates)  # Short position
        }, index=dates)
        
        results = {
            'equity_curve': equity,
            'weights': {'total': weights}
        }
        
        report = make_report(results, outdir=str(tmp_path / "test15"))
        
        # Gross exposure = 0.5 + 0.3 + 0.2 = 1.0
        # Net exposure = |0.5 + 0.3 - 0.2| = 0.6
        assert abs(report['metrics']['avg_gross_exposure'] - 1.0) < 1e-6
        assert abs(report['metrics']['avg_net_exposure'] - 0.6) < 1e-6


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_equity_curve(self, tmp_path):
        """Test handling of empty equity curve."""
        equity = pd.Series(dtype=float)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test16"))
        
        # Should return empty metrics
        assert report['metrics']['cagr'] == 0.0
        assert report['metrics']['sharpe'] == 0.0
        assert report['files'] == {}
    
    def test_missing_equity_curve(self, tmp_path):
        """Test error when equity_curve is missing."""
        results = {}
        
        with pytest.raises(ValueError, match="equity_curve"):
            make_report(results, outdir=str(tmp_path / "test17"))
    
    def test_single_point_equity_curve(self, tmp_path):
        """Test handling of equity curve with single point."""
        dates = pd.date_range(start='2020-01-01', periods=1, freq='B')
        equity = pd.Series([1.0], index=dates)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test18"))
        
        # Should handle gracefully (no returns to compute)
        assert 'metrics' in report
    
    def test_minimal_results(self, tmp_path):
        """Test with minimal results (only equity_curve)."""
        equity = create_synthetic_equity_curve(n_days=50)
        
        results = {
            'equity_curve': equity
        }
        
        report = make_report(results, outdir=str(tmp_path / "test19"))
        
        # Should generate report with available data
        assert 'metrics' in report
        assert 'equity' in report['files']
        
        # Exposure metrics should be 0 (no weights provided)
        assert report['metrics']['avg_gross_exposure'] == 0.0
        assert report['metrics']['avg_net_exposure'] == 0.0


class TestTimeAlignment:
    """Test time alignment across panels."""
    
    def test_consistent_date_alignment(self, tmp_path):
        """Test that all outputs use consistent date indices."""
        equity = create_synthetic_equity_curve(n_days=100)
        weights = create_synthetic_weights(equity.index, n_symbols=3)
        
        results = {
            'equity_curve': equity,
            'weights': {'total': weights}
        }
        
        report = make_report(results, outdir=str(tmp_path / "test20"))
        
        # Load CSVs
        equity_df = pd.read_csv(report['files']['equity'], index_col=0, parse_dates=True)
        weights_df = pd.read_csv(report['files']['weights_total'], index_col=0, parse_dates=True)
        
        # Dates should align
        pd.testing.assert_index_equal(equity_df.index, equity.index)
        pd.testing.assert_index_equal(weights_df.index, weights.index)


class TestDeterminism:
    """Test that diagnostics produces deterministic outputs."""
    
    def test_deterministic_metrics(self, tmp_path):
        """Test that repeated calls with same inputs give same metrics."""
        equity = create_synthetic_equity_curve(n_days=100, seed=42)
        weights = create_synthetic_weights(equity.index, n_symbols=3, seed=42)
        
        results = {
            'equity_curve': equity,
            'weights': {'total': weights}
        }
        
        # Run twice
        report1 = make_report(results, outdir=str(tmp_path / "run1"))
        report2 = make_report(results, outdir=str(tmp_path / "run2"))
        
        # Metrics should be identical
        for key in report1['metrics']:
            assert report1['metrics'][key] == report2['metrics'][key], \
                f"Metric {key} differs: {report1['metrics'][key]} vs {report2['metrics'][key]}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

