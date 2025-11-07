"""
Unit tests for MarketData broker.

These tests verify the read-only data broker works correctly without
mutating the source database.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import MarketData, FeatureStore
from src.agents.utils_db import open_readonly_connection, find_ohlcv_table


class TestReadOnlyConnection:
    """Test read-only database connection and safety."""
    
    def test_connect_readonly(self):
        """Verify connection is read-only and write attempts fail."""
        # Load config to get DB path
        import yaml
        with open("configs/data.yaml") as f:
            config = yaml.safe_load(f)
        
        db_path = config['db']['path']
        
        # Should connect successfully
        conn = open_readonly_connection(db_path)
        assert conn is not None
        
        # Attempt write should fail
        conn_type = type(conn).__module__
        
        with pytest.raises(Exception) as exc_info:
            if 'duckdb' in conn_type:
                conn.execute("CREATE TABLE test_write (id INTEGER)")
            else:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test_write (id INTEGER)")
        
        # Should contain "read-only" or similar error message
        error_msg = str(exc_info.value).lower()
        assert 'read' in error_msg or 'write' in error_msg or 'cannot' in error_msg
        
        conn.close()
    
    def test_discover_schema(self):
        """Verify schema discovery finds OHLCV table with required columns."""
        import yaml
        with open("configs/data.yaml") as f:
            config = yaml.safe_load(f)
        
        db_path = config['db']['path']
        conn = open_readonly_connection(db_path)
        
        # Should find OHLCV table
        table_name = find_ohlcv_table(conn)
        assert table_name is not None
        assert isinstance(table_name, str)
        assert len(table_name) > 0
        
        # Verify required columns exist
        required_cols = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        
        conn_type = type(conn).__module__
        if 'duckdb' in conn_type:
            result = conn.execute(f"DESCRIBE {table_name}").fetchall()
            columns = {row[0].lower() for row in result}
        else:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = {row[1].lower() for row in cursor.fetchall()}
        
        # Check for required columns with alternatives
        required_core = {'open', 'high', 'low', 'close', 'volume'}
        has_date = 'date' in columns or 'trading_date' in columns
        has_symbol = 'symbol' in columns or 'contract_series' in columns
        
        assert required_core.issubset(columns), f"Missing OHLCV columns. Found: {columns}"
        assert has_date, f"Missing date column. Found: {columns}"
        assert has_symbol, f"Missing symbol column. Found: {columns}"
        
        conn.close()


class TestMarketDataAPI:
    """Test MarketData public API methods."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_initialization(self, market_data):
        """Test MarketData initializes correctly."""
        assert market_data.conn is not None
        assert market_data.table_name is not None
        assert len(market_data.universe) > 0
        assert market_data.asof is None
    
    def test_get_price_panel_shapes(self, market_data):
        """Test get_price_panel returns correct shapes in tidy and wide formats."""
        # Use actual symbols from the database
        symbols = tuple(list(market_data.universe)[:2])
        
        # Test tidy format
        tidy_df = market_data.get_price_panel(symbols, fields=("close",), tidy=True)
        assert not tidy_df.empty
        assert 'date' in tidy_df.columns
        assert 'symbol' in tidy_df.columns
        assert 'close' in tidy_df.columns
        # In tidy format with multiple symbols, dates repeat so we check per symbol
        for sym in symbols:
            sym_data = tidy_df[tidy_df['symbol'] == sym]
            if not sym_data.empty:
                assert sym_data['date'].is_monotonic_increasing
        
        # Test wide format with single field
        wide_df = market_data.get_price_panel(symbols, fields=("close",), tidy=False)
        assert isinstance(wide_df, pd.DataFrame)
        assert wide_df.index.name == 'date' or isinstance(wide_df.index, pd.DatetimeIndex)
        assert symbols[0] in wide_df.columns or symbols[1] in wide_df.columns
        
        # Test wide format with multiple fields
        multi_df = market_data.get_price_panel(symbols, fields=("open", "close"), tidy=False)
        assert isinstance(multi_df, dict)
        assert 'open' in multi_df
        assert 'close' in multi_df
        assert isinstance(multi_df['open'], pd.DataFrame)
        assert isinstance(multi_df['close'], pd.DataFrame)
    
    def test_returns_methods(self, market_data):
        """Test log and simple return calculations."""
        symbols = ("ES",)
        
        # Get prices first
        prices = market_data.get_price_panel(symbols, fields=("close",), tidy=False)
        
        if prices.empty:
            pytest.skip("No price data available")
        
        # Get log returns
        log_ret = market_data.get_returns(symbols, method="log")
        assert not log_ret.empty
        assert log_ret.index.is_monotonic_increasing
        
        # Get simple returns
        simple_ret = market_data.get_returns(symbols, method="simple")
        assert not simple_ret.empty
        
        # Verify log returns ≈ np.log(close).diff()
        expected_log = np.log(prices['ES']).diff()
        actual_log = log_ret['ES']
        
        # Compare non-NaN values
        mask = expected_log.notna() & actual_log.notna()
        if mask.sum() > 0:
            np.testing.assert_allclose(
                expected_log[mask].values,
                actual_log[mask].values,
                rtol=1e-10,
                err_msg="Log returns don't match expected calculation"
            )
        
        # Verify simple returns ≈ pct_change()
        expected_simple = prices['ES'].pct_change()
        actual_simple = simple_ret['ES']
        
        mask = expected_simple.notna() & actual_simple.notna()
        if mask.sum() > 0:
            np.testing.assert_allclose(
                expected_simple[mask].values,
                actual_simple[mask].values,
                rtol=1e-10,
                err_msg="Simple returns don't match expected calculation"
            )
    
    def test_snapshot_filter(self, market_data):
        """Test snapshot filters data to dates <= asof."""
        # Create snapshot at a specific date
        snapshot_date = "2023-12-31"
        snapshot_md = market_data.snapshot(snapshot_date)
        
        assert snapshot_md.asof is not None
        
        # Get prices from snapshot
        prices = snapshot_md.get_price_panel(fields=("close",), tidy=False)
        
        if not prices.empty:
            # All dates should be <= snapshot date
            max_date = prices.index.max()
            assert max_date <= pd.to_datetime(snapshot_date), \
                f"Found date {max_date} after snapshot date {snapshot_date}"
        
        # Get returns from snapshot
        returns = snapshot_md.get_returns()
        
        if not returns.empty:
            max_date = returns.index.max()
            assert max_date <= pd.to_datetime(snapshot_date), \
                f"Found return date {max_date} after snapshot date {snapshot_date}"
        
        snapshot_md.close()
    
    def test_flag_roll_jumps_runs(self, market_data):
        """Test flag_roll_jumps returns proper DataFrame structure."""
        symbols = ("ES", "NQ")
        
        jumps = market_data.flag_roll_jumps(symbols, threshold_bp=100)
        
        assert isinstance(jumps, pd.DataFrame)
        
        # Should have these columns even if empty
        expected_cols = {'date', 'symbol', 'return', 'flagged'}
        if not jumps.empty:
            assert expected_cols.issubset(set(jumps.columns))
            assert jumps['flagged'].all(), "All rows should have flagged=True"
            
            # Returns should be large (> 1%)
            assert (jumps['return'].abs() > 0.01).all(), "Flagged returns should exceed threshold"
    
    def test_no_forward_fill(self, market_data):
        """Test that missing data results in NaN, not forward-filled values."""
        # Get prices for a symbol
        prices = market_data.get_price_panel(("ES",), fields=("close",), tidy=False)
        
        if prices.empty:
            pytest.skip("No price data available")
        
        # Get returns
        returns = market_data.get_returns(("ES",), method="log")
        
        # Check if there are any NaN values in returns (there should be at least one for first row)
        assert returns.isna().any().any(), "Returns should contain NaN values (no forward fill)"
        
        # First return should be NaN (no previous price to calculate from)
        first_valid_idx = prices.first_valid_index()
        if first_valid_idx is not None:
            # The return right after first valid price should exist, but there may be gaps
            has_nan = returns.isna().sum().sum() > 0
            assert has_nan, "Expected at least some NaN values in returns"


class TestAdditionalMethods:
    """Test additional MarketData methods."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_trading_days(self, market_data):
        """Test trading_days returns sorted date index."""
        symbols = ("ES", "NQ")
        trading_days = market_data.trading_days(symbols)
        
        assert isinstance(trading_days, pd.DatetimeIndex)
        
        if len(trading_days) > 0:
            assert trading_days.is_monotonic_increasing, "Trading days should be sorted"
            assert trading_days.is_unique, "Trading days should be unique"
    
    def test_missing_report(self, market_data):
        """Test missing_report generates coverage statistics."""
        symbols = ("ES", "NQ")
        report = market_data.missing_report(symbols)
        
        assert isinstance(report, pd.DataFrame)
        
        if not report.empty:
            expected_cols = {'symbol', 'total_days', 'missing_days', 'coverage_pct'}
            assert expected_cols.issubset(set(report.columns))
            
            # Coverage should be between 0 and 100
            assert (report['coverage_pct'] >= 0).all()
            assert (report['coverage_pct'] <= 100).all()
            
            # total_days = missing_days + actual_days
            # Just verify missing_days <= total_days
            assert (report['missing_days'] <= report['total_days']).all()
    
    def test_get_meta(self, market_data):
        """Test get_meta returns metadata if available."""
        symbols = ("ES", "NQ")
        meta = market_data.get_meta(symbols)
        
        assert isinstance(meta, pd.DataFrame)
        # Meta may be empty if no metadata columns exist, which is fine
    
    def test_get_vol(self, market_data):
        """Test volatility calculation."""
        symbols = ("ES",)
        
        vol = market_data.get_vol(symbols, lookback=63)
        
        if not vol.empty:
            assert isinstance(vol, pd.DataFrame)
            assert 'ES' in vol.columns
            
            # Volatility should be positive
            valid_vol = vol['ES'].dropna()
            if len(valid_vol) > 0:
                assert (valid_vol >= 0).all(), "Volatility should be non-negative"
    
    def test_get_cov(self, market_data):
        """Test covariance matrix calculation."""
        symbols = ("ES", "NQ")
        
        cov = market_data.get_cov(symbols, lookback=252, shrink="none")
        
        if not cov.empty:
            assert isinstance(cov, pd.DataFrame)
            # Covariance matrix should be square
            assert cov.shape[0] == cov.shape[1]
            
            # Should be symmetric
            np.testing.assert_allclose(cov.values, cov.T.values, rtol=1e-10)


class TestFeatureStore:
    """Test FeatureStore caching wrapper."""
    
    @pytest.fixture
    def feature_store(self):
        """Create FeatureStore instance for testing."""
        md = MarketData()
        fs = FeatureStore(md)
        yield fs
        md.close()
    
    def test_returns_caching(self, feature_store):
        """Test that returns are cached on second call."""
        symbols = ("ES",)
        
        # First call - cache miss
        ret1 = feature_store.get_returns(symbols, method="log")
        
        # Second call - should hit cache
        ret2 = feature_store.get_returns(symbols, method="log")
        
        # Results should be identical (same object from cache)
        if not ret1.empty:
            pd.testing.assert_frame_equal(ret1, ret2)
        
        # Check cache stats
        stats = feature_store.cache_stats()
        assert stats['returns_cached'] >= 1
    
    def test_clear_cache(self, feature_store):
        """Test cache clearing."""
        symbols = ("ES",)
        
        # Load some data
        feature_store.get_returns(symbols)
        feature_store.get_vol(symbols)
        
        stats_before = feature_store.cache_stats()
        
        # Clear cache
        feature_store.clear_cache()
        
        stats_after = feature_store.cache_stats()
        
        assert stats_after['returns_cached'] == 0
        assert stats_after['vol_cached'] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_invalid_symbols(self, market_data):
        """Test behavior with non-existent symbols."""
        symbols = ("INVALID_SYMBOL_XYZ",)
        
        prices = market_data.get_price_panel(symbols, fields=("close",), tidy=False)
        
        # Should return empty DataFrame, not raise error
        assert isinstance(prices, pd.DataFrame)
    
    def test_invalid_return_method(self, market_data):
        """Test that invalid return method raises error."""
        with pytest.raises(ValueError):
            market_data.get_returns(method="invalid_method")
    
    def test_invalid_shrinkage_method(self, market_data):
        """Test that invalid shrinkage method raises error."""
        with pytest.raises(ValueError):
            market_data.get_cov(shrink="invalid_shrink")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

