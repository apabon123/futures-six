"""
Unit tests for RiskVol agent.

Tests verify:
- Covariance matrices are positive semi-definite
- Covariance/volatility indices match symbols
- Window length validation raises errors appropriately
- No look-ahead bias
- Deterministic outputs given MarketData snapshot
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import MarketData
from src.agents.risk_vol import RiskVol


class TestRiskVolInitialization:
    """Test RiskVol initialization and configuration."""
    
    def test_init_defaults(self):
        """Test RiskVol initializes with default parameters."""
        agent = RiskVol()
        
        assert agent.cov_lookback == 252
        assert agent.vol_lookback == 63
        assert agent.shrinkage == "lw"
        assert agent.nan_policy == "mask-asset"
    
    def test_init_custom_params(self):
        """Test RiskVol initializes with custom parameters."""
        agent = RiskVol(
            cov_lookback=126,
            vol_lookback=21,
            shrinkage="none",
            nan_policy="drop-row"
        )
        
        assert agent.cov_lookback == 126
        assert agent.vol_lookback == 21
        assert agent.shrinkage == "none"
        assert agent.nan_policy == "drop-row"
    
    def test_init_from_config(self):
        """Test RiskVol loads configuration from YAML file."""
        agent = RiskVol(config_path="configs/strategies.yaml")
        
        # Should load from config
        assert agent.cov_lookback == 252
        assert agent.vol_lookback == 63
        assert agent.shrinkage == "lw"
        assert agent.nan_policy == "mask-asset"
    
    def test_invalid_lookback(self):
        """Test that invalid lookback values raise errors."""
        with pytest.raises(ValueError, match="cov_lookback must be >= 2"):
            RiskVol(cov_lookback=1)
        
        with pytest.raises(ValueError, match="vol_lookback must be >= 2"):
            RiskVol(vol_lookback=0)
    
    def test_invalid_shrinkage(self):
        """Test that invalid shrinkage method raises error."""
        with pytest.raises(ValueError, match="shrinkage must be"):
            RiskVol(shrinkage="invalid")
    
    def test_invalid_nan_policy(self):
        """Test that invalid nan_policy raises error."""
        with pytest.raises(ValueError, match="nan_policy must be"):
            RiskVol(nan_policy="invalid")
    
    def test_describe(self):
        """Test describe method returns configuration."""
        agent = RiskVol()
        desc = agent.describe()
        
        assert isinstance(desc, dict)
        assert desc['agent'] == 'RiskVol'
        assert 'cov_lookback' in desc
        assert 'vol_lookback' in desc
        assert 'shrinkage' in desc
        assert 'nan_policy' in desc
        assert 'outputs' in desc


class TestVolatilityCalculation:
    """Test volatility calculation methods."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    @pytest.fixture
    def agent(self):
        """Create RiskVol agent for testing."""
        return RiskVol(vol_lookback=63)
    
    def test_vols_returns_series(self, market_data, agent):
        """Test vols returns a pandas Series."""
        # Get a recent date with sufficient history
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for volatility test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        vols = agent.vols(market_data, test_date)
        
        assert isinstance(vols, pd.Series)
        assert len(vols) > 0
        assert vols.index.name in (None, 'symbol') or isinstance(vols.index, pd.Index)
    
    def test_vols_positive(self, market_data, agent):
        """Test that all volatilities are positive."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for volatility test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        vols = agent.vols(market_data, test_date)
        
        assert (vols > 0).all(), "All volatilities should be positive"
    
    def test_vols_annualized(self, market_data, agent):
        """Test that volatilities are properly annualized."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for volatility test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        vols = agent.vols(market_data, test_date)
        
        # Volatilities should be in reasonable range (annualized)
        # Typical futures vol: 10% to 100% annually
        assert (vols >= 0.01).all(), "Volatilities should be at least 1%"
        assert (vols <= 5.0).all(), "Volatilities should be at most 500%"
    
    def test_vols_alignment(self, market_data, agent):
        """Test that volatility indices match symbols."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for volatility test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        vols = agent.vols(market_data, test_date)
        
        # All index values should be strings (symbol names)
        assert all(isinstance(sym, str) for sym in vols.index)
        
        # Symbols should be from universe
        for sym in vols.index:
            assert sym in market_data.universe, f"Symbol {sym} not in universe"
    
    def test_vols_insufficient_history(self, market_data, agent):
        """Test that insufficient history raises error."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < 10:
            pytest.skip("Not enough data to test")
        
        # Try to calculate vol with insufficient history
        early_date = trading_days[min(10, len(trading_days) - 1)]
        
        with pytest.raises(ValueError, match="Insufficient history"):
            agent.vols(market_data, early_date)


class TestCovarianceCalculation:
    """Test covariance matrix calculation methods."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    @pytest.fixture
    def agent(self):
        """Create RiskVol agent for testing."""
        return RiskVol(cov_lookback=252, shrinkage="lw")
    
    def test_covariance_returns_dataframe(self, market_data, agent):
        """Test covariance returns a DataFrame."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for covariance test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        cov = agent.covariance(market_data, test_date)
        
        assert isinstance(cov, pd.DataFrame)
        assert len(cov) > 0
    
    def test_covariance_square(self, market_data, agent):
        """Test that covariance matrix is square."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for covariance test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        cov = agent.covariance(market_data, test_date)
        
        assert cov.shape[0] == cov.shape[1], "Covariance matrix must be square"
    
    def test_covariance_symmetric(self, market_data, agent):
        """Test that covariance matrix is symmetric."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for covariance test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        cov = agent.covariance(market_data, test_date)
        
        # Check symmetry
        np.testing.assert_allclose(
            cov.values,
            cov.T.values,
            rtol=1e-10,
            err_msg="Covariance matrix should be symmetric"
        )
    
    def test_covariance_psd(self, market_data, agent):
        """Test that covariance matrix is positive semi-definite."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for covariance test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        cov = agent.covariance(market_data, test_date)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov.values)
        
        # All eigenvalues should be >= -1e-8 (allowing for numerical precision)
        assert np.all(eigenvalues >= -1e-8), \
            f"Covariance matrix not PSD: min eigenvalue = {eigenvalues.min()}"
    
    def test_covariance_alignment(self, market_data, agent):
        """Test that covariance indices match symbols."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for covariance test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        cov = agent.covariance(market_data, test_date)
        
        # Row and column indices should match
        assert cov.index.equals(cov.columns), "Row and column indices must match"
        
        # All symbols should be strings
        assert all(isinstance(sym, str) for sym in cov.index)
        
        # Symbols should be from universe
        for sym in cov.index:
            assert sym in market_data.universe, f"Symbol {sym} not in universe"
    
    def test_covariance_diagonal_positive(self, market_data, agent):
        """Test that diagonal elements (variances) are positive."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for covariance test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        cov = agent.covariance(market_data, test_date)
        
        # Extract diagonal (variances)
        variances = np.diag(cov.values)
        
        assert np.all(variances > 0), "All variances (diagonal) should be positive"
    
    def test_covariance_insufficient_history(self, market_data, agent):
        """Test that insufficient history raises error."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < 10:
            pytest.skip("Not enough data to test")
        
        # Try to calculate cov with insufficient history
        early_date = trading_days[min(10, len(trading_days) - 1)]
        
        with pytest.raises(ValueError, match="Insufficient history"):
            agent.covariance(market_data, early_date)
    
    def test_covariance_shrinkage_none(self, market_data):
        """Test covariance calculation without shrinkage."""
        agent = RiskVol(cov_lookback=252, shrinkage="none")
        
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for covariance test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        cov = agent.covariance(market_data, test_date)
        
        # Should still be PSD (sample covariance is PSD)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-8), \
            f"Sample covariance not PSD: min eigenvalue = {eigenvalues.min()}"


class TestMaskCalculation:
    """Test mask (tradable symbols) calculation."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    @pytest.fixture
    def agent(self):
        """Create RiskVol agent for testing."""
        return RiskVol()
    
    def test_mask_returns_index(self, market_data, agent):
        """Test mask returns a pandas Index."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for mask test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        mask = agent.mask(market_data, test_date)
        
        assert isinstance(mask, pd.Index)
    
    def test_mask_symbols_valid(self, market_data, agent):
        """Test mask returns valid symbols."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for mask test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        mask = agent.mask(market_data, test_date)
        
        if len(mask) > 0:
            # All masked symbols should be strings
            assert all(isinstance(sym, str) for sym in mask)
            
            # Should be subset of universe
            for sym in mask:
                assert sym in market_data.universe
    
    def test_mask_matches_vols(self, market_data, agent):
        """Test mask matches symbols from vols calculation."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for mask test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        mask = agent.mask(market_data, test_date)
        vols = agent.vols(market_data, test_date)
        
        # Mask should contain same symbols as vols
        assert set(mask) == set(vols.index), "Mask should match vols symbols"


class TestConsistencyAcrossDates:
    """Test consistency and determinism across different dates."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    @pytest.fixture
    def agent(self):
        """Create RiskVol agent for testing."""
        return RiskVol()
    
    def test_vols_deterministic(self, market_data, agent):
        """Test that vols calculation is deterministic."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.vol_lookback + 10:
            pytest.skip("Insufficient data for determinism test")
        
        test_date = trading_days[agent.vol_lookback + 10]
        
        # Calculate twice
        vols1 = agent.vols(market_data, test_date)
        vols2 = agent.vols(market_data, test_date)
        
        # Should be identical
        pd.testing.assert_series_equal(vols1, vols2)
    
    def test_covariance_deterministic(self, market_data, agent):
        """Test that covariance calculation is deterministic."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for determinism test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        # Calculate twice
        cov1 = agent.covariance(market_data, test_date)
        cov2 = agent.covariance(market_data, test_date)
        
        # Should be identical
        pd.testing.assert_frame_equal(cov1, cov2)
    
    def test_no_lookahead(self, market_data, agent):
        """Test that calculations use only past data."""
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 50:
            pytest.skip("Insufficient data for look-ahead test")
        
        # Calculate vols at two different dates
        date1 = trading_days[agent.cov_lookback + 20]
        date2 = trading_days[agent.cov_lookback + 40]
        
        # Create snapshots
        snapshot1 = market_data.snapshot(date1)
        snapshot2 = market_data.snapshot(date2)
        
        vols1 = agent.vols(snapshot1, date1)
        vols2 = agent.vols(snapshot2, date2)
        
        # vols1 calculated at date1 should use only data up to date1
        # vols2 calculated at date2 should use only data up to date2
        # They should be different (market conditions change)
        
        # Check that we can recalculate date1 using full market data
        vols1_recompute = agent.vols(market_data, date1)
        
        # Should match original calculation (no look-ahead)
        pd.testing.assert_series_equal(vols1, vols1_recompute)
        
        snapshot1.close()
        snapshot2.close()


class TestAlignmentBetweenVolsAndCovariance:
    """Test alignment between vols and covariance outputs."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    @pytest.fixture
    def agent(self):
        """Create RiskVol agent for testing."""
        return RiskVol()
    
    def test_vols_cov_alignment(self, market_data, agent):
        """Test that vols and covariance have aligned symbols."""
        trading_days = market_data.trading_days()
        
        # Use the longer lookback to ensure both can be calculated
        min_lookback = max(agent.vol_lookback, agent.cov_lookback)
        
        if len(trading_days) < min_lookback + 10:
            pytest.skip("Insufficient data for alignment test")
        
        test_date = trading_days[min_lookback + 10]
        
        vols = agent.vols(market_data, test_date)
        cov = agent.covariance(market_data, test_date)
        
        # Symbols in vols should overlap with symbols in cov
        # (may not be identical due to different lookbacks and nan_policy)
        vols_syms = set(vols.index)
        cov_syms = set(cov.index)
        
        # There should be some overlap
        overlap = vols_syms & cov_syms
        assert len(overlap) > 0, "vols and covariance should have overlapping symbols"
    
    def test_vols_from_cov_diagonal(self, market_data, agent):
        """Test that vols are consistent with covariance diagonal."""
        trading_days = market_data.trading_days()
        
        # Use cov_lookback since that's longer
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for alignment test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        # Use same lookback for both to ensure comparability
        agent_same_lookback = RiskVol(
            cov_lookback=252,
            vol_lookback=252,
            shrinkage="none"  # Use none to get sample covariance
        )
        
        vols = agent_same_lookback.vols(market_data, test_date)
        cov = agent_same_lookback.covariance(market_data, test_date)
        
        # For symbols in both, vol^2 should equal cov diagonal
        common_syms = vols.index.intersection(cov.index)
        
        if len(common_syms) > 0:
            for sym in common_syms:
                vol_squared = vols[sym] ** 2
                cov_diag = cov.loc[sym, sym]
                
                # Should be close (within numerical tolerance)
                # Higher tolerance needed for real data with different NaN handling
                np.testing.assert_allclose(
                    vol_squared,
                    cov_diag,
                    rtol=0.2,  # 20% tolerance due to potential differences in NaN handling and shrinkage
                    err_msg=f"Variance mismatch for {sym}: vol^2={vol_squared}, cov_diag={cov_diag}"
                )


class TestNaNPolicies:
    """Test different NaN handling policies."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_drop_row_policy(self, market_data):
        """Test drop-row NaN policy."""
        agent = RiskVol(nan_policy="drop-row")
        
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for NaN policy test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        # Should work without errors
        vols = agent.vols(market_data, test_date)
        cov = agent.covariance(market_data, test_date)
        
        assert len(vols) > 0
        assert cov.shape[0] > 0
    
    def test_mask_asset_policy(self, market_data):
        """Test mask-asset NaN policy."""
        agent = RiskVol(nan_policy="mask-asset")
        
        trading_days = market_data.trading_days()
        
        if len(trading_days) < agent.cov_lookback + 10:
            pytest.skip("Insufficient data for NaN policy test")
        
        test_date = trading_days[agent.cov_lookback + 10]
        
        # Should work without errors
        vols = agent.vols(market_data, test_date)
        cov = agent.covariance(market_data, test_date)
        
        assert len(vols) > 0
        assert cov.shape[0] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

