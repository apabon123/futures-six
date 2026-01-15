"""
Contract tests for Allocator v1 (Layer 6).

Tests ensure:
1. Required features computable when enabled (rvol/dd/corr)
2. State feature frame has finite values after warmup
3. Regime values in allowed set
4. Scalar in [min,max], finite, not NaN
5. Date alignment: scalar index aligns with returns index (daily)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.allocator.state_v1 import AllocatorStateV1
from src.allocator.regime_v1 import RegimeClassifierV1
from src.allocator.risk_v1 import RiskTransformerV1, create_risk_transformer_from_profile


class TestAllocatorV1RequiredFeatures:
    """Test that required features are computable when enabled."""
    
    def test_rvol_features_computable(self):
        """Portfolio volatility features (rvol_20d, rvol_60d) should be computable."""
        state_computer = AllocatorStateV1()
        
        # Create synthetic portfolio returns (enough for 60d lookback)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        equity_curve = (1 + portfolio_returns).cumprod()
        
        # Create asset returns (for correlation features)
        asset_returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        }, index=dates)
        
        state_df = state_computer.compute(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_returns=asset_returns
        )
        
        assert not state_df.empty, "State DataFrame should not be empty"
        assert 'port_rvol_20d' in state_df.columns, "port_rvol_20d should be present"
        assert 'port_rvol_60d' in state_df.columns, "port_rvol_60d should be present"
        
        # After warmup, should have finite values
        if len(state_df) > 0:
            assert state_df['port_rvol_20d'].notna().any(), "port_rvol_20d should have some finite values"
            assert state_df['port_rvol_60d'].notna().any(), "port_rvol_60d should have some finite values"
    
    def test_drawdown_features_computable(self):
        """Drawdown features (dd_level, dd_slope) should be computable."""
        state_computer = AllocatorStateV1()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        equity_curve = (1 + portfolio_returns).cumprod()
        asset_returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        }, index=dates)
        
        state_df = state_computer.compute(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_returns=asset_returns
        )
        
        assert 'dd_level' in state_df.columns, "dd_level should be present"
        assert 'dd_slope_10d' in state_df.columns, "dd_slope_10d should be present"
        
        if len(state_df) > 0:
            assert state_df['dd_level'].notna().any(), "dd_level should have some finite values"
    
    def test_correlation_features_computable(self):
        """Correlation features (corr_20d, corr_60d, corr_shock) should be computable."""
        state_computer = AllocatorStateV1()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        equity_curve = (1 + portfolio_returns).cumprod()
        
        # Create correlated asset returns
        np.random.seed(42)
        asset_returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01 + 0.5 * np.random.randn(100) * 0.01
        }, index=dates)
        
        state_df = state_computer.compute(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_returns=asset_returns
        )
        
        assert 'corr_20d' in state_df.columns, "corr_20d should be present"
        assert 'corr_60d' in state_df.columns, "corr_60d should be present"
        assert 'corr_shock' in state_df.columns, "corr_shock should be present"


class TestAllocatorV1StateFinite:
    """Test that state features have finite values after warmup."""
    
    def test_state_features_finite_after_warmup(self):
        """All state features should be finite after warmup period."""
        state_computer = AllocatorStateV1()
        
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        portfolio_returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
        equity_curve = (1 + portfolio_returns).cumprod()
        asset_returns = pd.DataFrame({
            'ES': np.random.randn(200) * 0.01,
            'NQ': np.random.randn(200) * 0.01
        }, index=dates)
        
        state_df = state_computer.compute(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_returns=asset_returns
        )
        
        # After warmup (60 days), all features should be finite
        if len(state_df) > 60:
            # Drop rows with any NaN (canonical rule)
            state_clean = state_df.dropna()
            
            if len(state_clean) > 0:
                # All values should be finite
                for col in state_clean.columns:
                    assert state_clean[col].apply(np.isfinite).all(), \
                        f"Column {col} has non-finite values"


class TestAllocatorV1RegimeValues:
    """Test that regime values are in allowed set."""
    
    def test_regime_values_in_allowed_set(self):
        """Regime classifications should be in {NORMAL, ELEVATED, STRESS, CRISIS}."""
        state_computer = AllocatorStateV1()
        classifier = RegimeClassifierV1()
        
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        portfolio_returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
        equity_curve = (1 + portfolio_returns).cumprod()
        asset_returns = pd.DataFrame({
            'ES': np.random.randn(200) * 0.01,
            'NQ': np.random.randn(200) * 0.01
        }, index=dates)
        
        state_df = state_computer.compute(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_returns=asset_returns
        )
        
        if not state_df.empty:
            regime = classifier.classify(state_df)
            
            if not regime.empty:
                allowed_regimes = {'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS'}
                regime_values = set(regime.unique())
                
                assert regime_values.issubset(allowed_regimes), \
                    f"Regime values {regime_values} not all in allowed set {allowed_regimes}"


class TestAllocatorV1ScalarConstraints:
    """Test that scalar values are within valid bounds."""
    
    def test_scalar_finite_and_bounded(self):
        """Scalar should be finite, in [min, max], not NaN."""
        state_computer = AllocatorStateV1()
        classifier = RegimeClassifierV1()
        transformer = create_risk_transformer_from_profile('H')
        
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        portfolio_returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
        equity_curve = (1 + portfolio_returns).cumprod()
        asset_returns = pd.DataFrame({
            'ES': np.random.randn(200) * 0.01,
            'NQ': np.random.randn(200) * 0.01
        }, index=dates)
        
        state_df = state_computer.compute(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_returns=asset_returns
        )
        
        if not state_df.empty:
            regime = classifier.classify(state_df)
            
            if not regime.empty:
                risk_scalars = transformer.transform(state_df, regime)
                
                if not risk_scalars.empty and 'risk_scalar' in risk_scalars.columns:
                    scalars = risk_scalars['risk_scalar']
                    
                    # All should be finite
                    assert scalars.apply(np.isfinite).all(), "All scalars should be finite"
                    
                    # All should be in [min, max]
                    assert (scalars >= transformer.risk_min).all(), \
                        f"Scalars should be >= {transformer.risk_min}"
                    assert (scalars <= transformer.risk_max).all(), \
                        f"Scalars should be <= {transformer.risk_max}"
                    
                    # None should be NaN
                    assert scalars.notna().all(), "No scalars should be NaN"


class TestAllocatorV1DateAlignment:
    """Test that scalar index aligns with returns index."""
    
    def test_scalar_index_alignment(self):
        """Scalar index should align with portfolio returns index (daily)."""
        state_computer = AllocatorStateV1()
        classifier = RegimeClassifierV1()
        transformer = create_risk_transformer_from_profile('H')
        
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        portfolio_returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
        equity_curve = (1 + portfolio_returns).cumprod()
        asset_returns = pd.DataFrame({
            'ES': np.random.randn(200) * 0.01,
            'NQ': np.random.randn(200) * 0.01
        }, index=dates)
        
        state_df = state_computer.compute(
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            asset_returns=asset_returns
        )
        
        if not state_df.empty:
            regime = classifier.classify(state_df)
            
            if not regime.empty:
                risk_scalars = transformer.transform(state_df, regime)
                
                if not risk_scalars.empty:
                    # Scalar index should be a subset of or equal to state_df index
                    # (after warmup, some dates may be dropped)
                    assert risk_scalars.index.isin(state_df.index).all(), \
                        "Scalar index should align with state index"
                    
                    # Scalar index should be DatetimeIndex
                    assert isinstance(risk_scalars.index, pd.DatetimeIndex), \
                        "Scalar index should be DatetimeIndex"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
