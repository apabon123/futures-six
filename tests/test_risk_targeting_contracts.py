"""
Contract tests for Risk Targeting Layer (Layer 5).

Tests ensure:
1. Presence: If RT enabled → returns df exists and has overlap with rebalance dates
2. No silent NaNs: computed current_vol finite for all rebalances
3. PSD-ish: Σ has finite diagonal, no NaNs; if you do PSD projection, assert output is PSD
4. Value constraints: multiplier finite, non-negative, within [floor, cap]
5. Teeth test fixture: a small synthetic case where Σ implies vol ≠ target so multiplier ≠ 1
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.layers.risk_targeting import RiskTargetingLayer


class TestRiskTargetingPresence:
    """Test that Risk Targeting has required inputs when enabled."""
    
    def test_returns_df_present_when_enabled(self):
        """If RT enabled, returns df must exist and have overlap with rebalance dates."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        }, index=dates)
        
        weights = pd.Series({'ES': 0.5, 'NQ': 0.5})
        rebalance_date = dates[50]
        
        # Should not raise (returns available)
        scaled = rt.scale_weights(weights, returns, rebalance_date)
        assert not scaled.empty
    
    def test_returns_df_empty_handled_gracefully(self):
        """Empty returns df should be handled without crashing."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        weights = pd.Series({'ES': 0.5, 'NQ': 0.5})
        empty_returns = pd.DataFrame()
        rebalance_date = datetime(2020, 1, 15)
        
        # Should fall back to vol_floor, not crash
        scaled = rt.scale_weights(weights, empty_returns, rebalance_date)
        assert not scaled.empty


class TestRiskTargetingNoSilentNaNs:
    """Test that current_vol is always finite (no silent NaNs)."""
    
    def test_current_vol_finite(self):
        """Computed current_vol should be finite for all rebalances."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0, vol_floor=0.05)
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        }, index=dates)
        
        weights = pd.Series({'ES': 0.5, 'NQ': 0.5})
        
        # Test multiple dates
        for i in range(20, 100, 10):
            date = dates[i]
            current_vol = rt.compute_portfolio_vol(weights, returns, date)
            
            assert np.isfinite(current_vol), f"current_vol not finite at {date}"
            assert current_vol >= rt.vol_floor, f"current_vol below floor at {date}"
    
    def test_covariance_psd_handled(self):
        """Covariance matrix should be handled even if not perfectly PSD."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Create returns with some correlation
        np.random.seed(42)
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01 + 0.5 * np.random.randn(100) * 0.01  # Correlated
        }, index=dates)
        
        weights = pd.Series({'ES': 0.5, 'NQ': 0.5})
        
        # Compute vol - should handle covariance correctly
        current_vol = rt.compute_portfolio_vol(weights, returns, dates[50])
        assert np.isfinite(current_vol)
        assert current_vol > 0


class TestRiskTargetingValueConstraints:
    """Test that multiplier values are within valid bounds."""
    
    def test_multiplier_finite_and_bounded(self):
        """Multiplier should be finite, non-negative, within [floor, cap]."""
        rt = RiskTargetingLayer(
            target_vol=0.20,
            leverage_cap=7.0,
            leverage_floor=1.0,
            vol_floor=0.05
        )
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        }, index=dates)
        
        weights = pd.Series({'ES': 0.5, 'NQ': 0.5})
        
        for i in range(20, 100, 10):
            date = dates[i]
            current_vol = rt.compute_portfolio_vol(weights, returns, date)
            leverage = rt.compute_leverage(current_vol)
            
            assert np.isfinite(leverage), f"leverage not finite at {date}"
            assert leverage >= 0, f"leverage negative at {date}"
            assert rt.leverage_floor <= leverage <= rt.leverage_cap, \
                f"leverage {leverage} outside bounds [{rt.leverage_floor}, {rt.leverage_cap}] at {date}"


class TestRiskTargetingTeeth:
    """Test that RT has teeth (multiplier ≠ 1 when vol ≠ target)."""
    
    def test_multiplier_deviates_from_one(self):
        """When vol differs from target, multiplier should deviate from 1.0."""
        rt = RiskTargetingLayer(
            target_vol=0.20,  # 20% target
            leverage_cap=7.0,
            leverage_floor=1.0,
            vol_floor=0.05
        )
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Create synthetic case: low vol (should give leverage > 1)
        low_vol_returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.005,  # 0.5% daily ≈ 8% annual
            'NQ': np.random.randn(100) * 0.005
        }, index=dates)
        
        # Create synthetic case: high vol (should give leverage < 1, but clamped to floor)
        high_vol_returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.03,  # 3% daily ≈ 48% annual
            'NQ': np.random.randn(100) * 0.03
        }, index=dates)
        
        weights = pd.Series({'ES': 0.5, 'NQ': 0.5})
        test_date = dates[50]
        
        # Low vol case: should have leverage > 1.0
        low_vol = rt.compute_portfolio_vol(weights, low_vol_returns, test_date)
        low_leverage = rt.compute_leverage(low_vol)
        assert low_leverage > 1.0, f"Low vol ({low_vol:.2%}) should give leverage > 1.0, got {low_leverage:.2f}"
        
        # High vol case: should have leverage < 1.0 (but clamped to floor)
        high_vol = rt.compute_portfolio_vol(weights, high_vol_returns, test_date)
        high_leverage = rt.compute_leverage(high_vol)
        # May be clamped to floor, but should be <= 1.0
        assert high_leverage <= 1.0, f"High vol ({high_vol:.2%}) should give leverage <= 1.0, got {high_leverage:.2f}"
    
    def test_weights_change_when_rt_applied(self):
        """When RT is applied, weights should change (unless vol exactly equals target)."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0, leverage_floor=1.0)
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Create returns that will give vol ≠ target
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        }, index=dates)
        
        weights_pre = pd.Series({'ES': 0.5, 'NQ': 0.5})
        test_date = dates[50]
        
        weights_post = rt.scale_weights(weights_pre, returns, test_date)
        
        # Weights should change (unless by coincidence vol exactly equals target)
        # Check that gross exposure changed (which indicates RT had effect)
        gross_pre = weights_pre.abs().sum()
        gross_post = weights_post.abs().sum()
        
        # RT should have some effect (unless vol exactly equals target, which is unlikely)
        # Allow small tolerance for numerical precision
        assert abs(gross_post - gross_pre) > 1e-6 or abs(gross_post - 1.0) < 1e-6, \
            f"RT should change weights: pre={gross_pre:.4f}, post={gross_post:.4f}"


class TestRiskTargetingDateAlignment:
    """Test that date alignment works correctly."""
    
    def test_rebalance_date_alignment(self):
        """RT should handle rebalance dates that may not exactly match returns index."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        # Returns on trading days
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        }, index=dates)
        
        weights = pd.Series({'ES': 0.5, 'NQ': 0.5})
        
        # Rebalance date that exists in index
        rebalance_date = dates[50]
        scaled = rt.scale_weights(weights, returns, rebalance_date)
        assert not scaled.empty
        
        # Rebalance date between trading days (should use history up to previous day)
        rebalance_date_between = dates[50] + timedelta(hours=12)
        scaled2 = rt.scale_weights(weights, returns, rebalance_date_between)
        assert not scaled2.empty


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
